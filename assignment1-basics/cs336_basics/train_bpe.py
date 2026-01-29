from __future__ import annotations

import os
import json
import regex as re
import multiprocessing
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import tqdm
import mmap
from collections import Counter

from cs336_basics.bpe_data_structures import WordObject, InvertedIndex, FrequencyBuckets

# GPT-2 使用的预分词正则：用于将文本切分为基础单词块，防止跨单词合并
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _get_chunk_boundaries(input_path: str, num_chunks: int, special_tokens: List[str]) -> List[Tuple[int, int]]:
    """
    【基于内存映射的文件读取 (Memory-Mapped I/O)】辅助函数
    #把大文件切成 N 块，每块边界落在换行符处，防止截断单词。
    作用：
        在不读取整个文件到 RAM 的情况下，计算出每个 Worker 应该处理的字节范围 (start, end)。
        它会通过 mmap 快速扫描，并确保切分点落在换行符（\n）处，防止截断单词或 UTF-8 字符。

    返回：
        List[(start_offset, end_offset)]
    """
    file_size = os.path.getsize(input_path)
    chunk_size = file_size // num_chunks
    boundaries = []
    delimiters = [b'\n']
    if special_tokens:
        for st in special_tokens:
            delimiters.append(st.encode('utf-8'))

    with open(input_path, "r+b") as f:  # 二进制模式打开
        # 使用 mmap 避免大文件 IO
        # 如果文件过大无法即使 mmap，可以改用 seek + read 小 buffer 的方式
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            start = 0
            for i in range(num_chunks):
                if i == num_chunks - 1:
                    # 最后一个块直接读到文件末尾
                    end = file_size
                else:
                    target_end = start + chunk_size
                    if target_end >= file_size:
                        end = file_size
                    else:
                        # --- 竞速逻辑开始 ---
                        # 在 target_end 之后，寻找最近的一个合法切分点
                        best_end = -1
                        min_dist = float('inf')

                        for delim in delimiters:
                            # 从 target_end 开始找 delim
                            pos = mm.find(delim, target_end)

                            if pos != -1:
                                # 计算距离
                                dist = pos - target_end
                                if dist < min_dist:
                                    min_dist = dist
                                    # 核心细节：
                                    # 如果是 \n，我们切在 \n 之后 (pos + 1)，包含换行符
                                    # 如果是 special_token，我们也切在它之后 (pos + len)，包含该 token
                                    # 这样确保 token 完整地留在当前块，或者完整地进入下一块（取决于你的偏好，这里保留在当前块比较直观）
                                    best_end = pos + len(delim)

                        # 2. 决策
                        if best_end != -1:
                            end = best_end
                        else:
                            # 极其罕见：后面既没有换行符也没有特殊token了，直接读到头
                            end = file_size
                        # --- 竞速逻辑结束 ---

                if start < end:
                    boundaries.append((start, end))
                start = end

    return boundaries  # 计算每一个文件的分界


def _pre_tokenize_worker(args) -> Dict[bytes, int]:
    """
    【Worker 进程】
    实现了：
    1. Memory-Mapped I/O 逻辑：只读取分配给自己的字节片段。
    2. Special Token Isolation：先切分特殊 Token。
    """
    input_path, start, end, pattern_str, special_tokens = args

    local_freqs = Counter()  # <--- 修正：使用 Counter
    gpt2_pat = re.compile(pattern_str)  # 规则

    # 构造特殊 Token 的切分正则 (用于隔离)
    # 使用括号 () 捕获，这样 split 后特殊 token 也会保留在列表中（如果有需要统计的话）
    # 但根据 BPE 逻辑，我们通常不希望特殊 Token 干扰普通单词的合并，所以通常是 Split 之后跳过它们，
    # 或者把它们作为单独的单位。作业要求"Special tokens... will always be kept as a single token"。
    # 这里我们用 split 将其物理隔绝。
    if special_tokens:
        special_pattern_str = '|'.join(re.escape(st) for st in special_tokens)
        special_pat = re.compile(special_pattern_str)
    else:
        special_pat = None

    # --- 1. 基于偏移量的局部读取 (IO 优化) ---
    with open(input_path, 'rb') as f:
        f.seek(start)
        # 只读取分配的大小
        chunk_bytes = f.read(end - start)
        if b'\r\n' in chunk_bytes:
            chunk_bytes = chunk_bytes.replace(b'\r\n', b'\n')
        # 解码 (如果有截断风险，上一步的 boundary 计算必须保证落在字符边界，\n 是安全的)
        text_chunk = chunk_bytes.decode('utf-8', errors='replace')

    # --- 2. 特殊 Token 的严格隔离 (Isolation) ---
    # 先按照特殊 Token 将文本切碎，形成 "纯文本片段" 列表
    # 这样 GPT-2 正则永远跑不到特殊 Token 身上，也跑不到跨越特殊 Token 的边界上
    if special_pat:
        # split 后，列表里会有 [text, special_token, text, special_token...]
        # 我们可以只对非特殊 Token 的部分跑 GPT-2 正则
        segments = special_pat.split(text_chunk)
    else:
        segments = [text_chunk]

    # --- 3. 内存友好的正则匹配 (Iterator Optimization) ---
    for segment in segments:
        if not segment: continue

        # 检查 segment 是否正好是特殊 token (因为 split 可能会把分隔符去掉，或者我们假设 split 只是为了隔离)
        # 如果 split 逻辑没有保留分隔符，我们其实不需要统计特殊 token 的频率（因为它们不参与 BPE merge）
        # 这里假设 segments 主要是纯文本
        # 如果 segment 是特殊 token 列表中的，直接跳过 (因为它不参与 BPE 训练统计)  前面的special_tokens保证special_tokens非空
        if special_tokens and segment in special_tokens:
            continue

        # 1. 只要内存允许（Chunk别太大），findall 比 finditer 快得多
        tokens_str = gpt2_pat.findall(segment)

        # 2. 批量 Encode：列表推导式通常比显式 for 循环快
        tokens_bytes = [s.encode('utf-8') for s in tokens_str]

        # 3. 批量统计：利用 C 语言实现的 update
        local_freqs.update(tokens_bytes)

    return local_freqs


def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: List[str],
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    BPE 训练主函数 (高度优化版)
    """
    # ——————————————————————————预分词————————————————————————————
    print(f"正在分析文件: {input_path}")
    # 1. 计算文件切分边界 (主进程计算，不读内容)
    num_procs = max(1, multiprocessing.cpu_count() - 1)
    target_num_chunks = num_procs * 4
    boundaries = _get_chunk_boundaries(str(input_path), target_num_chunks, special_tokens)
    # 准备 Worker 参数

    worker_args = []
    for start, end in boundaries:
        worker_args.append((str(input_path), start, end, GPT2_SPLIT_PATTERN, special_tokens))

    print(f"启动 {num_procs} 个 Worker 进行流式预分词...")
    global_freqs = defaultdict(int)

    total_chunks = len(worker_args)
    # 2. Map-Reduce 归约策略
    # 主进程只负责汇总 Dict[bytes, int]，内存压力极小
    with multiprocessing.Pool(num_procs) as pool:
        # 【修改点 2】：添加 tqdm 进度条
        # unit='chunk' 表示单位是“块”，desc 设置描述文本
        with tqdm.tqdm(total=total_chunks, desc="Pre-tokenization Workers", unit="chunk") as pbar:
            # 使用 imap_unordered 减少等待
            for local_freqs in pool.imap_unordered(_pre_tokenize_worker, worker_args):
                for k, v in local_freqs.items():
                    global_freqs[k] += v

                # 【修改点 3】：每处理完一个 Worker 的返回结果，更新一次进度
                pbar.update(1)

    # --- 第二阶段：初始化 BPE 数据结构 ---
    train_words: List[WordObject] = []  # 词表
    stats_engine = FrequencyBuckets()  # 频率桶+频率表
    inverted_index = InvertedIndex()  # 倒排索引
    # 初始词表0-255
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # 按byte排序 key,保证可复现性
    sorted_pretokens = sorted(global_freqs.items(), key=lambda x: x[0])

    print("正在构建倒排索引和频率桶...")
    for idx, (b_seq, count) in enumerate(sorted_pretokens):
        tokens = list(b_seq)
        word_obj = WordObject(tokens, count, b_seq)
        train_words.append(word_obj)
        # 找到相邻巅峰字符
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            stats_engine.update(pair, count)
            inverted_index.add_occurrence(pair, idx)
    merges: List[Tuple[bytes, bytes]] = []
    next_token_id = 256  # 扩展词表位置
    target_merges = vocab_size - 256 - len(special_tokens)  # 此表大小
    # 关键：先给特殊 Token 分配 ID
    for st in special_tokens:
        vocab[next_token_id] = st.encode('utf-8')
        next_token_id += 1
    batch_removed = defaultdict(int)
    batch_added = defaultdict(int)
    # --- 第三阶段：训练主循环 ---
    # (使用 O(1) 查找 + 懒惰更新 + 局部 diff 计算)
    with tqdm.tqdm(total=target_merges, desc="BPE Training") as pbar:
        while len(merges) < target_merges:
            # O(1) 获取最大
            best_pair = stats_engine.get_max_pair(vocab)
            if best_pair is None:
                break

            stats_engine.remove_entry(best_pair)
            # 添加词表
            token_byte_0 = vocab[best_pair[0]]  # 存的是byte对应的id ，更高效
            token_byte_1 = vocab[best_pair[1]]

            new_token_byte = token_byte_0 + token_byte_1

            vocab[next_token_id] = new_token_byte
            merges.append((token_byte_0, token_byte_1))
            # 定位
            affected_word_ids = inverted_index.get_word_ids(best_pair)

            # 批量更新
            batch_removed.clear()
            batch_added.clear()
            # 转换为 list 避免 set 迭代时修改 (虽然这里只读)
            current_ids_list = list(affected_word_ids)
            # 彻底删除这个key
            inverted_index.clear_pair(best_pair)

            for word_id in current_ids_list:
                word = train_words[word_id]
                # word.apply_merge 内部实现了 Lookahead 检查，防止 Banana 重复统计
                w_added, w_removed = word.apply_merge(best_pair, next_token_id)

                for p, c in w_removed.items():
                    batch_removed[p] += c
                for p, c in w_added.items():
                    batch_added[p] += c
                    # 懒惰更新：只增不减
                    inverted_index.add_occurrence(p, word_id)

            for pair, count in batch_removed.items():
                if pair == best_pair:
                    continue  # <--- 核心：既然已经删全家了，就别再减它了
                stats_engine.update(pair, -count)
            for pair, count in batch_added.items():
                stats_engine.update(pair, count)
                inverted_index.add_occurrence(pair, word_id)

            next_token_id += 1
            pbar.update(1)

    return vocab, merges






