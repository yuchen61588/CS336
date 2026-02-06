from __future__ import annotations

import os
import json
import regex as re
import heapq
import multiprocessing
import tqdm
from functools import partial
from typing import List, Dict, Tuple, Optional, Iterable, Iterator,Union

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# -----------------------------------------------------------------------------
# 并行化 Worker 全局变量与函数 (必须定义在类外部以支持 Pickle)
# -----------------------------------------------------------------------------
_worker_tokenizer = None

def _worker_initializer(serialized_vocab: Dict[int, List[int]],
                        merges_list: List[Tuple[bytes, bytes]],
                        special_tokens: List[str]):
    """
    子进程初始化函数：在每个 Worker 进程启动时重建 Tokenizer 实例。
    避免每处理一个 Chunk 就传递一次巨大的 Tokenizer 对象。
    """
    global _worker_tokenizer
    # 重建 Vocab (List[int] -> bytes)
    vocab = {k: bytes(v) for k, v in serialized_vocab.items()} #比 bytes更好序列化
    _worker_tokenizer = Tokenizer(vocab, merges_list, special_tokens)

def _worker_encode(text: str) -> List[int]:
    """子进程实际执行的编码任务"""
    return _worker_tokenizer.encode(text)
class Tokenizer:
    def __init__(self,max_cache: int,vocab:Dict[int,bytes],merge:List[Tuple[bytes,bytes]],special_tokens: List[str] = None, ):
        """
                初始化分词器。
                Args:
                    vocab: Token ID 到 bytes 的映射。
                    merges: 合并规则列表，顺序代表优先级。
                    special_tokens: 特殊 Token 列表 (如 <|endoftext|>)。
                """


        self.vocab = { int(k) : v for k, v in vocab.items()} # id->byte
        self.vocab_inv = {v: k for k, v in vocab.items()} #byte->id

        # 将 merges 列表转换为 ranks 字典，实现 O(1) 查找优先级
        # Key: (byte_token_1, byte_token_2), Value: Rank (越小越优先)
        self.ranks = {pair : i for i, pair in enumerate(merge)}
        self.special_tokens = special_tokens if special_tokens else []
        # 预编译正则
        self.gpt2_pat = re.compile(GPT2_SPLIT_PATTERN)
        # 构造特殊 Token 正则，用于在预分词前进行物理隔离
        # 注意，对于两个连续的，要优先匹配长的正则
        self.special_pat = None
        if self.special_tokens:
            # 必须使用 re.escape 确保特殊字符被当作字面量
            # 使用捕获组 () 确保 split 后保留分隔符
            # 排序：优先匹配较长的特殊 Token，防止 "<|endoftext|>" 被 "<|end|>" 截断
            # 是从左到右尝试匹配的，而且一旦匹配成功就立即停止（Eager Matching）。它不会自动寻找“最长”的匹配，而是寻找“最早”的匹配。
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            pattern_str = "|".join(re.escape(st) for st in sorted_specials)
            self.special_pat = re.compile(f"({pattern_str})")
        else:
            special_tokens = None

            # 缓存: 将 原始bytes片段 映射为 Token ID 列表
            # 对应 markdown 中的 cache 优化
        self.cache: Dict[bytes, List[int]] = {}
        self.MAX_CACHE = max_cache

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] = None) -> Tokenizer:
        """
        从文件加载词表和合并规则。
        """
        with open(vocab_filepath,'r',encoding='utf-8') as f:
            # json load 出来的 key 默认是 str，需要转 int
            raw_vocab = json.load(f)
            # 兼容：有些保存是将 bytes 存为 unicode string (latin-1) 或 16进制
            # 这里假设保存格式符合 assignment 要求，value 是 bytes 对应的 string 表达
            # 如果是 train_bpe.py 生成的，通常需要确保存储格式正确。
            # 为了通用性，这里假设 json 存的是 {id: "str_repr"}，需要 encode 回 bytes
            # 或者如果是 pickle/pt 则直接读取。这里遵循标准 JSON 文本格式。
            # 这里的转换逻辑取决于 save 的方式，假设 save 时是用 latin-1 解码保存的

        vocab = {}
        for k,v in raw_vocab.items():
             # JSON 不支持 bytes，通常存为 latin1 字符串
             if isinstance(v, list):
                 vocab[int(k)] = bytes(v)
             elif isinstance(v, str):
                 vocab[int(k)] = v.encode('latin-1')
             else:
                 vocab[int(k)] = v #train用的是旧版存储格式

        # 加载merge
        # merges.txt 通常每行是一个 pair，例如 "u g"
        # 或者存为 json list
        merges = []
        with open(merges_filepath,'r',encoding='utf-8') as f:
            for line_num,line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                # 变化点：解析 "[...]" 格式
                # 寻找中间的分隔符 "] [" 来切分 p0 和 p1
                split_idx = line.find('] [')
                if split_idx == -1:
                    print(f"Warning: Line {line_num + 1} format error, skipping: {line}")
                    continue
                try:
                    # 提取字符串部分: "[32, 116]"
                    p1_str = line[:split_idx + 1]
                    p2_str = line[split_idx + 1:].strip()

                    # 使用 json.loads 将字符串 "[32, 116]" 变回列表 [32, 116]
                    p1_ids = json.loads(p1_str)
                    p2_ids = json.loads(p2_str)

                    # 转回 bytes 对象，这是 Tokenizer 内部逻辑需要的格式
                    merges.append((bytes(p1_ids), bytes(p2_ids)))
                except Exception as e:
                    print(f"Warning: Failed to parse line {line_num + 1}: {line} | Error: {e}")
                    continue

        return cls(vocab, merges, special_tokens)
    def _merge_tokens(self, token_ids: List[int]) -> List[int]:
        """
        【核心优化】推理阶段的合并逻辑。
        使用 最小堆 (Heap) + 双向链表 (Linked List) 策略。
        复杂度：O(N log N) 而非 O(N^2)。
        """
        #小于2无法合并
        if len(token_ids)<2:
            return token_ids

        # 初始化双向链表结构
        # id从0开始，next表示下一个有效下标，prev指向上一个，没有就表示-1
        n = len(token_ids)
        next_pos = [i+1 for i in range(n)]
        next_pos[-1] = -1
        prev_pos = [i-1 for i in range(n)]

        # 标记当前位置的 token 是否有效 (Lazy Delete 需要用到)
        # 我们直接修改 token_ids 列表，如果合并了，原来的位置就不读了

        # 2. 初始化堆
        # 堆元素: (rank, start_index)
        # start_index 是 pair 左侧 token 的下标
        pq = []

        def get_rank(idx):
            """尝试获取 (tokens[idx], tokens[next_pos[idx]]) 的 rank"""
            if idx == -1 or next_pos[idx] == -1:
                return None

            # 从 vocab 还原 bytes 才能查 rank
            # 注意：ranks 里的 key 是 (bytes, bytes)
            # 优化：在初始化时可以将 ranks 转为 (id, id) -> rank，避免 lookup vocab
            # 但由于 assignment 要求 vocab 是 bytes，这里做一次转换，或者假设 ranks 已经做了转换
            # 为了严谨，我们查 vocab。为了性能，可以在 __init__ 里把 ranks 映射成 id。
            # 这里遵循文档的 ranks 定义。

            pair_bytes = (self.vocab[token_ids[idx]], self.vocab[token_ids[next_pos[idx]]])
            return self.ranks.get(pair_bytes)

        # 初始扫描 构建频率堆
        for i in range(n-1):
            rank = get_rank(i)
            if rank is not None:
                heapq.heappush(pq, (rank, i))

        #循环合并
        while pq:
            rank, i = heapq.heappop(pq)
            # --- 懒惰删除校验 (Lazy Check) ---
            # 检查链表连通性：如果 i 的下一个节点的上一个节点不是 i，说明 i 已经被跳过或断裂
            # 用于频率堆的假删除，就是通过跳过来不访问删除的频率堆
            if next_pos[i] == -1 or prev_pos[next_pos[i]] != i:
                continue
            # 二次校验 Rank (防止同一个位置的 Pair 被多次推入堆，rank 发生了变化)
            # 虽然 BPE 这里的 Rank 是固定的，但为了逻辑严密
            # 检验那种虽然合并了，但是还保持连接的那种（合并但是邻居变了）
            current_rank = get_rank(i)
            if current_rank != rank:
                continue

            # 执行合并
            j = next_pos[i]
            pair_bytes = (self.vocab[token_ids[i]], self.vocab[token_ids[j]])
            new_bytes = pair_bytes[0] + pair_bytes[1]
            if new_bytes in self.vocab_inv:
                new_token_id = self.vocab_inv[new_bytes]
            else:
                # 异常情况：merges 里有，但 vocab 里没有 (训练数据不一致)
                continue

            # 更新当前位置i为token 这里不使用表格的方法，使用原地修改法,前面的索引不用删
            token_ids[i] = new_token_id


            # 更新链表指针
            k = next_pos[j]
            next_pos[i] = k
            if k !=-1:
                # 已经没有下一个了 这个k就已经毫无意义了
                prev_pos[k] = i

            # 检查新产生的邻居
            # 1. 检查左邻居: (prev_pos[i], i)
            if prev_pos[i] !=-1:
                left_neighbor = prev_pos[i]
                new_rank_left = get_rank(left_neighbor)
                if new_rank_left is not None:
                    heapq.heappush(pq,(new_rank_left,left_neighbor))

            # 2. 检查右邻居: (i, next_pos[i])
            # 注意：i 现在是新 token，next_pos[i] 是原来的 k
            if next_pos[i]!=-1:
                new_rank_right = get_rank(i)
                if new_rank_left is not None:
                    heapq.heappush((pq,(new_rank_right,i)))

        # 4. 重组结果列表
        # 从头(0)或第一个有效节点开始遍历 next_pos
        result = []

        # 寻找头节点 找到prev[i] = -1的 0永远是链表头
        cur = 0
        while cur!=-1:
            result.append(token_ids[curr])
            curr = next_pos[curr]

        return result

    def encode(self,text:str)->List[int]:
        """
        并行编码多个文本字符串（适用于处理数据集，如 OpenWebText）。
        Args:
            texts: 字符串列表 (List[str])
            num_processes: 进程数，默认使用 CPU 核心数 - 1

        Returns:
            List[int]: 所有文本展平后的 Token ID 列表
        """
        if not text:
            return []
            # 1. 特殊 Token 切分
        if self.special_pat:
             # 使用 split 后，保留分隔符(在括号里)，所以列表里会有 special tokens
            raw_segments = self.special_pat.split(text)
        else:
            raw_segments = [text]

        ids = []
        for segment in raw_segments:
            if not segment: #K空的
                continue
            if segment in self.special_tokens:
                seg_bytes = segment.encode('utf-8')
                if seg_bytes in self.vocab_inv:
                    ids.append(self.vocab_inv[seg_bytes])
                continue #防止放到下面

            pre_tokens = self.gpt2_pat.findall(segment)
            for token_str in pre_tokens:
                token_bytes  = token_str.encode('utf-8') #一个词进行序列化

                #缓存
                if token_bytes in self.cache:
                    ids.extend(self.cache[token_bytes])
                    continue
                # 3. 转初始 ID (Bytes -> Ints)
                # 使用 vocab_inv 将每个字节映射为基础 ID (0-255)
                current_ids = [self.vocab_inv[bytes([b])] for b in token_bytes]

                # bpe合并
                merged_ids = self._merge_tokens(current_ids) #已经合并完成了
                # 更新 Cache
                if len(self.cache) < self.MAX_CACHE:  # 简单容量限制
                    self.cache[token_bytes] = merged_ids

                ids.extend(merged_ids)

                #z装入缓存
                if len(self.cache) < self.MAX_CACHE:
                    self.cache[token_bytes] = merged_ids

        return ids

    def decode(self,ids:List[int])->str:
        """解码：IDs -> Bytes -> String (with replacement)"""
        byte_parts = []
        for i in ids:
            if i in self.vocab:
                byte_parts.append(self.vocab[i])
        all_bytes = b"".join(byte_parts)
        return all_bytes.decode('utf-8', errors='replace')
    # =========================================================================
    #  并行化接口
    # =========================================================================

    def _chunk_string(self, text: str, num_chunks: int) -> List[str]:
        """
        模仿 train_bpe.py 的逻辑，将内存中的大字符串切分为 num_chunks 份。
        关键点：
        1. 寻找最近的换行符 '\n' 作为切分点，防止切断单词。
        2. 相比 splitlines()，它保留了换行符，且不会产生数百万个小碎片。
        """
        length = len(text)
        if length == 0:
            return []

            # 基础块大小
        chunk_size = max(1, length // num_chunks)
        chunks = []
        start = 0

        for i in range(num_chunks):
            if i == num_chunks - 1:
                    # 最后一个块直接取到末尾
                end = length
            else:
                target_end = start + chunk_size
                if target_end >= length:
                    end = length
                else:
                    # 在 target_end 之后寻找第一个换行符
                    # find 的第二个参数是起始搜索位置
                    newline_pos = text.find('\n', target_end)

                    if newline_pos != -1:
                        # 切在换行符之后，包含换行符 (保留格式)
                        end = newline_pos + 1
                    else:
                        # 如果后面没有换行符了，就不得不强制切分
                        # 或者为了安全，直接读到末尾 (取决于数据分布，通常不建议强切)
                        end = length

            # 添加切片
            if start < end:
                chunks.append(text[start:end])

            start = end
            if start >= length:
                break

        return chunks

        # -------------------------------------------------------------------------
        # 更新后的 encode_parallel
        # -------------------------------------------------------------------------

    def encode_parallel(self, texts: Union[str, List[str]], num_processes: int = None) -> List[int]:
        """
                并行编码接口。

                优化点：
                1. 如果输入是 str，使用智能分块策略（保留换行符，控制任务数量）。
                2. 任务数量控制在 cpu_count * 4 左右，最大化吞吐量。
                """
        if num_processes is None:
            num_processes = max(1, multiprocessing.cpu_count() - 1)

            # --- 1. 类型兼容性与分块策略优化 ---
        if isinstance(texts, str):
            print("Input is a single string. Chunking strategies activated...")

            # 策略：将大字符串切分为 核心数 * 4 份
            # 这样既能充分利用多核，又不会因为任务太碎导致通信阻塞
            target_chunks = num_processes * 4
            texts = self._chunk_string(texts, target_chunks)

            if not texts:
                return []
            print(f"-> Split into {len(texts)} large chunks (preserving '\\n').")

        print(f"Parallel encoding {len(texts)} chunks with {num_processes} processes...")
        # 2. 序列化必要数据 (Lightweight Serialization)
        # vocab: 转为 {int: list[int]}，体积更小且兼容 pickle
        serialized_vocab = {k: list(v) for k, v in self.vocab.items()}
        # merges: 重建为有序列表，因为 ranks 字典是无序的
        merges_list = [None] * len(self.ranks) #反向查 ，使用数字索引，开销小。
        for pair, rank in self.ranks.items():
            merges_list[rank] = pair
         # 3. 启动进程池
        with multiprocessing.Pool(
                processes=num_processes,
                initializer=_worker_initializer,
                initargs=(serialized_vocab, merges_list, self.special_tokens)
        ) as pool:

            # 计算 chunksize
            chunk_size = max(1, len(texts) // (num_processes * 4))
            results = []

            # 使用 imap 保证结果顺序
            for res in tqdm.tqdm(
                    pool.imap(_worker_encode, texts, chunksize=chunk_size),
                    total=len(texts),
                    desc="Encoding"
            ):
                results.extend(res)

        return results























