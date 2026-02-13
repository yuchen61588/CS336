import json
import time

from .adapters import run_train_bpe
from .common import FIXTURES_PATH,TINY_PATH, gpt2_bytes_to_unicode
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from tests.common import gpt2_bytes_to_unicode
import os
import pytest


def test_train_bpe_speed():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    确保 BPE 训练相对高效：在这个小数据集上测量训练时间，若耗时超过 1.5 秒就抛出错误。
    1.5 秒是相当宽松的上限——参考实现在我笔记本上只需 0.38 秒；而玩具级实现大约要 3 秒。
    """
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    print(end_time - start_time)
    assert end_time - start_time < 1.5


def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    current_test_dir = Path(__file__).parent.resolve()
    project_root = current_test_dir.parent
    output_dir = project_root / "merge_vocab"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    assert merges == reference_merges

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently, we'll make sure that the vocab keys and values match)
    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())

    # ==========================================
    # 3. 保存部分 (只有测试通过才会执行到这里)
    # ==========================================
    print("测试通过！结果与标准一致，正在保存 Vocab 和 Merges...")

    # 确保输出目录存在 (使用 pathlib)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 【在这里调用你的保存函数】
    # 假设你的函数需要 vocab, merges 和保存路径
    # 请将 save_my_bpe_model 替换为你实际的函数名
    save_tokenizer(
        vocab=vocab,
        merges=merges,
        output_dir=output_dir,
        model_name="test"
    )

    print(f"模型已保存至: {output_dir}")


def test_train_bpe_special_tokens(snapshot):
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    snapshot.assert_match(
        {
            "vocab_keys": set(vocab.keys()),
            "vocab_values": set(vocab.values()),
            "merges": merges,
        },
    )

def save_tokenizer(
    vocab: Dict[int, bytes],
    merges: List[Tuple[bytes, bytes]],
    output_dir: str,
    model_name: str
) -> None:
    """
    持久化保存 BPE 分词器 (转换为你想要的 GPT-2 字符串格式)
    """
    os.makedirs(output_dir, exist_ok=True)
    file_prefix = os.path.join(output_dir, model_name)

    # 1. 获取字节到 Unicode 的映射字典
    b2u = gpt2_bytes_to_unicode()

    # 2. 保存 Vocab: 格式为 {"Ġt": 123, "a": 64, ...}
    vocab_export = {}
    for token_id, token_bytes in vocab.items():
        # 将每个 byte 映射为对应的 unicode 字符并拼接为字符串
        token_str = "".join([b2u[b] for b in token_bytes])
        vocab_export[token_str] = token_id

    vocab_path = f"{file_prefix}_vocab.json"
    print(f"Saving vocab to: {vocab_path}")
    with open(vocab_path, "w", encoding="utf-8") as f:
        # ensure_ascii=False 确保直接输出 Ġ 这类字符而不是 \uXXXX
        json.dump(vocab_export, f, ensure_ascii=False)

    # 3. 保存 Merges: 格式为 "Ġ t\n"
    merges_path = f"{file_prefix}_merges.txt"
    print(f"Saving merges to: {merges_path}")
    with open(merges_path, "w", encoding="utf-8") as f:
        for p0, p1 in merges:
            str0 = "".join([b2u[b] for b in p0])
            str1 = "".join([b2u[b] for b in p1])
            f.write(f"{str0} {str1}\n")
@pytest.mark.skip(reason="已通过,分词完成")
def test_train_bpe_tinystory():
    
    input_path = TINY_PATH / "TinyStoriesV2-GPT4-train.txt"
    current_test_dir = Path(__file__).parent.resolve()
    project_root = current_test_dir.parent
    output_dir = project_root / "merge_vocab"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    save_tokenizer(
        vocab=vocab,
        merges=merges,
        output_dir=output_dir,
        model_name="TinyStoriesV2-GPT4-train"
    )
@pytest.mark.skip(reason="已通过,分词完成")
def test_train_bpe_tinystory():
    
    input_path = TINY_PATH / "TinyStoriesV2-GPT4-valid.txt"
    current_test_dir = Path(__file__).parent.resolve()
    project_root = current_test_dir.parent
    output_dir = project_root / "merge_vocab"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=4096,
        special_tokens=["<|endoftext|>"],
    )

    save_tokenizer(
        vocab=vocab,
        merges=merges,
        output_dir=output_dir,
        model_name="TinyStoriesV2-GPT4-valid")


    
@pytest.mark.skip(reason="耗时太长，已通过，当前专注于大数据集处理")
def test_train_bpe_openWebText():
    
    input_path = TINY_PATH / "owt_train.txt"
    current_test_dir = Path(__file__).parent.resolve()
    project_root = current_test_dir.parent
    output_dir = project_root / "merge_vocab"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )

    save_tokenizer(
        vocab=vocab,
        merges=merges,
        output_dir=output_dir,
        model_name="owt_train"
    )
@pytest.mark.skip(reason="已数据集过大，使用服务器完成")
def test_train_bpe_openWebText100MB():
    
    input_path = TINY_PATH / "owt_train.txt"
    current_test_dir = Path(__file__).parent.resolve()
    project_root = current_test_dir.parent
    output_dir = project_root / "merge_vocab"
    
    data_dir = project_root / "data"
    sampled_input_path = data_dir / "owt_train_100mb.txt" # 采样文件存放在 data 目录

    # 2. 采样 100MB 数据逻辑 (如果文件不存在则进行分割)
    target_size_bytes = 100 * 1024 * 1024  # 100MB
    if not sampled_input_path.exists():
        if not input_path.exists():
            pytest.fail(f"原始数据文件不存在: {input_path}，请检查路径。")
            
        print(f"正在从原始数据采样 100MB 到: {sampled_input_path}...")
        with open(input_path, 'rb') as f_in:
            chunk = f_in.read(target_size_bytes)
            with open(sampled_input_path, 'wb') as f_out:
                f_out.write(chunk)
    else:
        print(f"检测到已存在的采样文件: {sampled_input_path}，跳过采样步骤。")
    
    print(f"开始 BPE 训练，输入文件: {sampled_input_path.name}")
    vocab, merges = run_train_bpe(
        input_path=str(sampled_input_path),
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )

    save_tokenizer(
        vocab=vocab,
        merges=merges,
        output_dir=output_dir,
        model_name="owt_train_100MB"
    )
