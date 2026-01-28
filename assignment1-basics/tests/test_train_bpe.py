import json
import time

from .adapters import run_train_bpe
from .common import FIXTURES_PATH,TINY_PATH, gpt2_bytes_to_unicode
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import os
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
    output_dir = project_root / "output"
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
    output_dir: str,  # 变化1：指定目录
    model_name: str  # 变化2：指定模型名字（区分实验结果）
) -> None:
    """
    持久化保存 BPE 分词器
    会自动生成:
      - output_dir/model_name_vocab.json
      - output_dir/model_name_merges.txt
    """
    # 1. 自动创建目录（如果不存在）
    # 这一步非常重要，防止因为文件夹不存在而报错
    os.makedirs(output_dir, exist_ok=True)

    # 2. 拼接完整的文件路径前缀
    # 例如: output/run_v1
    file_prefix = os.path.join(output_dir, model_name)

    # 3. 保存 Vocab
    vocab_export = {k: list(v) for k, v in vocab.items()}
    vocab_path = f"{file_prefix}_vocab.json"
    print(f"Saving vocab to: {vocab_path}") # 增加一点日志
    with open(vocab_path, "w") as f:
        json.dump(vocab_export, f)

    # 4. 保存 Merges
    merges_path = f"{file_prefix}_merges.txt"
    print(f"Saving merges to: {merges_path}")
    with open(merges_path, "w", encoding="utf-8") as f:
        for p0, p1 in merges:
            f.write(f"{list(p0)} {list(p1)}\n")
def test_train_bpe_speed2():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    确保 BPE 训练相对高效：在这个小数据集上测量训练时间，若耗时超过 1.5 秒就抛出错误。
    1.5 秒是相当宽松的上限——参考实现在我笔记本上只需 0.38 秒；而玩具级实现大约要 3 秒。
    """
    input_path = TINY_PATH / "TinyStoriesV2-GPT4-train.txt"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    print(end_time - start_time)
    assert end_time - start_time < 1.5