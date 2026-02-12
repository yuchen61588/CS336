import os
import numpy as np
from tests.common import gpt2_bytes_to_unicode
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path

# 如果在 test_tokenizer.py 中，你需要导入你自己的 Tokenizer
# from cs336_basics.tokenizer import Tokenizer

def process_and_save(
        vocab_path: str,
        merges_path: str,
        raw_txt_path: str,
        output_npy_path: str,
        name: str
):
    print(f"\n" + "=" * 60)
    print(f"🚀 任务开始: 处理 {name} 数据集")
    print("=" * 60)

    # 1. 加载并还原分词器
    tokenizer = get_tokenizer_from_vocab_merges_path(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
 

    # 2. 调用你的并行编码接口
    print(f"  -> 启动并行分词 (encode_parallel)...")
    ids = tokenizer.encode_parallel(raw_txt_path)

    # 4. 存为 uint16 .npy 格式
    # 使用 uint16 极大节省后续训练时的显存和磁盘占用
    print(f"  -> 转换为 uint16 并保存...")
    np_ids = np.array(ids, dtype=np.uint16)


    # os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
    # np.save(output_npy_path, np_ids)

    print(f"✅ 完成！Token总数: {len(np_ids):,}")
    print(f"✅ 文件已保存至: {output_npy_path}")


if __name__ == "__main__":
    # 配置路径
    PROJECT_ROOT = "."
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "merge_vocab")

    # 1. TinyStories 任务
    process_and_save(
        vocab_path=os.path.join(OUTPUT_DIR, "TinyStoriesV2-GPT4-train_vocab.json"),
        merges_path=os.path.join(OUTPUT_DIR, "TinyStoriesV2-GPT4-train_merges.txt"),
        raw_txt_path=os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt"),
        output_npy_path=os.path.join(DATA_DIR, "TinyStories_train.npy"),
        name="TinyStories-Train"
    )

    # 2. OpenWebText 任务
    process_and_save(
        vocab_path=os.path.join(OUTPUT_DIR, "owt_train_vocab.json"),
        merges_path=os.path.join(OUTPUT_DIR, "owt_train_merges.txt"),
        raw_txt_path=os.path.join(DATA_DIR, "owt_train.txt"),
        output_npy_path=os.path.join(DATA_DIR, "owt_train.npy"),
        name="OWT-Train"
    )
