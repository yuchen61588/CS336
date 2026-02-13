import os
import glob
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


def train_bpe_tokenizer(input_path, vocab_size=10000, save_path="tokenizer.json"):
    """
    使用 HuggingFace tokenizers 库训练 BPE 分词器
    对应 PDF Section 2.5: BPE Training on TinyStories [cite: 248]
    """
    print(f"正在从 {input_path} 训练 BPE Tokenizer (Vocab Size: {vocab_size})...")

    # 1. 初始化 BPE 模型
    tokenizer = Tokenizer(models.BPE())

    # 2. 预分词 (Pre-tokenization)
    # 使用 ByteLevel，这对 GPT-2/Llama 类模型是标准的
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 3. 解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 4. 训练器配置
    # 注意：PDF 要求包含 <|endoftext|>
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 5. 开始训练
    # 如果文件很大，这里会自动处理流式读取
    files = [input_path]
    tokenizer.train(files, trainer)

    # 6. 保存
    tokenizer.save(save_path)
    print(f"Tokenizer 已保存至 {save_path}")
    return tokenizer


def encode_and_save(tokenizer, input_path, output_path):
    """
    读取文本文件，编码为 Token ID，并保存为 numpy uint16 格式。
    对应 PDF Section 2.7 (d) [cite: 328, 329]
    """
    print(f"正在编码 {input_path} ...")

    # 获取特殊 token ID
    eot_token_id = tokenizer.token_to_id("<|endoftext|>")

    token_ids = []

    # 逐行读取以节省内存
    with open(input_path, 'r', encoding='utf-8') as f:
        # 使用 tqdm 显示进度
        for line in tqdm(f, desc="Processing lines"):
            line = line.strip()
            if not line:
                continue

            # 编码
            # 注意：不添加 special tokens (如 BOS/EOS)，我们手动控制 <|endoftext|>
            encoded = tokenizer.encode(line).ids

            token_ids.extend(encoded)
            # 在每段故事/文本后添加 <|endoftext|>
            token_ids.append(eot_token_id)

    print(f"编码完成，总 Token 数: {len(token_ids)}")
    print("正在转换为 uint16 numpy 数组...")

    # PDF 建议使用 uint16 [cite: 330]
    # 因为 vocab_size (10000) < 65535，uint16 足够且节省一半内存
    arr = np.array(token_ids, dtype=np.uint16)

    print(f"正在保存至 {output_path} ...")
    np.save(output_path, arr)
    print("保存完成！")


def main():
    # === 配置路径 ===
    # 假设你的原始数据是 txt 格式
    raw_train_path = "data/TinyStoriesV2-GPT4-train.txt"
    raw_val_path = "data/TinyStoriesV2-GPT4-valid.txt"

    # 输出路径
    npy_train_out = "output/TinyStories_train.npy"
    npy_val_out = "output/TinyStories_val.npy"
    tokenizer_save_path = "merge_vocab/tokenizer.json"

    # 确保 data 目录存在
    os.makedirs("data", exist_ok=True)

    # === 1. 训练 Tokenizer ===
    # 如果已经训练过，可以直接加载
    if os.path.exists(tokenizer_save_path):
        print(f"加载现有 Tokenizer: {tokenizer_save_path}")
        tokenizer = Tokenizer.from_file(tokenizer_save_path)
    else:
        # 通常只用训练集训练 Tokenizer
        tokenizer = train_bpe_tokenizer(raw_train_path, vocab_size=10000, save_path=tokenizer_save_path)

    # === 2. 生成训练集 .npy ===
    if not os.path.exists(npy_train_out):
        encode_and_save(tokenizer, raw_train_path, npy_train_out)
    else:
        print(f"{npy_train_out} 已存在，跳过。")

    # === 3. 生成验证集 .npy ===
    if not os.path.exists(npy_val_out):
        encode_and_save(tokenizer, raw_val_path, npy_val_out)
    else:
        print(f"{npy_val_out} 已存在，跳过。")


if __name__ == "__main__":
    main()