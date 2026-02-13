import os
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

def train_bpe_tokenizer(input_path, vocab_size=10000, save_path="tokenizer.json"):
    """
    (保持不变) 使用 HuggingFace tokenizers 库训练 BPE 分词器
    """
    print(f"正在从 {input_path} 训练 BPE Tokenizer (Vocab Size: {vocab_size})...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tokenizer.train([input_path], trainer)
    tokenizer.save(save_path)
    print(f"Tokenizer 已保存至 {save_path}")
    return tokenizer

def encode_and_save(tokenizer, input_path, output_path, batch_size=10000):
    """
    优化版：使用 encode_batch 并行处理，极大提升速度
    """
    print(f"正在编码 {input_path} (Batch Size: {batch_size})...")

    # 获取特殊 token ID
    eot_token_id = tokenizer.token_to_id("<|endoftext|>")
    
    # 确保并行开启 (虽然默认是开启的，显式设置一下更保险)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    all_token_ids = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        batch_lines = []
        
        # tqdm 用于显示进度，我们按行读取，但按批次处理
        # total 只是个估算，或者你可以预先计算行数
        for line in tqdm(f, desc="Processing"):
            line = line.strip()
            if not line:
                continue
            
            batch_lines.append(line)
            
            # 当攒够一个 batch，或者是最后一部分时
            if len(batch_lines) >= batch_size:
                # === 核心加速点 ===
                # encode_batch 会在底层 Rust 中并行处理列表中的所有文本
                batch_encoded = tokenizer.encode_batch(batch_lines)
                
                # 快速将结果取回 Python
                for enc in batch_encoded:
                    all_token_ids.extend(enc.ids)
                    all_token_ids.append(eot_token_id)
                
                # 清空缓冲区
                batch_lines = []
        
        # 处理剩余的不足一个 batch 的行
        if batch_lines:
            batch_encoded = tokenizer.encode_batch(batch_lines)
            for enc in batch_encoded:
                all_token_ids.extend(enc.ids)
                all_token_ids.append(eot_token_id)

    print(f"编码完成，总 Token 数: {len(all_token_ids)}")
    print("正在转换为 uint16 numpy 数组...")

    # 转换为 numpy
    arr = np.array(all_token_ids, dtype=np.uint16)

    print(f"正在保存至 {output_path} ...")
    np.save(output_path, arr)
    print("保存完成！")

def main():
    # === 配置路径 ===
    raw_train_path = "data/TinyStoriesV2-GPT4-train.txt"
    raw_val_path = "data/TinyStoriesV2-GPT4-valid.txt"
    npy_train_out = "output/TinyStories_train.npy"
    npy_val_out = "output/TinyStories_val.npy"
    tokenizer_save_path = "merge_vocab/tokenizer.json"

    os.makedirs("output", exist_ok=True)
    os.makedirs("merge_vocab", exist_ok=True)

    # === 1. 训练 Tokenizer ===
    if os.path.exists(tokenizer_save_path):
        print(f"加载现有 Tokenizer: {tokenizer_save_path}")
        tokenizer = Tokenizer.from_file(tokenizer_save_path)
    else:
        tokenizer = train_bpe_tokenizer(raw_train_path, vocab_size=10000, save_path=tokenizer_save_path)

    # === 2. 生成训练集 .npy (带并行优化) ===
    if not os.path.exists(npy_train_out):
        encode_and_save(tokenizer, raw_train_path, npy_train_out, batch_size=20000) # 调大 batch_size 榨干 CPU
    else:
        print(f"{npy_train_out} 已存在，跳过。")

    # === 3. 生成验证集 .npy (带并行优化) ===
    if not os.path.exists(npy_val_out):
        encode_and_save(tokenizer, raw_val_path, npy_val_out, batch_size=20000)
    else:
        print(f"{npy_val_out} 已存在，跳过。")

if __name__ == "__main__":
    main()