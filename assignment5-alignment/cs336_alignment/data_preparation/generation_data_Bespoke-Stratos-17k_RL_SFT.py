import os
import re
import json
import random
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

# 从你的评估脚本中导入 grade 和 extract_answer
from cs336_alignment.drgrpo_grader import grade, extract_answer
from cs336_alignment.data_preparation.clear import clean_and_repair_response
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 配置区 ---
MIN_LEN = 500
MAX_LEN = 2560
MAX_ANSWER_LEN = 100
TOKENIZER_PATH = "Qwen/Qwen2.5-Math-1.5B"

R1_ZERO_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
"""


def format_bespoke_data(item: dict, tokenizer, stats_dict: dict) -> tuple:
    # (此部分逻辑与上一版基本一致，主要是组装 SFT 数据)
    messages = item.get("messages", [])
    if not messages:
        messages = item.get("conversations", [])

    question = ""
    raw_response = ""

    for msg in messages:
        if msg["role"].lower() == "user":
            question = msg["content"].strip()
        elif msg["role"].lower() == "assistant":
            raw_response = msg["content"].strip()

    if not question or not raw_response:
        stats_dict["missing_qa"] += 1
        return None, None

    # 标准化标签
    response = raw_response.replace("<|begin_of_thought|>\n\n", "<think>")
    response = response.replace("<|begin_of_thought|>", "<think>")
    response = response.replace("\n\n<|end_of_thought|>\n\n", "</think>")
    response = response.replace("<|end_of_thought|>", "</think>")
    response = response.replace("<|begin_of_solution|>\n\n", "<answer>")
    response = response.replace("<|begin_of_solution|>", "<answer>")
    response = response.replace("\n\n<|end_of_solution|>", "</answer>")
    response = response.replace("<|end_of_solution|>", "</answer>")

    # 提取 Ground Truth（原始答案，无需强行套 boxed，因为 grade() 会处理）
    ground_truth = item.get("solution", "")
    if not ground_truth:
        ground_truth = item.get("answer", "")

    if not ground_truth:
        stats_dict["missing_ground_truth"] += 1
        return None, None

    # 调用重构后的清洗黑盒
    cleaned_response, status = clean_and_repair_response(response, ground_truth)
    stats_dict[status] = stats_dict.get(status, 0) + 1

    if cleaned_response is None:
        return None, None

    # 长度限制检查
    prompt = R1_ZERO_PROMPT.replace("{question}", question)
    full_text = prompt + cleaned_response

    tokens = tokenizer.encode(full_text)
    if not (MIN_LEN <= len(tokens) <= MAX_LEN):
        stats_dict["length_out_of_range"] = stats_dict.get("length_out_of_range", 0) + 1
        return None, None

    rl_record = {
        "prompt": prompt,
        "question": question,
        "answer": ground_truth
    }

    sft_record = {
        "prompt": prompt,
        "response": cleaned_response,
        "ground_truth": ground_truth
    }

    return rl_record, sft_record


def main():
    output_dir = "train_data/datasets/Bespoke17k_Cleaned"
    os.makedirs(output_dir, exist_ok=True)

    train_rl_path = os.path.join(output_dir, "train.jsonl")
    val_rl_path = os.path.join(output_dir, "validation_rl.jsonl")
    train_sft_path = os.path.join(output_dir, "sft.jsonl")
    val_sft_path = os.path.join(output_dir, "validation.jsonl")

    print("正在通过 HF-Mirror 加载 Bespoke-Stratos-17k 数据集...")
    dataset = load_dataset("HuggingFaceH4/Bespoke-Stratos-17k", split="train")

    all_data = []

    # 建立统计字典
    stats_dict = {"total_processed": 0}

    print("正在严格清洗并解析数据格式 (单进程+纯字符长度验证)...")
    for item in tqdm(dataset, desc="Processing Data"):
        rl_record, sft_record = format_bespoke_data(item, stats_dict)
        if rl_record and sft_record:
            all_data.append((rl_record, sft_record))

    random.seed(42)
    random.shuffle(all_data)

    val_size = min(1000, len(all_data) // 10)
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]

    # ==========================================
    # 📊 打印极其详细的统计报告
    # ==========================================
    total = stats_dict.get("total_processed", 1)
    kept_clean = stats_dict.get("clean_passed", 0)
    kept_fixed = stats_dict.get("fixed_and_passed", 0)
    total_kept = kept_clean + kept_fixed

    print("\n" + "=" * 50)
    print("📊 数据清洗统计报告")
    print("=" * 50)
    print(f"原始数据总量: {total} 条")
    print(f"最终保留数据: {total_kept} 条 (存活率: {total_kept / total:.2%})")

    print("\n✅ 保留详情:")
    print(f"  - 完美格式 (无需修复直接放行) : {kept_clean} 条")
    print(f"  - 成功修复 (截断超长/清理换行) : {kept_fixed} 条")

    print("\n❌ 过滤/丢弃原因分布:")
    # 排除掉正常通过的 key，只打印拒绝的原因
    reject_keys = {k: v for k, v in stats_dict.items() if
                   k not in ["total_processed", "clean_passed", "fixed_and_passed"] and v > 0}
    for k, v in sorted(reject_keys.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {k:<25}: {v} 条")
    print("=" * 50 + "\n")

    print("正在写入文件...")
    with open(train_rl_path, "w", encoding="utf-8") as f_rl_train, \
            open(train_sft_path, "w", encoding="utf-8") as f_sft_train:
        for rl_rec, sft_rec in train_data:
            f_rl_train.write(json.dumps(rl_rec, ensure_ascii=False) + "\n")
            f_sft_train.write(json.dumps(sft_rec, ensure_ascii=False) + "\n")

    with open(val_rl_path, "w", encoding="utf-8") as f_rl_val, \
            open(val_sft_path, "w", encoding="utf-8") as f_sft_val:
        for rl_rec, sft_rec in val_data:
            f_rl_val.write(json.dumps(rl_rec, ensure_ascii=False) + "\n")
            f_sft_val.write(json.dumps(sft_rec, ensure_ascii=False) + "\n")

    print(f"✅ 生成完毕！所有结构纯净的数据保存在 {output_dir}/ 目录下。")


if __name__ == "__main__":
    main()