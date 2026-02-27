import os
import json
import random
from datasets import load_dataset

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

R1_ZERO_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
"""

def format_gsm8k_item(item: dict) -> tuple:
    question = item["question"].strip()
    original_answer = item["answer"]
    
    # 提取推理和纯文本答案
    if "####" in original_answer:
        reasoning, ground_truth = original_answer.split("####")
        reasoning = reasoning.strip()
        ground_truth = ground_truth.strip()
    else:
        reasoning = original_answer.strip()
        ground_truth = reasoning

    prompt = R1_ZERO_PROMPT.replace("{question}", question)
    
    # RL 格式：对应基线评估和 GRPO
    rl_record = {
        "prompt": prompt,
        "question": question,
        "answer": ground_truth
    }

    # SFT 格式：对应训练
    # 模拟 R1 响应：[reasoning]\n</think>\n<answer> [answer] </answer>
    response = f"{reasoning}\n</think>\n<answer> {ground_truth} </answer>"
    sft_record = {
        "prompt": prompt,
        "response": response,
        "ground_truth": ground_truth  
    }

    return rl_record, sft_record

def main():
    output_dir = "train_data/datasets/GSM8K"
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在通过镜像站加载 GSM8K 训练集...")
    dataset = load_dataset("parquet", data_files={"train": "https://hf-mirror.com/datasets/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet"})["train"]

    all_data = [format_gsm8k_item(item) for item in dataset]
    
    # 写入文件
    with open(os.path.join(output_dir, "train.jsonl"), "w", encoding="utf-8") as f_rl, \
         open(os.path.join(output_dir, "sft.jsonl"), "w", encoding="utf-8") as f_sft:
        for rl_rec, sft_rec in all_data:
            f_rl.write(json.dumps(rl_rec, ensure_ascii=False) + "\n")
            f_sft.write(json.dumps(sft_rec, ensure_ascii=False) + "\n")

    print(f"✅ GSM8K 训练数据生成完毕，保存在 {output_dir}")

if __name__ == "__main__":
    main()