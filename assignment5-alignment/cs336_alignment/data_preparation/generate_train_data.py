import os
import json
from datasets import load_dataset

# 强制使用国内镜像并禁用可能报错的 XetHub
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

# r1_zero 提示词模板
R1_ZERO_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
"""

def format_for_sft(question: str, original_answer: str) -> dict:
    """
    将数据格式化为 SFT 需要的 {"prompt": str, "response": str} 格式
    """
    if "####" in original_answer:
        parts = original_answer.split("####")
        reasoning = parts[0].strip()
        final_answer = parts[1].strip()
    else:
        reasoning = original_answer.strip()
        final_answer = ""

    prompt = R1_ZERO_PROMPT.replace("{question}", question)
    # 模拟 R1 模型的输出格式
    response = f"{reasoning}\n</think>\n<answer> {final_answer} </answer>"
    
    return {
        "prompt": prompt,
        "response": response
    }

def main():
    output_dir = "data/datasets/MATH"
    os.makedirs(output_dir, exist_ok=True)
    
    train_output_path = os.path.join(output_dir, "train.jsonl")
    sft_output_path = os.path.join(output_dir, "sft.jsonl")

    print("正在通过 HF-Mirror 镜像站直接加载 GSM8K 训练集...")
    # 绕过 builder，直接拉取 parquet 文件
    dataset = load_dataset(
        "parquet", 
        data_files={"train": "https://hf-mirror.com/datasets/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet"}
    )
    train_data = dataset["train"]

    print(f"正在生成 RL 训练集 (用于 Expert Iteration 和 GRPO)，保存至 {train_output_path} ...")
    print(f"正在生成 SFT 训练集 (严格 prompt/response 格式)，保存至 {sft_output_path} ...")
    
    with open(train_output_path, "w", encoding="utf-8") as f_train, \
         open(sft_output_path, "w", encoding="utf-8") as f_sft:
        
        for item in train_data:
            question = item["question"]
            original_answer = item["answer"]
            ground_truth = original_answer.split("####")[1].strip() if "####" in original_answer else original_answer.strip()
            
            # 1. 写入 RL 训练集 (train.jsonl)
            train_record = {
                "question": question,
                "answer": ground_truth
            }
            f_train.write(json.dumps(train_record, ensure_ascii=False) + "\n")
            
            # 2. 写入 SFT 训练集 (sft.jsonl)
            sft_record = format_for_sft(question, original_answer)
            f_sft.write(json.dumps(sft_record, ensure_ascii=False) + "\n")

    print("✅ 训练集 (train.jsonl) 和 SFT 数据集 (sft.jsonl) 生成完毕！")

if __name__ == "__main__":
    main()