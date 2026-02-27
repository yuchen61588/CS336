import os
import json
from datasets import load_dataset

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# r1_zero 提示词模板
R1_ZERO_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
"""

def main():
    output_dir = "train_data/datasets/GSM8K"
    os.makedirs(output_dir, exist_ok=True)
    val_output_path = os.path.join(output_dir, "validation.jsonl")

    print("正在加载 GSM8K 测试集作为验证集...")
    dataset = load_dataset("parquet", data_files={"test": "https://hf-mirror.com/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet"})["test"]

    with open(val_output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            question_text = item["question"].strip()
            original_answer = item["answer"]
            
            # 解析 GSM8K 的推理过程和最终答案
            if "####" in original_answer:
                reasoning, ground_truth = original_answer.split("####")
                reasoning = reasoning.strip()
                ground_truth = ground_truth.strip()
            else:
                reasoning = original_answer.strip()
                ground_truth = reasoning
            
            # 构造 prompt
            prompt = R1_ZERO_PROMPT.replace("{question}", question_text)
            
            # 构造带有标准标签的 response
            response = f"{reasoning}\n</think>\n<answer> {ground_truth} </answer>"
            
            # 严格按照要求：使用 prompt 形式，丢弃原有的 question 和 answer 键
            val_record = {
                "prompt": prompt,
                "response": response,
                "ground_truth": ground_truth
            }
            f.write(json.dumps(val_record, ensure_ascii=False) + "\n")

    print(f"✅ 验证集生成完毕: {val_output_path}")

if __name__ == "__main__":
    main()