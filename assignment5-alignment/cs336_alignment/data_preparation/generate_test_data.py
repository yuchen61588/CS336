import os
import json
from datasets import load_dataset

# 强制使用国内镜像并禁用可能报错的 XetHub
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

def main():
    # 设定输出目录
    output_dir = "data/datasets/MATH"
    os.makedirs(output_dir, exist_ok=True)
    val_output_path = os.path.join(output_dir, "validation.jsonl")

    print("正在通过 HF-Mirror 镜像站直接加载 GSM8K 测试集...")
    # 绕过 builder，直接拉取 parquet 文件
    dataset = load_dataset(
        "parquet", 
        data_files={"test": "https://hf-mirror.com/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet"}
    )
    test_data = dataset["test"]

    print(f"正在生成验证集，保存至 {val_output_path} ...")
    with open(val_output_path, "w", encoding="utf-8") as f:
        for item in test_data:
            # GSM8K 的答案格式通常是 "推理过程 #### 最终答案"
            original_answer = item["answer"]
            ground_truth = original_answer.split("####")[1].strip() if "####" in original_answer else original_answer.strip()
            
            # 组装为字典
            val_record = {
                "question": item["question"],
                "answer": ground_truth,
                # 保留完整的原始答案以便参考
                "original_solution": original_answer 
            }
            f.write(json.dumps(val_record, ensure_ascii=False) + "\n")

    print("✅ 验证集/测试集 (validation.jsonl) 生成完毕！")

if __name__ == "__main__":
    main()