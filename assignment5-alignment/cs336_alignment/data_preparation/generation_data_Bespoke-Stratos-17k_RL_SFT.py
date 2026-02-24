import os
import json
import random
from datasets import load_dataset

# 强制使用国内镜像并禁用可能报错的 XetHub
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

# r1_zero 提示词模板
R1_ZERO_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
"""


def format_bespoke_data(item: dict) -> tuple:
    """
    解析 Bespoke 数据，返回 RL 格式和 SFT 格式的字典
    返回: (rl_record, sft_record)
    """
    # 增加容错：兼容 messages 或 conversations 字段
    messages = item.get("messages", [])
    if not messages:
        messages = item.get("conversations", [])
        
    question = ""
    assistant_response = ""

    # 提取问题和回答
    for msg in messages:
        if msg["role"] == "user":
            question = msg["content"].strip()
        elif msg["role"] == "assistant":
            assistant_response = msg["content"].strip()

    # === 关键修复：统一 Bespoke 的特殊标签到标准 R1 标签 ===
    assistant_response = assistant_response.replace("<|begin_of_thought|>", "<think>")
    assistant_response = assistant_response.replace("<|end_of_thought|>", "</think>")
    assistant_response = assistant_response.replace("<|begin_of_solution|>", "<answer>")
    assistant_response = assistant_response.replace("<|end_of_solution|>", "</answer>")

    # 提取标准答案
    ground_truth = item.get("solution", "")
    if not ground_truth:
        ground_truth = item.get("answer", "")

    # 从 response 的 </think> 后面提取答案 (此时标签已经替换完毕，可以命中)
    if not ground_truth and "</think>" in assistant_response:
        ground_truth = assistant_response.split("</think>")[-1]

    # 清理 ground_truth 中多余的 <answer> 标签，只保留纯文本答案
    ground_truth = ground_truth.replace("<answer>", "").replace("</answer>", "").strip()

    # 构造 SFT prompt
    prompt = R1_ZERO_PROMPT.replace("{question}", question)

    # 构造 SFT response
    response = assistant_response.strip()
    # 裁掉开头的 <think>，防止与 prompt 重复
    if response.startswith("<think>"):
        response = response[len("<think>"):].lstrip()

    # 如果 Bespoke 的结尾没有 <answer> 标签，顺手给它包上
    if "<answer>" not in response and "</think>" in response:
        parts = response.split("</think>")
        reasoning = parts[0].strip()
        final_answer = parts[1].strip()
        response = f"{reasoning}\n</think>\n<answer> {final_answer} </answer>"

    rl_record = {
        "question": question,
        "answer": ground_truth
    }

    sft_record = {
        "prompt": prompt,
        "response": response,
        "ground_truth": ground_truth  
    }

    return rl_record, sft_record


def main():
    output_dir = "train_data/datasets/Bespoke17k"
    os.makedirs(output_dir, exist_ok=True)

    # 准备保存的文件路径
    train_rl_path = os.path.join(output_dir, "train.jsonl")
    val_rl_path = os.path.join(output_dir, "validation_rl.jsonl")
    train_sft_path = os.path.join(output_dir, "sft.jsonl")
    val_sft_path = os.path.join(output_dir, "validation.jsonl")

    print("正在通过 HF-Mirror 加载 Bespoke-Stratos-17k 数据集...")
    # 使用 load_dataset 自动通过镜像拉取
    dataset = load_dataset(
    "HuggingFaceH4/Bespoke-Stratos-17k", 
    split="train",
    download_mode="force_redownload", # 强制重新下载，忽略报错的本地缓存
    )

    all_data = []
    print("正在解析数据格式...")
    for item in dataset:
        rl_record, sft_record = format_bespoke_data(item)
        if rl_record["question"] and rl_record["answer"]:  # 确保数据完整
            all_data.append((rl_record, sft_record))

    # 打乱数据以切分验证集
    random.seed(42)
    random.shuffle(all_data)

    # 划分验证集大小（例如取 1000 条作为验证集）
    val_size = 1000
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]

    print(f"切分完成：训练集 {len(train_data)} 条，验证集 {len(val_data)} 条。正在写入文件...")

    # 写入训练集 (train.jsonl 和 sft.jsonl)
    with open(train_rl_path, "w", encoding="utf-8") as f_rl_train, \
            open(train_sft_path, "w", encoding="utf-8") as f_sft_train:
        for rl_rec, sft_rec in train_data:
            f_rl_train.write(json.dumps(rl_rec, ensure_ascii=False) + "\n")
            f_sft_train.write(json.dumps(sft_rec, ensure_ascii=False) + "\n")

    # 写入验证集 (validation_rl.jsonl 和 validation.jsonl)
    with open(val_rl_path, "w", encoding="utf-8") as f_rl_val, \
            open(val_sft_path, "w", encoding="utf-8") as f_sft_val:
        for rl_rec, sft_rec in val_data:
            f_rl_val.write(json.dumps(rl_rec, ensure_ascii=False) + "\n")
            f_sft_val.write(json.dumps(sft_rec, ensure_ascii=False) + "\n")

    print(f"✅ 生成完毕！所有文件保存在 {output_dir}/ 目录下。")


if __name__ == "__main__":
    main()