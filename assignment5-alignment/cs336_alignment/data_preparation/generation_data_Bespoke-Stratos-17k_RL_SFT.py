import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免分词器多进程警告

import json
import random
from tqdm import tqdm  # 引入进度条，方便观察过滤进度
from datasets import load_dataset
from transformers import AutoTokenizer
from cs336_alignment.drgrpo_grader import extract_answer

# --- 新增的过滤配置 (与铂金级要求对齐) ---
MIN_LEN = 500
MAX_LEN = 2560
# 分词器路径，建议填你实际用来微调的底座模型名称，这里默认用 Qwen
TOKENIZER_PATH = "Qwen/Qwen2.5-Math-1.5B"

# r1_zero 提示词模板 (保持你的原样)
R1_ZERO_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
"""


def format_bespoke_data(item: dict, tokenizer) -> tuple:
    """
    解析 Bespoke 数据，进行极其严苛的质量过滤，
    返回 RL 格式和 SFT 格式的字典。如果数据不合格，返回 (None, None)。
    """
    messages = item.get("messages", [])
    if not messages:
        messages = item.get("conversations", [])

    question = ""
    assistant_response = ""

    for msg in messages:
        if msg["role"] == "user":
            question = msg["content"].strip()
        elif msg["role"] == "assistant":
            assistant_response = msg["content"].strip()

    if not question or not assistant_response:
        return None, None

    # === 1. 标签标准化 (保留你的逻辑) ===
    assistant_response = assistant_response.replace("<|begin_of_thought|>", "<think>")
    assistant_response = assistant_response.replace("<|end_of_thought|>", "</think>")
    assistant_response = assistant_response.replace("<|begin_of_solution|>", "<answer>")
    assistant_response = assistant_response.replace("<|end_of_solution|>", "</answer>")

    # === 2. 严苛的结构检查 (Structure Check) ===
    # 必须有且仅有一对完整的标签
    tags = ["<think>", "</think>", "<answer>", "</answer>"]
    for tag in tags:
        if assistant_response.count(tag) != 1:
            return None, None  # 标签数量不对，比如提前中断，直接丢弃

    t_start = assistant_response.find("<think>")
    t_end = assistant_response.find("</think>")
    a_start = assistant_response.find("<answer>")
    a_end = assistant_response.find("</answer>")

    # 标签顺序必须绝对正确
    if not (t_start < t_end < a_start < a_end):
        return None, None

    # 提取区域内容
    think_content = assistant_response[t_start + 7: t_end].strip()
    answer_content = assistant_response[a_start + 8: a_end].strip()

    if not think_content or not answer_content:
        return None, None  # 防止空思考或空回答

    # === 3. 防作弊与硬性要求 ===
    # 思考过程中不准出现最终答案(防思维泄露)
    if "\\boxed" in think_content:
        return None, None

    # 回答部分必须包含 \boxed{}
    if "\\boxed" not in answer_content:
        return None, None

    # === 4. 精确提取 Ground Truth (仅保留答案) ===
    ground_truth = item.get("solution", "")
    if not ground_truth:
        ground_truth = item.get("answer", "")
    if not ground_truth:
        ground_truth = answer_content  # 兜底：从助手的回答中找

    # 使用你的 extract_answer 函数强制清洗
    extracted = extract_answer(ground_truth)
    if not extracted:
        # 如果提取失败（例如没有明确的 \boxed 或者解析崩溃），丢弃该数据，否则 RL 打分必为 0
        return None, None
    ground_truth = extracted.strip()

    # === 5. 构造 SFT prompt 与 response  ===
    prompt = R1_ZERO_PROMPT.replace("{question}", question)
    response = assistant_response.strip()

    # 裁掉开头的 <think>，防止与 prompt 重复
    if response.startswith("<think>"):
        response = response[len("<think>"):].lstrip()

    # === 6. 长度检查 (Length Check) ===
    # 将完整的 prompt + response 送入 Tokenizer 计算总长度
    full_text = prompt + response
    tokens = tokenizer.encode(full_text)
    if not (MIN_LEN <= len(tokens) <= MAX_LEN):
        return None, None  # 长度超标或过短，抛弃

    # === 7. 组装并返回你的标准结构 ===
    rl_record = {
        "prompt": prompt,
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

    train_rl_path = os.path.join(output_dir, "train.jsonl")
    val_rl_path = os.path.join(output_dir, "validation_rl.jsonl")
    train_sft_path = os.path.join(output_dir, "sft.jsonl")
    val_sft_path = os.path.join(output_dir, "validation.jsonl")

    # 初始化 Tokenizer (必须提前下好或者网络畅通)
    print(f"正在加载分词器: {TOKENIZER_PATH} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"分词器加载失败: {e}\n请确保已配置网络或替换为本地模型路径。")
        return

    print("正在通过 HF-Mirror 加载 Bespoke-Stratos-17k 数据集...")
    dataset = load_dataset("HuggingFaceH4/Bespoke-Stratos-17k", split="train")

    all_data = []
    print("正在严格清洗并解析数据格式...")

    # 增加 tqdm 进度条，因为加上 Tokenizer 编码后处理时间会变长
    for item in tqdm(dataset, desc="Processing Data"):
        rl_record, sft_record = format_bespoke_data(item, tokenizer)

        # 只有当数据通过了所有苛刻条件，才会被加入最终数据集
        if rl_record and sft_record:
            all_data.append((rl_record, sft_record))

    # 打乱数据以切分验证集
    random.seed(42)
    random.shuffle(all_data)

    # 划分验证集大小
    val_size = min(1000, len(all_data) // 10)  # 加上防护：如果合格数据不够，按比例切
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]

    print(f"\n--- 过滤总结 ---")
    print(f"原始数据总量: {len(dataset)} 条")
    print(f"清洗后剩余量: {len(all_data)} 条 (存活率: {len(all_data) / len(dataset):.2%})")
    print(f"切分完成：训练集 {len(train_data)} 条，验证集 {len(val_data)} 条。")

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

    print(f"✅ 生成完毕！所有纯净数据保存在 {output_dir}/ 目录下。")


if __name__ == "__main__":
    main()