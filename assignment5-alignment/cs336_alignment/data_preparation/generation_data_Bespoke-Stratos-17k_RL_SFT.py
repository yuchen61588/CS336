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