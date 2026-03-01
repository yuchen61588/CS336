import json
import os
from datasets import load_dataset
from tqdm import tqdm
from cs336_alignment.drgrpo_grader import extract_answer

# --- 全局配置 ---
BASE_DATA_DIR = "train_data/datasets"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# r1_zero 提示词模板
R1_ZERO_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
"""

def save_records(f_std, f_rl, question, reasoning, ground_truth):
    """
    通用保存函数：用于构造 prompt 和 response，并写入标准集和 RL 测试集。
    """
    question = str(question).strip() if question else ""
    reasoning = str(reasoning).strip() if reasoning else ""
    ground_truth = str(ground_truth).strip() if ground_truth else ""

    # 构造 prompt
    prompt = R1_ZERO_PROMPT.replace("{question}", question)
    
    # 构造带有标准标签的 response
    response = f"{reasoning}\n</think>\n<answer> {ground_truth} </answer>"

    # 1. 标准记录格式
    val_record = {
        "prompt": prompt,
        "response": response,
        "ground_truth": ground_truth
    }
    
    # 2. RL记录格式
    rl_record = {
        "prompt": prompt,
        "question": question,
        "answer": ground_truth
    }

    f_std.write(json.dumps(val_record, ensure_ascii=False) + "\n")
    f_rl.write(json.dumps(rl_record, ensure_ascii=False) + "\n")


def prepare_aime_datasets():
    """准备 AIME 数据集 (2024, 2025)"""
    print("\n" + "="*20 + " 准备 AIME 数据集 " + "="*20)
    
    tasks = {
        "AIME_2024": "HuggingFaceH4/aime_2024",
        "AIME_2025": "yentinglin/aime_2025"
    }

    for folder_name, dataset_name in tasks.items():
        print(f"正在处理: {dataset_name} ...")
        # 每个年份单独建文件夹，名字统一叫 validation
        output_dir = os.path.join(BASE_DATA_DIR, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        std_path = os.path.join(output_dir, "validation.jsonl")
        rl_path = os.path.join(output_dir, "validation_rl.jsonl")
        
        try:
            # 注：HF 上 AIME 纯评测集挂在 train split 下
            ds = load_dataset(dataset_name, split="train")
            with open(std_path, 'w', encoding='utf-8') as f_std, \
                 open(rl_path, 'w', encoding='utf-8') as f_rl:
                for item in tqdm(ds, desc=f"Converting {folder_name}"):
                    q = item.get('problem')
                    r = item.get('solution')
                    gt = item.get('answer', r) 
                    
                    save_records(f_std, f_rl, q, r, gt)
                    
            print(f"✅ {folder_name} 已保存至 {output_dir} ({len(ds)} 条)")
        except Exception as e:
            print(f"❌ 下载或处理 {dataset_name} 失败: {e}")

def prepare_math_datasets():
    """准备 MATH 数据集 (math-500 和 lighteval-MATH 测试集)"""
    print("\n" + "="*20 + " 准备 MATH 数据集 " + "="*20)

    # 1. 准备 math-500
    print("正在处理: HuggingFaceH4/math-500 ...")
    output_dir_500 = os.path.join(BASE_DATA_DIR, "MATH-500")
    os.makedirs(output_dir_500, exist_ok=True)
    std_path_500 = os.path.join(output_dir_500, "validation.jsonl")
    rl_path_500 = os.path.join(output_dir_500, "validation_rl.jsonl")
    try:
        ds_500 = load_dataset("HuggingFaceH4/math-500", split="test")
        with open(std_path_500, 'w', encoding='utf-8') as f_std, \
             open(rl_path_500, 'w', encoding='utf-8') as f_rl:
            for item in tqdm(ds_500, desc="Converting math-500"):
                q = item.get("problem")
                r = item.get("solution")
                gt = item.get("answer", r)
                # 加上 isinstance(r, str) 保护，防止 float 报错
                if (not gt or gt == r) and isinstance(r, str):
                    extracted = extract_answer(r)
                    gt = extracted if extracted is not None else r
                
                save_records(f_std, f_rl, q, r, gt)
        print(f"✅ math-500 已保存至 {output_dir_500} ({len(ds_500)} 条)")
    except Exception as e:
        print(f"❌ 下载或处理 math-500 失败: {e}")

    # 2. 准备轻量版 MATH (只拉取 test split)
    print("\n正在处理: xDAN2099/lighteval-MATH (train and test splits) ...")
    output_dir_math = os.path.join(BASE_DATA_DIR, "MATH")
    os.makedirs(output_dir_math, exist_ok=True)
    
    splits_to_process = ["train", "test"]
    for split_name in splits_to_process:
        print(f"--> 正在处理 {split_name} split...")
        
        # 按照要求针对不同 split 命名
        if split_name == "train":
            std_path = os.path.join(output_dir_math, "sft.jsonl")
            rl_path = os.path.join(output_dir_math, "train.jsonl")
        else:
            std_path = os.path.join(output_dir_math, "validation.jsonl")
            rl_path = os.path.join(output_dir_math, "validation_rl.jsonl")
            
        try:
            ds = load_dataset("xDAN2099/lighteval-MATH", split=split_name)
            with open(std_path, 'w', encoding='utf-8') as f_std, \
                 open(rl_path, 'w', encoding='utf-8') as f_rl:
                for item in tqdm(ds, desc=f"Converting lighteval-MATH {split_name}"):
                    q = item.get("problem")
                    r = item.get("solution")
                    gt = item.get("answer", r)
                    
                    # 加上 isinstance(r, str) 保护
                    if (not gt or gt == r) and isinstance(r, str):
                        extracted = extract_answer(r)
                        gt = extracted if extracted is not None else r
                    
                    save_records(f_std, f_rl, q, r, gt)
            print(f"✅ MATH {split_name} 集 已保存至 {output_dir_math} ({len(ds)} 条)")
        except Exception as e:
            print(f"❌ 下载或处理 lighteval-MATH {split_name} 失败: {e}")


def prepare_gsm8k_dataset():
    """准备 GSM8K 测试集"""
    print("\n" + "="*20 + " 准备 GSM8K 数据集 " + "="*20)
    output_dir = os.path.join(BASE_DATA_DIR, "GSM8K")
    os.makedirs(output_dir, exist_ok=True)
    
    std_path = os.path.join(output_dir, "validation.jsonl")
    rl_path = os.path.join(output_dir, "validation_rl.jsonl")
    
    print("正在处理: GSM8K test split ...")
    try:
        # 仅拉取 test
        ds = load_dataset("gsm8k", "main", split="test")
        with open(std_path, 'w', encoding='utf-8') as f_std, \
             open(rl_path, 'w', encoding='utf-8') as f_rl:
            for item in tqdm(ds, desc="Converting GSM8K (test)"):
                q = item.get("question")
                original_answer = item.get("answer", "")
                
                if "####" in original_answer:
                    r, gt = original_answer.split("####")
                else:
                    r = original_answer
                    gt = original_answer
                
                save_records(f_std, f_rl, q, r, gt)
        print(f"✅ GSM8K 测试集已保存至 {output_dir} ({len(ds)} 条)")
    except Exception as e:
        print(f"❌ 处理 GSM8K 失败: {e}")


def prepare_amc_dataset():
    """准备 AMC (AI-MO format) 验证集"""
    print("\n" + "="*20 + " 准备 AMC 数据集 " + "="*20)
    output_dir = os.path.join(BASE_DATA_DIR, "AMC")
    os.makedirs(output_dir, exist_ok=True)
    
    std_path = os.path.join(output_dir, "validation.jsonl")
    rl_path = os.path.join(output_dir, "validation_rl.jsonl")
    
    print("正在处理: AI-MO/aimo-validation-amc ...")
    try:
        ds = load_dataset("AI-MO/aimo-validation-amc", split="train")
        with open(std_path, 'w', encoding='utf-8') as f_std, \
             open(rl_path, 'w', encoding='utf-8') as f_rl:
            for item in tqdm(ds, desc="Converting AMC"):
                q = item.get("problem") or item.get("question")
                r = item.get("solution") or item.get("answer")
                gt = item.get("answer") or item.get("final_answer") or r
                
                # [修正]：添加 isinstance(r, str) 判断，过滤掉本身就是数字类型的字段
                if gt == r and isinstance(r, str):
                    extracted = extract_answer(r)
                    gt = extracted if extracted is not None else r

                save_records(f_std, f_rl, q, r, gt)
        print(f"✅ AMC 已保存至 {output_dir} ({len(ds)} 条)")
    except Exception as e:
        print(f"❌ 处理 AMC 失败: {e}")


def prepare_omnimath_dataset():
    """准备 OmniMATH 测试集"""
    print("\n" + "="*20 + " 准备 OmniMATH 数据集 " + "="*20)
    output_dir = os.path.join(BASE_DATA_DIR, "OmniMATH")
    os.makedirs(output_dir, exist_ok=True)
    
    std_path = os.path.join(output_dir, "validation.jsonl")
    rl_path = os.path.join(output_dir, "validation_rl.jsonl")
    
    print("正在处理: KbsdJames/Omni-MATH ...")
    try:
        # 直接拉取 test
        ds = load_dataset("KbsdJames/Omni-MATH", split="test")
        with open(std_path, 'w', encoding='utf-8') as f_std, \
             open(rl_path, 'w', encoding='utf-8') as f_rl:
            for item in tqdm(ds, desc="Converting OmniMATH"):
                q = item.get("problem") or item.get("question")
                r = item.get("solution") or item.get("answer")
                gt = item.get("answer") or item.get("final_answer") or r
                
                # OmniMATH 也加上 extract_answer
                if gt == r:
                    extracted = extract_answer(r)
                    gt = extracted if extracted is not None else r

                save_records(f_std, f_rl, q, r, gt)
        print(f"✅ OmniMATH 已保存至 {output_dir} ({len(ds)} 条)")
    except Exception as e:
        print(f"❌ 处理 OmniMATH 失败: {e}")


if __name__ == "__main__":
    print("--- 开始准备并格式化所有评测数据集 ---")
    
    prepare_aime_datasets()
    prepare_math_datasets()
    prepare_gsm8k_dataset()
    prepare_amc_dataset()
    prepare_omnimath_dataset()
    
    print("\n--- 所有测试集提取并准备完成！---")