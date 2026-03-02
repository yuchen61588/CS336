import json
import os
import yaml
import argparse
from typing import Callable, List
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    dataset: List[dict],
    output_path: str,
    output_log: str,
    model_name: str,
    dataset_path: str
) -> None:
    """
    评估语言模型，计算指标，并将结果序列化到磁盘（包含 JSON 详情和 TXT 统计日志）。
    """
    print("开始使用 vLLM 生成回复...")
    outputs = vllm_model.generate(prompts, sampling_params=eval_sampling_params)

    print("计算奖励与统计指标...")
    results = []
    
    # 指标累加器
    total_rewards, format_rewards, answer_rewards = [], [], []
    all_lengths, correct_lengths, incorrect_lengths = [], [], []

    # 类别计数器
    category_1 = 0  # 格式对, 答案对
    category_2 = 0  # 格式对, 答案错
    category_3 = 0  # 格式错, 答案错
    category_4 = 0  # 格式错, 答案对 (兜底)

    for item, output in zip(dataset, outputs):
        generated_text = output.outputs[0].text
        # 直接使用 vLLM 返回的 token_ids 计算长度，最准确且无需额外 Tokenizer
        resp_len = len(output.outputs[0].token_ids) 
        
        ground_truth = item.get("ground_truth", "")
        reward = reward_fn(generated_text, ground_truth)

        r_format = reward.get("format_reward", 0.0)
        r_answer = reward.get("answer_reward", 0.0)
        r_total = reward.get("reward", 0.0)

        # 记录分数
        format_rewards.append(r_format)
        answer_rewards.append(r_answer)
        total_rewards.append(r_total)

        # 逻辑判断 (假设分数 > 0 代表正确)
        is_format_correct = r_format > 0
        is_answer_correct = r_answer > 0

        if is_format_correct and is_answer_correct:
            category_1 += 1
        elif is_format_correct and not is_answer_correct:
            category_2 += 1
        elif not is_format_correct and not is_answer_correct:
            category_3 += 1
        else:
            category_4 += 1

        # 统计长度
        all_lengths.append(resp_len)
        if r_total > 0:
            correct_lengths.append(resp_len)
        else:
            incorrect_lengths.append(resp_len)

        # 保存单条详情
        results.append({
            "prompt": output.prompt,
            "generated_text": generated_text,
            "ground_truth": ground_truth,
            "format_reward": r_format,
            "answer_reward": r_answer,
            "total_reward": r_total,
            "generation_length": resp_len
        })
   
    total_samples = len(results)
    
    # 辅助计算均值的内部函数
    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    # ===============================
    # 1. 终端打印统计结果
    # ===============================
    print(f"\n=== {model_name} 在 {os.path.basename(dataset_path)} 的评估结果 ===")
    print(f"总样本数: {total_samples}")
    if total_samples > 0:
        print(f"类别 1 (格式对, 答案对): {category_1} ({category_1/total_samples*100:.2f}%)")
        print(f"类别 2 (格式对, 答案错): {category_2} ({category_2/total_samples*100:.2f}%)")
        print(f"类别 3 (格式错, 答案错): {category_3} ({category_3/total_samples*100:.2f}%)")
        if category_4 > 0:
            print(f"类别 4 (格式错, 答案对): {category_4} ({category_4/total_samples*100:.2f}%)")
    else:
        print("警告：验证集样本数为 0！")
    print("===============================\n")

    # ===============================
    # 2. 保存详细的 JSON/JSONL 结果
    # ===============================
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            if output_path.endswith('.jsonl'):
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + '\n')
            else:
                json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[*] 详情结果已保存至 {output_path}")

    # ===============================
    # 3. 追加模式写入 TXT 统计日志
    # ===============================
    if output_log:
        os.makedirs(os.path.dirname(output_log), exist_ok=True)
        with open(output_log, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"🚀 模型: {model_name} | 数据集: {dataset_path}\n")
            f.write(f"{'='*50}\n")
            
            f.write("【🏆 奖励得分统计】\n")
            f.write(f"平均总奖励 (Total Reward):   {mean(total_rewards):.4f}\n")
            f.write(f"平均格式奖励 (Format Reward): {mean(format_rewards):.4f}\n")
            f.write(f"平均答案奖励 (Answer Reward): {mean(answer_rewards):.4f}\n\n")
            
            f.write("【📏 生成长度统计 (Tokens)】\n")
            f.write(f"平均生成总长度:     {mean(all_lengths):.2f} tokens\n")
            f.write(f"正确回答平均长度:   {mean(correct_lengths):.2f} tokens\n")
            f.write(f"错误回答平均长度:   {mean(incorrect_lengths):.2f} tokens\n\n")
            
            f.write("【📊 模型表现占比】\n")
            f.write(f"✅ 格式对 & 答案对: {(category_1 / total_samples * 100) if total_samples > 0 else 0.0:.2f}%\n")
            f.write(f"⚠️ 格式对 & 答案错: {(category_2 / total_samples * 100) if total_samples > 0 else 0.0:.2f}%\n")
            f.write(f"❌ 格式错 & 答案错: {(category_3 / total_samples * 100) if total_samples > 0 else 0.0:.2f}%\n")
            f.write(f"🍀 格式错 & 答案对: {(category_4 / total_samples * 100) if total_samples > 0 else 0.0:.2f}%\n\n")
            
            f.write("【🔢 具体生成数量】\n")
            f.write(f"✅ 格式对 & 答案对: {category_1} 条\n")
            f.write(f"⚠️ 格式对 & 答案错: {category_2} 条\n")
            f.write(f"❌ 格式错 & 答案错: {category_3} 条\n")
            f.write(f"🍀 格式错 & 答案对: {category_4} 条\n")
        
        print(f"[*] 统计指标已追加至日志 {output_log}")


def main():
    parser = argparse.ArgumentParser(description="运行数据集的零样本基线评估")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    args = parser.parse_args()

    # 1. 加载 YAML 配置
    print(f"[*] 正在加载配置文件: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 2. 提取配置参数
    data_path = config.get("example_path")
    prompt_file_path = config.get("prompt_path")
    output_path = config.get("output_path")
    output_log = config.get("output_log")  # 新增的 TXT 日志路径
    model_path = config.get("model_path")
    max_tokens = config.get("max_tokens", 4096)
    temperature = config.get("temperature", 0)
    top_p = config.get("top_p", 0.95)

    print(f"[*] 模型路径: {model_path}")
    print(f"[*] 数据集路径: {data_path}")
    print(f"[*] 详情结果将保存至: {output_path}")
    print(f"[*] 统计日志将追加至: {output_log}")

    # 3. 加载数据集
    dataset = []
    print(f"正在加载数据集: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 过滤空行
                dataset.append(json.loads(line))
                
    # 4. 构建 prompts
    prompts = [item["prompt"] for item in dataset]

    # 5. 设置 vLLM 采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    
    # 6. 启动推理
    print(f"正在启动 vLLM 引擎加载模型 ...")
    llm = LLM(model=model_path)
   
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        eval_sampling_params=sampling_params,
        dataset=dataset,
        output_path=output_path,
        output_log=output_log,
        model_name=os.path.basename(model_path), # 提取模型文件夹名作为日志标识
        dataset_path=data_path
    )

if __name__ == "__main__":
    main()