import json
import os
from typing import Callable,List
from vllm import LLM, SamplingParams
import argparse

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


MODEL_REGISTRY = {
    "qwen_1.5b": "data/models/Qwen2.5-Math-1.5B",
    # 兼容图片和文档的 Llama 3.1 路径，请根据集群实际情况微调最后的文件夹名
    "llama_3.1_8b": "data/models/Llama-3.1", 
    "llama_3.3_70b": "data/models/Llama-3.3-70B-Instruct"
}

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    dataset: List[dict],
    output_path: str
) -> None:
    """
    Evaluate a language model on a list of prompts, 
    compute evaluation metrics, and serialize results to disk. [cite: 156, 157]
    """
    print("开始使用 vLLM 生成回复...")
    outputs = vllm_model.generate(prompts, sampling_params=eval_sampling_params)

    print("计算奖励...")
    results = []
    category_1 = 0  # format=1, answer=1 
    category_2 = 0  # format=1, answer=0 
    category_3 = 0  # format=0, answer=0
    for item, output in zip(dataset, outputs):
       generated_text = output.outputs[0].text

       ground_truth = item["answer"]
       reward = reward_fn(generated_text, ground_truth)

       format_reward = reward.get("format_reward", 0.0)
       answer_reward = reward.get("answer_reward", 0.0)

       if format_reward == 1.0 and answer_reward == 1.0:
           category_1 += 1
       elif format_reward == 1.0 and answer_reward == 0.0:
           category_2 += 1
       elif format_reward == 0.0 and answer_reward == 0.0:
           category_3 += 1

       results.append({
           "question": item["question"],
           "generated_text": generated_text,
           "prompt": output.prompt,
           "ground_truth": ground_truth,
           "format_reward": format_reward,
           "answer_reward": answer_reward,
           "total_reward": reward.get("reward", 0.0)
       })
    
    total_samples = len(results)
    print("\n=== 评估结果 ===")
    print(f"总样本数: {total_samples}")
    print(f"类别 1 (格式对, 答案对): {category_1} ({category_1/total_samples*100:.2f}%)")
    print(f"类别 2 (格式对, 答案错): {category_2} ({category_2/total_samples*100:.2f}%)")
    print(f"类别 3 (格式错, 答案错): {category_3} ({category_3/total_samples*100:.2f}%)")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至 {output_path}")

def main():
    
    parser = argparse.ArgumentParser(description="运行 MATH 数据集的零样本基线评估") 
    parser.add_argument("--model_name", 
                        type=str, required=True, 
                        choices=MODEL_REGISTRY.keys(), 
                        help="要评估的模型名称")
    args = parser.parse_args()
    model_path = MODEL_REGISTRY[args.model_name]
    output_dir = "data/output/math_baseline_results"
    data_path = "data/datasets/MATH/validation.jsonl" 
    prompt_file_path = "cs336_alignment/prompts/r1_zero.prompt"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.model_name}.json")

    print(f"[*] 选定模型: {args.model_name}")
    print(f"[*] 模型路径: {model_path}")
    print(f"[*] 结果将保存至: {output_path}")

    print(f"正在读取提示词模板: {prompt_file_path}")
    #提示词模板
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    dataset = []
    print(f"正在加载数据集: {data_path}")
    #把数据集弄出来
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    # 将格式替换为问题
    prompts = [prompt_template.replace("{question}", item["question"]) for item in dataset]
    #预设要求
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    # 7. 启动推理
    print(f"正在启动 vLLM 引擎加载模型 ...")
    llm = LLM(model=model_path)
    
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        eval_sampling_params=sampling_params,
        dataset=dataset,
        output_path=output_path
    )

if __name__ == "__main__":
    main()


    


    
    
    
