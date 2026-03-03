import wandb
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from typing import List,Dict,Optional,Tuple,Callable,Literal
import torch
from transformers import  PreTrainedTokenizer
import numpy as np
import os



def log_train_metrics(step, reward_mean, loss, grad_norm, lr, token_entropy):
    """阶段 1：训练过程中的极简画图统计"""
    wandb.log({
        "train_step": step,
        "train/reward_mean": reward_mean,  # 核心指标
        "train/entropy": token_entropy,  # 核心指标
        "train/loss": loss,  # 仅供调试防飞
        "train/grad_norm": grad_norm,
        "train/lr": lr,
    })


def log_periodic_eval_metrics(step, llm, valid_data, prompt_template, sampling_params):
    """阶段 2：定期验证的极简画图统计 (纯算分数，不跑前向模型算熵)"""
    val_prompts = [prompt_template.format(question=item["question"]) for item in valid_data]
    val_truths = [item.get("ground_truth", item.get("answer", "")) for item in valid_data]

    eval_outputs = llm.generate(val_prompts, sampling_params)

    total_eval_reward = 0.0
    for i, out in enumerate(eval_outputs):
        reward_res = r1_zero_reward_fn(out.outputs[0].text, val_truths[i])
        total_eval_reward += reward_res["reward"]

    avg_eval_reward = total_eval_reward / len(valid_data)
    wandb.log({"eval_step": step, "eval/reward_mean": avg_eval_reward})
    print(f"=== 定期评估 Step {step} | Eval Reward: {avg_eval_reward:.3f} ===")


def log_generations(
        prompts: List[str],
        generated_responses: List[str],
        ground_truths: List[str],
        token_entropies: List[torch.Tensor],
        reward_fn,
        step: int,
        save_logs: Optional[str] = None,
        yaml_config_name: Optional[str] = None,
        tokenizer: PreTrainedTokenizer = None
):
    """阶段 3：最终详尽统计 (你提供的函数，略作合并)"""
    log_data, total_rewards, format_rewards, answer_rewards = [], [], [], []
    correct_lengths, incorrect_lengths, all_lengths = [], [], []
    category_1, category_2, category_3, category_4 = 0, 0, 0, 0

    for prompt, response, truth, entropies in zip(prompts, generated_responses, ground_truths, token_entropies):
        rewards = reward_fn(response, truth)
        r_total = rewards.get("reward", 0.0)
        r_format = rewards.get("format_reward", 0.0)
        r_answer = rewards.get("answer_reward", 0.0)

        total_rewards.append(r_total)
        format_rewards.append(r_format)
        answer_rewards.append(r_answer)

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

        avg_entropy = entropies.mean().item() if len(entropies) > 0 else 0.0

        resp_len = len(tokenizer.encode(response))
        all_lengths.append(resp_len)
        if r_total > 0:
            correct_lengths.append(resp_len)
        else:
            incorrect_lengths.append(resp_len)

        log_data.append([prompt, response, truth, r_format, r_answer, r_total, avg_entropy])

    total_samples = len(prompts)
    print(f"\n[Final Eval] === {yaml_config_name} 最终评估结果统计 ===")
    print(f"总样本数: {total_samples}")
    if total_samples > 0:
        print(f"类别 1 (格对, 答对): {category_1} ({category_1 / total_samples * 100:.2f}%)")
        print(f"类别 2 (格对, 答错): {category_2} ({category_2 / total_samples * 100:.2f}%)")
        print(f"类别 3 (格错, 答错): {category_3} ({category_3 / total_samples * 100:.2f}%)")
        if category_4 > 0:
            print(f"类别 4 (格错, 答对): {category_4} ({category_4 / total_samples * 100:.2f}%)")
    print("===============================\n")

    if save_logs:
        os.makedirs(os.path.dirname(save_logs), exist_ok=True)
        with open(save_logs, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 50}\n")
            f.write(f"🚀 实验名称: {yaml_config_name} | 最终阶段 (Step): {step}\n")
            f.write(f"{'=' * 50}\n")
            f.write("【🏆 奖励得分统计】\n")
            f.write(f"平均总奖励:   {np.mean(total_rewards) if total_rewards else 0.0:.4f}\n")
            f.write(f"平均格式奖励: {np.mean(format_rewards) if format_rewards else 0.0:.4f}\n")
            f.write(f"平均答案奖励: {np.mean(answer_rewards) if answer_rewards else 0.0:.4f}\n\n")
            f.write("【📏 生成长度统计】\n")
            f.write(f"平均生成总长度:     {np.mean(all_lengths) if all_lengths else 0.0:.2f} tokens\n")
            f.write(f"正确回答平均长度:   {np.mean(correct_lengths) if correct_lengths else 0.0:.2f} tokens\n")
            f.write(f"错误回答平均长度:   {np.mean(incorrect_lengths) if incorrect_lengths else 0.0:.2f} tokens\n\n")
            f.write("【📊 模型表现占比】\n")
            f.write(f"✅ 格对&答对: {(category_1 / total_samples * 100) if total_samples > 0 else 0:.2f}%\n")
            f.write(f"⚠️ 格对&答错: {(category_2 / total_samples * 100) if total_samples > 0 else 0:.2f}%\n")
            f.write(f"❌ 格错&答错: {(category_3 / total_samples * 100) if total_samples > 0 else 0:.2f}%\n")
            f.write(f"🍀 格错&答对: {(category_4 / total_samples * 100) if total_samples > 0 else 0:.2f}%\n")

    table = wandb.Table(data=log_data,
                        columns=["Prompt", "Response", "Truth", "Format R", "Answer R", "Total R", "Entropy"])
    wandb.log({
        "final_eval/generations": table,
        "final_eval/mean_reward": np.mean(total_rewards) if total_rewards else 0.0,
        "final_eval/category_1_ratio": (category_1 / total_samples * 100) if total_samples > 0 else 0.0,
        "final_eval/category_2_ratio": (category_2 / total_samples * 100) if total_samples > 0 else 0.0,
        "final_eval/category_3_ratio": (category_3 / total_samples * 100) if total_samples > 0 else 0.0,
    }, step=step)