import os
from typing import List,Dict,Optional,Tuple
import torch
import torch.nn.functional as F
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
import wandb
import numpy as np




# 掩码归一化,将对数熵与掩码结合求和
def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Returns:
        torch.Tensor: 归一化后的总和张量。
        在求和过程中，被掩码的元素（即 mask == 0 的位置）不会产生任何贡献。
    """
    masked_tensor = tensor * mask
    if dim is None:
        return masked_tensor.sum()/normalize_constant
    else:
        return masked_tensor.sum(dim=dim)/normalize_constant
# SFT 微批次训练步骤
def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
        - loss (torch.Tensor): 标量张量（Scalar tensor）。
          当前微批次（microbatch）的损失，并且已经除以了 gradient_accumulation_steps 进行缩放。
          返回它主要是为了在训练循环中累加或记录日志。
        - metadata (dict[str, torch.Tensor]): 字典格式。
          包含底层损失调用的元数据（例如未经过梯度累积缩放的原始 loss），以及你希望记录的任何其他统计数据。
    """
    # SFT 的目标是最小化负对数似然 (Negative Log-Likelihood)
    # 利用 response_mask 提取目标输出的 log_probs，求和并取负
    nll_sum = -masked_normalize(
        tensor=policy_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim = None # 在整个微批次所有 response token 上求和
    )
    #根据batch缩放
    batch_size = policy_log_probs.shape[0]
    # 考虑到梯度累积的影响，对 loss 进行缩放
    scaled_loss = nll_sum / gradient_accumulation_steps / batch_size
    
    scaled_loss.backward()
    # metadata 中返回原始 loss (剥离计算图) 方便 logging
    metadata = {
        "loss": nll_sum.detach()
    }

    return scaled_loss.detach(), metadata

# 记录迭代数量



def log_generations(
        prompts: List[str],
        generated_responses: List[str],
        ground_truths: List[str],
        token_entropies: List[torch.Tensor],  # 每个元素的形状为 (seq_len,)
        reward_fn,
        step: int,
        save_logs: Optional[str] = None,
        yaml_config_name: Optional[str] = None,
        tokenizer:PreTrainedTokenizer =None
):
    """
    一个日志辅助函数，用于将模型生成结果及分类统计记录到 wandb (或终端)。
    """
    log_data = []
    total_rewards = []
    format_rewards = []
    answer_rewards = []

    correct_lengths = []
    incorrect_lengths = []
    all_lengths = []

    # ===============================
    # 新增：定义各类别的计数器
    # ===============================
    category_1 = 0  # 格式对, 答案对
    category_2 = 0  # 格式对, 答案错
    category_3 = 0  # 格式错, 答案错
    category_4 = 0  # 格式错, 答案对 (兜底情况)

    for prompt, response, truth, entropies in zip(prompts, generated_responses, ground_truths, token_entropies):
        # 1. 计算各项奖励分数
        rewards = reward_fn(response, truth)
        r_total = rewards.get("reward", 0.0)
        r_format = rewards.get("format_reward", 0.0)
        r_answer = rewards.get("answer_reward", 0.0)
        
        total_rewards.append(r_total)
        format_rewards.append(r_format)
        answer_rewards.append(r_answer)

        # ===============================
        # 新增：判断类别逻辑 (假设分数 > 0 代表正确)
        # ===============================
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

        # 2. 计算平均 token 熵
        avg_entropy = entropies.mean().item() if len(entropies) > 0 else 0.0

        # 3. 统计生成长度
        resp_len = len(tokenizer.encode(response))
        all_lengths.append(resp_len)
        if r_total > 0:
            correct_lengths.append(resp_len)
        else:
            incorrect_lengths.append(resp_len)

        # 4. 汇总到单条记录，用于展示
        log_data.append([
            prompt, response, truth,
            r_format, r_answer, r_total,
            avg_entropy
        ])

    # ===============================
    # 新增：终端打印统计结果
    # ===============================
    total_samples = len(prompts)
    print(f"\n[Step {step}] === {yaml_config_name} 评估结果统计 ===")
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
    # 📝 新增：写入本地文本文件 (追加模式)
    # ===============================
    if save_logs:
        # 自动创建父级目录 (例如如果 save_logs 是 "output/sft_logs.txt"，就会自动创建 output 文件夹)
        os.makedirs(os.path.dirname(save_logs), exist_ok=True)
        
        # 使用 'a' 模式（追加模式）打开文件
        with open(save_logs, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"🚀 实验名称: {yaml_config_name} | 训练步数 (Step): {step}\n")
            f.write(f"{'='*50}\n")
            
            f.write("【🏆 奖励得分统计】\n")
            f.write(f"平均总奖励 (Total Reward):   {np.mean(total_rewards) if total_rewards else 0.0:.4f}\n")
            f.write(f"平均格式奖励 (Format Reward): {np.mean(format_rewards) if format_rewards else 0.0:.4f}\n")
            f.write(f"平均答案奖励 (Answer Reward): {np.mean(answer_rewards) if answer_rewards else 0.0:.4f}\n\n")
            
            f.write("【📏 生成长度统计】\n")
            f.write(f"平均生成总长度:     {np.mean(all_lengths) if all_lengths else 0.0:.2f} tokens\n")
            f.write(f"正确回答平均长度:   {np.mean(correct_lengths) if correct_lengths else 0.0:.2f} tokens\n")
            f.write(f"错误回答平均长度:   {np.mean(incorrect_lengths) if incorrect_lengths else 0.0:.2f} tokens\n\n")
            
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
            

    # 将文本数据转换为 WandB Table
    table = wandb.Table(
        data=log_data,
        columns=["Prompt", "Response", "Ground Truth", "Format Reward", "Answer Reward", "Total Reward", "Avg Entropy"]
    )

    # 记录统计平均值和占比到 WandB
    wandb.log({
        "eval/generations": table,
        "eval/mean_reward": np.mean(total_rewards) if total_rewards else 0.0,
        "eval/mean_format_reward": np.mean(format_rewards) if format_rewards else 0.0,
        "eval/mean_answer_reward": np.mean(answer_rewards) if answer_rewards else 0.0,
        "eval/mean_response_length": np.mean(all_lengths) if all_lengths else 0.0,
        "eval/mean_correct_response_length": np.mean(correct_lengths) if correct_lengths else 0.0,
        "eval/mean_incorrect_response_length": np.mean(incorrect_lengths) if incorrect_lengths else 0.0,
        
        # 新增：将占比也推送到 WandB (方便查看折线图趋势)
        "eval/category_1_ratio(%)": (category_1 / total_samples * 100) if total_samples > 0 else 0.0,
        "eval/category_2_ratio(%)": (category_2 / total_samples * 100) if total_samples > 0 else 0.0,
        "eval/category_3_ratio(%)": (category_3 / total_samples * 100) if total_samples > 0 else 0.0,


        # 📊 绝对数量指标 (Count) - 用于排查验证集抽样是否有偏差
        "eval/count_format_O_answer_O": category_1,
        "eval/count_format_O_answer_X": category_2,
        "eval/count_format_X_answer_X": category_3,
        "eval/count_format_X_answer_O": category_4,
    }, step=step)





