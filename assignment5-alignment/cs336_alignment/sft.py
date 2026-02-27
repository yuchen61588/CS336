import os
from typing import List,Dict,Optional,Tuple
import torch
import torch.nn.functional as F
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
import wandb
import numpy as np


def tokenize_prompt_and_output(
        prompt_strs:List[str],
        output_strs:List[str],
        tokenizer:PreTrainedTokenizer,max_seq_length: int = 1024)->Dict[str, torch.Tensor]:
          
    """
    Returns:
        dict[str, torch.Tensor]: 包含分词后数据的字典。假设 `prompt_and_output_lens` 是合并后字符串的长度列表：
        - 'input_ids' (torch.Tensor): 形状为 (batch_size, max(prompt_and_output_lens) - 1)。
          拼接后的提示和输出 token，切掉了最后一个 token（因为最后一个 token 不需要作为输入去预测下一个词）。
        - 'labels' (torch.Tensor): 形状为 (batch_size, max(prompt_and_output_lens) - 1)。
          平移后的 input_ids，即去掉了第一个 token 的序列，用作交叉熵损失的目标标签。
        - 'response_mask' (torch.Tensor): 形状为 (batch_size, max(prompt_and_output_lens) - 1)。
          布尔掩码（0或1），在 labels 中仅针对模型生成的回答（response tokens）部分为 1，提示或填充部分为 0。
    """
    input_ids_list = []
    labels_list = []
    response_mask_list = []
    
    
    for prompt, output in zip(prompt_strs,output_strs):
        #tokenizer()
        # {
        #     "input_ids": [101, 2023, 3034, ...],      # ←  token ID 列表（你需要的）
        #     "attention_mask": [1, 1, 1, ...],          #  注意力掩码
        #     # 可能还有其他字段...
        # }
        prompt_tokens = tokenizer(prompt,add_special_tokens=False)["input_ids"]
        output_tokens = tokenizer(output, add_special_tokens=False)["input_ids"]
        
        # 拼接 tokens
        combined_tokens = prompt_tokens + output_tokens
        # # input_ids: 截掉最后一个 token (形状: L-1)
        # inp = combined_tokens[:-1]
        # # labels: 截掉第一个 token (形状: L-1)，整体平移了一格
        # lbl = combined_tokens[1:]
        # response_mask: 对于 prompt 部分为 0，对于 output 部分为 1
        # 注意预测第一个 output_token 是基于最后一个 prompt_token 的，所以在 label 中对应位置应该置为 1
        # 先生成等长的 raw_mask，然后跟 labels 一样截掉第一个 token
        raw_mask = [0] * len(prompt_tokens) + [1] * len(output_tokens)
        if len(combined_tokens) > max_seq_length:
            combined_tokens = combined_tokens[:max_seq_length]
            raw_mask = raw_mask[:max_seq_length]

        input_ids_list.append(torch.tensor(combined_tokens,dtype = torch.long))# nn.Embedding类型规定
        labels_list.append(torch.tensor( combined_tokens, dtype=torch.long))
        response_mask_list.append(torch.tensor(raw_mask, dtype=torch.long))
        # 只要是有真实 token 的地方，attention_mask 就是 1
        
    # 获取掩码token 默认为0
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # 填充处理，组装batch， 默认右侧补齐
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids_list,batch_first=True,padding_value=pad_token_id
    )
    # label 的 padding_value 
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels_list, batch_first=True, padding_value=pad_token_id
    )
    mask_padded = torch.nn.utils.rnn.pad_sequence(
        response_mask_list, batch_first=True, padding_value=0
    ) 
   
    shifted_inputs = input_ids_padded[:, :-1]
    
    shifted_labels = labels_padded[:, 1:]
    shifted_response_masks = mask_padded[:, 1:]

    return {
        "input_ids": shifted_inputs,
       
        "labels": shifted_labels,
        "response_mask": shifted_response_masks
    }

def compute_entropy(all_log_probs: torch.Tensor) -> torch.Tensor:
    """
    Returns:
        他声称的时一个vocab词表，对每个词表的概率
        torch.Tensor: 形状为 (batch_size, sequence_length)。
        计算出的每个 next-token 预测的熵值（Entropy）。
    """
    probs = torch.exp(all_log_probs)
    entropy = -torch.sum(probs * all_log_probs, dim=-1)
    # 处理可能的 NaN (如 log(0) 带来的问题)
    return torch.nan_to_num(entropy, nan=0.0)

def get_response_log_probs(  model: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor, return_token_entropy: bool = False, ) -> dict[str, torch.Tensor]:
    """
    Returns:
        dict[str, torch.Tensor]: 包含对数概率和（可选的）熵的字典。
        - 'log_probs' (torch.Tensor): 形状为 (batch_size, sequence_length)。
          模型对于标准答案的条件对数概率 \log p_θ(x_t | x_{<t})。
        - 'token_entropy' (torch.Tensor, 可选): 形状为 (batch_size, sequence_length)。
          每个位置上模型预测分布的逐 token 熵（仅在参数 return_token_entropy=True 时返回）。
    """
    outputs = model(input_ids)
    logits = outputs.logits  # (batch_size, seq_len, vocab_size)
    batch_size, seq_len, vocab_size = logits.shape
   

    # 准备 gather_labels：防止 labels 中的 padding 值 (如 -100) 导致 gather 越界
    gather_labels = labels.clone()
    gather_labels[gather_labels == -100] = 0

    if not return_token_entropy:
        # 🌟 极致优化版：关闭 Entropy 时，直接走 CrossEntropy，节约 80% 显存！
        # logits 展平: (B*L, V), labels 展平: (B*L)
        per_token_loss = F.cross_entropy(
            logits.view(-1, vocab_size), 
            labels.view(-1), 
            reduction='none',
            ignore_index=-100  # 直接让 PyTorch 忽略 padding
        )
        # 变回 (B, L) 并取负号得到 log_prob
        selected_log_probs = -per_token_loss.view(batch_size, seq_len)
        return {"log_probs": selected_log_probs}
        
    else:
        # ⚠️ 必须开启 Entropy 时：转 FP32 防溢出，且坚决只算一次 log_softmax
        logits_fp32 = logits.to(torch.float32)
        all_log_probs = F.log_softmax(logits_fp32, dim=-1)
        
        # 1. 提取目标 token 概率
        selected_log_probs = torch.gather(
            all_log_probs, dim=-1, index=gather_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # 2. 复用 all_log_probs 计算熵，而不是重新传入 logits
        token_entropy = compute_entropy(all_log_probs)
        
        return {
            "log_probs": selected_log_probs,
            "token_entropy": token_entropy
        }

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
        yaml_config_name: Optional[str] = None
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
        resp_len = len(response)
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





