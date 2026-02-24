import os
from typing import List,Dict,Optional,Tuple
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer,PreTrainedModel
import wandb
import numpy as np


def tokenize_prompt_and_output(
        prompt_strs:List[str],
        output_strs:List[str],
        tokenizer:PreTrainedTokenizer)->Dict[str, torch.Tensor]:
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

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns:
        他声称的时一个vocab词表，对每个词表的概率
        torch.Tensor: 形状为 (batch_size, sequence_length)。
        计算出的每个 next-token 预测的熵值（Entropy）。
    """
    log_probs = F.log_softmax(logits , dim=-1)
    probs = torch.exp(log_probs)

    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def get_response_log_probs(  model: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor, return_token_entropy: bool = False, ) -> dict[str, torch.Tensor]:
    """
    Returns:
        dict[str, torch.Tensor]: 包含对数概率和（可选的）熵的字典。
        - 'log_probs' (torch.Tensor): 形状为 (batch_size, sequence_length)。
          模型对于标准答案的条件对数概率 \log p_θ(x_t | x_{<t})。
        - 'token_entropy' (torch.Tensor, 可选): 形状为 (batch_size, sequence_length)。
          每个位置上模型预测分布的逐 token 熵（仅在参数 return_token_entropy=True 时返回）。
    """
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits,dim=-1)

    # 准备 gather_labels：防止 labels 中的 padding 值 (如 -100) 导致 gather 越界
    gather_labels = labels.clone()
    gather_labels[gather_labels == -100] = 0

    gathered_log_probs = torch.gather(
        log_probs, dim=-1, index=gather_labels.unsqueeze(-1)
    ).squeeze(-1)

    result = {"log_probs": gathered_log_probs}

    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result

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
        step: int
):
    """
    一个日志辅助函数，用于将模型生成结果记录到 wandb (或终端)。
    这里假设模型输出已经由 vLLM 或 HF generate 获得。
    """
    log_data = []
    total_rewards = []
    format_rewards = []
    answer_rewards = []

    correct_lengths = []
    incorrect_lengths = []
    all_lengths = []

    for prompt, response, truth, entropies in zip(prompts, generated_responses, ground_truths, token_entropies):
        # 1. 计算各项奖励分数
        rewards = reward_fn(response, truth)
        total_rewards.append(rewards["reward"])
        format_rewards.append(rewards["format_reward"])
        answer_rewards.append(rewards["answer_reward"])

        # 2. 计算平均 token 熵
        avg_entropy = entropies.mean().item() if len(entropies) > 0 else 0.0

        # 3. 统计生成长度
        resp_len = len(response)  # 这里可替换为 tokenizer(response) 的长度
        all_lengths.append(resp_len)
        if rewards["reward"] > 0:
            correct_lengths.append(resp_len)
        else:
            incorrect_lengths.append(resp_len)

        # 4. 汇总到单条记录，用于展示
        log_data.append([
            prompt, response, truth,
            rewards["format_reward"], rewards["answer_reward"], rewards["reward"],
            avg_entropy
        ])

    # 将文本数据转换为 WandB Table
    table = wandb.Table(
        data=log_data,
        columns=["Prompt", "Response", "Ground Truth", "Format Reward", "Answer Reward", "Total Reward", "Avg Entropy"]
    )

    # 记录统计平均值
    wandb.log({
        "eval/generations": table,
        "eval/mean_reward": np.mean(total_rewards),
        "eval/mean_format_reward": np.mean(format_rewards),
        "eval/mean_answer_reward": np.mean(answer_rewards),
        "eval/mean_response_length": np.mean(all_lengths),
        "eval/mean_correct_response_length": np.mean(correct_lengths) if correct_lengths else 0.0,
        "eval/mean_incorrect_response_length": np.mean(incorrect_lengths) if incorrect_lengths else 0.0,
    }, step=step)






