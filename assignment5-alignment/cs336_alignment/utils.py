
from typing import List,Dict
import torch
import torch.nn.functional as F
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel




def tokenize_prompt_and_output(
        prompt_strs: List[str],
        output_strs: List[str],
        tokenizer: PreTrainedTokenizer, max_seq_length: int = 1024) -> Dict[str, torch.Tensor]:
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

    for prompt, output in zip(prompt_strs, output_strs):
        # tokenizer()
        # {
        #     "input_ids": [101, 2023, 3034, ...],      # ←  token ID 列表（你需要的）
        #     "attention_mask": [1, 1, 1, ...],          #  注意力掩码
        #     # 可能还有其他字段...
        # }
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
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

        input_ids_list.append(torch.tensor(combined_tokens, dtype=torch.long))  # nn.Embedding类型规定
        labels_list.append(torch.tensor(combined_tokens, dtype=torch.long))
        response_mask_list.append(torch.tensor(raw_mask, dtype=torch.long))
        # 只要是有真实 token 的地方，attention_mask 就是 1

    # 获取掩码token 默认为0
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # 填充处理，组装batch， 默认右侧补齐
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=pad_token_id
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


def get_response_log_probs(model: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor,
                           return_token_entropy: bool = False, ) -> dict[str, torch.Tensor]:
    """
    Returns:
        dict[str, torch.Tensor]: 包含对数概率和（可选的）熵的字典。
        - 'log_probs' (torch.Tensor): 形状为 (batch_size, sequence_length)。
          模型对于标准答案的条件对数概率 \log p_θ(x_t | x_{<t})。
        - 'token_entropy' (torch.Tensor, 可选): 形状为 (batch_size, sequence_length)。
          每个位置上模型预测分布的逐 token 熵（仅在参数 return_token_entropy=True 时返回）。
    """
    outputs = model(input_ids)  # 调用forward方法
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
        )
        # 变回 (B, L) 并取负号得到 log_prob
        selected_log_probs = -per_token_loss.view(batch_size, seq_len)
        return {"log_probs": selected_log_probs}

    else:
        # ⚠️ 必须开启 Entropy 时：转 FP32 防溢出，且坚决只算一次 log_softmax
        logits_fp32 = logits.to(torch.float32)
        all_log_probs = F.log_softmax(logits_fp32, dim=-1)

        labels_expanded = gather_labels.unsqueeze(-1)
        # 1. 提取目标 token 概率
        selected_log_probs = torch.gather(
            all_log_probs, dim=-1, index=labels_expanded
        ).squeeze(-1)

        # 2. 复用 all_log_probs 计算熵，而不是重新传入 logits
        token_entropy = compute_entropy(all_log_probs)

        return {
            "log_probs": selected_log_probs,
            "token_entropy": token_entropy
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