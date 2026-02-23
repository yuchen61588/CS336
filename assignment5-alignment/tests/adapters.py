from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from cs336_alignment.sft import tokenize_prompt_and_output,masked_normalize,compute_entropy,get_response_log_probs,sft_microbatch_train_step


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """对提示词（prompt）和输出（output）字符串进行分词，并构建一个掩码（mask）。
该掩码在回答（response）令牌处为 1，在其他令牌（提示词或填充位/padding）处为 0。

参数:
    prompt_strs: list[str]，提示词字符串列表。
    output_strs: list[str]，输出字符串列表。
    tokenizer: PreTrainedTokenizer，所使用的分词器。

返回:
    dict[str, torch.Tensor]:
        "input_ids": 形状为 (batch_size, max(prompt_and_output_lens) - 1) 的张量：
            分词后的提示词与输出字符串，且切掉了最后一个令牌。
        "labels": 形状为 (batch_size, max(prompt_and_output_lens) - 1) 的张量：
            偏移后的 input_ids（即去掉了第一个令牌的 input_ids）。
        "response_mask": 形状为 (batch_size, max(prompt_and_output_lens) - 1) 的张量：
            对应 `labels` 中回答（response）部分的令牌掩码。
"""

    return tokenize_prompt_and_output(prompt_strs,output_strs,tokenizer)
    raise NotImplementedError


def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
计算每一组采样回答（rollout responses）的奖励值，
并按组大小（group size）进行归一化处理。

关于 GRPO 的更多信息，请参阅：
    DeepSeekMath: https://arxiv.org/abs/2402.03300
    DeepSeek-R1: https://arxiv.org/abs/2501.12948

参数:
    reward_fn: Callable[[str, str], dict[str, float]]，
        根据标准答案（ground truths）对采样回答进行打分，
        生成包含 "reward"、"format_reward" 和 "answer_reward" 键的字典。
    rollout_responses: list[str]，策略模型生成的采样回答列表。
        该列表的长度为 `rollout_batch_size = n_prompts_per_rollout_batch * group_size`。
    repeated_ground_truths: list[str]，样本的标准答案列表。
        该列表长度为 `rollout_batch_size`，因为每个样本的标准答案
        都重复了 `group_size` 次（以对应同组内的多个采样）。
    group_size: int，每个组（每个 Prompt）对应的采样回答数量。
    advantage_eps: float，用于组归一化时防止除以零的 epsilon 值。
    normalize_by_std: bool，是否使用标准差（std）对奖励值进行归一化。

返回:
    tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        形状为 (rollout_batch_size,) 的 torch.Tensor：
            每个采样回答经组归一化后的奖励值（通常作为 Advantage）。
        形状为 (rollout_batch_size,) 的 torch.Tensor：
            每个采样回答的原始奖励值（Raw rewards）。
        dict[str, float]: 采样批次（rollout batch）奖励值的元数据。
            你可以选择在此记录任何统计信息（如奖励值的均值、方差等）。
"""
    raise NotImplementedError


def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    return compute_entropy(logits)
    raise NotImplementedError


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """获取给定提示词（prompt）下回答（response）的条件对数概率，
并可选择性地返回下个令牌预测的信息熵（entropy）。

参数:
    model: PreTrainedModel，用于评分的模型。
    input_ids: 形状为 (batch_size, sequence_length) 的 torch.Tensor：
        分词后的提示词与输出内容。
    labels: 形状为 (batch_size, sequence_length) 的 torch.Tensor：
        偏移后的 input_ids（用于对齐预测目标）。
    return_token_entropy: bool，是否返回下个令牌预测的信息熵。

返回:
    dict[str, torch.Tensor]:
        "log_probs": 形状为 (batch_size, sequence_length) 的 torch.Tensor：
            在给定提示词条件下，回答部分的条件对数概率。
            注意：此处尚未屏蔽（mask）掉提示词或填充位（padding）对应的令牌索引；
            该处理步骤将在训练循环中完成。
        "token_entropy": 形状为 (batch_size, sequence_length) 的可选 torch.Tensor：
            下个令牌预测的信息熵。与 log-probs 一样，此处尚未屏蔽掉提示词
            或填充位对应的令牌索引；该处理步骤将在训练循环中完成。
"""
    return get_response_log_probs(model,input_ids,labels,return_token_entropy)
    raise NotImplementedError


def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    raise NotImplementedError


def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    raise NotImplementedError


def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    raise NotImplementedError


def run_masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """计算张量在指定维度上的均值，
且仅考虑掩码（mask）值为 1 的元素。

参数:
    tensor: torch.Tensor，要计算均值的张量。
    mask: torch.Tensor，掩码张量。我们仅对掩码值为 1
        的元素进行均值计算。
    dim: int | None，计算均值所沿的维度。
        如果为 None，则对所有未被掩码（non-masked）的元素求和，
        并除以它们的总数来计算平均值。

返回:
    torch.Tensor，在指定维度上仅考虑掩码值为 1
        的元素所得到的张量均值。
"""

    raise NotImplementedError

def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    return sft_microbatch_train_step(policy_log_probs,response_mask,gradient_accumulation_steps,normalize_constant)
    raise NotImplementedError

    
def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """

    raise NotImplementedError


def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """在指定维度上求和并除以一个常数进行归一化，
且仅考虑掩码（mask）值为 1 的元素。

参数:
    tensor: torch.Tensor，要进行求和及归一化的张量。
    mask: torch.Tensor，掩码张量。我们仅考虑掩码值为 1
        的元素。
    dim: int | None，归一化前进行求和操作的维度。
        如果为 None，则在所有维度上进行求和。
    normalize_constant: float，用于归一化除法的常数。

返回:
    torch.Tensor，归一化后的各项之和，
        其中被掩码的元素（mask=0）不对求和结果产生贡献。
"""
    masked_normalize(tensor, mask, normalize_constant,dim)
    raise NotImplementedError


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    raise NotImplementedError


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    raise NotImplementedError


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    raise NotImplementedError
