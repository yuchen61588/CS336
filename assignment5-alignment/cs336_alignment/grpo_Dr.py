import os
from typing import List,Dict,Optional,Tuple,Callable,Literal
import torch
import torch.nn.functional as F
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
import wandb
import numpy as np


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    核心逻辑：计算每组 rollout 回复的奖励，并在同组内部执行归一化操作。
    参数:
         reward_fn (Callable): 评分函数，用于对比模型回复与标准答案并返回得分。
         rollout_responses (list[str]): 策略生成的回答文本列表。
         repeated_ground_truths (list[str]): 与回答一一对应的标准答案列表。
         group_size (int): 每个问题对应的回答数量，即分组大小。
         advantage_eps (float): 防止除零异常的微小常数。
         normalize_by_std (bool): 是否需要除以组内标准差进行归一化的开关。
    返回:
         advantages (torch.Tensor): 组归一化后的优势值张量。
         raw_rewards (torch.Tensor): 未经归一化的原始奖励得分张量。
         metadata (Dict): 记录均值、标准差等额外统计特征的字典。
         最里面得作为一维数值张量。
         这个数据结构就是优势特征、原始分数和附加统计信息的集合，它明确告知调用者，这个函数固定返回三样东西：第一样是“用于优化的组归一化优势”，第二样是“监控表现用的原始奖励”，第三样是“记录中间统计量的数据字典”。
    算法细节:
            采用分组统计归一化策略。先调用评分函数获取所有原始奖励，将其变形为二维张量按组对齐，计算组内均值（及可选的标准差）执行标准化，最后展平回一维返回。
    """
    #计算每个分数的原始奖励
    raw_rewards_list = []
    for resp,gt in zip(rollout_responses,repeated_ground_truths):
        reward_dict = reward_fn(resp,gt)
        raw_rewards_list.append(reward_dict["reward"])

    # 创建奖励得分张量
    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)
    rollout_size = len(raw_rewards)

    # 分离batch_size(问题数量)，提出grope_size
    reshaped_rewards = raw_rewards.view(-1,group_size)

    group_means = reshaped_rewards.mean(dim=1, keepdim=True)

    if normalize_by_std:
        group_stds = reshaped_rewards.std(dim=-1,keepdim=True)
        advantages  = (reshaped_rewards-group_means)/(group_stds+advantage_eps)

    else:
        advantages = (reshaped_rewards - group_means)
    #恢复形状
    advantages = advantages.view(-1)

    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std().item(),
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std().item()
    }
    return advantages, raw_rewards, metadata

# 基于PG的损失函数PG_LOSS
def compute_naive_policy_gradient_loss( raw_rewards_or_advantages: torch.Tensor, policy_log_probs: torch.Tensor, ) -> torch.Tensor:
    """
        核心逻辑：在单 Token 粒度上执行朴素策略梯度损失的计算。
        参数:
             raw_rewards_or_advantages (torch.Tensor): 对应每个回答的标量原始奖励或组归一化优势值。
             policy_log_probs (torch.Tensor): 策略模型针对每个 Token 生成的对数概率。
        返回:
             loss (torch.Tensor): 每个 Token 位置的策略梯度损失。
             最里面得作为二维数值张量。
             这个数据结构就是每 Token 损失矩阵，它明确告知调用者，这个函数固定返回一样东西：“待在序列和批次维度聚合的未缩放策略梯度损失”。
        算法细节:
            采用张量广播相乘策略。将标量级别的奖励或优势值扩展维度后，与对数概率矩阵逐元素相乘，并整体取负值以满足梯度下降的最小化要求。
        """
    if raw_rewards_or_advantages.dim() == 1:
        adv = raw_rewards_or_advantages.unsqueeze(1)
    else:
        adv = raw_rewards_or_advantages

        # 每 token 损失： - A_t * log_p(o_t | q, o_{<t}) [cite: 675]
    return -adv * policy_log_probs

# GRPO-Clip loss.
def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float,) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    """
        核心逻辑：在单 Token 粒度上执行带有 PPO 风格裁剪机制的组相对策略优化损失计算。
        参数:
             advantages (torch.Tensor): 预先计算出的每个回答的优势值。
             policy_log_probs (torch.Tensor): 正在训练的新策略生成的每个 Token 对数概率。
             old_log_probs (torch.Tensor): 收集数据时旧策略生成的每个 Token 对数概率。
             cliprange (float): 限制新旧策略概率比例变动幅度的裁剪超参数。
        返回:
             loss (torch.Tensor): 每个 Token 经过裁剪处理后的损失张量。
             metadata (Dict): 记录裁剪比例、概率比值等统计信息的字典。
             最里面得作为二维数值张量。
             这个数据结构就是损失张量与统计字典的元组，它明确告知调用者，这个函数固定返回两样东西：第一样是“限制了更新幅度的裁剪损失矩阵”，第二样是“包含裁剪触发频率等日志信息的字典”。
        算法细节:
                采用重要性采样与裁剪限制策略。通过新旧对数概率相减取指数得到概率比值，分别计算未裁剪目标和限制在设定区间内的裁剪目标，取两者中较小的一方再取负转化为损失。
        """
    if advantages.dim()==1:
        advantages = advantages.unsqueeze(1)

    # 计算概率壁纸 因为这个是Log，除法就得exp
    radio = torch.exp(policy_log_probs-old_log_probs)

    unclipped_obj = radio* advantages
    clipped_ratio = torch.clamp(radio, 1.0 - cliprange, 1.0 + cliprange)
    clipped_obj = clipped_ratio * advantages

    # 目标是最大化
    unclipped_loss = -unclipped_obj
    clipped_loss = -clipped_obj

    loss = torch.max(unclipped_loss,clipped_loss)
    # 记录有多少 token 触发了裁剪限制
    is_clipped = (clipped_loss > unclipped_loss).float() #为了做平均

    metadata = {
        "clip_fraction": is_clipped.mean(),
        "ratio_mean":radio.mean()
    }
    return loss ,metadata


# 策略梯度损失包装器
# 策略梯度损失包装器。我们将运行消融比较三种不同版本的策略梯度：
# 朴素策略梯度，带基线的，GRPO

def compute_policy_gradient_loss( policy_log_probs: torch.Tensor,
                                  loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
                                  raw_rewards: torch.Tensor | None = None,
                                  advantages: torch.Tensor | None = None,
                                  old_log_probs: torch.Tensor | None = None,
                                  cliprange: float | None = None,

                                  ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    核心逻辑：在不同类型的策略梯度损失计算方法之间执行路由分发 [cite: 727, 729]。
        参数:
        policy_log_probs (torch.Tensor): 当前策略生成的各个 Token 的对数概率 [cite: 745]。
             loss_type (Literal): 指定当前使用的损失计算策略，支持无基线、带基线或 GRPO 裁剪 [cite: 746]。
             raw_rewards (Optional[torch.Tensor]): 原始奖励，仅当类型为无基线时需要 [cite: 747]。
             advantages (Optional[torch.Tensor]): 优势值，仅当类型为带基线或 GRPO 裁剪时需要 [cite: 747, 748]。
             old_log_probs (Optional[torch.Tensor]): 旧策略对数概率，仅当类型为 GRPO 裁剪时需要 [cite: 749]。
             cliprange (Optional[float]): 裁剪幅度阈值，仅当类型为 GRPO 裁剪时需要 [cite: 749]。
        返回:
             loss (torch.Tensor): 路由目标算法计算出的每 Token 损失结果 [cite: 752]。
             metadata (Dict): 路由目标算法返回的附加统计信息字典 [cite: 753]。
             最里面的 loss 作为二维张量。
             这个数据结构就是底层函数返回值的透传，它明确告知调用者，该包装器统一了不同强化学习变体的调用接口和返回值格式。
        算法细节:
                采用条件分支匹配策略。验证传入参数的完备性后，
                一旦匹配到指定的 loss_type，即调用底层对应的独立计算例程并返回结果 [cite: 755]。
        """
    if loss_type == 'no_baseline':
        assert raw_rewards is not None
        loss = compute_naive_policy_gradient_loss(raw_rewards,policy_log_probs)
        metadata = {}

    elif loss_type == 'reinforce_with_baseline':
        assert advantages is not None
        loss = compute_naive_policy_gradient_loss(advantages,policy_log_probs)
        metadata = {}

    elif loss_type == 'grpo_clip':
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None
        loss,metadata = compute_grpo_clip_loss(advantages,policy_log_probs,old_log_probs,cliprange)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss, metadata

# Dr.GRPO专用方法，总和不归一化
def masked_normalize(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        normalize_constant: float,
        dim: int | None = None,
) -> torch.Tensor:
    """
    核心逻辑：在遵守有效性掩码的前提下，执行指定维度的求和，并统一除以常数进行归一化。
    参数:
         tensor (torch.Tensor): 待执行求和与归一化操作的原始数据张量。
         mask (torch.Tensor): 标记各位置数据有效性的布尔掩码张量（1 代表包含）。
         normalize_constant (float): 固定的全局归一化常数（如 max_gen_len）。
         dim (Optional[int]): 设定要沿其聚合的张量维度，为空则在全局求和。
    返回:
         normalized_result (torch.Tensor): 过滤无效位置并除以常数后得出的张量。
         最里面得作为数值张量。
         这个数据结构就是固定标量归一化后的聚合结果，它明确告知调用者，该聚合方式剥离了样本自身长度对梯度权重的干扰。
    算法细节:
            采用定额缩放策略。将输入张量与布尔掩码逐元素相乘消除无效位数据，随后在指定维度上对有效数值求和，最后统一除以传入的 normalize_constant 标量进行放缩。
    """
    masked_tensor = tensor * mask

    if dim is None:
        summed = masked_tensor.sum()
    else:
        summed = masked_tensor.sum(dim=dim)

    return summed / normalize_constant
#
def masked_mean( tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None, ) -> torch.Tensor:
    """
        核心逻辑：在遵守布尔掩码的限制下执行指定维度上的平均值聚合操作 。
        参数:
             tensor (torch.Tensor): 待执行求均值操作的原始数据张量 。
             mask (torch.Tensor): 标记各位置数据有效性的布尔掩码张量，1 代表计算时需包含 。
             dim (Optional[int]): 设定要沿其求平均的张量维度，为空则求全局平均 。
        返回:
             masked_mean_result (torch.Tensor): 过滤掉无效位置后得出的均值张量 。
             最里面得作为数值张量。
             这个数据结构就是平均后的数值结果，它明确告知调用者，返回张量的形状语义等同于对原张量调用 mean(dim) 函数 。
        算法细节:
                采用先过滤后求均值的策略。将输入张量与掩码逐元素相乘消除无效数据，
                再对有效数据进行求和，最后除以掩码中有效标记的总数量 。
        """
    masked_tensor = tensor * mask
    if dim is None:
        return masked_tensor.sum() / torch.clamp(mask.sum(), min=0)
    else:
        # 对指定维度求均值，例如 dim=1 对每个序列求平均
        return masked_tensor.sum(dim=dim) / torch.clamp(mask.sum(dim=dim), min=0)

# GRPO 微批次训练步
def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    remove_length_norm: bool = False,
    normalize_constant: float = 1.0,

) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
        核心逻辑：在一个微批次内部依次执行损失计算、根据开关选择聚合方式、梯度缩放及反向传播操作。
        参数:
             policy_log_probs (torch.Tensor): 训练策略输出的每个 Token 对数概率。
             response_mask (torch.Tensor): 标识回复 Token 所在有效位置的掩码张量。
             gradient_accumulation_steps (int): 一次优化器更新所需的微批次累加总数。
             remove_length_norm (bool): Dr. GRPO 长度归一化消融开关。False 则使用 masked_mean（按自身长度均值），True 则使用 masked_normalize（除以定值）。
             normalize_constant (float): 当 remove_length_norm 为 True 时生效的除数常数。
             normalize_by_std:控制是否进行标准化计算。
             其余参数: 透传给策略梯度损失路由函数的对应参数。
        返回:
             loss (torch.Tensor): 当前微批次产生的、已针对梯度累加缩放过的标量损失。
             metadata (Dict): 包含底层损失计算所传递上来的各类监控特征字典。
             最里面得作为零维标量张量。
             这个数据结构就是标量损失与统计字典的元组，它明确告知调用者，这个函数固定返回两样东西：第一样是“用于打印监控且已触发反向传播的微批次总损失”，第二样是“汇总各步骤指标的数据字典”。
        算法细节:
                采用前向与反向串联执行策略。先调用包装器算出每 Token 损失。接着检查 remove_length_norm：若为 False，用 masked_mean 获取按实际长度平摊的样本损失；若为 True，用 masked_normalize 以全局常数获取样本损失。之后在批次维度求均值，除以累加步数缩放，调用 backward 触发求导。
        """
    per_token_loss,metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    # 序列为每条得的损失
    if not remove_length_norm:
        per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)
    else:
        per_example_loss = masked_normalize(
            per_token_loss,
            response_mask,
            normalize_constant=normalize_constant,
            dim=1
        )

    # 对一个match求平均
    batch_loss = per_example_loss.mean()
    # 4. 梯度累加缩放 [cite: 786, 817]
    scaled_loss = batch_loss / gradient_accumulation_steps

    # 5. 反向传播 [cite: 817]
    scaled_loss.backward()

    return scaled_loss, metadata


