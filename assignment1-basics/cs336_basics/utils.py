from __future__ import annotations

import math
from typing import Iterable

import torch


def cross_entropy(logits:torch.Tensor,targets:torch.Tensor)->torch.Tensor:
    """
        计算交叉熵损失 (Cross Entropy Loss)，使用 Log-Sum-Exp 技巧保证数值稳定性。

        公式推导:
        Loss = -log(softmax(x)[y])
             = -log( exp(x_y) / sum(exp(x_j)) )
             = log(sum(exp(x_j))) - x_y

        为了稳定性，引入 c = max(x):
        Loss = log(sum(exp(x_j - c))) - (x_y - c)

        Args:
            logits: 模型输出的未归一化分数，形状为 (batch_size, seq_len, vocab_size)
                    或者 (..., vocab_size)
            targets: 真实标签索引，形状为 (batch_size, seq_len)
                    或者 (...)

        Returns:
            loss: 标量 (Scalar)，所有样本损失的平均值
        """
    # 1. 数值稳定性 (Numerical Stability)
    # 找到最后一个维度 (vocab维度) 的最大值，保持维度以便广播
    max_logits = logits.max(dim=-1, keepdim=True).values

    # 全局平移：logits_shifted = x - c
    logits_shifted = logits - max_logits

    # 2. 计算第一项: Log-Sum-Exp
    # log(sum(exp(x_j - c)))
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_shifted), dim=-1))

    # 3. 计算第二项: Target Logits (x_y - c)
    # targets 的形状通常是 (Batch, Seq)，我们需要把它扩展成 (Batch, Seq, 1) 才能在最后一维做 gather
    # gather 出来的结果形状是 (Batch, Seq, 1)，最后 squeeze 掉多余的 1
    target_logits_shifted = torch.gather(logits_shifted, -1, targets.unsqueeze(-1)).squeeze(-1)

    # 4. 计算 Loss
    # Loss = Term1 - Term2
    loss = log_sum_exp - target_logits_shifted

    # 5. 返回平均值
    return loss.mean() #两个维度全部mean了



def get_lr_cosine_schedule(t:int,alpha_max:float,alpha_min:float,T_w:int,T_c:int)->float:
    """
        计算给定步数 t 的余弦退火学习率（包含预热）。

        Args:
            t (int): 当前迭代步数。
            alpha_max (float): 最大学习率 (预热结束后的峰值)。
            alpha_min (float): 最小学习率 (退火结束后的值)。
            T_w (int): 预热迭代次数 (Warm-up iterations)。
            T_c (int): 余弦退火结束的总迭代次数 (Cosine annealing iterations)。
                       注意：这是指整个训练过程的时间轴节点，即退火在第 T_c 步结束。

        Returns:
            float: 当前步数 t 对应的学习率 alpha_t。
        """
    if t<T_w:
        return (t/T_w)*alpha_max
    elif t<T_c:
        return alpha_min+0.5*(1+math.cos((t-T_w)/(T_c-T_w)*math.pi))*(alpha_max-alpha_min)

    else:
        return alpha_min


def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6)->None:
    """
    逻辑:
        1. 计算所有参数梯度的全局 L2 范数 ||g||_2。
        2. 如果 ||g||_2 > max_norm:
           则对梯度进行缩放: g <- g * (max_norm / (||g||_2 + eps))
        3. 如果 ||g||_2 <= max_norm:
           不做任何改变。

        Args:
            params: 需要裁剪梯度的参数列表 (例如 model.parameters())。
            max_norm: 允许的最大 L2 范数 (M)。
            eps: 数值稳定性项 (默认 1e-6)。
        """

    # 遍历两次，转化为列表
    p_list = [p for p in params if p.grad is not None]

    if not p_list:
        return
    # total_norm = sqrt( sum( ||p.grad||^2 ) )
    total_norm_sq = sum(p.grad.detach().norm(2).item() **2 for p in p_list)
    total_norm = math.sqrt(total_norm_sq)

    if total_norm> max_norm:
        scale_coef = max_norm/(total_norm+eps)

        for p in p_list:
            p.grad.detach().mul_(scale_coef)



