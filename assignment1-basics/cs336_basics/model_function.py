import torch
import torch.nn as nn
import math
from einops import einsum,repeat,rearrange



def softmax(tensor: torch.Tensor, dim: int)-> torch.Tensor:
    """
        实现 Softmax 函数，包含数值稳定性技巧 (Subtract Max)。
        Args:
            tensor (torch.Tensor): 输入张量
            dim (int): 需要进行 softmax 的维度
        Returns:
            torch.Tensor: 输出张量，形状与输入相同，指定维度的和为 1
    """
    # 计算最大值，方便广播
    max_val = tensor.max(dim=dim,keepdim=True).values
    # 相减保证安全
    x_safe = tensor-max_val

    # 计算维度
    exp_x = torch.exp(x_safe)

    # 总维度
    sum_exp_x = torch.sum(dim=dim,keepdim=True)

    return exp_x/sum_exp_x

def scaled_dot_product_attention(
        q:torch.Tensor,
        k:torch.Tensor,
        v:torch.Tensor,
        mask:torch.Tensor)->torch.Tensor:
    """
    实现 Scaled Dot-Product Attention。
        Args:
            query: 形状 (batch_size, ..., seq_len_q, d_k)
            key:   形状 (batch_size, ..., seq_len_k, d_k)
            value: 形状 (batch_size, ..., seq_len_k, d_v)
            mask:  可选，形状 (seq_len, seq_len)。
                   True 表示参与 Attention，False 表示被屏蔽 (masked out)。
        Returns:
            output: 形状 (batch_size, ..., seq_len, d_v)
        """
    d_k = q.shape[-1]
    scale = 1.0/math.sqrt(d_k)
    # 2. 计算相似度分数 (Q @ K^T)
    #   ... : 自动匹配所有前面的维度 (Batch, Heads)
    #   i   : Query 的序列长度
    #   j   : Key 的序列长度
    #   d   : 这里的 d 就是 d_k，我们要在这个维度上做点积 (消掉它)
    scores = einsum(q,k,'... i d, ... j d -> ... i j')
    scores = scores*scale
    # 设置掩码
    if mask is not None:
        # 文档要求: 把 Mask 为 False 的位置填为 -inf
        # 这里的 mask 会自动广播以匹配 scores 的维度
        scores = scores.masked_fill(mask == False, float('-inf'))

    attn_probs = scores.softmax(dim=-1)

    # pattern: 把 Attention 权重 (i, j) 和 Value (j, v) 结合
    # 消掉 j 维度，得到最终结果 (i, v)
    output = einsum(attn_probs,v,'... i j, ... j v -> ... i v')

    return output
