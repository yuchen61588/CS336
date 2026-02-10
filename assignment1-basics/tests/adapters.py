from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.train_bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer
from typing import List, Tuple, Dict, Optional, Any

from cs336_basics.model import (
    Linear,
    Embedding,
    SwiGLUFeedForward,
    ROPE,
    CausalMultiHeadSelfAttention,
    Transformer_block,
    TransformerLM,
    RMSNorm
)
from cs336_basics.model_function import scaled_dot_product_attention, softmax
from cs336_basics.training_tools import cross_entropy,AdamW,gradient_clipping,get_lr_cosine_schedule
import torch.nn.functional as F
def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
        给定线性层的权重，计算批量输入的变换。

        参数:
            in_dim (int): 输入维度的大小
            out_dim (int): 输出维度的大小
            weights (Float[Tensor, "d_out d_in"]): 要使用的线性权重
            in_features (Float[Tensor, "... d_in"]): 要应用函数的输出张量

        返回:
            Float[Tensor, "... d_out"]: 线性模块的变换输出。
        """
    linear = Linear(d_in,d_out)
    linear.weight.data = weights
    return linear(in_features)



    raise NotImplementedError


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    给定 Embedding 层的权重，获取一批 token ID 对应的嵌入向量。

    参数:
        vocab_size (int): 词汇表中嵌入向量的数量（词表大小）
        d_model (int): 嵌入维度的尺寸（向量长度）
        weights (Float[Tensor, "vocab_size d_model"]): 要从中提取的嵌入向量矩阵
        token_ids (Int[Tensor, "..."]): 要从 Embedding 层查询的 token ID 集合

    返回:
        Float[Tensor, "... d_model"]: Embedding 层返回的一批嵌入向量。
    """
    embed = Embedding(vocab_size,d_model)
    embed.weight.data = weights
    return embed(token_ids)

    raise NotImplementedError


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
        给定 SwiGLU 网络的权重，返回使用这些权重的实现输出。

        参数:
            d_model (int): 前馈输入和输出的维度
            d_ff (int): SwiGLU 内部升维投影的维度
            w1_weight (Float[Tensor, "d_ff d_model"]): W1 的存储权重
            w2_weight (Float[Tensor, "d_model d_ff"]): W2 的存储权重
            w3_weight (Float[Tensor, "d_ff d_model"]): W3 的存储权重
            in_features (Float[Tensor, "... d_model"]): 输入到前馈层的嵌入向量

        返回:
            Float[Tensor, "... d_model"]: 与输入嵌入形状相同的输出嵌入
        """
    # 示例:
    # 如果状态字典键匹配，可以使用 `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # 也可以手动赋值权重
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    swiglu = SwiGLUFeedForward(d_model,d_ff)
    swiglu.w1.weight.data = w1_weight
    swiglu.w2.weight.data=w2_weight
    swiglu.w3.weight.data=w3_weight
    return swiglu(in_features)

    raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
        给定键（K）、查询（Q）和值（V）张量，返回缩放点积注意力的实现输出。

        参数:
            Q (Float[Tensor, " ... queries d_k"]): 查询张量
            K (Float[Tensor, " ... keys d_k"]): 键张量
            V (Float[Tensor, " ... values d_v"]): 值张量
            mask (Bool[Tensor, " ... queries keys"] | None): 掩码张量

        返回:
            Float[Tensor, " ... queries d_v"]: SDPA（缩放点积注意力）的输出
        """
    return scaled_dot_product_attention(Q,K,V,mask)
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
        给定一个朴素非批量化多头注意力实现的键、查询和值投影权重，
        返回优化批量化实现的输出。该实现应在**单次矩阵乘法中**
        处理所有头的键、查询和值投影。
        此函数不应使用 RoPE。
        参见 Vaswani et al., 2017 论文第 3.2.2 节。

        参数:
            d_model (int): 前馈输入和输出的维度
            num_heads (int): 多头注意力中使用的头数
            max_seq_len (int): 如果实现需要预缓存，最大序列长度
            q_proj_weight (Float[Tensor, "d_k d_in"]): Q 投影的权重
            k_proj_weight (Float[Tensor, "d_k d_in"]): K 投影的权重
            v_proj_weight (Float[Tensor, "d_k d_in"]): V 投影的权重
            o_proj_weight (Float[Tensor, "d_model d_v"]): 输出投影的权重
            in_features (Float[Tensor, "... sequence_length d_in"]): 要在其上运行实现的张量

        返回:
            Float[Tensor, " ... sequence_length d_out"]: 使用给定的 QKV 投影权重和输入特征，
            运行优化批量化多头注意力实现后的输出张量。
        """
    mha = CausalMultiHeadSelfAttention(d_model,num_heads)

    qkv_weight =torch.cat([q_proj_weight,k_proj_weight,v_proj_weight],dim=0)
    mha.qkv_proj.weight.data = qkv_weight
    mha.o_proj.weight.data = o_proj_weight
    # 这是一个不带 RoPE 的测试，传入 None
    return mha(in_features, rope_embed=None)

    raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    给定一个朴素非批量化多头注意力实现的键、查询和值投影权重，
    返回优化批量化实现的输出。该实现应在**单次矩阵乘法中**
    处理所有头的键、查询和值投影。
    此版本的 MHA 应包含 RoPE。
    在此情况下，RoPE 嵌入维度必须等于头的嵌入维度（d_model // num_heads）。
    参见 Vaswani et al., 2017 论文第 3.2.2 节。

    参数:
        d_model (int): 前馈输入和输出的维度
        num_heads (int): 多头注意力中使用的头数
        max_seq_len (int): 如果实现需要预缓存，最大序列长度
        theta (float): RoPE 参数（频率基数，通常 10000）
        q_proj_weight (Float[Tensor, "d_k d_in"]): Q 投影的权重
        k_proj_weight (Float[Tensor, "d_k d_in"]): K 投影的权重
        v_proj_weight (Float[Tensor, "d_k d_in"]): V 投影的权重
        o_proj_weight (Float[Tensor, "d_model d_v"]): 输出投影的权重
        in_features (Float[Tensor, "... sequence_length d_in"]): 要在其上运行实现的张量
        token_positions (Int[Tensor, " ... sequence_length"] | None): 可选的 token 位置张量

    返回:
        Float[Tensor, " ... sequence_length d_out"]: 使用给定的 QKV 投影权重和输入特征，
        运行优化批量化多头注意力实现后的输出张量。
    """
    mha = CausalMultiHeadSelfAttention(d_model, num_heads)

    # 同样进行权重合并
    qkv_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
    mha.qkv_proj.weight.data = qkv_weight
    mha.o_proj.weight.data = o_proj_weight

    d_k = d_model//num_heads
    rope = ROPE(theta,d_k,max_seq_len)
    return mha(in_features, rope_embed=rope, token_positions=token_positions)




    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    对给定输入张量运行 RoPE。

    参数:
        d_k (int): 查询或键张量的嵌入维度大小
        theta (float): RoPE 参数（频率基数）
        max_seq_len (int): 如果实现需要预缓存，最大序列长度
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): 要运行 RoPE 的输入张量
        token_positions (Int[Tensor, "... sequence_length"]): 形状为 (batch_size, sequence_length) 的张量，
                                                              包含 token 的位置信息
    返回:
        Float[Tensor, " ... sequence_length d_k"]: 经过 RoPE 处理的输入张量
    """
    rope = ROPE(theta, d_k, max_seq_len)
    return rope(in_query_or_key,token_positions)
    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
        给定预归一化 Transformer 块的权重和输入特征，返回在输入特征上运行 Transformer 块的输出。

        此函数应使用 RoPE。
        根据你的实现，你可能只需将相关参数传递给 TransformerBlock 构造函数，
        或者你可能需要初始化自己的 RoPE 类并传递它。

        参数:
            d_model (int): Transformer 块输入的维度
            num_heads (int): 多头注意力中使用的头数。`d_model` 必须能被 `num_heads` 整除。
            d_ff (int): 前馈网络内层的维度
            max_seq_len (int): 如果实现需要预缓存，最大序列长度
            theta (float): RoPE 参数
            weights (dict[str, Tensor]):
                参考实现的状态字典。键包括：
                - `attn.q_proj.weight`
                    所有 `num_heads` 注意力头的查询投影。
                    形状为 (d_model, d_model)。
                    行按 (num_heads, d_k) 形状的矩阵排序，
                    即 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
                - `attn.k_proj.weight`
                    所有 `num_heads` 注意力头的键投影。
                    形状为 (d_model, d_model)。
                    行按 (num_heads, d_k) 形状的矩阵排序，
                    即 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
                - `attn.v_proj.weight`
                    所有 `num_heads` 注意力头的值投影。
                    形状为 (d_model, d_model)。
                    行按 (num_heads, d_v) 形状的矩阵排序，
                    即 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
                - `attn.output_proj.weight`
                    多头自注意力输出投影的权重。
                    形状为 (d_model, d_model)。
                - `ln1.weight`
                    Transformer 块中第一个 RMSNorm 仿射变换的权重。
                    形状为 (d_model,)。
                - `ffn.w1.weight`
                    FFN 中第一个线性变换的权重。
                    形状为 (d_model, d_ff)。
                - `ffn.w2.weight`
                    FFN 中第二个线性变换的权重。
                    形状为 (d_ff, d_model)。
                - `ffn.w3.weight`
                    FFN 中第三个线性变换的权重。
                    形状为 (d_model, d_ff)。
                - `ln2.weight`
                    Transformer 块中第二个 RMSNorm 仿射变换的权重。
                    形状为 (d_model,)。
            in_features (Float[Tensor, "batch sequence_length d_model"]):
                要在其上运行实现的张量。

        返回:
            Float[Tensor, "batch sequence_length d_model"]: 使用 RoPE 在输入特征上运行 Transformer 块的输出张量。
        """
    block = Transformer_block(d_model, num_heads, d_ff)

    # 构造 RoPE
    d_k = d_model // num_heads
    rope = ROPE(theta, d_k, max_seq_len)

    # 手动加载权重以处理 QKV 合并
    q = weights['attn.q_proj.weight']
    k = weights['attn.k_proj.weight']
    v = weights['attn.v_proj.weight']
    block.mha.qkv_proj.weight.data = torch.cat([q, k, v], dim=0)

    block.mha.o_proj.weight.data = weights['attn.output_proj.weight']
    block.rms_norm1.weight.data = weights['ln1.weight']
    block.rms_norm2.weight.data = weights['ln2.weight']
    block.ffn.w1.weight.data = weights['ffn.w1.weight']
    block.ffn.w2.weight.data = weights['ffn.w2.weight']
    block.ffn.w3.weight.data = weights['ffn.w3.weight']

    return block(in_features, rope_embed=rope)
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """
        给定 Transformer 语言模型的权重和输入索引，返回在输入索引上运行前向传播的输出。

        此函数应使用 RoPE。

        参数:
            vocab_size (int): 输出词汇表中要预测的唯一项数量（词表大小）
            context_length (int): 一次处理的最大 token 数（上下文长度）
            d_model (int): 模型嵌入和子层输出的维度
            num_layers (int): 使用的 Transformer 层数
            num_heads (int): 多头注意力中使用的头数。`d_model` 必须能被 `num_heads` 整除。
            d_ff (int): 前馈网络内层的维度（论文第 3.3 节）
            rope_theta (float): RoPE 的 $\Theta$ 参数（频率基数）
            weights (dict[str, Tensor]):
                参考实现的状态字典。{num_layers} 表示介于 `0` 和 `num_layers - 1` 之间的整数（层索引）。
                字典的键包括：
                - `token_embeddings.weight`
                    Token 嵌入矩阵。形状为 (vocab_size, d_model)。
                - `layers.{num_layers}.attn.q_proj.weight`
                    所有 `num_heads` 注意力头的查询投影。
                    形状为 (num_heads * (d_model / num_heads), d_model)。
                    行按 (num_heads, d_k) 形状的矩阵排序，
                    即 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
                - `layers.{num_layers}.attn.k_proj.weight`
                    所有 `num_heads` 注意力头的键投影。
                    形状为 (num_heads * (d_model / num_heads), d_model)。
                    行按 (num_heads, d_k) 形状的矩阵排序，
                    即 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
                - `layers.{num_layers}.attn.v_proj.weight`
                    所有 `num_heads` 注意力头的值投影。
                    形状为 (num_heads * (d_model / num_heads), d_model)。
                    行按 (num_heads, d_v) 形状的矩阵排序，
                    即 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
                - `layers.{num_layers}.attn.output_proj.weight`
                    多头自注意力输出投影的权重。
                    形状为 ((d_model / num_heads) * num_heads, d_model)。
                - `layers.{num_layers}.ln1.weight`
                    Transformer 块中第一个 RMSNorm 仿射变换的权重。
                    形状为 (d_model,)。
                - `layers.{num_layers}.ffn.w1.weight`
                    FFN 中第一个线性变换的权重。
                    形状为 (d_model, d_ff)。
                - `layers.{num_layers}.ffn.w2.weight`
                    FFN 中第二个线性变换的权重。
                    形状为 (d_ff, d_model)。
                - `layers.{num_layers}.ffn.w3.weight`
                    FFN 中第三个线性变换的权重。
                    形状为 (d_model, d_ff)。
                - `layers.{num_layers}.ln2.weight`
                    Transformer 块中第二个 RMSNorm 仿射变换的权重。
                    形状为 (d_model,)。
                - `ln_final.weight`
                    应用于最终 Transformer 块输出的 RMSNorm 仿射变换权重。
                    形状为 (d_model,)。
                - `lm_head.weight`
                    语言模型输出嵌入的权重。
                    形状为 (vocab_size, d_model)。
            in_indices (Int[Tensor, "batch_size sequence_length"]):
                要在其上运行语言模型的输入索引张量。形状为 (batch_size, sequence_length)，
                其中 `sequence_length` 最多为 `context_length`。

        返回:
            Float[Tensor, "batch_size sequence_length vocab_size"]:
                每个 token 预测的未归一化下一个词分布张量（logits）。
        """
    lm = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)

    # 你的 TransformerLM __init__ 中硬编码了 theta=10000.0
    # 为了通过可能使用不同 theta 的测试，我们需要在这里覆盖它


    # 构造新的 state_dict 以匹配你的模型结构（主要是 QKV 合并和层命名）
    # 你的模型中层列表叫 self.layers，这与测试权重的键 'layers.0...' 匹配
    # 但你的 MHA 内部叫 qkv_proj，测试权重是分开的

    new_state_dict = {}
    # 复制非层权重
    new_state_dict['embedding.weight'] = weights['token_embeddings.weight']
    new_state_dict['final_norm.weight'] = weights['ln_final.weight']
    new_state_dict['output_head.weight'] = weights['lm_head.weight']

    # 处理每一层
    for i in range(num_layers):
        src_prefix = f'layers.{i}.'
        tgt_prefix = f'layers.{i}.'

        # 1. 合并 QKV
        q = weights[f'{src_prefix}attn.q_proj.weight']
        k = weights[f'{src_prefix}attn.k_proj.weight']
        v = weights[f'{src_prefix}attn.v_proj.weight']
        new_state_dict[f'{tgt_prefix}mha.qkv_proj.weight'] = torch.cat([q, k, v], dim=0)

        # 2. 其他权重直接映射
        # 注意：你的模型中属性名是 mha, rms_norm1, ffn, rms_norm2
        new_state_dict[f'{tgt_prefix}mha.o_proj.weight'] = weights[f'{src_prefix}attn.output_proj.weight']
        new_state_dict[f'{tgt_prefix}rms_norm1.weight'] = weights[f'{src_prefix}ln1.weight']
        new_state_dict[f'{tgt_prefix}rms_norm2.weight'] = weights[f'{src_prefix}ln2.weight']

        new_state_dict[f'{tgt_prefix}ffn.w1.weight'] = weights[f'{src_prefix}ffn.w1.weight']
        new_state_dict[f'{tgt_prefix}ffn.w2.weight'] = weights[f'{src_prefix}ffn.w2.weight']
        new_state_dict[f'{tgt_prefix}ffn.w3.weight'] = weights[f'{src_prefix}ffn.w3.weight']

    lm.load_state_dict(new_state_dict)

    return lm(in_indices)
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
        给定 RMSNorm 仿射变换的权重，返回在输入特征上运行 RMSNorm 的输出。

        参数:
            d_model (int): RMSNorm 输入的维度
            eps (float): 添加到分母以保证数值稳定性的值
            weights (Float[Tensor, "d_model"]): RMSNorm 权重（可学习的缩放参数 γ）
            in_features (Float[Tensor, "... d_model"]): 要运行 RMSNorm 的输入特征。
                可以有任意的前导维度。

        返回:
            Float[Tensor, "... d_model"]: 与 `in_features` 形状相同的张量，
                包含对 `in_features` 运行 RMSNorm 后的输出。
        """
    norm = RMSNorm(d_model,eps=eps)
    norm.weight.data = weights
    return norm(in_features)

    raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """
    给定输入张量，返回对每个元素应用 SiLU 后的输出。

    参数:
        in_features (Float[Tensor, "..."]): 要运行 SiLU 的输入特征。形状任意。

    返回:
        Float[Tensor, "..."]: 与 `in_features` 形状相同的张量，
            包含对每个元素应用 SiLU 后的输出。
    """
    return F.silu(in_features)
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
        给定输入张量，返回对输入指定 `dim` 维度进行 softmax 后的输出。

        参数:
            in_features (Float[Tensor, "..."]): 要进行 softmax 的输入特征。形状任意。
            dim (int): 要对 `in_features` 应用 softmax 的维度。

        返回:
            Float[Tensor, "..."]: 与 `in_features` 形状相同的张量，
                包含对指定 `dim` 进行 softmax 归一化后的输出。
        """


    return softmax(in_features, dim)
    raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    return cross_entropy(inputs,targets)
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    return gradient_clipping(parameters,max_l2_norm)
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    根据余弦学习率衰减调度（带线性预热）的参数和迭代次数，返回指定迭代次数下的学习率。

    参数:
        it (int): 要获取学习率的迭代次数。
        max_learning_rate (float): alpha_max，余弦学习率调度（带预热）的最大学习率。
        min_learning_rate (float): alpha_min，余弦学习率调度（带预热）的最小/最终学习率。
        warmup_iters (int): T_w，线性预热学习率的迭代次数。
        cosine_cycle_iters (int): T_c，余弦退火的迭代次数。

    返回:
        指定调度下给定迭代次数的学习率。
    """
    return get_lr_cosine_schedule(it,max_learning_rate,min_learning_rate,warmup_iters,cosine_cycle_iters)

    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    tokenizer = Tokenizer(100000,vocab, merges, special_tokens)
    # 为了通过 test_encode_iterable_* 系列测试
    # 测试文件要求 tokenizer 必须有一个 encode_iterable(iterable) 方法
    # 如果你的类里没写这个方法，我们在这里动态给它补上（Monkey Patching），
    # 这样你不需要去修改原来的 Tokenizer 类代码。
    if not hasattr(tokenizer, "encode_iterable"):
        import types
        def encode_iterable(self, iterable):
            for text in iterable:
                # 调用你已实现的 encode
                token_ids = self.encode(text)
                for token_id in token_ids:
                    yield token_id

        tokenizer.encode_iterable = types.MethodType(encode_iterable, tokenizer)
    return tokenizer

    raise NotImplementedError


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
    **kwargs,

) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
    input_path (str | os.PathLike): BPE 分词器训练数据文件的路径。
    vocab_size (int): 分词器词汇表中的条目总数（包含特殊 token）。
    special_tokens (list[str]):
    要加入词汇表的特殊 token 字符串列表。
    这些字符串永远不会被切分成多个子 token，而会始终作为一个完整的 token 存在。
    如果训练数据文件 input_path 中出现了这些特殊 token，它们会被当作普通字符串处理（不会被特殊对待）。

    Returns:
        vocab:训练后的分词器词汇表，是一个从 int（词汇表中的 token ID）到 bytes（token 的字节表示）的映射。
        merges:BPE 合并序列。列表中的每一项都是一个 bytes 元组 (<token1><token2><token1><token2>
    """
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )


    # 3. 返回结果 (保持不变，确保测试脚本能拿到返回值)
    return vocab, merges


    raise NotImplementedError
