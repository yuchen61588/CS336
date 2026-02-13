from __future__ import annotations
import torch
import torch.nn as nn
import math
from einops import einsum,repeat,rearrange
from cs336_basics.model_function import softmax,scaled_dot_product_attention

class Linear(nn.Module):
    def __init__(self,
                 in_features:int,
                 out_features:int,
                 device:torch.device | None = None,  #可以是torch.device | None 。= None表示默认值为None
                 dtype:torch.dtype|None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty((out_features,in_features),**factory_kwargs))
        # 使用截断正态分布 (Truncated Normal Distribution) 进行初始化
        # 均值 mean = 0
        # 截断范围为 [-3 * std, 3 * std] (即 3倍标准差)
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self,x: torch.Tensor)->torch.Tensor:
        """
                前向传播函数。

                参数:
                    x: 输入张量，形状可以是 (batch_size, sequence_length, in_features)
                       或者包含任意数量的 batch 维度 (..., in_features)

                返回:
                    输出张量，形状为 (..., out_features)
                """
        # 使用 einops.einsum 进行矩阵乘法 [cite: 420-423]
        # 这里的模式字符串解释:
        # '... in_feat': 代表输入 x，'...' 自动匹配任意 batch 维度，'in_feat' 是输入特征维
        # 'out_feat in_feat': 代表权重 W 的形状 (out_features, in_features)
        # -> '... out_feat': 代表输出，保持 batch 维度不变，特征维变为 out_feat
        return  einsum(x, self.weight, '... in_feat, out_feat in_feat -> ... out_feat')

class Embedding(nn.Module):
    """
    实现一个嵌入层 (Embedding Layer)，用于将 token ID 映射为密集向量。
    """
    def __init__(self,
                 num_embeddings:int,
                 embedding_dim:int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
                 ):
        super().__init__()
        self.num_embeddings = num_embeddings #vocab
        self.embedding_dim = embedding_dim  #d_model

        # 1. 创建嵌入矩阵
        # 形状为 (num_embeddings, embedding_dim)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim),**factory_kwargs))

        std = 1.0
        # 截断范围为 [-3, 3] (3倍标准差)
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std
        )
    def forward(self, token_ids: torch.Tensor)->torch.Tensor:
        """
            前向传播函数。
            参数:
            token_ids: 整数类型的 token ID 张量，形状为 (batch_size, sequence_length)
            返回:
                嵌入向量张量，形状为 (batch_size, sequence_length, embedding_dim)
            """
        return self.weight[token_ids] #对于没一个向量的id进行存储，取行，然后最后自动拼接 里面的参数保留，外面少一点东西


class RMSNorm(nn.Module):
    def __init__(self,d_model: int,
                 eps: float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        # g0训练
        self.weight = nn.Parameter(torch.ones(d_model,**factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        Args:
        x: 输入张量，形状通常为 (batch_size, sequence_length, d_model)
        """
        # 类型保存与转换
        input_dtype = x.dtype
        x = x.to(torch.float32)
        # 3. 计算 RMS (均方根)
        # 公式: RMS(a) = sqrt(mean(a^2) + eps)
        # dim=-1: 在最后一个维度 (d_model) 上进行计算 [cite: 587]
        # keepdim=True: 保持维度以便进行广播 (Broadcasting)
        variance = x.pow(2).mean(dim=-1, keepdim=True) #只在最后一个维度作平均
        # torch.rsqrt(x) 等价于 1 / sqrt(x)，计算效率更高
        rms = torch.rsqrt(variance + self.eps)

        x = x*rms*self.weight #pytorch使用

        return x.to(input_dtype)

# 标准化layNorm 器最大特点就是需要减去均值u，RMS没有这个步骤

class LayerNorm(nn.Module):
    """
    Standard Layer Normalization (用于原始 Transformer 架构)
    公式: y = ((x - mean) / sqrt(var + eps)) * gamma + beta
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}

        # Learnable parameters: Gamma (weight) and Beta (bias)
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.bias = nn.Parameter(torch.zeros(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        # 计算均值 (Mean)
        mean = x.mean(dim=-1, keepdim=True)

        # 计算方差 (Variance) - 使用无偏估计或总体方差均可，通常 LayerNorm 使用简单的 mean((x-mu)^2)
        # 这里为了数值稳定性，通常写作 x.var(dim=-1, keepdim=True, unbiased=False)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 应用仿射变换 (Affine Transform)
        return x_norm * self.weight + self.bias

class SwiGLUFeedForward(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff:int| None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
                 ):
        """
            SwiGLU 前馈神经网络 (Position-wise Feed-Forward Network).
            Args:
                    d_model: 模型的输入/输出维度
                    device: 设备
                    dtype: 数据类型
                """
        super().__init__()
        if d_ff is not None:
            hidden_dim = d_ff
        else:
            # 否则使用 Llama 的默认计算公式 (Fallback)
            hidden_dim = int(d_model * 8 / 3)
            # 向上取整到 64 的倍数 (硬件亲和性优化)
            hidden_dim = ((hidden_dim + 63) // 64) * 64

        # 2. 定义三个线性层
        #GUI门控的两个权重
        self.w1 = Linear(d_model,hidden_dim,device=device,dtype=dtype)
        self.w3 = Linear(d_model,hidden_dim,device=device,dtype=dtype)
        self.w2 = Linear(hidden_dim,d_model,device=device,dtype=dtype)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        """
        前向传播
        公式: FFN(x) = W2( SiLU(W1x) * W3x )
        """
        w1_out = self.w1(x)

        silu_out = w1_out*torch.sigmoid(w1_out)

        w3_out = self.w3(x)

        hidden = silu_out*w3_out
        return self.w2(hidden)

# 标准化前馈网络，使用全连接层+relu +dropout+全连接层的配比方式
class StandardFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation_type: str = 'relu', dropout: float = 0.0, device=None, dtype=None):
        super().__init__()
        # Original Transformer: d_ff = 4 * d_model [cite: 625]
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        if activation_type == 'relu':
            self.act = nn.ReLU()
        elif activation_type == 'silu':
            self.act = nn.SiLU()
        elif activation_type == 'gelu':
            self.act = nn.GELU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN(x) = ReLU(xW1)W2
        return self.w2(self.dropout(self.relu(self.w1(x))))


class ROPE(nn.Module):
    def __init__(self,
                 theta:float,
                 d_k:int,
                 max_seq_len:int,
                 device: torch.device | None = None,
                 ):
        super().__init__()

        # 1. 计算频率 (Frequencies)
        # 依据公式: theta_k = 1 / (theta ** (2k/d))
        # 生成序列 [0, 2, 4, ... d_k-2]
        indices = torch.arange(0,d_k,2,device=device).float() #索引向量
        freqs = 1.0/(theta**(indices/d_k))

        # 生成位置索引t
        t = torch.arange(max_seq_len,device=device).float()

        # 3. 计算角度 (Angles) = m * theta
        # 外积: (seq_len) x (d_k/2) -> (seq_len, d_k/2)
        freqs = torch.outer(t, freqs)
        # 根据文档矩阵公式，x_2k 和 x_2k+1 使用相同的角度 theta_k
        # 因此我们需要将频率重复一遍: [theta_0, theta_1, ...] -> [theta_0, theta_0, theta_1, theta_1, ...]
        freqs_expanded = repeat(freqs, 'seq dim -> seq (dim a)',a=2)
        # 5. 预计算并注册 Buffer (非持久化)
        # 形状均为 (max_seq_len, d_k)
        self.register_buffer("cos_cached", freqs_expanded.cos(), persistent=False) # 预计算
        self.register_buffer("sin_cached", freqs_expanded.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
                辅助函数: 执行 [-x_odd, x_even] 的操作
                使用 einops.rearrange 显式地处理成对关系，比 .view() 更易读
                """
        x_pairs = rearrange(x, '... (d a) -> ... d a',a=2)
        # 2. 解包：获取偶数位(x1)和奇数位(x2)
        # x1: [x0, x2, x4...]
        # x2: [x1, x3, x5...]
        x1, x2 = x_pairs.unbind(dim=-1) #表示按照最后一个维度拆分，有两行

        # 3. 交换并取反: [-x2, x1]
        x_rotated_pairs = torch.stack((-x2, x1), dim=-1) #拼接

        # 4. 还原形状: 把对子拼回去
        # '... d 2 -> ... (d 2)'
        return rearrange(x_rotated_pairs, '... d a -> ... (d a)',a=2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        # 1. 根据位置索引获取 cos / sin
        # token_positions: [batch, seq]
        # cos: [batch, seq, d_k]
        # 对于每一个位置的token，取出它对应d_model的位置向量
        cos = self.cos_cached[token_positions] # 绝对位置
        sin = self.sin_cached[token_positions]

        # 只有一种情况需要我们手动插入维度：
        # 当 x 是 4D (Batch, Head, Seq, Dim) 且 cos 是 3D (Batch, Seq, Dim) 时。
        # 这种情况下，cos 的 Batch 维会错误地对应到 x 的 Head 维，所以我们需要在 Head 位置(dim=1)插一个 1。
        if x.ndim == 4 and cos.ndim == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        # 其他情况交给 PyTorch 自动广播即可：
        # - x(3D) * cos(2D): (B,S,D) * (S,D) -> OK
        # - x(4D) * cos(2D): (B,H,S,D) * (S,D) -> OK
        # - x(3D) * cos(3D): (B,S,D) * (B,S,D) -> OK

        return (x*cos)+(self._rotate_half(x)*sin)

# 绝对位置编码
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        # 创建一个足够长的 PE 矩阵
        pe = torch.zeros(max_seq_len, d_model, device=device, dtype=dtype)
        position = torch.arange(0, max_seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 注册为 buffer：不是可训练参数，但随模型 state_dict 保存
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Embedding 后的输入，形状 (batch, seq_len, d_model)
        Returns:
            对应的位置编码，形状 (1, seq_len, d_model)
        """
        seq_len = x.size(1)
        # 截取当前序列长度对应的编码，并增加 batch 维度以便广播
        return self.pe[:seq_len, :].unsqueeze(0)

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model:int,num_heads:int,device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        """
           实现因果多头自注意力 (Causal Multi-Head Self-Attention)
               Args:
                    d_model (int): 输入维度 (Transformer block inputs)
                    num_heads (int): 头的数量
         """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # 验证 d_model 能被 num_heads 整除
        # assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads

        # 1. 定义线性层 (Projections)
        # 按照文档脚注5的 "Stretch goal"，我们将 W_Q, W_K, W_V 合并为一个矩阵 就是变为三个矩阵
        # 形状: [d_model, 3 * d_model] -> 对应输出 [Q, K, V]
        self.qkv_proj = Linear(d_model,3*d_model,device,dtype)

        #融合的线性层
        self.o_proj = Linear(d_model,d_model,device,dtype)
    def forward(self,x:torch.Tensor,rope_embed: nn.Module = None,token_positions: torch.Tensor | None = None,)->torch.Tensor:
        """
        前向传播
         Args:
            x: 输入张量，形状 (batch_size, seq_len, d_model)
            rope_embed: 自定义的 ROPE 模块实例 (可选)
         """
        batch_size ,seq_len , _ = x.shape

        # Step 1: 投影并分割 Q, K, V
        qkv = self.qkv_proj(x)
        # 拆解
        q, k, v = rearrange(qkv, 'b s (c h d) -> c b h s d', c=3,h=self.num_heads, d=self.head_dim)

        # Step 2: 应用 RoPE (如果提供)
        if rope_embed is not None:
            if token_positions is None:
            # 1. 生成基础位置索引: [0, 1, ..., seq_len-1]
            # 形状: (seq_len,)
                ids = torch.arange(seq_len, device=x.device)
            # 2. 扩展到 Batch 维度，以匹配你的 ROPE 输入要求
            # 形状: (batch_size, seq_len)
                token_positions = ids.unsqueeze(0).expand(batch_size, -1)

            # q, k: (batch, num_heads, seq_len, head_dim)
            # token_positions: (batch, seq_len)
            q = rope_embed(q, token_positions)
            k = rope_embed(k, token_positions)
        # Step 3: 掩码计算
        mask = torch.tril(torch.ones((seq_len,seq_len),device = x.device,dtype = torch.bool))

        # Step 4: 计算注意力
        attn_output = scaled_dot_product_attention(q,k,v,mask = mask)

        # Step 5： 拼接并输出
        output = rearrange(attn_output,'b h s d -> b s (h d)')
        #输出投影
        return self.o_proj(output)

class Transformer_block(nn.Module):
    def __init__(self,
                 d_model:int,
                 num_heads:int,
                 d_ff:int,
                 config: dict,  # 传入配置字典
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
                 ):
        """
            实现 Transformer 块 (Pre-norm 结构)
            包含两个子层: Multi-Head Self-Attention 和 Feed-Forward Network.
            Args:
                d_model: 输入维度
                num_heads: 注意力头数
                d_ff: FFN 隐藏层维度 (注意：你之前的 SwiGLUFeedForward 是内部计算 hidden_dim 的，
                        如果需要严格使用这个 d_ff，你需要修改 SwiGLUFeedForward 的 __init__ 来接收它。
                        这里我暂时按照标准的 SwiGLU 逻辑，传入 d_model)
            # 注意，为了实现消融实验，我们传入了config字典，便于配置的调整
            """
        super().__init__()
        self.norm_location = config.get('norm_location', 'pre')  # 'pre' or 'post'
        norm_type = config.get('norm_type', 'rmsnorm')  # 'rmsnorm' or 'layernorm'
        ffn_type = config.get('ffn_type', 'swiglu')  # 'swiglu' or 'relu'
        dropout = config.get('dropout', 0.0)

        NormClass = RMSNorm if norm_type == 'rmsnorm' else nn.LayerNorm

        # 修改：增加 'none' 支持，返回 Identity
        if norm_type == 'rmsnorm':
            NormClass = lambda d: RMSNorm(d, eps=1e-5, device=device, dtype=dtype)
        elif norm_type == 'layernorm':
            NormClass = lambda d: nn.LayerNorm(d, eps=1e-5, device=device, dtype=dtype)
        elif norm_type == 'none':
            NormClass = lambda d: nn.Identity()  # 不做任何操作

        self.norm1 = NormClass(d_model)
        self.norm2 = NormClass(d_model)

        self.mha = CausalMultiHeadSelfAttention(d_model,num_heads,device=device,dtype=dtype)

        # 工厂模式：选择 FFN
        # 修改：FFN 分发逻辑，支持 swiglu, relu, silu gelu
        if ffn_type == 'swiglu':
            self.ffn = SwiGLUFeedForward(d_model, d_ff, device=device, dtype=dtype)
        elif ffn_type == 'relu':
            # 原始架构
            self.ffn = StandardFeedForward(d_model, d_ff, activation_type='relu', dropout=dropout, device=device,
                                           dtype=dtype)
        elif ffn_type == 'silu':
            # 用于对比 SwiGLU 的非门控版本
            self.ffn = StandardFeedForward(d_model, d_ff, activation_type='silu', dropout=dropout, device=device,
                                           dtype=dtype)
        else:
            self.ffn = StandardFeedForward(d_model, d_ff, activation_type='gelu', dropout=dropout, device=device,
                                           dtype=dtype)
    def forward(self,x:torch.Tensor,rope_embed:nn.Module)->torch.Tensor:
        """
            Args:
            x: (batch, seq_len, d_model)
             rope_embed: ROPE 实例，传递给 MHA
        """
        # 子层 1: MHA
        # y = x + MultiHeadSelfAttention(RMSNorm(x))
        # 注意：MHA 需要接收 rope_embed
        # 注意，下面有两种模式，一种正常模式，一种llama模式，根据配置文件来

        # Pre-Norm (Llama/Modern)
        if self.norm_location =='pre':
            x = x + self.mha(self.norm1(x), rope_embed=rope_embed)
            x = x+self.ffn(self.norm2(x))

        # Post-Norm (Original/Vaswani)

        else:
            # Sublayer 1: Attention
            # 注意：Post-Norm 中，RoPE 应用在 Attention 内部，但 Norm 在残差之后
            norm_input = x+ self.mha(x,rope_embed=rope_embed)
            x =self.norm1(norm_input)

            norm_input = x+self.ffn(x)
            x =self.norm2(norm_input)

        return x



class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size:int,
                 context_length:int,
                 d_model:int,
                 num_layers:int,
                 num_heads:int,
                 d_ff:int,
                 rope_theta: float,
                 config: dict,  # 接收完整配置
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
                 ):
        """
        实现完整的 Transformer 语言模型
        Args:
                    vocab_size: 词表大小
                    context_length: 最大上下文长度 (用于 RoPE)
                    d_model: 模型维度
                    num_layers: 层数
                    num_heads: 头数
                    d_ff: FFN 维度
        """
        super().__init__()
        # Embedding
        self.embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )
        # 位置编码选择
        self.pos_emb_type = config.get('pos_emb_type', 'rope')

        # 初始化占位符
        self.rope = None #这个是用来在Attention判断是否需要旋转编码的，不然会出问题
        self.abs_pos_embed = None

        if self.pos_emb_type == 'rope':
            # RoPE 只需要初始化一次，然后在所有层之间共享
            d_k = d_model // num_heads
            self.rope = ROPE(
                theta=rope_theta,
                d_k=d_k,
                max_seq_len=context_length,
                device=device
            )

        elif self.pos_emb_type == 'learned':
            # Learned Absolute Positional Embedding (原始 GPT-2/3 风格)
            # 可学习参数，直接加在 input embedding 上
            self.abs_pos_embed = nn.Parameter(torch.zeros(1, context_length, d_model, device=device, dtype=dtype))
            nn.init.trunc_normal_(self.abs_pos_embed, std=0.02)

        elif self.pos_emb_type == 'sinusoidal':
            # Fixed Sinusoidal Positional Embedding (原始 Transformer 论文风格)
            self.abs_pos_embed = SinusoidalPositionalEmbedding(
                d_model=d_model,
                max_seq_len=context_length,
                device=device,
                dtype=dtype
            )

        elif self.pos_emb_type == 'none':
            # NoPE (No Positional Embedding)
            pass
        else:
            raise ValueError(f"Unknown pos_emb_type: {self.pos_emb_type}")

        # 堆叠Transformer Blocks
        self.layers = nn.ModuleList([
            Transformer_block(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                config=config,
                device=device,
                dtype=dtype
            ) for _ in range(num_layers)
        ])
        # Final Norm
        norm_type = config.get('norm_type', 'rmsnorm')
        if norm_type == 'rmsnorm':
            self.final_norm = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        elif norm_type == 'layernorm':
            self.final_norm = nn.LayerNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        elif norm_type == 'none':
            self.final_norm = nn.Identity()  # 不做归一化
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        self.output_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype
       )
        # 核心：权重绑定 (Weight Tying)
        if config.get('weight_tying', False):
            # PyTorch 中 Embedding 和 Linear 的权重 shape 都是 (vocab, d_model)
            # Embedding.weight: (num_embeddings, embedding_dim)
            # Linear.weight: (out_features, in_features) -> (vocab, d_model)
            # 所以可以直接赋值共享内存
            self.output_head.weight = self.embedding.weight

    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        """
            Args:
                token_ids: (batch, seq_len)

            Returns:
                logits: (batch, seq_len, vocab_size) - 未归一化的概率分布
            """
        x = self.embedding(token_ids)
        # 2. 穿过所有 Transformer 层
        # 2. 应用绝对位置编码 (如果是 Learned 或 Sinusoidal)
        # RoPE 是在 Attention 层内部作用的，所以这里不处理
        if self.pos_emb_type in ['learned', 'sinusoidal']:
            if self.pos_emb_type == 'learned':
                # 截取当前长度并相加
                seq_len = token_ids.size(1)
                x = x + self.abs_pos_embed[:, :seq_len, :]
            else:
                # Sinusoidal 模块内部处理了截取逻辑
                x = x + self.abs_pos_embed(x)
        for layer in self.layers:
            # 将 rope 实例传给每个 block
            x = layer(x, rope_embed=self.rope)
        #输出层
        x = self.final_norm(x)
        logits = self.output_head(x)

        return logits































