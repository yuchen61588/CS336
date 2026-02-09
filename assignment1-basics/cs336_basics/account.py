import math

class TransformerResourceAnalyzer:
    """
    专门针对 model.py 中定义的 TransformerLM 架构进行资源核算。
    完全镜像 model.py 中的类结构和初始化逻辑。
    """

    def __init__(self, config):
        """
        Args:
            config: 字典，包含 vocab_size, context_length, d_model, num_layers, num_heads, d_ff
        """
        self.cfg = config

    def _calculate_swiglu_hidden_dim(self, d_model, d_ff):
        """
        镜像 model.py 中 SwiGLUFeedForward 的 hidden_dim 计算逻辑
        """
        if d_ff is not None:
            return d_ff
        else:
            # Fallback logic provided in your SwiGLU class
            hidden_dim = int(d_model * 8 / 3)
            # 向上取整到 64 的倍数
            hidden_dim = ((hidden_dim + 63) // 64) * 64
            return hidden_dim

    def attention_resources(self):
        """
        对应 CausalMultiHeadSelfAttention
        """
        d = self.cfg['d_model']
        s = self.cfg['context_length']

        # 1. Linear Projections (qkv_proj + o_proj)
        # qkv_proj: Linear(d, 3d) -> Params: 3d^2
        # o_proj: Linear(d, d)    -> Params: d^2
        proj_params = (3 * d * d) + (d * d)

        # FLOPs (Batch=1):
        # qkv: 2 * s * d * 3d = 6sd^2
        # o:   2 * s * d * d  = 2sd^2
        proj_flops = 8 * s * (d ** 2)

        # 2. Attention Mechanism Operations (scaled_dot_product_attention)
        # Score calculation (Q @ K^T): [s, d] @ [d, s] -> [s, s]
        # Ops: 2 * s^2 * d
        # Weighted sum (Score @ V): [s, s] @ [s, d] -> [s, d]
        # Ops: 2 * s^2 * d
        attn_ops_flops = 4 * (s ** 2) * d

        return {
            "params": proj_params,
            "flops": proj_flops + attn_ops_flops,
            "flops_details": {"proj": proj_flops, "ops": attn_ops_flops}
        }

    def ffn_resources(self):
        """
        对应 SwiGLUFeedForward
        """
        d = self.cfg['d_model']
        d_ff = self.cfg.get('d_ff', None)  # 可能为 None
        s = self.cfg['context_length']

        # 获取实际的 hidden_dim (h)
        h = self._calculate_swiglu_hidden_dim(d, d_ff)

        # 3个 Linear 层: w1(d->h), w3(d->h), w2(h->d)
        # Params: 3 * d * h
        params = 3 * d * h

        # FLOPs: 3次矩阵乘法
        # 3 * (2 * s * d * h)
        flops = 6 * s * d * h

        return {"params": params, "flops": flops}

    def rmsnorm_resources(self):
        """
        对应 RMSNorm
        通常 FLOPs 忽略不计（非矩阵乘法），但有参数
        """
        d = self.cfg['d_model']
        return {"params": d, "flops": 0}

    def block_resources(self):
        """
        对应 Transformer_block
        包含: 1 MHA, 1 FFN, 2 RMSNorm
        """
        attn = self.attention_resources()
        ffn = self.ffn_resources()
        norm = self.rmsnorm_resources()

        total_params = attn['params'] + ffn['params'] + (2 * norm['params'])
        total_flops = attn['flops'] + ffn['flops'] + (2 * norm['flops'])

        return {
            "params": total_params,
            "flops": total_flops,
            "submodules": {"attn": attn, "ffn": ffn}
        }

    def lm_resources(self):
        """
        对应 TransformerLM (整个模型)
        """
        V = self.cfg['vocab_size']
        d = self.cfg['d_model']
        s = self.cfg['context_length']
        L = self.cfg['num_layers']

        # 1. Embedding
        # Params: V * d
        embed_params = V * d
        embed_flops = 0  # Lookup has 0 FLOPs

        # 2. Transformer Blocks (x L)
        block = self.block_resources()
        blocks_params = L * block['params']
        blocks_flops = L * block['flops']

        # 3. Final RMSNorm
        norm = self.rmsnorm_resources()
        final_norm_params = norm['params']

        # 4. Output Head (Linear)
        # Linear(d, V)
        # Params: d * V
        head_params = d * V
        # FLOPs: 2 * s * d * V
        head_flops = 2 * s * d * V

        total_params = embed_params + blocks_params + final_norm_params + head_params
        total_flops = embed_flops + blocks_flops + head_flops

        return {
            "total_params": total_params,
            "total_flops": total_flops,
            "breakdown": {
                "embedding_params": embed_params,
                "block_params": blocks_params,
                "head_params": head_params,
                "block_flops": blocks_flops,
                "head_flops": head_flops,
                "attn_flops_fraction": (L * block['submodules']['attn']['flops']) / total_flops,
                "ffn_flops_fraction": (L * block['submodules']['ffn']['flops']) / total_flops,
                "non_embedding_params": total_params - embed_params
            }
        }


def print_resource_analysis(config, model_name="Model"):
    analyzer = TransformerResourceAnalyzer(config)
    res = analyzer.lm_resources()

    print(f"--- Analysis for {model_name} ---")
    print(
        f"Config: d_model={config['d_model']}, layers={config['num_layers']}, ctx={config['context_length']}, vocab={config['vocab_size']}")
    print(f"Total Parameters: {res['total_params']:,} ({res['total_params'] / 1e6:.2f} M)")
    print(
        f"Non-Embedding Params: {res['breakdown']['non_embedding_params']:,} ({res['breakdown']['non_embedding_params'] / 1e6:.2f} M)")
    print(f"Total FLOPs (per forward pass): {res['total_flops']:,} ({res['total_flops'] / 1e9:.2f} GFLOPs)")
    print(f"  - Attention: {res['breakdown']['attn_flops_fraction']:.1%}")
    print(f"  - FFN:       {res['breakdown']['ffn_flops_fraction']:.1%}")
    print(f"  - Output:    {res['breakdown']['head_flops'] / res['total_flops']:.1%}")
    print("-" * 30)