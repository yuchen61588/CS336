from __future__ import annotations
import torch
import math
from typing import Tuple,Iterable

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

class AdamW(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 lr:float=1e-3,
                 betas:Tuple[float,float] = (0.9,0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,

                 ):
        """
                实现了 AdamW 算法。
                Args:
                    params: 待优化的参数集合或者参数组字典
                    lr: 学习率 (默认: 1e-3)
                    betas: 用于计算梯度及其平方的运行平均值的系数 (默认: (0.9, 0.999))
                    eps: 增加到分母中以提高数值稳定性的项 (默认: 1e-8)
                    weight_decay: 权重衰减系数 (默认: 0.0)
                """
        #检验传参合法性的
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults) #
    def step(self,closure=None):
        """执行一步优化过程。"""
        loss = None
        if closure is not None:
            with torch.enable_grad():  # ← 关键！强制开启梯度计算
                loss = closure()  # 计算损失

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad == None:
                    continue
                    # 保持梯度不被修改，使用 .data 或者 detached tensor
                grad = p.grad.data

                state = self.state[p] #优化器类参数，对于当前步数，给出state的字典，包括
                # 'exp_avg': tensor([...]),      # 一阶动量   'exp_avg_sq': tensor([...]),   # 二阶矩  'step': 100                     # 当前步数
                if len(state) == 0:
                    state['step'] = 0
                    # 一阶矩向量的初始值 (m <- 0)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)#内存格式相同
                    # 二阶矩向量的初始值 (v <- 0)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                # 1. 更新一阶矩估计
                # m <- beta1 * m + (1 - beta1) * g
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 2. 更新二阶矩估计
                # v <- beta2 * v + (1 - beta2) * g^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # 3. 计算偏差修正后的 alpha_t
                # alpha_t <- alpha * sqrt(1 - beta2^t) / (1 - beta1^t)
                bias_correction1 = 1-beta2**t
                bias_correction2 = 1 - beta1**t

                step_size = lr*math.sqrt(bias_correction1)/bias_correction2

                # 4. 更新参数 theta
                # theta <- theta - alpha_t * m / (sqrt(v) + eps)

                denom = exp_avg_sq.sqrt().add_(eps)

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                if weight_decay>0:
                    p.data.mul_(1 - lr * weight_decay)

        return loss

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



