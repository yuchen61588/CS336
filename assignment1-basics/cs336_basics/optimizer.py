from __future__ import annotations
import torch
import math
from typing import Tuple


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