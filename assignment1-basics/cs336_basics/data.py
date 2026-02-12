
from __future__ import annotations
import os
import torch
import numpy as np
from typing import Tuple, Optional,Union


class TextDataLoader:
    def __init__(self, input_data: Union[str,np.ndarray], batch_size: int, context_length: int, device: str):
        """
        初始化数据加载器。

        Args:
            input_path: .npy tokenized 数据文件的路径
            batch_size: 每次 get_batch 返回的序列数量
            context_length: 每个序列的长度 (seq_len)
            device: 数据加载后存放的目标设备 ('cpu', 'cuda', 'mps')
        """
        if isinstance(input_data,str):
            if not os.path.exists(input_data):
                raise FileNotFoundError(f"Data file not found at: {input_data}")

            # 1. 加载数据 (Memory Mapping)
            # mmap_mode='r' 保证大文件不会一次性读入内存
            self.data = np.load(input_data, mmap_mode='r')
            print(f"Loaded dataset from {input_data} with {len(self.data)} tokens.")
        elif isinstance(input_data,np.ndarray):
            self.data = input_data
        else:
            raise TypeError("input_data must be either a file path (str) or a numpy array")

        # 2. 存储配置
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

        # 3. 基础校验
        if len(self.data) <= self.context_length + 1:
            raise ValueError(f"数据过短 ({len(self.data)})，无法满足 context_length ({self.context_length}) 的需求。")



    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        随机采样一个批次的数据。
        使用初始化时设定的 batch_size 和 context_length。

        Returns:
            x (Inputs): (B, T) LongTensor
            y (Targets): (B, T) LongTensor
        """
        # 1. 确定有效索引范围
        # 有效起点索引范围: [0, len(data) - context_length - 1)
        max_idx = len(self.data) - self.context_length - 1

        # 2. 随机生成 batch_size 个起始索引
        ix = torch.randint(low=0, high=max_idx+1, size=(self.batch_size,))

        # 3. 切片并转换为 Tensor
        # 这里会触发实际的磁盘 I/O (因为 data 是 mmap)
        # 必须转为 int64 以适配 PyTorch Embedding
        x_list = [
            torch.from_numpy((self.data[i: i + self.context_length]).astype(np.int64))
            for i in ix
        ]
        y_list = [
            torch.from_numpy((self.data[i + 1: i + 1 + self.context_length]).astype(np.int64))
            for i in ix
        ]

        # 4. 堆叠
        x = torch.stack(x_list)
        y = torch.stack(y_list)

        # 5. 移动到设备
        # 如果是 CUDA，使用 pin_memory 加速传输
        if self.device.startswith('cuda'):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)

        return x, y

    def __len__(self):
        """返回数据集的总 token 数量"""
        return len(self.data)

