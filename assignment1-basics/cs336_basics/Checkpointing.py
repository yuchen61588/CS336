import torch
import os
from typing import Union,BinaryIO,IO

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]]
):
    """
    保存模型检查点。
        Args:
            model: PyTorch 模型
            optimizer: PyTorch 优化器
            iteration: 当前迭代步数
            out: 输出路径或文件对象
        """
    # 构建一个包含所有必要信息的字典
    checkpoint_state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    # 使用 torch.save 保存
    # torch.save 内部会自动处理 str 路径或 file-like object
    torch.save(checkpoint_state, out)

def load_checkpoint(src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
                    model:torch.nn.Module,
                    optimizer:torch.optim.Optimizer)->int:
    """
        加载模型检查点并恢复状态。
        Args:
            src: 检查点文件路径或文件对象
            model: 需要恢复状态的模型实例 (将会被原地修改)
            optimizer: 需要恢复状态的优化器实例 (将会被原地修改)

        Returns:
            int: 恢复的迭代步数
        """
    # 1. 加载检查点字典
    # map_location='cpu' 是一种防御性编程，防止在没有 GPU 的机器上加载 GPU 保存的模型报错
    if torch.cuda.is_available():
        checkpoint_state = torch.load(src)
    else:
        checkpoint_state = torch.load(src, map_location='cpu')

    # 2. 恢复模型参数
    # strict=True 是默认值，要求 checkpoint 中的 key 和 model 中的 key 严格对应
    model.load_state_dict(checkpoint_state['model_state_dict'])

    # 3. 恢复优化器状态 (包含动量 buffer 等)
    optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])

    return checkpoint_state['iteration']