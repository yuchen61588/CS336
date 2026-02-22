import os
from huggingface_hub import snapshot_download

local_model_dir = "data/models/Qwen2.5-Math-1.5B"
os.makedirs(local_model_dir, exist_ok=True)

print(f"正在下载 Qwen/Qwen2.5-Math-1.5B 到 {local_model_dir} ...")

# 从 HuggingFace 下载模型权重和配置文件
snapshot_download(
    repo_id="Qwen/Qwen2.5-Math-1.5B",
    local_dir=local_model_dir,
    # 忽略一些不需要的大文件，比如非safetensors格式的权重，以节省时间
    ignore_patterns=["*.pt", "*.bin", "*.h5"] 
)

print("模型下载完成！")