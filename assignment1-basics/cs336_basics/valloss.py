import os
import argparse
import yaml
import torch
import math
import glob
import re
import wandb
from tqdm import tqdm

# 引入你的模块
from cs336_basics.model import TransformerLM
from cs336_basics.data import TextDataLoader
from cs336_basics.utils import cross_entropy

def extract_step(filename):
    """从文件名 ckpt_2000.pt 中提取出数字"""
    match = re.search(r'ckpt_(\d+)\.pt', filename)
    return int(match.group(1)) if match else -1

def evaluate_checkpoints(exp_config: dict, common_config: dict, project_name: str):
    run_name = exp_config['run_name']
    
    # 1. 初始化 WandB
    wandb.init(
        project=project_name,
        name=f"{run_name}-eval",
        group=run_name,
        job_type="evaluation",
        config={"experiment": exp_config, "common": common_config}
    )

    device = common_config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    print(f"\n[{run_name}] 开始评估 (Custom Mapping Mode)...")

    # 2. 准备数据
    val_path = common_config['data'].get('val_path')
    if not val_path:
        print(f"❌ 错误: 配置文件中未找到 'val_path'")
        wandb.finish()
        return

    val_loader = TextDataLoader(
        input_data=val_path,
        batch_size=common_config['training']['batch_size'],
        context_length=exp_config['model']['context_length'],
        device=device
    )

    # 3. 初始化模型
    model_config = exp_config['model']
    model = TransformerLM(
        vocab_size=model_config['vocab_size'],
        context_length=model_config['context_length'],
        d_model=model_config['d_model'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        d_ff=model_config['d_ff'],
        rope_theta=model_config.get('rope_theta', 10000.0),
        config=model_config,
        device=device,
        dtype=torch.float32
    )
    model.to(device)
    model.eval()

    # 4. 查找文件
    out_dir = os.path.join(common_config['checkpoint']['out_dir_base'], run_name)
    if not os.path.exists(out_dir):
        print(f"❌ 找不到目录: {out_dir}")
        wandb.finish()
        return

    ckpt_files = glob.glob(os.path.join(out_dir, "ckpt_*.pt"))
    ckpt_files.sort(key=extract_step) # 按步数排序

    print(f"📂 找到 {len(ckpt_files)} 个 checkpoint 文件。")

    # 5. 循环评估
    eval_iters = 200 

    for ckpt_path in tqdm(ckpt_files, desc=f"Evaluating {run_name}"):
        file_step = extract_step(ckpt_path)
        
        # === 🔥 核心修改：自定义映射逻辑 ===
        if file_step == 25000:
            # 这里的需求是：把 25000 强行画在 24000 的位置上
            wandb_step = 24000
        else:
            # 其他的按原计划：向前平移 2000 (例如 2000->0, 4000->2000)
            wandb_step = file_step - 2000
        
        # 如果是 ckpt_0.pt (file_step=0 -> wandb_step=-2000)，跳过
        if wandb_step < 0:
            continue
            
        # 如果不需要超过 24000 的点，也可以在这里截断
        # if wandb_step > 24000: break

        try:
            # 加载权重 (修复了 missing key 问题)
            checkpoint = torch.load(ckpt_path, map_location=device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
                
        except Exception as e:
            print(f"⚠️ 无法加载 {ckpt_path}: {e}")
            continue

        # 计算 Validation Loss
        losses = []
        with torch.no_grad():
            for _ in range(eval_iters):
                x, y = val_loader.get_batch()
                logits = model(x)
                loss = cross_entropy(logits, y)
                losses.append(loss.item())

        mean_loss = sum(losses) / len(losses)
        perplexity = math.exp(mean_loss)

        # 记录到 WandB
        wandb.log({
            "val/loss": mean_loss,
            "val/ppl": perplexity,
            "source_ckpt": file_step  # 记录一下原始是哪个文件，方便查验
        }, step=wandb_step)

    wandb.finish()
    print(f"✅ [{run_name}] 评估完成！")


def run_wandb_evaluation(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    project_name = config.get('project_name', 'cs336-transformer-ablation')
    
    # WandB Key
    my_wandb_key = "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"
    try:
        wandb.login(key=my_wandb_key)
    except:
        pass

    print(f"=== 正在处理配置文件: {config_path} ===")
    
    for exp in config['experiments']:
        evaluate_checkpoints(exp, config['common'], project_name)

if __name__ == "__main__":
    my_wandb_key = "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"
    if my_wandb_key == "这里粘贴你的API_KEY":
        print("警告: 你还没有设置 API Key。如果是第一次运行，请在代码最后一行填入 key，或者手动运行 wandb login。")
    else:
        wandb.login(key=my_wandb_key)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config file')
    args = parser.parse_args()

    run_wandb_evaluation(args.config)