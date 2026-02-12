import os
import argparse
import yaml
import time
import math
import torch
import wandb
from tqdm import tqdm
import gc

# 导入你自己编写的模块
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.data import TextDataLoader
from cs336_basics.Checkpointing import save_checkpoint

# 导入你纯手写的工具函数
from cs336_basics.utils import cross_entropy, get_lr_cosine_schedule, gradient_clipping


def train_single_model(exp_config: dict, common_config: dict, project_name: str):
    """负责训练单个模型的独立函数"""
    run_name = exp_config['run_name']

    # === 关键点 1：每次循环都开启一个全新的 WandB Run ===
    # 只要 project 相同，指标名字相同，WandB 网页端会自动把它们画在同一张图里
    wandb.init(
        project=project_name,
        name=run_name,
        config={"experiment": exp_config, "common": common_config}
    )

    device = common_config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    print(f"\n[{run_name}] 开始训练 | 使用设备: {device}")

    train_loader = TextDataLoader(
        input_data=common_config['data']['train_path'],
        batch_size=common_config['training']['batch_size'],
        context_length=exp_config['model']['context_length'],
        device=device
    )

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

    train_config = common_config['training']
    optimizer = AdamW(
        model.parameters(),
        lr=train_config['learning_rate'],
        betas=(train_config['beta1'], train_config['beta2']),
        eps=train_config['eps'],
        weight_decay=train_config.get('weight_decay', 0.0)
    )

    out_dir = os.path.join(common_config['checkpoint']['out_dir_base'], run_name)
    os.makedirs(out_dir, exist_ok=True)

    model.train()
    max_iters = train_config['max_iters']
    warmup_iters = train_config['warmup_iters']
    alpha_max = train_config['learning_rate']
    alpha_min = train_config['min_lr']
    grad_clip = train_config.get('grad_clip', 1.0)
    save_interval = common_config['checkpoint']['save_interval']

    pbar = tqdm(range(1, max_iters + 1), total=max_iters, desc=f"Training {run_name}")

    for step in pbar:
        x, y = train_loader.get_batch()

        lr = get_lr_cosine_schedule(step, alpha_max, alpha_min, warmup_iters, max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)

        loss = cross_entropy(logits, y)
        loss.backward()

        # 计算梯度范数
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.detach().norm(2).item() ** 2
        grad_norm = math.sqrt(total_norm_sq)

        if grad_clip > 0.0:
            gradient_clipping(model.parameters(), max_norm=grad_clip, eps=1e-6)

        optimizer.step()

        # === 关键点 2：记录名字必须完全一样 ===
        # 因为所有模型都向 "train/loss" 和 "train/grad_norm" 写入数据
        # WandB 就会把它们叠加在一起，用不同颜色区分
        metrics = {
            "train/loss": loss.item(),
            "train/grad_norm": grad_norm,
            "train/lr": lr,
        }
        wandb.log(metrics, step=step)

        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'grad_norm': f"{grad_norm:.2f}"})

        if step % save_interval == 0 or step == max_iters:
            ckpt_path = os.path.join(out_dir, f"ckpt_{step}.pt")
            save_checkpoint(model, optimizer, step, ckpt_path)

    # === 关键点 3：结束当前 Run，并清空显存 ===
    wandb.finish()
    print(f"[{run_name}] 训练完成！\n" + "=" * 50)

    del model, optimizer, logits, loss, x, y
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_all_experiments(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    project_name = config.get('project_name', 'cs336-transformer-ablation')
    common_config = config['common']
    experiments = config['experiments']

    # 依次训练配置文件中的所有模型
    for exp in experiments:
        train_single_model(exp, common_config, project_name)

    print("\n所有消融实验均已顺利完成！请前往 WandB 网页端查看对比图。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to yaml')
    args = parser.parse_args()

    run_all_experiments(args.config)