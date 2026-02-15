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


# === 修改点 1：函数增加 train_mode 参数 ===
def train_single_model(exp_config: dict, common_config: dict, project_name: str, train_mode: str):
    """
    负责训练单个模型的独立函数
    train_mode: '1' 表示重新训练, '2' 表示续训
    """
    run_name = exp_config['run_name']

    # 设定续训的目标步数 (你可以根据需要修改这里，或者写在 config 里)
    RESUME_STEP = 25000

    wandb.init(
        project=project_name,
        name=run_name,
        # 记录一下当前的训练模式
        config={"experiment": exp_config, "common": common_config, "mode": "resume" if train_mode == '2' else "new"}
    )

    device = common_config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    print(f"\n[{run_name}] 开始{'续训' if train_mode == '2' else '新训练'} | 使用设备: {device}")

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
        lr=float(train_config['learning_rate']),
        betas=(float(train_config['beta1']), float(train_config['beta2'])),
        eps=float(train_config['eps']),
        weight_decay=train_config.get('weight_decay', 0.0)
    )

    out_dir = os.path.join(common_config['checkpoint']['out_dir_base'], run_name)
    os.makedirs(out_dir, exist_ok=True)

    # === 修改点 2：处理续训逻辑 (加载权重 + 设置起始步数) ===
    start_step = 1  # 默认为 1

    if train_mode == '2':
        ckpt_path = os.path.join(out_dir, f"ckpt_{RESUME_STEP}.pt")
        if os.path.exists(out_dir):
            all_files = os.listdir(out_dir)
            ckpt_steps = []
            for f in all_files:
                if f.startswith("ckpt_") and f.endswith(".pt"):
                    try:
                        step_num = int(f.split('_')[1].split('.')[0])
                        ckpt_steps.append(step_num)
                    except ValueError:
                        continue

            if len(ckpt_steps) > 0:
                latest_step = max(ckpt_steps)
                ckpt_path = os.path.join(out_dir, f"ckpt_{latest_step}.pt")
                print(f"检测到最新 Checkpoint: {ckpt_path}")

                # 加载文件
                checkpoint = torch.load(ckpt_path, map_location=device)

                # === 关键修改点：根据你的保存格式读取 ===
                # 你的保存格式是: {'model_state_dict': ..., 'optimizer_state_dict': ..., 'iteration': ...}

                # 1. 加载模型权重
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("成功加载模型权重 (model_state_dict)。")
                elif 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                else:
                    # 最后的尝试：假设整个文件就是权重
                    try:
                        model.load_state_dict(checkpoint)
                    except Exception as e:
                        print(f"加载模型权重失败: {e}")
                        raise e

                # 2. 加载优化器状态
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("成功加载优化器状态 (optimizer_state_dict)。")
                elif 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])

                # 3. 确定起始步数
                if 'iteration' in checkpoint:
                    start_step = checkpoint['iteration'] + 1
                else:
                    start_step = latest_step + 1

                print(f"续训准备就绪！将从 Step {start_step} 继续训练。")

            else:
                print(f"未在 {out_dir} 找到任何 checkpoint 文件，从头开始训练。")
        else:
            print(f"目录 {out_dir} 不存在，从头开始训练。")

    model.train()
    max_iters = int(train_config['max_iters'])
    warmup_iters = int(train_config['warmup_iters'])
    alpha_max = float(train_config['learning_rate'])
    alpha_min = float(train_config['min_lr'])
    grad_clip = float(train_config.get('grad_clip', 1.0))
    save_interval = int(common_config['checkpoint']['save_interval'])

    # === 修改点 3：调整 tqdm 的 range ===
    # 如果是续训，进度条会直接从 25001 开始走
    pbar = tqdm(range(start_step, max_iters + 1), total=max_iters, initial=start_step, desc=f"Training {run_name}")

    for step in pbar:
        x, y = train_loader.get_batch()

        lr = get_lr_cosine_schedule(step, alpha_max, alpha_min, warmup_iters, max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)

        loss = cross_entropy(logits, y)
        loss.backward()

        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.detach().norm(2).item() ** 2
        grad_norm = math.sqrt(total_norm_sq)

        if grad_clip > 0.0:
            gradient_clipping(model.parameters(), max_norm=grad_clip, eps=1e-6)

        optimizer.step()

        metrics = {
            "train/grad_norm": grad_norm,
            "train/lr": lr,
        }

        # Loss 记录逻辑：
        # 如果是续训，step 肯定大于 300，会直接记录
        if step > 300:
            metrics["train/loss"] = loss.item()

        wandb.log(metrics, step=step)
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'grad_norm': f"{grad_norm:.2f}"})

        if step % save_interval == 0 or step == max_iters:
            ckpt_path = os.path.join(out_dir, f"ckpt_{step}.pt")
            # 建议：保存时把 optimizer 也存进去，方便下次继续续训
            save_checkpoint(model, optimizer, step, ckpt_path)

    wandb.finish()
    print(f"[{run_name}] 训练完成！\n" + "=" * 50)

    del model, optimizer, logits, loss, x, y
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# === 修改点 4：传递 mode 参数 ===
def run_all_experiments(config_path: str, train_mode: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    project_name = config.get('project_name', 'cs336-transformer-ablation')
    common_config = config['common']
    experiments = config['experiments']

    for exp in experiments:
        # 将 mode 传进去
        train_single_model(exp, common_config, project_name, train_mode)

    print("\n所有消融实验均已顺利完成！请前往 WandB 网页端查看对比图。")


if __name__ == "__main__":
    my_wandb_key = "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"
    # 这里为了演示方便没有加校验逻辑，直接 login
    wandb.login(key=my_wandb_key)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to yaml')
    args = parser.parse_args()

    # === 修改点 5：增加用户输入交互 ===
    print("=" * 30)
    print("请选择训练模式：")
    print("  [1] 全新训练 (Start from scratch)")
    print("  [2] 断点续训 (Resume from ckpt_25000)")
    print("=" * 30)

    mode = input("请输入数字 (1 或 2): ").strip()

    if mode not in ['1', '2']:
        print("输入无效，默认执行 [1] 全新训练")
        mode = '1'

    run_all_experiments(args.config, train_mode=mode)