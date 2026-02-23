# sft_run.py
import os
import sys
import json
import yaml
import torch
import wandb
import random
import argparse
import collections.abc
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

# ==========================================
# 导入你的辅助模块 (请确保这些文件在同级目录)
# ==========================================
from cs336_alignment.sft import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
    log_generations
)
from drgrpo_grader import r1_zero_reward_fn


# ==========================================
# 1. 严格配置校验系统
# ==========================================
def update_dict(d, u):
    """递归合并字典"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def load_and_validate_config(args):
    with open(args.base_config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    with open(args.exp_config, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)

    cfg = update_dict(base_cfg, exp_cfg)

    try:
        assert "model" in cfg, "缺少 model 字段"
        assert "output_dir" in cfg["model"], "缺少 model.output_dir (最终模型路径)"
        assert "checkpoint_dir" in cfg["model"], "缺少 model.checkpoint_dir (断点保存路径)"
        assert "data" in cfg, "缺少 data 字段"
        assert "training" in cfg, "缺少 training 字段"
        assert "eval_batch_size" in cfg["training"], "缺少 eval_batch_size (防OOM必须)"
    except AssertionError as e:
        print(f"❌ 启动中止! {e}")
        sys.exit(1)

    print(f"✅ 配置校验通过！单卡模式准备就绪。")
    return cfg


# ==========================================
# 2. 数据加载与整理
# ==========================================
def load_and_filter_data(data_path, num_examples=None, filter_correct_only=False):
    dataset = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))

    if filter_correct_only:
        filtered_dataset = []
        for item in dataset:
            ground_truth = item.get("ground_truth", "")
            rewards = r1_zero_reward_fn(item["response"], ground_truth)
            if rewards["reward"] > 0:
                filtered_dataset.append(item)
        dataset = filtered_dataset
        print(f"   - 过滤后数据集大小 (Correct Only): {len(dataset)}")

    random.seed(42)
    random.shuffle(dataset)
    if num_examples is not None and num_examples < len(dataset):
        dataset = dataset[:num_examples]

    return dataset


def string_collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    responses = [item.get("response", "") for item in batch]
    truths = [item.get("ground_truth", "") for item in batch]
    return {"prompt": prompts, "response": responses, "ground_truth": truths}


# ==========================================
# 3. 核心主循环
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Single GPU SFT Training Pipeline")
    parser.add_argument("--base_config", type=str, required=True, help="基础硬件与超参配置")
    parser.add_argument("--exp_config", type=str, required=True, help="具体实验变量配置")

    # [新增] 断点续训参数
    parser.add_argument("--resume_from", type=str, default=None, help="从某个 Checkpoint 恢复训练的路径")
    parser.add_argument("--wandb_id", type=str, default=None, help="需要恢复的 WandB run ID")
    parser.add_argument("--start_epoch", type=int, default=0, help="从第几个 Epoch 开始 (跳过已完成的)")
    args = parser.parse_args()

    cfg = load_and_validate_config(args)

    # ==========================================
    # WandB 初始化 (支持断点续传)
    # ==========================================
    my_wandb_key = "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"
    # 这里为了演示方便没有加校验逻辑，直接 login
    wandb.login(key=my_wandb_key)
    if args.wandb_id:
        print(f"🔄 正在恢复 WandB 运行记录: {args.wandb_id}")
        wandb.init(
            project=cfg["wandb"]["project_name"],
            id=args.wandb_id,
            resume="must",
            config=cfg
        )
    else:
        wandb.init(
            project=cfg["wandb"]["project_name"],
            name=cfg["wandb"]["run_name"],
            config=cfg
        )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    device = cfg["training"]["device"]
    batch_size = cfg["training"]["batch_size"]
    eval_batch_size = cfg["training"]["eval_batch_size"]
    grad_acc_steps = cfg["training"]["gradient_accumulation_steps"]
    lr = float(cfg["training"]["learning_rate"])
    epochs = cfg["training"]["epochs"]
    max_grad_norm = cfg["training"]["max_grad_norm"]
    eval_every = cfg["training"]["eval_every_n_steps"]

    # ==========================================
    # 模型与 Tokenizer 加载 (支持断点权重加载)
    # ==========================================
    model_load_path = args.resume_from if args.resume_from else cfg["model"]["model_id"]
    print(f"\n📦 加载模型权重自: {model_load_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_load_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    original_padding_side = tokenizer.padding_side

    print(f"🚀 将策略模型部署至 {device}...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_load_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)

    # ==========================================
    # 数据加载
    # ==========================================
    print("\n📚 加载并处理数据集...")
    train_data = load_and_filter_data(
        cfg["data"]["train_path"],
        cfg["data"]["num_train_examples"],
        cfg["data"]["filter_correct_only"]
    )
    val_data = load_and_filter_data(cfg["data"]["val_path"], cfg["data"]["num_val_examples"])

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=string_collate_fn, drop_last=True
    )

    # ==========================================
    # 优化器与 Cosine Scheduler
    # ==========================================
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)

    steps_per_epoch = len(train_loader) // grad_acc_steps
    total_global_steps = steps_per_epoch * epochs
    warmup_steps = int(total_global_steps * cfg["training"].get("warmup_ratio", 0.05))

    global_step = args.start_epoch * steps_per_epoch
    eval_step = (global_step // eval_every)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_global_steps,
        last_epoch=global_step - 1 if global_step > 0 else -1
    )

    print(f"\n=== 开始 SFT 单卡训练 ===")
    print(f"总 Global Steps: {total_global_steps} | Warmup Steps: {warmup_steps}")
    policy_model.train()

    # ==========================================
    # 主训练循环
    # ==========================================
    for epoch in range(args.start_epoch, epochs):
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            ncols=130,
            leave=True
        )

        accumulated_loss = 0.0

        for step_idx, batch in enumerate(progress_bar):
            prompts = batch["prompt"]
            responses = batch["response"]

            # (A) 分词与生成掩码
            tokens_dict = tokenize_prompt_and_output(prompts, responses, tokenizer)
            input_ids = tokens_dict["input_ids"].to(device)
            labels = tokens_dict["labels"].to(device)
            response_mask = tokens_dict["response_mask"].to(device)

            # (B) 前向传播
            log_probs_dict = get_response_log_probs(policy_model, input_ids, labels)

            # (C) 算 Loss
            loss, metrics = sft_microbatch_train_step(
                policy_log_probs=log_probs_dict["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=grad_acc_steps
            )

            # 累加缩放后的微批次损失
            accumulated_loss += loss.item()

            # (D) 梯度累积与参数更新
            if (step_idx + 1) % grad_acc_steps == 0 or (step_idx + 1) == len(train_loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                current_lr = scheduler.get_last_lr()[0]

                # 更新 tqdm 进度条
                progress_bar.set_postfix({
                    "Step": f"{global_step}/{total_global_steps}",
                    "Loss": f"{accumulated_loss:.4f}",
                    "Norm": f"{grad_norm.item():.2f}",
                    "LR": f"{current_lr:.2e}"
                })

                wandb.log({
                    "train/loss": accumulated_loss,
                    "train/grad_norm": grad_norm.item(),
                    "train/lr": current_lr,
                    "train_step": global_step,
                    "train/epoch": epoch + (step_idx + 1) / len(train_loader)
                })

                accumulated_loss = 0.0
                # save_every = cfg["training"].get("save_every_n_steps", 0)
                # if save_every > 0 and global_step % save_every == 0:
                #     ckpt_dir = os.path.join(cfg["model"]["checkpoint_dir"], f"step_{global_step}")
                #     os.makedirs(ckpt_dir, exist_ok=True)
                #     progress_bar.write(f"💾 触发高频存盘：正在保存 Step {global_step} 的断点至 {ckpt_dir} ...")
                #     policy_model.save_pretrained(ckpt_dir)
                #     tokenizer.save_pretrained(ckpt_dir)

                # ==========================================
                # (E) 验证与模型生成阶段 (4090 防 OOM 单卡原生推理)
                # ==========================================
                if global_step % eval_every == 0:
                    torch.cuda.empty_cache()
                    progress_bar.write(f"\n[Step {global_step}] 开始原生批量生成验证...")
                    policy_model.eval()

                    val_prompts = [item["prompt"] for item in val_data]
                    val_truths = [item["ground_truth"] for item in val_data]

                    generated_texts = []
                    val_entropies = []

                    # 验证必须左侧填充 。这样模型才能找到结尾 输入 3 个不同的 Prompt 进行批量生成，为了补齐长度在右边塞满了 <pad>，模型就不知道真正的结尾在哪了(通过结尾预测)。换成 left，所有的真实提问都会靠右对齐，模型就能准确地顺着最后一个词开始生成。
                    tokenizer.padding_side = "left"

                    with torch.no_grad():
                        for i in range(0, len(val_prompts), eval_batch_size):
                            batch_prompts = val_prompts[i: i + eval_batch_size]
                            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
                            input_lengths = inputs.input_ids.shape[1]

                            output_ids = policy_model.generate(
                                **inputs,
                                max_new_tokens=1024,
                                temperature=1.0,
                                top_p=1.0,
                                do_sample=True,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                            )

                            generated_ids = output_ids[:, input_lengths:]
                            batch_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                            # 清理幻觉数据 如果模型生成了 ...我的回答完毕。</answer> 用户提问：...，这行代码会以 </answer> 为界把字符串切开，只保留前半部分，然后再把 </answer> 拼回去。
                            cleaned_texts = [
                                text.split("</answer>")[0] + "</answer>" if "</answer>" in text else text
                                for text in batch_texts
                            ]
                            generated_texts.extend(cleaned_texts)

                            # 算熵时恢复右侧填充
                            tokenizer.padding_side = original_padding_side
                            for vp, vg in zip(batch_prompts, cleaned_texts):
                                t_data = tokenize_prompt_and_output([vp], [vg], tokenizer)
                                out_dict = get_response_log_probs(
                                    policy_model,
                                    t_data["input_ids"].to(device),
                                    t_data["labels"].to(device),
                                    return_token_entropy=True
                                )
                                mask = t_data["response_mask"].to(device)
                                val_entropies.append(out_dict["token_entropy"][mask == 1])
                            tokenizer.padding_side = "left"

                    tokenizer.padding_side = original_padding_side
                    eval_step += 1
                    log_generations(
                        prompts=val_prompts,
                        generated_responses=generated_texts,
                        ground_truths=val_truths,
                        token_entropies=val_entropies,
                        reward_fn=r1_zero_reward_fn,
                        step=eval_step
                    )
                    progress_bar.write(

                        "[Evaluation] 验证完成，继续训练...")
                    torch.cuda.empty_cache()
                    policy_model.train()

        # ==========================================
        # (F) Epoch 结束，保存 Checkpoint
        # ==========================================
        ckpt_dir = os.path.join(cfg["model"]["checkpoint_dir"], f"epoch_{epoch + 1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        progress_bar.write(f"💾 正在保存 Epoch {epoch + 1} 的断点至 {ckpt_dir} ...")
        policy_model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

    # ==========================================
    # (G) 训练彻底结束，保存最终模型
    # ==========================================
    final_output_dir = cfg["model"]["output_dir"]
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"\n🎉 训练圆满结束！正在保存终极模型至: {final_output_dir}")
    policy_model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    wandb.finish()


if __name__ == "__main__":
    main()