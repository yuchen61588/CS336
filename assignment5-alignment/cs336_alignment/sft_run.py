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
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup
import gc
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
                    

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

    max_seq_length = cfg["model"].get("max_length", 2048)
    print(f"📏 当前实验配置的最大截断长度 (max_length) 为: {max_seq_length}")

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
    num_examples = cfg["data"]["num_train_examples"]
    eval_every = epochs * num_examples // (batch_size * grad_acc_steps) if "eval_every_n_steps" not in cfg["training"] else cfg["training"]["eval_every_n_steps"]

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

    policy_model.gradient_checkpointing_enable()  #GPU 内存优化

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
        train_data, batch_size=batch_size, shuffle=True, collate_fn=string_collate_fn, drop_last=True # type: ignore
    )

    # ==========================================
    # 优化器与 Cosine Scheduler
    # ==========================================
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)

    if args.resume_from:
        opt_path = os.path.join(args.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            print(f"🔄 正在恢复优化器状态: {opt_path}")
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            # 防止优化器里的 tensor 不在当前设备上
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
    

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
            tokens_dict = tokenize_prompt_and_output(prompts, responses, tokenizer, max_seq_length=max_seq_length)
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
                if epoch == epochs - 1 and (step_idx + 1) == len(train_loader):
                    
                    progress_bar.write(f"\n[Step {global_step}] 准备启动 vLLM 验证，正在进行显存置换...")
                    
                    # 1. 保存当前实时权重到一个临时目录
                    temp_eval_dir = os.path.join(cfg["model"]["checkpoint_dir"], "temp_vllm_eval")
                    os.makedirs(temp_eval_dir, exist_ok=True)
                    policy_model.save_pretrained(temp_eval_dir)
                    tokenizer.save_pretrained(temp_eval_dir)

                    # 2. 释放训练模型显存 (将其移至 CPU 并清空 CUDA 缓存)
                    policy_model.to("cpu")
                    optimizer_state_dict = optimizer.state_dict() # 如果极度缺显存，优化器也可能需要处理，这里暂时只动模型
                    torch.cuda.empty_cache()
                    gc.collect()

                    progress_bar.write(f"显存已释放，启动 vLLM 引擎...")

                    # 3. 初始化 vLLM (关键：必须严格限制 gpu_memory_utilization，防止 OOM)
                    # 4090 是 24G，设为 0.4 意味着 vLLM 最多只能用约 9.6G 显存
                    llm = LLM(
                        model=temp_eval_dir,
                        trust_remote_code=True,
                        tensor_parallel_size=1,
                        gpu_memory_utilization=0.4 # ⚠️ 如果报错 OOM，调低这个值(如 0.3)；如果提示 KV cache 空间不足，调高这个值(如 0.5)
                    )

                    # vLLM 直接支持 Stop Token，不需要事后切分字符串了
                    sampling_params = SamplingParams(
                        temperature=1.0,
                        top_p=1.0,
                        max_tokens=1024,
                        stop=["</answer>"] 
                    )

                    val_prompts = [item["prompt"] for item in val_data]
                    val_truths = [item["ground_truth"] for item in val_data]

                    # 4. vLLM 批量生成 (自带进度条，极其丝滑)
                    outputs = llm.generate(val_prompts, sampling_params)
                    
                    # 提取生成的文本并自动补齐我们设定的 Stop Token
                    generated_texts = [
                        (out.outputs[0].text + "</answer>") if out.outputs[0].finish_reason == "stop" else out.outputs[0].text
                        for out in outputs
                    ]

                    # 5. 销毁 vLLM，归还显存！
                    progress_bar.write("vLLM 生成完毕，正在销毁推理引擎并恢复训练环境...")
                    del llm
                    destroy_model_parallel()
                    torch.cuda.empty_cache()
                    gc.collect()

                    # 6. 把策略模型搬回 GPU，恢复原样
                    policy_model.to(device)
                    policy_model.eval()

                    # 7. 算熵阶段：统一切换回右侧填充 (与原来逻辑一致)
                    tokenizer.padding_side = original_padding_side
                    val_entropies = []
                    
                    with torch.no_grad():
                        for vp, vg in zip(val_prompts, generated_texts):
                            t_data = tokenize_prompt_and_output([vp], [vg], tokenizer)
                            out_dict = get_response_log_probs(
                                policy_model,
                                t_data["input_ids"].to(device),
                                t_data["labels"].to(device),
                                return_token_entropy=True
                            )
                            mask = t_data["response_mask"].to(device)
                            val_entropies.append(out_dict["token_entropy"][mask == 1])

                    eval_step += 1
                    save_logs = cfg["training"].get("save_logs","output/sft_logs.txt")
                    log_generations(
                        prompts=val_prompts,
                        generated_responses=generated_texts,
                        ground_truths=val_truths,
                        token_entropies=val_entropies,
                        reward_fn=r1_zero_reward_fn,
                        step=global_step,
                        save_logs=save_logs,
                        yaml_config_name = os.path.basename(args.exp_config).replace(".yaml", "")
                    )
                    
                    progress_bar.write("[Evaluation] 验证完成，继续训练...")
                    torch.cuda.empty_cache()
                    policy_model.train()

        # ==========================================
        # (F) Epoch 结束，保存 Checkpoint
        # ==========================================
        current_epoch = epoch + 1
        save_freq = cfg["training"].get("save_every_n_epochs", 1)
        if current_epoch % save_freq == 0:
            ckpt_dir = os.path.join(cfg["model"]["checkpoint_dir"], f"epoch_{current_epoch}")
            os.makedirs(ckpt_dir, exist_ok=True)
            progress_bar.write(f"💾 触发保存：正在保存 Epoch {current_epoch} 的断点至 {ckpt_dir} ...")
            policy_model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
        else:
            progress_bar.write(f"⏩ 跳过保存：当前是 Epoch {current_epoch}，策略设置为每 {save_freq} 个 Epoch 保存一次。")

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