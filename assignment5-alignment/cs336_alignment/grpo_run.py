# grpo_run.py
import os
import sys
import json
import yaml
import torch
import wandb
import random
import argparse

import collections.abc

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup
from vllm import LLM, SamplingParams

from cs336_alignment.utils import tokenize_prompt_and_output, get_response_log_probs
from cs336_alignment.grpo_Dr import compute_group_normalized_rewards, grpo_microbatch_train_step
from drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.grpo_log import log_train_metrics,log_periodic_eval_metrics,log_generations


# ==========================================
# 辅助工具函数
# ==========================================
def update_dict(d, u):
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
    return update_dict(base_cfg, exp_cfg)

# 最大训练样本
def load_and_filter_data(data_path, max_samples=None):
    dataset = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    random.seed(42)
    random.shuffle(dataset)
    if max_samples and max_samples < len(dataset):
        dataset = dataset[:max_samples]
    return dataset

# 权重迁移
def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# ==========================================
# 三大评估与日志函数
# ==========================================




# ==========================================
# 主训练循环
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, required=True)
    parser.add_argument("--exp_config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_and_validate_config(args)

    if cfg["model"].get("wandb_id"):
        wandb.init(project=cfg["wandb"]["project"], id=cfg["model"]["wandb_id"], resume="must", config=cfg)
    else:
        wandb.init(project=cfg["wandb"]["project"], name=cfg["wandb"]["run_name"], config=cfg)
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    wandb.define_metric("final_eval/*", step_metric="train_step")  # Final 绑在最新的 train_step 上

    base_model_path = cfg["model"]["model_path"]
    # 最终模型直接使用 output_dir
    final_model_dir = cfg["training"]["output_dir"]
    # 断点保存路径
    checkpoint_dir = cfg["training"].get("checkpoint_dir", "checkpoints/grpo_steps")
    n_grpo_steps = cfg["training"]["n_grpo_steps"]
    group_size = cfg["training"]["group_size"]
    rollout_batch_size = cfg["training"]["rollout_batch_size"]
    micro_batch_size = cfg["training"]["micro_batch_size"]
    epochs_per_rollout_batch = cfg["training"]["epochs_per_rollout_batch"]
    gradient_accumulation_steps = rollout_batch_size // micro_batch_size

    normalize_by_std = cfg["training"]["normalize_by_std"]
    remove_length_norm = cfg["training"]["remove_length_norm"]
    advantage_eps = float(cfg["training"]["advantage_eps"])
    clip_range = float(cfg["training"]["clip_range"])
    loss_type = cfg["training"]["loss_type"]
    # 是否跳过训练部分
    skip_training = os.path.exists(final_model_dir)

    if skip_training:
        print(f"\n✅ 检测到已训练完成的模型: {final_model_dir}，将跳过训练直接评估！\n")
        active_model_path = final_model_dir 
    else:
        print(f"\n🚀 未检测到最终模型，将从 {base_model_path} 开始训练...")
        active_model_path = base_model_path
    

    prompt_template = "{question}"
    if "prompt_path" in cfg["data"] and os.path.exists(cfg["data"]["prompt_path"]):
        with open(cfg["data"]["prompt_path"], "r", encoding="utf-8") as f:
            prompt_template = f.read()
    #找到train_data
    train_data = load_and_filter_data(cfg["data"]["train_path"], cfg["data"]["max_samples"])
    #中途评测
    periodic_valid_data = load_and_filter_data(cfg["validation"]["valid_path"], cfg["validation"].get("max_samples", 100))
    # 最后评估
    final_eval_data = load_and_filter_data(cfg["evaluation"]["valid_path"], cfg["evaluation"].get("max_samples", None))
    n_prompts_per_batch = rollout_batch_size // group_size
    # logits用的
    tokenizer = AutoTokenizer.from_pretrained(active_model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    # 模型初始化
    policy_model = AutoModelForCausalLM.from_pretrained(
        active_model_path,
        torch_dtype=torch.bfloat16 if cfg["model"]["dtype"] == "bfloat16" else torch.float32,
        attn_implementation=cfg["model"]["attn_implementation"]
    ).cuda()
    if cfg["training"]["gradient_checkpointing"]: policy_model.gradient_checkpointing_enable()
    # 两次回答的LLM与sampling_params
    llm = LLM(
        model=active_model_path,
        trust_remote_code=True,
        gpu_memory_utilization=cfg["training"]["gpu_memory_utilization"],
        enable_prefix_caching=cfg["training"]["enable_prefix_caching"],
        enforce_eager=cfg["training"]["enforce_eager"]
    )

    train_sampling_params = SamplingParams(
        temperature=cfg["training"]["sampling_temperature"],
        repetition_penalty=cfg["training"]["repetition_penalty"],
        min_tokens=cfg["training"]["sampling_min_tokens"],
        max_tokens=cfg["training"]["sampling_max_tokens"],
        n=group_size,
        stop=cfg["training"]["stop_tokens"],
        include_stop_str_in_output=True
    )

    eval_sampling_params = SamplingParams(
        temperature=cfg["evaluation"]["temperature"],
        top_p=cfg["evaluation"]["top_p"],
        max_tokens=cfg["evaluation"]["max_new_tokens"],
        n=1,
        stop=cfg["training"]["stop_tokens"],
        include_stop_str_in_output=True
    )
    if not skip_training:
        # 初始化优化器与学习率
        optimizer = torch.optim.AdamW(policy_model.parameters(), lr=float(cfg["training"]["learning_rate"]))
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(n_grpo_steps * cfg["training"]["warmup_ratio"]),
                                                num_training_steps=n_grpo_steps)

        # ==========================================
        # 开始训练
        # ==========================================
        policy_model.train()

        for step in range(cfg["model"]["start_step"], n_grpo_steps):
            #导入最新权重
            load_policy_into_vllm_instance(policy_model, llm)
            # 抽取数据集
            batch_samples = random.sample(train_data, n_prompts_per_batch)
            prompts = [
                item.get("prompt", prompt_template.format(question=item.get("question", ""))) 
                for item in batch_samples
            ]
            ground_truths = [item.get("ground_truth", item.get("answer", "")) for item in batch_samples]

            outputs = llm.generate(prompts, train_sampling_params)
            #
            rollout_prompts, rollout_responses, repeated_ground_truths = [], [], []
            for i, output in enumerate(outputs):
                for gen in output.outputs:
                    rollout_prompts.append(prompts[i])
                    rollout_responses.append(gen.text)
                    repeated_ground_truths.append(ground_truths[i])
            # 计算基线
            advantages, raw_rewards, adv_metadata = compute_group_normalized_rewards(
                reward_fn=r1_zero_reward_fn,
                rollout_responses=rollout_responses,
                repeated_ground_truths=repeated_ground_truths,
                group_size=group_size,
                advantage_eps=advantage_eps,
                normalize_by_std=normalize_by_std
            )
            # 生成掩码
            tokens_dict = tokenize_prompt_and_output(rollout_prompts, rollout_responses, tokenizer,
                                                    max_seq_length=cfg["data"]["max_seq_length"])

            input_ids = tokens_dict["input_ids"].cuda()
            labels = tokens_dict["labels"].cuda()
            response_mask = tokens_dict["response_mask"].cuda()

            normalize_constant = cfg["training"]["fixed_norm_length"] if cfg["training"][
                "fixed_norm_length"] else response_mask.sum(dim=1).max().item()

            with torch.no_grad():
                policy_model.eval()
                # 提取真实对数概率
                old_log_probs_dict = get_response_log_probs(policy_model, input_ids, labels, return_token_entropy=True)
                old_log_probs = old_log_probs_dict["log_probs"]
                token_entropy = old_log_probs_dict.get("token_entropy", torch.tensor(0.0))
            policy_model.train()

            avg_entropy = token_entropy.mean().item() if isinstance(token_entropy, torch.Tensor) else 0.0

            # 🌟 修复点 1：在循环外提前初始化用于跨 Epoch 统计的变量
            total_loss = 0.0
            total_clip_fraction = 0.0
            total_ratio_mean = 0.0
            last_grad_norm = 0.0
            # 微批次
            for epoch in range(epochs_per_rollout_batch):
                accumulated_loss = 0.0
                epoch_clip_fraction = 0.0
                epoch_ratio_mean = 0.0
                num_micro_batches = 0
                indices = list(range(rollout_batch_size))
                random.shuffle(indices)
                # 微批次
                for i in range(0, rollout_batch_size, micro_batch_size):
                    mb_indices = indices[i:i + micro_batch_size]
                    mb_input_ids, mb_labels, mb_mask = input_ids[mb_indices], labels[mb_indices], response_mask[mb_indices]
                    mb_advantages, mb_old_log_probs = advantages[mb_indices].cuda(), old_log_probs[mb_indices]
                    # 虽然是64个微批次，但是由于跑完才更新，因此一直是1，如果epoch等于1的话
                    new_log_probs_dict = get_response_log_probs(policy_model, mb_input_ids, mb_labels,
                                                                return_token_entropy=False)
                    loss,meta = grpo_microbatch_train_step(
                        policy_log_probs=new_log_probs_dict["log_probs"],
                        response_mask=mb_mask,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        loss_type=loss_type,
                        advantages=mb_advantages,
                        old_log_probs=mb_old_log_probs,
                        cliprange=clip_range,
                        remove_length_norm=remove_length_norm,
                        normalize_constant=normalize_constant
                    )
                    accumulated_loss += loss.item()

                    epoch_clip_fraction += meta.get("clip_fraction", 0.0)
                    epoch_ratio_mean += meta.get("ratio_mean", 1.0)
                    num_micro_batches += 1

                grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), cfg["training"]["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # 计算当前 Epoch 的平均裁剪比例和概率比值
                total_loss += accumulated_loss
                total_clip_fraction += (epoch_clip_fraction / max(num_micro_batches, 1))
                total_ratio_mean += (epoch_ratio_mean / max(num_micro_batches, 1))

            # 【画图】记录训练集的平均真实Reward与熵
            actual_epochs = max(epochs_per_rollout_batch, 1)
            final_loss = total_loss / actual_epochs
            final_clip_fraction = total_clip_fraction / actual_epochs
            final_ratio_mean = total_ratio_mean / actual_epochs

            train_metrics = {
                "train/loss": final_loss,
                "train/reward_mean": adv_metadata.get("reward_mean", 0.0),
                "train/reward_std": adv_metadata.get("reward_std", 0.0),
                "train/advantage_mean": adv_metadata.get("advantage_mean", 0.0),
                "train/advantage_std": adv_metadata.get("advantage_std", 0.0),
                "train/clip_fraction": final_clip_fraction,
                "train/ratio_mean": final_ratio_mean,
                "train/entropy": avg_entropy,
                "train/grad_norm": last_grad_norm,
                "train/lr": scheduler.get_last_lr()[0]
            }



            log_train_metrics(step, train_metrics)
            print(f"Train Step {step}/{n_grpo_steps} | Train Reward: {adv_metadata['reward_mean']:.3f}")

            # 【画图】定期进行验证集评估（只看重Reward均值）
            if (step + 1) % cfg["evaluation"]["eval_every_steps"] == 0 and (step + 1) != n_grpo_steps:
                # 权重迁移
                load_policy_into_vllm_instance(policy_model, llm)
                
                # 兼容 Prompt 拼接，使用 periodic_valid_data
                val_prompts_periodic = [item.get("prompt", prompt_template.format(question=item.get("question", ""))) for item in periodic_valid_data]
                val_truths_periodic = [item.get("ground_truth", item.get("answer", "")) for item in periodic_valid_data]

                log_periodic_eval_metrics(step, llm, val_prompts_periodic, val_truths_periodic, eval_sampling_params)

                ckpt_dir = os.path.join(checkpoint_dir, f"step_{step + 1}")
                os.makedirs(ckpt_dir, exist_ok=True)
                policy_model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

    # ==========================================
    # 终极评估 (训练彻底结束后触发)
    # ==========================================
    print("\n🚀 训练全部结束！正在启动对验证集的终极详尽评估...")
    load_policy_into_vllm_instance(policy_model, llm)

    val_prompts_raw = [
        item.get("prompt", prompt_template.format(question=item.get("question", ""))) 
        for item in final_eval_data
    ]
    val_truths = [item.get("ground_truth", item.get("answer", "")) for item in final_eval_data]

    eval_outputs = llm.generate(val_prompts_raw, eval_sampling_params)
    generated_texts = [out.outputs[0].text for out in eval_outputs]

    # 为了获取 token_entropies，需要用 PyTorch 模型过一次前向
    print("获取生成结果的熵特征...")
    token_entropies_list = []
    policy_model.eval()
    with torch.no_grad():
        for vp, vg in zip(val_prompts_raw, generated_texts):
            t_data = tokenize_prompt_and_output([vp], [vg], tokenizer, max_seq_length=cfg["data"]["max_seq_length"])
            out_dict = get_response_log_probs(
                policy_model,
                t_data["input_ids"].cuda(),
                t_data["labels"].cuda(),
                return_token_entropy=True
            )
            mask = t_data["response_mask"].cuda()
            token_entropies_list.append(out_dict["token_entropy"][mask == 1])

    # 调用你的详尽打印写入函数
    save_logs_path = cfg["evaluation"].get("save_logs", "output/grpo_final_eval_logs.txt")
    log_generations(
        prompts=val_prompts_raw,
        generated_responses=generated_texts,
        ground_truths=val_truths,
        token_entropies=token_entropies_list,
        reward_fn=r1_zero_reward_fn,
        step=n_grpo_steps,
        save_logs=save_logs_path,
        yaml_config_name=os.path.basename(args.exp_config),
        tokenizer=tokenizer
    )

    # 保存最终模型
    final_dir = os.path.join(cfg["training"]["output_dir"], "final_model")
    policy_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    wandb.finish()


if __name__ == "__main__":
    main()