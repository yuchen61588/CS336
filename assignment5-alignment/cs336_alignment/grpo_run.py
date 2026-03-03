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

    model_path = cfg["model"]["model_path"]
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

    with open(cfg["data"]["prompt_path"], "r", encoding="utf-8") as f:
        prompt_template = f.read()

    train_data = load_and_filter_data(cfg["data"]["train_path"], cfg["data"]["max_samples"])
    valid_data = load_and_filter_data(cfg["evaluation"]["valid_path"], cfg["evaluation"]["max_samples"])
    n_prompts_per_batch = rollout_batch_size // group_size

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if cfg["model"]["dtype"] == "bfloat16" else torch.float32,
        attn_implementation=cfg["model"]["attn_implementation"]
    ).cuda()
    if cfg["training"]["gradient_checkpointing"]: policy_model.gradient_checkpointing_enable()

    llm = LLM(
        model=model_path,
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

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=float(cfg["training"]["learning_rate"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(n_grpo_steps * cfg["training"]["warmup_ratio"]),
                                                num_training_steps=n_grpo_steps)

    # ==========================================
    # 开始训练
    # ==========================================
    policy_model.train()

    for step in range(cfg["model"]["start_step"], n_grpo_steps):
        load_policy_into_vllm_instance(policy_model, llm)

        batch_samples = random.sample(train_data, n_prompts_per_batch)
        prompts = [prompt_template.format(question=item["question"]) for item in batch_samples]
        ground_truths = [item.get("ground_truth", item.get("answer", "")) for item in batch_samples]

        outputs = llm.generate(prompts, train_sampling_params)

        rollout_prompts, rollout_responses, repeated_ground_truths = [], [], []
        for i, output in enumerate(outputs):
            for gen in output.outputs:
                rollout_prompts.append(prompts[i])
                rollout_responses.append(gen.text)
                repeated_ground_truths.append(ground_truths[i])

        advantages, raw_rewards, adv_metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=normalize_by_std
        )

        tokens_dict = tokenize_prompt_and_output(rollout_prompts, rollout_responses, tokenizer,
                                                 max_seq_length=cfg["data"]["max_seq_length"])
        # 生成掩码padding

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

        for epoch in range(epochs_per_rollout_batch):
            accumulated_loss = 0.0
            indices = list(range(rollout_batch_size))
            random.shuffle(indices)
            for i in range(0, rollout_batch_size, micro_batch_size):
                mb_indices = indices[i:i + micro_batch_size]
                mb_input_ids, mb_labels, mb_mask = input_ids[mb_indices], labels[mb_indices], response_mask[mb_indices]
                mb_advantages, mb_old_log_probs = advantages[mb_indices].cuda(), old_log_probs[mb_indices]

                new_log_probs_dict = get_response_log_probs(policy_model, mb_input_ids, mb_labels,
                                                            return_token_entropy=False)
                loss, _ = grpo_microbatch_train_step(
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

            grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), cfg["training"]["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # 【画图】记录训练集的平均真实Reward与熵
        avg_entropy = token_entropy.mean().item() if isinstance(token_entropy, torch.Tensor) else 0.0
        log_train_metrics(step, adv_metadata["reward_mean"], accumulated_loss, grad_norm.item(),
                          scheduler.get_last_lr()[0], avg_entropy)
        print(f"Train Step {step}/{n_grpo_steps} | Train Reward: {adv_metadata['reward_mean']:.3f}")

        # 【画图】定期进行验证集评估（只看重Reward均值）
        if (step + 1) % cfg["evaluation"]["eval_every_steps"] == 0 and (step + 1) != n_grpo_steps:
            load_policy_into_vllm_instance(policy_model, llm)
            log_periodic_eval_metrics(step, llm, valid_data, prompt_template, eval_sampling_params)

            ckpt_dir = os.path.join(cfg["training"]["output_dir"], f"step_{step + 1}")
            policy_model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    # ==========================================
    # 终极评估 (训练彻底结束后触发)
    # ==========================================
    print("\n🚀 训练全部结束！正在启动对验证集的终极详尽评估...")
    load_policy_into_vllm_instance(policy_model, llm)

    val_prompts_raw = [prompt_template.format(question=item["question"]) for item in valid_data]
    val_truths = [item.get("ground_truth", item.get("answer", "")) for item in valid_data]

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