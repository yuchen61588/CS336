# CS336 Spring 2025 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)

We include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
安装软件包：先安装除 flash-attn 以外的所有包，然后再安装全部（因为 flash-attn 的安装机制比较特殊/古怪）：
```
uv sync --no-install-package flash-attn
uv sync
uv pip install --upgrade datasets fsspec huggingface_hub
uv pip install -U datasets
uv pip install --upgrade wandb

export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
uv cache clean
实在不行把venv删了
```

2. Run unit tests:

``` sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).
初始状态下，所有测试都会因为 NotImplementedError（未实现错误）而失败。
为了将你的代码实现与测试框架连接起来，请完成 ./tests/adapters.py 中的函数编写。

source .venv/bin/activate
/bin/vim ~/.bashrc
vim ~/.bashrc

# baseline命令
python cs336_alignment/math_baseline.py --model_name qwen_1.5b --dataset_type GSM8K
python cs336_alignment/math_baseline.py --model_name qwen_1.5b --dataset_type Bespoke17k
# sft 数据集训练
python cs336_alignment/data_preparation/generation_data_Bespoke-Stratos-17k_RL_SFT.py
python cs336_alignment/sft_run.py --base_config config/sft/sft_base.yaml --exp_config config/sft/qwen_2.5B_bespoke_128_false.yaml
python cs336_alignment/run_all_yaml.py \
  --run_script cs336_alignment/sft_run.py \
  --base_config config/sft/sft_base.yaml \
  --exp_pattern "config/sft/qwen_2.5B_GSM8K*_*.yaml"

python cs336_alignment/run_all_yaml.py \
  --run_script cs336_alignment/sft_run.py \
  --base_config config/sft/sft_base.yaml \
  --exp_pattern "config/sft/qwen_2.5B_Bespoke17k*_*.yaml"



