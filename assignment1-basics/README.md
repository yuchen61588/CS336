# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

有关作业的完整说明，请参阅作业讲义文件 `cs336_spring2025_assignment1_basics.pdf`。

如果您发现作业讲义或代码有任何问题，请随时提交 GitHub Issue 或发起 Pull Request 进行修复。

### 设置 (Setup)

#### 环境 (Environment)

我们使用 `uv` 来管理环境，以确保可复现性、可移植性和易用性。 请[在此处]安装 `uv`（推荐），或者运行 `pip install uv` / `brew install uv`。 我们建议阅读一些关于使用 `uv` 管理项目的内容[点击此处]（绝对值得一读！）。

您现在可以使用以下命令运行仓库中的任何代码：

Bash

```
uv run <python_file_path>
```

环境将在必要时自动解析并激活。

#### 运行单元测试 (Run unit tests)

Bash

```
uv run pytest
```

最初，所有测试都应该失败并报错 `NotImplementedErrors`。 要将您的实现与测试连接起来，请完成 `./tests/adapters.py` 中的函数。

#### 下载数据 (Download data)

下载 TinyStories 数据集以及 OpenWebText 的子样本。

rm -rf .venv
# 使用 Python 3.12（推荐，torch 2.6.0 支持）
uv venv --python 3.12
source .venv/bin/activate

# 测试BPE分词器训练部分
pytest tests/test_train_bpe.py -v
# 测试BPE分词器推理部分
pytest tests/test_tokenizer.py -v
# 测试model部分
pytest tests/test_model.py -v
