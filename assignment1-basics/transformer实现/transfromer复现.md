# minst_clip

### Transformer结构解读：

Transformer由且仅由Attention和Feed Forward [Neural Network](https://so.csdn.net/so/search?q=Neural Network&spm=1001.2101.3001.7020)(也称FFN)组成，其中Attention包含self Attention与Mutil-Head Attention，如下图：

#### ![image-20251020170514839](C:\Users\15231\AppData\Roaming\Typora\typora-user-images\image-20251020170514839.png)

![image-20251027181731449](D:\typora-user-images\image-20251027181731449.png)

![image-20251027181746558](D:\typora-user-images\image-20251027181746558.png)

#### **左侧：编码器 (Encoder)**

编码器的核心任务是接收一个输入序列（例如一个句子），并将其转换为一组富含上下文信息的连续表示（向量）。

1. **Inputs -> Input Embedding**: 首先，输入的文本序列被分解为词元（tokens）。接着，**Input Embedding** 层将每个词元转换为一个固定维度的数字向量。

2. **Positional Encoding**: Transformer 模型本身不包含任何关于序列顺序的信息。为了解决这个问题，**Positional Encoding** (位置编码) 会为每个词元的嵌入向量添加一个独特的位置向量。这使得模型能够知道每个词元在序列中的具体位置。

3. **Nx 编码器层**: 输入经过嵌入和位置编码后，会进入一个堆叠了 N 次的编码器层。每一层都由两个核心子模块组成：

   - **Multi-Head Attention (多头注意力机制)**: 这是一个自注意力（Self-Attention）模块。它允许序列中的每个词元关注（attend to）序列中的所有其他词元，并计算它们之间的相关性得分。根据这些得分，它会为每个词元生成一个新的、融合了全局上下文信息的表示。所谓“多头”，是指这个过程会并行地在多个不同的表示子空间中进行，从而捕捉更多样的关联特征。

   - **Feed Forward (前馈神经网络)**: 这是一个简单的全连接神经网络，它独立地应用于每个词元位置的表示上。它的作用是对注意力模块输出的信息进行一次非线性变换，进一步增强模型的表达能力。（就是MLP）

   - **Add & Norm**: 每个子模块（注意力层和前馈网络）的周围都有一个 **Add & Norm** 环节。"Add" 指的是**残差连接**，它将子模块的输入直接加到其输出上，这有助于防止在深度网络中梯度消失，使模型更容易训练。"Norm" 指的是**层归一化**，它稳定了每层的数据分布，加速了训练过程。

     （LN适合小批量数据，对每个样本的所有特征维度进行归一化，BN对一个batch中的所有样本的同一特征维度进行归一化，再Batch维度归一化，即对于张量（N,C,H,W）,他要对C进行归一化，累加N,H,W,计算均值与方差。LN是对N进行归一化，累加通道，H和W）

![image-20251026213416733](D:\typora-user-images\image-20251026213416733.png)

**编码器最终输出**的是一组向量，这组向量是输入序列的最终上下文表示，它将被传递给解码器。

------

#### **右侧：解码器 (Decoder)**

解码器的任务是利用编码器生成的上下文表示，来生成目标输出序列。

1. **Outputs (shifted right) -> Output Embedding**: 在训练时，解码器接收的是目标序列，但会向右偏移一位（即在开头加上一个起始符）。**Output Embedding** 和 **Positional Encoding** 的作用与编码器端相同，将目标序列转换为带有位置信息的向量。
2. **Nx 解码器层**: 解码器同样由 N 个相同的层堆叠而成，但每层包含三个核心子模块：
   - **Masked Multi-Head Attention (带掩码的多头注意力)**: 这是解码器内部的自注意力模块。它与编码器的自注意力类似，但增加了一个“掩码”（Mask）。这个掩码会阻止任何位置关注其后续的位置。这是为了确保在预测当前词元时，模型只能依赖于已经生成的部分，而不能“偷看”未来的答案。
   - **Multi-Head Attention (交叉注意力)**: 这是连接编码器和解码器的**关键桥梁**。该模块的**查询（Query）** 来自于解码器自身的上一层（Masked Multi-Head Attention的输出），而**键（Key）和值（Value）** 则来自于**编码器的最终输出**。这使得解码器在生成每个词元时，都能有效地关注输入序列中的所有相关部分。
   - **Feed Forward (前馈神经网络)**: 其结构和功能与编码器中的前馈网络完全相同。
   - **Add & Norm**: 同样，每个子模块都配有残差连接和层归一化。

#### Attention计算

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

![image-20251020172458151](C:\Users\15231\AppData\Roaming\Typora\typora-user-images\image-20251020172458151.png)

在自注意力（Self-Attention）机制中，*Q*、*K*和*V*通常来源于同一组输入，而在编码器-解码器注意力（Encoder-Decoder Attention）中，*Q*来自解码器的上一层输出，而*K*和*V*来自编码器的输出。

对于每个输入，我们通过以下方式得到*Q*、*K*和*V*：

- *Q*=**$W^Q$**X
- *K*=*$W^K$*X
- *V*=$W^V$X

其中，*W**Q*、*W**K*和*W**V*是可学习的权重矩阵，*X*是输入特征。

![image-20251020173634770](C:\Users\15231\AppData\Roaming\Typora\typora-user-images\image-20251020173634770.png)

##### 1.Attention Score (AS)

$$
AttentionScore(AS) = QK^T = Q \times K^T
$$

这里 $Q$ 是一个大小为 $(n_q, d_q)$ 的矩阵，$K$ 是一个大小为 $(n_k, d_k)$ 的矩阵，其中 $d_q = d_k = d$，指的是 Key 或 Query 的维度。点乘结果为注意力得分矩阵 $AS$，大小为 $(n_q, n_k)$。因此 $AS_{i,j}$ 理解为第 $i$ 个 Query 对第 $j$ 个 Key 的注意力得分，这个值越大代表第 $i$ 个 Query 与第 $j$ 个 Key 的关联越大。

- **注**：如果是 Self-Attention，则 $n_q = n_k$，$AS$ 是一个方阵。

##### 2. 缩放（Scale）

$$
ScaledAttentionScores(SAS) = \frac{Q \times K^T}{\sqrt{d_k}} = \frac{AS}{\sqrt{d_k}}
$$

为了防止注意力值过大，Attention 机制中通常会对这个结果除以 $\sqrt{d_k}$ 进行缩放。这个缩放步骤是为了稳定训练过程。

- **注**：为什么是 $\sqrt{d_k}$？假设 $Q$ 和 $K$ 的方差都是 1，则 $Q \times K^T$ 的方差为 $\sqrt{d_k}$。

##### 3. 掩码 (Mask)

为了遮挡某些不需要关注的元素，会使用 Mask 操作。比如 NLP 问题，遮挡某个 token 对其之后出现的 token 的关注度。Mask 通常是一个与 Scores 矩阵相同维度的矩阵，包含 0 或 $-\infty$ 的值。

- 0：表示允许关注的元素。
- $-\infty$：表示需要遮挡的元素，使得这些元素在 Softmax 后变为 0 权重。

$$
MaskedScores = ScaledAttentionScores + Mask
$$

这样，在进行 Softmax 时，被遮挡的部分会被分配到接近 0 的权重。

##### 4. 归一化 (Softmax)

回顾Softmax公式：

$$
Softmax(z_i) = \frac{e^{z_i}}{\sum_{c=1}^{C} e^{z_c}}
$$

因此将 $ScaledAttentionScores$ 输入 $Softmax$，将缩放后的注意力分数归一化为一个概率分布。

$$
AttentionWeight_{i,j} = \frac{exp(\frac{Q_i \times K_j^T}{\sqrt{d_k}})}{\sum_{k=1}^{n_k} exp(\frac{Q_i \times K_k^T}{\sqrt{d_k}})}
$$

- **注**：这里的Softmax是针对每一行的操作，即对每个 Query 对所有 Key 的注意力得分进行归一化。

##### 5. 加权求和（MatMul 2）

$$
Attention = AttentionWeight \times V
$$

在3中得到的是一个服从概率分布的注意力权重矩阵 $AttentionWeight$，大小为 $(n_q, n_k)$。Value矩阵 $V$ 大小为 $(n_v, d_v)$，这里 $n_v = n_k$，可以理解为1个key对应1个value。因此，注意力矩阵大小为 $(n_q, d_v)$，可以理解为一条Query查到一个Value。$Attention_{(i,)}$ 代表了第 $i$ 个 Query 查询到的 Value 值。

- **注**：5与1一样，均为MatMul矩阵相乘操作，然而意义有所不同，1的目的是求Query和Key的相似度，而4的目的是对Value加权求和。

##### 6.矩阵大小的分析

总结以上，分析矩阵大小：

1. $$d_q = d_k$$。即 Query 与 Key 在维度上一致，Value 不一定保持一致。
2. $$n_k = n_v$$。即 Key 和 Value 在长度上一致，Query 不一定保持一致。
3. $$AttentionScore$$ 和 $$Attention$$，大小分别是 $$(n_q, n_k)$$ 和 $$(n_q, d_v)$$。

$$AttentionScore$$ 即为所有 Query 对于所有 Key 的相关程度；最终输出结果 $$Attention$$ 为所有 Query 根据注意力的信息查询到的 Value 结果。也代表整个 Attention 的最终输出矩阵大小为 $$(n_q, d_v)$$。

归根结底，Attention 是一个使用键（Key）进行查询（Query）以获取值（Value）的方法，神经网络训练无非是通过训练的方法拟合这个过程。

#### 多头注意力模块：

![image-20251020195956851](C:\Users\15231\AppData\Roaming\Typora\typora-user-images\image-20251020195956851.png)

#### FFN架构

FFN 层实际上就是一个线性变换层，用来完成输入数据到输出数据的维度变换（细节这里不介绍，相关链接暂时留白）。

这个FFN层是一个顺序结构：包括一个全连接层(FC) + relu激活层 + 第二个全连接层，其公式可以表示为：FFN(x) = max(0, xW1 + b1)W2 + b2。

上式中，xW1 + b1 为第一个全连接层的计算公式，max(0, xW1 + b1) 为 relu 的计算公式，max(0, xW1 + b1)W2 + b2 则为第二个全连接层的计算公式。

随后作者提到，添加这个 FFN 层的作用，主要是通过第一个FC层将输入词向量的512维变换到2048维，随后通过第二个FC层再将2048维变换回512维，从而保证 FFN 的输入输出维度一致。

FFN 层的结构展开可以表示如下：
![image-20251026211319774](D:\typora-user-images\image-20251026211319774.png)

作用很简单：

捕捉到更复杂的模式，增强表达能力

1. 增加了两个可学习的权值矩阵，也就是上面表达公式中的两个 **W** 矩阵。通过和权值矩阵的相乘将输入 512 维度向量映射到隐层的 2048 维度空间中，使得输入数据可以完成更加丰富的特征表达和运算。
2. 虽然FFN的输入输出维度都是512，但是输出的512维度特征和输入的512为特征是不一样的。输出的512维度特征是在隐层空间（2048）的基础上进一步融合得到的。**可以说，输出的512维比输入的512维具有更加丰富和准确的特征表示。**

可以用Relu  也可以用GELU 公式如下：

![image-20251026211425683](D:\typora-user-images\image-20251026211425683.png)

#### `Positional Encoding`

一句话概括，Positional Encoding就是将位置信息添加（嵌入）到Embedding词向量中，让Transformer保留词向量的**位置信息**，可以提高模型对序列的理解能力。

![image-20251026225415803](D:\typora-user-images\image-20251026225415803.png)

![image-20251026225712838](D:\typora-user-images\image-20251026225712838.png)

![](D:\typora-user-images\image-20251026230023282.png)

![image-20251026230202696](D:\typora-user-images\image-20251026230202696.png)

![image-20251026230213423](D:\typora-user-images\image-20251026230213423.png)

#### dropout层原理

Dropout 的核心思想是：**在模型训练时，随机地让一部分神经元“临时失业”，强迫剩下的神经元“承担更多责任”，从而让整个网络变得更强大、更不容易依赖少数几个“明星员工”。**

**调整输出**: 为了补偿训练时一部分神经元被丢弃而导致的输出总值偏小的问题，有一个关键步骤：

- **Inverted Dropout (反向 Dropout)**: 这是现在最主流的实现方式。在**训练时**，所有**未被丢弃**的神经元的输出会被放大 `1 / (1 - p)` 倍。例如，如果 `p=0.5`，那么留下的神经元输出会乘以 `1 / (1 - 0.5) = 2`。这样做的好处是，到了测试阶段，我们什么都不用做，直接让所有神经元“全员上岗”即可，输出的期望值与训练时保持一致。**您的代码和绝大多数框架（如 PyTorch, TensorFlow）都采用这种方式。**

ebedming

### 参数解释

- `vocab_size`：词汇表的大小，即词汇表中不同单词的数量加上一个特殊的 `<PAD>`（填充）标记。例如，如果词汇表中有10000个单词，那么 `vocab_size` 通常是 10001（假设有一个 `<PAD>` 标记）。
- `d_model`：嵌入向量的维度，即每个单词嵌入的大小。这个维度是模型的一个超参数，可以根据模型的复杂性和任务的需求进行调整。例如，在一个简单的模型中，`d_model` 可能是 128 或 256，而在一个更复杂的模型中，它可能是 512 或更大。

当模型接收到一个包含单词索引的输入序列时，`nn.Embedding` 模块会查找这些索引对应的嵌入向量，并将这些向量作为模型的输入。例如，如果输入是一个形状为 `[batch_size, seq_length]` 的张量，其中包含了单词索引，那么 `self.src_emb` 将输出一个形状为 `[batch_size, seq_length, d_model]` 的张量，其中包含了对应的嵌入向量。

### 示例

假设我们有一个词汇表，其中包含10000个单词，我们希望每个单词的嵌入维度为512。我们可以这样创建一个嵌入层：

```python
import torch.nn as nn

vocab_size = 10000  # 词汇表大小
d_model = 512        # 嵌入维度

model = nn.Module()
model.src_emb = nn.Embedding(vocab_size, d_model)
```

在这个例子中，`model.src_emb` 是一个嵌入层，它将词汇表中的每个单词索引转换为一个512维的向量。这个嵌入层可以用于模型的输入层，为模型提供连续的词嵌入表示。

#### crossattention注意

Q来自于序列A ,K,V来自于序列B 

![image-20251028223847711](D:\typora-user-images\image-20251028223847711.png)

![image-20251028224030451](D:\typora-user-images\image-20251028224030451.png)

编码器（Encoder）：使用 Self-Attention 处理输入序列（比如源语言句子），生成上下文表示。
解码器（Decoder）：分为两步：
Self-Attention：关注目标序列自身（比如已生成的目标语言单词）。（因此未来的词要屏蔽）
Cross Attention：用目标序列的 Query 去关注编码器的输出（K 和 V）。
这种设计让解码器能够动态地从源序列中提取信息，而不是一次性接收所有内容。例如，在翻译“I love you”到中文“我爱你”时，解码器生成“我”时会通过 Cross Attention 关注“I”，“爱”时关注“love”，从而实现精准对齐。

#### 解码器的自注意力掩码和交叉注意力掩码

自注意力除了pading 还需要屏蔽前瞻掩码，因为对于自注意力不能推断出后面的输出。Decoder 每一步已经可以**一次性看到整个 Encoder 输出**（源语言所有词），**不存在“未来”概念**——源语言序列早就完整存在，**没有“从左到右”的限制**。

对于交叉注意力掩码，在[机器翻译](https://so.csdn.net/so/search?q=机器翻译&spm=1001.2101.3001.7020)中，解码器需要关注编码器的输出。但如果编码器的句子被填充过，解码器就不应该关注这些填充位置。这就是交叉注意力掩码的作用。

### 训练

![image-20251105183818994](D:\typora-user-images\image-20251105183818994.png)

#### 损失函数训练：

模型根据已有的字符去猜下一个字符是什么，然后根据生成的logit的标签训练让正确的字符的概率最大进行训练。

**数值稳定性**

常用技巧，任何$o_1$先减去max($0_1$)，在做exp，这样不会出现梯度爆炸。

log(exp）直接约掉。

最终代码里通常写成：

```
 max_val = oi.max(dim=-1, keepdim=True)          # M
log_sum = (oi - max_val).exp().sum(dim=-1).log() + max_val.squeeze()  # 公式 B
loss_i = log_sum - oi[xi+1]                      公式 A 的右边
```

![image-20251105191819095](D:\typora-user-images\image-20251105191819095.png)

这个就是标准交叉熵，因为只有一个分类为1，所以前面的项没有，至于p是通过softmax出来的。

标签软化

![image-20251105195429869](D:\typora-user-images\image-20251105195429869.png)

标签软化要把pad的行与列完全封住，然后在进行软化（也可以提前KL会自己处理），会先用bacth模型归一化。

然后就是传入的张量不能进行训练。

### 学习率可变性

能使损失下降最快的学习率数值，往往会随着训练阶段而变化。在训练 Transformer 时，通常采用学习率调度策略：先使用较大的学习率，让初期更新更快；随后逐步衰减到较小值，使训练后期更稳定。

![image-20251105203648772](D:\typora-user-images\image-20251105203648772.png)

```python
scheduler = LambdaLR(optimizer, lr_lambda)   # 用户自定义缩放函数 ，这个函数里面就是乘子，会默认与初始学习率相乘
```

### 数据加载

#### 分词器实现

分词器用于将字符转化为相应的数字token，一般来说简易映射可以直接用字典完成

在实际应用中，你需要先用一个分词器（如 BPE, WordPiece）将文本转换为 token ID 序列。这里我们为了演示，先用简单的字符级分词。

```
# --- 简易的字符级分词器 ---
# 在实际项目中，你应该使用 Hugging Face Tokenizers 或 SentencePiece 训练一个 BPE 分词器
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i+1 for i, ch in enumerate(self.chars)} # 0 is reserved for padding
        self.itos = {i+1: ch for i, ch in enumerate(self.chars)}  #创建映射字典
        self.stoi['<pad>'] = 0
        self.itos[0] = '<pad>'
        config.VOCAB_SIZE = self.get_vocab_size() # 动态更新词汇表大小

    def get_vocab_size(self):
        return len(self.chars) + 1

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])
```

我们可以使用一个专业的tokenizers 库来训练一个 **Byte-Pair Encoding (BPE)** 分词器。

**数据预处理与缓存**：对于大型数据集，每次启动训练时都从头读取和分词是非常低效的。专业流程是**一次性地**将整个文本数据集分词，并将生成的 token ID 保存到一个二进制文件中。

**高效的内存加载**：在训练时，我们不会将整个 token ID 文件加载到 RAM 中，而是使用**内存映射（Memory Mapping）**技术。这使得操作系统可以智能地将需要的数据部分从磁盘加载到内存，即使数据集比你的 RAM 还大，也能高效运行。

首先训练一个 BPE 分词器，并将其保存到文件中以备后用。

先初始化一个空的BRE分词器。然后设置切分，最后设置训练器与特殊标记，最后就是拿训练集进行训练分词器，最后保存分词器。

```
# train_tokenizer.py

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import config

def train_bpe_tokenizer():
    """
    在原始文本数据上训练一个 BPE 分词器。
    """
    # 1. 初始化一个空的 BPE 分词器
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    

# 2. 设置预分词器 (按空格切分)

tokenizer.pre_tokenizer = Whitespace()

# 3. 设置训练器

# vocab_size 应该与 config.py 中的 VOCAB_SIZE 一致

# special_tokens 是模型需要理解的特殊标记

trainer = BpeTrainer(
    vocab_size=config.VOCAB_SIZE, 
    special_tokens=["<unk>", "<pad>", "<s>", "</s>"]
)

# 4. 训练分词器

# files 参数可以是一个文件列表

print("Training BPE tokenizer...")
tokenizer.train(files=[config.DATASET_PATH], trainer=trainer)

# 5. 保存分词器

tokenizer.save(config.TOKENIZER_PATH)
print(f"Tokenizer trained and saved to {config.TOKENIZER_PATH}")

if __name__ == '__main__':
    train_bpe_tokenizer()
```

![image-20251106224012274](D:\typora-user-images\image-20251106224012274.png)

[(38 封私信 / 80 条消息) BPE 算法原理及使用指南【深入浅出】 - 知乎](https://zhuanlan.zhihu.com/p/448147465)

古典分词缺点，会出现OOV问题：对于**未在词表中出现的词（Out Of Vocabulary, OOV** ），模型将无法处理（未知符号标记为 `[UNK]`）。

**Character embedding**：拆分为极端分词方法，字母，解决了OOV问题，但是训练成本太高。

**Subword 算法** ：

**基于子词的分词方法（Subword Tokenization）** ，简称为 Subword 算法，意思就是把一个词切成更小的一块一块的子词。如果我们能使用将一个 token 分成多个 subtokens，[上面的问题](https://link.zhihu.com/?target=https%3A//www.wolai.com/heU7bADzdRaJje1jdeoVLp%23vFA25b5i9XhnafxdBEhtX1)就能很好的解决。

这种方法的目的是通过**一个有限的词表** 来解决所有单词的分词问题，同时尽可能将结果中 token 的数目降到最低。例如，可以用更小的词片段来组成更大的词，例如：

“**unfortunately** ” = “**un** ” + “**for** ” + “**tun** ” + “**ate** ” + “**ly** ”。

可以看到，有点类似英语中的词根词缀拼词法，其中的这些小片段又可以用来构造其他词。可见这样做，既可以降低词表的大小，同时对相近词也能更好地处理。

**BPE算法：**

![image-20251107150849976](D:\typora-user-images\image-20251107150849976.png)

1. 准备语料库，确定期望的 subword 词表大小等参数
2. 通常在每个单词末尾添加后缀 `</w>`，统计每个单词出现的频率，例如，`low` 的频率为 5，那么我们将其改写为 `"l o w </ w>”：5`
   注：停止符 `</w>` 的意义在于标明 subword 是词后缀。举例来说：`st` 不加 `</w>` 可以出现在词首，如 `st ar`；加了 `</w>` 表明该子词位于词尾，如 `we st</w>`，二者意义截然不同
3. 将语料库中所有单词拆分为单个字符，用所有单个字符建立最初的词典，并统计每个字符的频率，本阶段的 subword 的粒度是字符
4. **挑出频次最高的符号对** ，比如说 `t` 和 `h` 组成的 `th`，将新字符加入词表，然后将语料中所有该字符对融合（merge），即所有 `t` 和 `h` 都变为 `th`。
   注：新字符依然可以参与后续的 merge，有点类似哈夫曼树，BPE 实际上就是一种**贪心算法** 。
5. 重复遍历 2 和 3 操作，直到**词表中单词数达到设定量** 或**下一个最高频数为 1** ，如果已经打到设定量，其余的词汇直接丢弃

注：看似我们要维护两张表，一个词表，一个字符表，实际上只有一张，词表只是为了我们方便理解。

[(38 封私信 / 80 条消息) 2.1、BPE分词器（手搓+库调用） - 知乎](https://zhuanlan.zhihu.com/p/1965109854315213748)
