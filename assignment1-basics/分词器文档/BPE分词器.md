## 基本原理

### Unicode转化为UTF-8

 Unicode 标准包含了超过 15 万个字符（Code points）。如果模型要认识每一个字符，词表（Vocabulary）就会变得非常巨大。

 不直接看“字符”，而是通过 UTF-8 编码将字符转换成**字节**，对于一个字符，可以由多个字节构成，每个字节的范围是0-255。

encode函数用于访问UTF-8值转化为字节值。转化为list，可以转化为列表，表视如下：decode用来将字节符转为UTF-8

![image-20260118170954947](D:\typora-user-images\image-20260118170954947.png)

不同字符的转化方式如下：

| **需要字节数** | **能够容纳的编号范围 (Unicode)**   | **二进制模板 (x 是填空位，红色的固定表头)** |
| -------------- | ---------------------------------- | ------------------------------------------- |
| **1 字节**     | 0 ~ 127 (ASCII 英文)               | `0xxxxxxx`                                  |
| **2 字节**     | 128 ~ 2,047 (欧洲语言、符号)       | `110xxxxx` `10xxxxxx`                       |
| **3 字节**     | 2,048 ~ 65,535 (常用汉字、日韩文)  | `1110xxxx` `10xxxxxx` `10xxxxxx`            |
| **4 字节**     | 65,536 ~ 1,114,111 (Emoji、生僻字) | `11110xxx` `10xxxxxx` `10xxxxxx` `10xxxxxx` |

对于这些不同的汉字，就是根据他们原本的二进制编码（ Unicode）编码，按照顺序填充，最后形成不同的地址序列，例如[228, 184, 173]。，同样对于UTF-8编码，可以根据表头进行反退（哈夫曼编码），通过这个来查表为Unicode字符

### 子词标记化

BPE 是一种**子词（Subword）分词**方法，它是“词级（word-level）分词”和“字节级（byte-level）分词”之间的一个折中方案 。

- **目的**：它通过牺牲一点词表大小（即增加词表项），来换取对输入序列更好的压缩效果（即缩短序列长度）。
- （就是加入一些常用词，让序列token变小）

2. 运作机制（算法流程）

BPE 的训练过程是一个迭代的压缩过程：

1. **初始化**：从一个包含 256 个基本条目（字节值 0-255）的词表开始 。
2. **统计频率**：在训练数据中，统计所有相邻字节对（byte pairs）出现的频率 。
3. **合并（Merge）**：找到出现频率最高的字节对（例如 `('t', 'h')`），将其合并为一个新的、未使用的 token（例如 `th`），并加入到词表中 。
4. **迭代**：重复上述步骤。新的 token（如 `th`）可以继续参与下一轮的合并（例如与 `e` 合并成 `the`）。

3. 最终效果

- **高频词**：如果一个单词在文本中出现得足够频繁，经过多次合并后，它最终会被表示为一个单独的 token（例如 `the`）。
- **低频词/生僻词**：对于不常见的词，它们会被保留为多个子词或原始字节的序列

### BPE训练步骤：

**词表初始化（Vocabulary initialization）** 分词器词表是从字节串 token 到整数 ID 的一一映射。因为我们要训练的是字节级 BPE 分词器，我们的初始词表仅仅是所有字节的集合。由于有 256 个可能的字节值，我们的初始词表大小为 256。

**预分词（Pre-tokenization）** 一旦你有了词表，原则上你可以统计字节在文本中相邻出现的频率，并从最频繁的字节对开始合并它们。然而，这在计算上非常昂贵，因为每次合并我们都必须遍历整个语料库。此外，直接在语料库上合并字节可能导致生成的 token 仅在标点符号上有所不同（例如 `dog!` 与 `dog.`）。尽管这些 token 可能具有很高的语义相似度（因为它们仅在标点上不同），它们却会得到完全不同的 token ID。

我们对语料库进行**预分词**。你可以将其视为对语料库进行的一种粗粒度分词，这有助于我们统计字符对出现的频率。例如，单词 `text` 可能是一个出现 10 次的预分词 token（pre-token）。在这种情况下，当我们统计字符 `t` 和 `e` 相邻出现的频率时，我们会看到单词 `text` 中 `t` 和 `e` 是相邻的，我们可以直接将它们的计数增加 10，而不是去遍历语料库。由于我们正在训练字节级 BPE 模型，每个预分词 token 都表示为 UTF-8 字节序列。

原始 BPE 实现通过简单地按空格拆分（即 `s.split(" ")`）来进行预分词。相比之下，我们将使用**基于正则表达式的预分词器**（GPT-2 使用；Radford 等人，2019），

但是在代码中使用它时，你应该使用 `re.finditer` 以避免在构建从预分词 token 到其计数的映射时存储所有预分词后的单词。

**计算 BPE 合并（Compute BPE merges）** 既然我们已经将输入文本转换为预分词 token，并将每个预分词 token 表示为 UTF-8 字节序列，我们就可以计算 BPE 合并了（即训练 BPE 分词器）。在高层面上，BPE 算法迭代地统计每对字节的频率，并识别出频率最高的一对（“A”, “B”）。每一个出现的最高频对（“A”, “B”）随后被**合并**，即替换为一个新的 token “AB”。这个新的合并 token 被添加到我们的词表中；因此，BPE 训练后的最终词表大小是初始词表大小（在这个例子中是 256），加上训练期间执行的 BPE 合并操作的数量。为了在 BPE 训练期间提高效率，我们不考虑跨越预分词 token 边界的配对。当计算合并时，通过**优先选择字典序较大的对**来确定性地打破频率平局。例如，如果对（“A”, “B”）、（“A”, “C”）、（“B”, “ZZ”）和（“BA”, “A”）都具有最高的频率，我们将合并（“BA”, “A”）：

**注意**：

**边界限制**：合并操作**不能**跨越预分词的边界。例如，如果 `hello world` 被预分词为 `['hello', 'world']`，那么 `hello` 的结尾 `o` 和 `world` 的开头 `w` 永远不会被合并，即使 `ow` 这个组合在全局很常见。

**打破平局（Tie-breaking）**：这是一个很容易被忽视的细节。如果有多个字节对的频率完全相同（例如 `('a', 'b')` 和 `('c', 'd')` 都出现了 100 次），标准做法是选择**字典序更大**的那一对（例如 `('c', 'd')` 会优于 `('a', 'b')`）。这是为了保证算法的确定性，让每次运行结果一致。

解释：由于每次合并都将**合并后的新 Token “追加”在现有词表的末尾**。因此要找最大的。



**特殊 Token（Special tokens）** 通常，一些字符串（例如 `<|endoftext|>`）被用来编码元数据（例如文档之间的边界）。在编码文本时，通常希望将某些字符串视为“特殊 token”，它们永远不应被分割成多个 token（即，将始终保留为单个 token）。例如，序列结束字符串 `<|endoftext|>` 应该始终被保留为单个 token（即单个整数 ID），以便我们知道何时停止从语言模型生成内容。这些特殊 token 必须添加到词表中，以便它们具有对应的固定 token ID。

Sennrich 等人 [2016] 的算法 1 包含了一个低效的 BPE 分词器训练实现（基本上遵循了我们上面概述的步骤）。作为第一个练习，实现并测试这个函数可能有助于检验你的理解。

## 高效代码写法（维护词频表）

三种优化策略：

注意，合并本身要针对的是元组，所以我们能要维护元组的每个合并对象，而不是针对每个词进行合并。

##### 倒排索引：

![image-20260119001620752](D:\typora-user-images\image-20260119001620752.png)

对于每一个合并，可以通过这种元组+集合的方式精准找到每一个单词，在后续便于合并。

##### 频率桶

不使用频率堆的原因：

1.它们不支持高效的“修改任意元素的值”，对于最大的一个频率堆，你无法让他能够任意的修改到其他值，这样难以插入。

2.代码逻辑里面要求对于相同频率的表

![image-20260119002904536](D:\typora-user-images\image-20260119002904536.png)

使用max指针的原因，_freq_buckets本质是一个哈希表，如果排序找的话很慢，但是这个合并词频是不可能增大的，因此可以用这个。

##### **单词合并逻辑**：双向链表逻辑

分词器最多只能找到自己的词进行合并，因此合并的时候通过倒排索引就可以查到tokenn，在token里面遍历

更新逻辑：编写函数给出全局的变化量，需要移除什么，移除多少，需要添加什么，添加多少。

对于频率表，根据change表进行修改。新生的关系有一个表会记录这个集合，

对于倒排索引，在上面的更新逻辑会出现新的id，添加就可以（外层循环迭代，每次都更新），对于死掉的旧关系，其实可以不用删的，因为单词序号变化了，没有这个pair了，这样就可以直接掠过了。

##### 多层重复逻辑

懒惰合并的原因：

![image-20260119010039660](D:\typora-user-images\image-20260119010039660.png)

![image-20260119010453812](D:\typora-user-images\image-20260119010453812.png)



## 问题

### 第二问

(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.  Deliverable: A one-to-two sentence response. 

我们更倾向于 UTF-8，因为它对于以 ASCII 为主的文本更节省内存（每个字符使用 1 个字节，而不是 2 或 4 个），避免了字节序问题，并且允许我们将输入建模为来自 256 个固定词表的原始字节序列。

 (b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.  

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):

 return "".join([bytes([b]).decode("utf-8") for b in bytestring])  >>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")) 'hello'  

**例输入：** `b'\xe7\x89\x9b'`（字符“牛”的 UTF-8 字节）。**解释：** 该函数不正确，因为它试图单独解码每个字节，但多字节 UTF-8 字符（如“牛”）使用必须一起解码的字节序列来表示单个字符，拆分它们会导致错误

Deliverable: An example input byte string for which decode_utf8_bytes_to_str_wrong produces incorrect output, with a one-sentence explanation of why the function is incorrect. 

 (c) Give a two byte sequence that does not decode to any Unicode character(s).  Deliverable: An example, with a one-sentence explanation.

不符合上面字节的都一样。