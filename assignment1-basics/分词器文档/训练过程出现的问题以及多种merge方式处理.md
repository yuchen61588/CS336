### get_max_pari出现的效率问题



原始代码

```
def get_max_pair(self, vocab: Dict[int, bytes]) -> Optional[Tuple[int, int]]:
    while self.max_freq > 0 and not self.buckets[self.max_freq]:                 self.max_freq -= 1        
    if self.max_freq == 0:        
        return
```





新代码
        

    def get_max_pair(self, vocab: Dict[int, bytes]) -> Optional[Tuple[int, int]]:
        # 使用 get() 方法，如果键不存在返回 None (或者空集合)，避免自动创建
        while self.max_freq > 0:
            bucket = self.buckets.get(self.max_freq)
            if bucket:  # 如果 bucket 存在且不为空
                break
            self.max_freq -= 1
            
        if self.max_freq == 0:
            return None

| **维度**     | **优化前 (Old Code)**                        | **优化后 (New Code)**               | **结果差异**                 |
| ------------ | -------------------------------------------- | ----------------------------------- | ---------------------------- |
| **查询方式** | `self.buckets[freq]` (方括号访问)            | `self.buckets.get(freq)` (方法访问) | **决定性差异**               |
| **字典行为** | **副作用读取**：不存在则创建空 Set           | **无副作用读取**：不存在则返回 None | 避免了数百万次无用对象创建   |
| **内存状态** | **只增不减**：保留所有历史空桶和查询过的空桶 | **动态清理**：桶空即删，查询不留痕  | 内存占用极低，Cache 命中率高 |
| **GC 压力**  | 极高 (需回收大量临时创建的 Set)              | 极低                                | CPU 专注于业务逻辑           |

Python 执行 `self.buckets.get(499)`。

`dict.get` 方法在底层 C 语言实现中，仅仅是计算哈希值，去表里看一眼。

发现没有，**直接返回 None**。

没有任何对象被创建，没有任何内存被分配，字典没有发生任何变化。

用 `[]` 硬拿：如果 key 不存在，Python 会当场报错（`KeyError`），程序崩溃。

用 `.get()` 试探：如果 key 不存在，Python 会温柔地返回 `None`（或者你指定的默认值），**程序继续平稳运行，绝不报错**。





### vocab存储的特征

训练阶段：

`vocab` 字典中存储的 Value 是 **`bytes` (字节序列)** 对象。

对应的就是int bytes

merge初始化的时候也是变为byte

推理阶段：

由于token_id转化的是整数，而merge是bytes,因此要查一次词表



### 推理过程的原地修改法

高校分词器里面merge的方式使用后面追加的方式，但是为了内存效率，我代码中进行原地修改。