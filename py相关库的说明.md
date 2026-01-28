```
from __future__ import annotations
```

**作用：** 启用**延迟类型注解评估**

```
# 使用前：会报错，因为 Tree 还没定义完
class Tree:
    left: Tree  # ❌ NameError: name 'Tree' is not defined

# 使用后：正常工作
from __future__ import annotations
class Tree:
    left: Tree  # ✅ 注解被存为字符串 "Tree"，稍后解析
```

- 解决**前向引用**问题（类在自身定义中引用自己）
- 避免循环导入问题
- 提升模块导入性能





```
from collections import defaultdict
```

**作用：** 导入**带默认值的字典**

- 普通 `dict` 访问不存在的键会抛出 `KeyError`
- `defaultdict` 会自动为不存在的键生成默认值
- **key不存在时，自动调用 `X()` 生成默认值 .**

```
from collections import defaultdict

# 不是"value必须是int"
# 而是"当key不存在时，自动调用 int() 生成默认值"
d = defaultdict(int)

print(d["a"])      # 0  ← 自动调用 int() 得到 0
d["b"] += 5        # 0 + 5 = 5
d["x"] = "hello"      # ✅ 完全OK！存字符串
d["y"] = [1, 2, 3]    # ✅ 存列表
d["z"] = {"a": 1}     # ✅ 存字典
self._index[pair]  ← pair 不存在？
      ↓
自动调用 set() 创建空集合
      ↓
存入 _index[pair]
      ↓
返回这个空集合
      ↓
.add(word_id)  添加成功！
```





```
from typing import List, Dict, Tuple, Set, Optional
```

**作用：** 导入**类型提示工具**（用于代码静态类型检查）：**工具和IDE帮你检查错误**



| 类型              | 含义                     | 示例           |
| :---------------- | :----------------------- | :------------- |
| `List[int]`       | 整数列表                 | `[1, 2, 3]`    |
| `Dict[str, int]`  | 字符串键、整数值的字典   | `{"a": 1}`     |
| `Tuple[int, str]` | 固定长度元组（int, str） | `(1, "a")`     |
| `Set[str]`        | 字符串集合               | `{"a", "b"}`   |
| `Optional[int]`   | int 或 None              | `42` 或 `None` |

Python

复制

```python
def greet(name: Optional[str] = None) -> str:
    if name is None:
        return "Hello!"
    return f"Hello, {name}!"

def process(items: List[int]) -> Dict[str, int]:
    return {str(i): i for i in items}
```



**`__slots__` = 限制类只能有哪些属性，省内存、防 typo，会直接存储为C语言结构体形式。不然对比普通类，Python 会为每个对象维护一个 `__dict__`，同样耗内存。如果您使用字典（如 `{'tokens': [...], 'count': 10}`）来表示一个词对象，Python 需要为每个对象维护一个哈希表，内存开销很大。**

普通类有dict因为可以动态添加字段。每个对象都有一个dict，但是shot就是相当于一个没dict得对象



| 特性       | 列表 `list`    | 元组 `tuple`           |
| ---------- | -------------- | ---------------------- |
| **可变性** | ✅ 可变（能改） | ❌ 不可变（不能改）     |
| **语法**   | `[1, 2, 3]`    | `(1, 2, 3)`            |
| **性能**   | 稍慢           | 更快                   |
| **内存**   | 更大           | 更小                   |
| **用途**   | 存储可变数据   | 存储固定数据、当字典键 |
| **安全性** | 低（易被修改） | 高（数据保护）         |

| 类型提示          | 实际初始化        | 用途       | 示例               |
| ----------------- | ----------------- | ---------- | ------------------ |
| `List[int]`       | `[]` 或 `list()`  | 可变序列   | `[1, 2, 3]`        |
| `Tuple[int, str]` | `()` 或 `tuple()` | 固定序列   | `(1, "a")`         |
| `Set[int]`        | `set()`           | 去重集合   | `{1, 2, 3}`        |
| `FrozenSet[int]`  | `frozenset()`     | 不可变集合 | `frozenset([1,2])` |
| `Dict[str, int]`  | `{}` 或 `dict()`  | 键值映射   | `{"a": 1}`         |

| 类型提示                | 实际初始化      | 含义               | 示例            |
| ----------------------- | --------------- | ------------------ | --------------- |
| `Optional[int]`         | `int` 或 `None` | 可能为None         | `42` 或 `None`  |
| `Union[int, str]`       | `int` 或 `str`  | 多种类型之一       | `1` 或 `"a"`    |
| `List[Union[int, str]]` | `[]`            | 混合列表           | `[1, "a", 2]`   |
| `Tuple[int, ...]`       | `()`            | 变长元组（同类型） | `(1, 2, 3)`     |
| `Dict[str, List[int]]`  | `{}`            | 嵌套结构           | `{"a": [1, 2]}` |

| 类型                | 初始化              | 特殊行为        | 示例                      |
| ------------------- | ------------------- | --------------- | ------------------------- |
| `defaultdict(int)`  | `defaultdict(int)`  | 默认值为0       | `d[k] += 1`               |
| `defaultdict(list)` | `defaultdict(list)` | 默认值为`[]`    | `d[k].append(x)`          |
| `defaultdict(set)`  | `defaultdict(set)`  | 默认值为`set()` | `d[k].add(x)`             |
| `defaultdict(dict)` | `defaultdict(dict)` | 默认值为`{}`    | `d[k][sub] = v`           |
| `Counter`           | `Counter()`         | 计数专用        | `c.update([1,1,2])`       |
| `OrderedDict`       | `OrderedDict()`     | 保持插入顺序    | Python 3.7+ dict 默认有序 |
| `deque`             | `deque()`           | 双端队列        | 快速 `popleft()`          |

**`typing` 中的类型 = 类型提示（注释），不是真正的类！**

**类型提示的参数 = "说明书"，不是"门禁系统"**

- 运行时：**完全不限制，不会报错**
- 静态检查（mypy）：**会报错**
- IDE：**会警告/标红**

```
┌─────────────────────────────────────────┐
│  编写代码 ──→ 保存文件 ──→ 运行程序        │
│     ↑           ↑           ↑           │
│     │           │           │           │
│   IDE提示    mypy检查      Python执行    │
│  (实时)     (手动/自动)    (真正运行)    │
│     ↑           ↑           │           │
│   静态检查期   静态检查期    运行时       │
│  （不执行代码）（不执行代码）（执行代码）  │
└─────────────────────────────────────────┘
```



remove与discard

| 方法             | 元素存在   | 元素不存在             |
| ---------------- | ---------- | ---------------------- |
| `set.remove(x)`  | ✅ 正常删除 | ❌ **报错 `KeyError`**  |
| `set.discard(x)` | ✅ 正常删除 | ✅ **静默忽略，不报错** |



**`tqdm` = 给你的循环加个进度条，直观显示处理进度**





**进程池**

```
with multiprocessing.Pool(num_procs) as pool:
#创建进程池 包含 num_procs 个独立进程  with 语句确保用完自动关闭
        # 使用 imap_unordered 减少等待
        for local_freqs in pool.imap_unordered(_pre_tokenize_worker, worker_args):
        ## imap_unordered: 无序迭代返回结果
            for k, v in local_freqs.items():
                global_freqs[k] += v  #全局计数
```

```
主进程                    多个 Worker 进程
   │                            │
   │  分发任务 (worker_args)     │
   ├──────────────────────────→ │
   │  ┌─────────────────────┐   │
   │  │ Worker1: 处理块1    │   │
   │  │ Worker2: 处理块2    │   │
   │  │ Worker3: 处理块3    │   │
   │  │ ...                 │   │
   │  └─────────────────────┘   │
   │ ←──────────────────────────┤
   │  结果逐个返回 (imap_unordered)│
   │                            │
   合并到 global_freqs           │
```

| 方法                   | 特点                       | 适用场景                   |
| ---------------------- | -------------------------- | -------------------------- |
| `map()`                | 阻塞，等全部完成才返回列表 | 小数据，要全部结果         |
| `imap()`               | 按提交顺序逐个yield        | 需要顺序一致               |
| **`imap_unordered()`** | **谁先完成谁先返回**       | **最快拿到结果，减少等待** |

上面说的完成就是分开的进程池。

```
准备阶段：
worker_args = [
    (file, 0, 2GB, pattern, specials),   # Worker1
    (file, 2GB, 4GB, pattern, specials), # Worker2  
    (file, 4GB, 6GB, pattern, specials), # Worker3
    ...
]

            ↓ pool.imap_unordered 分发

Worker1 处理中 [========>    ]  慢
Worker2 处理中 [====>        ]  更慢  
Worker3 处理完 [============>]  最快！→ 立即返回

            ↓ 主进程收到 local_freqs3

        for k, v in local_freqs3.items():
            global_freqs[k] += v   # 立即合并，不等待其他worker

            ↓ 继续等待...

Worker1 完成 → 返回 → 合并
Worker2 完成 → 返回 → 合并

            ↓ 全部完成

        global_freqs = {b"the": 1000, b"cat": 500, ...}
```

**线程安全说明：**

这里的work是完全隔离的，没有共享变量，。合并在主进程进行的，因此不存在锁的问题，他们都是合并完将结果发给主进程。for循环是**单线程串行**的，一次只处理一个 Worker 的结果！

不安全写法：

### ❌ 错误写法 1：Worker 直接改全局（进程版）

```python
from multiprocessing import Pool
from collections import defaultdict

# 全局变量（危险！多个进程会看到同一变量吗？）
global_freqs = defaultdict(int)

def bad_worker(text_chunk):
    # 每个进程都试图写这个 global_freqs
    for word in text_chunk.split():
        global_freqs[word] += 1  # ← 多个进程同时写！竞争！
    
    # 结果：数据丢失，统计错误
    # 进程1读global_freqs["the"]=100
    # 进程2同时读global_freqs["the"]=100  
    # 进程1写101，进程2写101 → 应该是102，实际101！

# 启动多个进程
with Pool(4) as pool:
    pool.map(bad_worker, chunks)  # 结果错误！
```

**问题**：多进程**不是共享内存**（除非用 `Manager`），但即使共享，同时写也会竞争！

### ❌ 错误写法 2：多线程 + 锁（线程版）

```python
import threading
from collections import defaultdict

global_freqs = defaultdict(int)
lock = threading.Lock()  # 需要锁来保护

def thread_worker(text_chunk):
    local = {}
    for word in text_chunk.split():
        local[word] = local.get(word, 0) + 1
    
    # 合并时需要加锁！
    with lock:  # ← 拿到锁才能改全局变量
        for k, v in local.items():
            global_freqs[k] += v

# 启动多个线程
threads = []
for chunk in chunks:
    t = threading.Thread(target=thread_worker, args=(chunk,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```
