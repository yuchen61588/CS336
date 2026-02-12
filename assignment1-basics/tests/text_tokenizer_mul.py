import time
from cs336_basics.tokenizer import Tokenizer # 假设你的类在 tokenizer.py

def test_parallel_correctness():
    # 1. 准备数据
    vocab = {0: b"a", 1: b"b"}
    merges = []
    # 搞一个简易 Tokenizer
    tokenizer = Tokenizer(100000,vocab, merges)
    
    # 造一点假数据 (1000 行)
    data = ["a" * 100, "b" * 100] * 500
    
    # 2. 跑普通 encode (基准)
    start = time.time()
    serial_ids = []
    for text in data:
        serial_ids.extend(tokenizer.encode(text))
    print(f"Serial time: {time.time() - start:.4f}s")
    
    # 3. 跑并行 encode
    start = time.time()
    # 强制使用 2 个进程测一下 IPC
    parallel_ids = tokenizer.encode_parallel(data, num_processes=2) 
    print(f"Parallel time: {time.time() - start:.4f}s")
    
    # 4. 核心校验：结果必须完全一致！
    assert serial_ids == parallel_ids, "Parallel result mismatch!"
    print("✅ Parallel implementation is correct!")

if __name__ == "__main__":
    test_parallel_correctness()