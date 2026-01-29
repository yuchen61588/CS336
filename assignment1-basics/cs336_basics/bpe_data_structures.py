from __future__ import annotations

from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional

class WordObject:
    """
    表示语料库中的一个唯一单词 (WordObject)。
    维护 Token 序列和词频。
    """
    __slots__ = ('tokens', 'count', 'id')

    def __init__(self,tokens:List[int],count:int,word_id:int):
        self.tokens = tokens
        self.count = count
        self.id = word_id

    def apply_merge(self,
                    pair: Tuple[int, int],
                    new_token_id:int)\
            -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]:
        """
        核心逻辑：在单词内部执行合并操作。
        参数:
             pair (Tuple[int, int]): 待合并的 Token 对，例如 (104, 101) 代表 ('h', 'e')。
             new_token_id (int): 合并后生成的新 Token ID。
        返回:
             changes_removed (Dict): 记录因合并而**消失/断裂**的相邻关系及其频率变化量。
                                    Key是断裂的Pair，Value是减少的次数(即单词的count)。
             changes_added (Dict): 记录因合并而**新产生**的相邻关系及其频率变化量。
                                     Key是新生成的Pair，Value是增加的次数。
             最里面得作为元组可哈希
             这个数据结构就是add和remove得元组集合，它明确告知调用者，这个函数固定返回两样东西：第一样是“移除的统计”，第二样是“新增的统计”。
        算法细节:
                采用从左到右的贪婪匹配策略。一旦匹配成功，跳过这两个 Token，
                记录左右邻居关系的断裂。
        """
        # new_tokens = []
        # i = 0
        # n =len(self.tokens)
        # change_removed = defaultdict(int)
        # change_added = defaultdict(int)

        # while i<n:
        #     # 检查：当前位置 i 和 i+1 是否正好构成了我们要合并的 pair
        #     if i<n-1 and self.tokens[i] ==pair[0]and self.tokens[i+1]==pair[1]:
        #         # 合并 记录左侧关系（如果有）(prev, pair[0])组合删去
        #         if i>0:
        #             prev_token = self.tokens[i-1]
        #             change_removed[(prev_token,self.tokens[i])] += self.count
        #         # 合并右侧 (pair[1], next)(如果下面两个是下一个将要合并得pair，那么这次循环就不能加入了，让下一次循环取)
        #         if i<n-2:
        #             # 检查下一个位置是否也是合并点
        #             is_next_merge = (i + 2 < n - 1) and \
        #                             (self.tokens[i + 2] == pair[0] and self.tokens[i + 3] == pair[1])
        #             if not is_next_merge:
        #                 next_token = self.tokens[i+2]
        #                 change_removed[(self.tokens[i+1],next_token)] += self.count
        #         # 处理添加：
        #         # 将新id加入i性能关系，并且如果new_tokens不为空，要把新id与前面得id构成新连接
        #         if new_tokens:
        #             change_added[(new_tokens[-1],new_token_id)]+=self.count
        #         new_tokens.append(new_token_id)
        #         # 跳过合并两个token
        #         i+=2
        #     else:
        #         current_token = self.tokens[i]
        #         # 处理添加 (Added) 如果前一个 token 是刚刚生成的"新 Token"，那么当前这个旧 Token 与它构成了新关系
        #         if new_tokens:
        #             last_tokens = new_tokens[-1]
        #             if last_tokens == new_token_id:
        #                 change_added[(last_tokens,current_token)] +=self.count
        #         new_tokens.append(current_token)
        #         i+=1

        # self.tokens = new_tokens
        # return change_added,change_removed
        #上面的实现直接导致内存爆掉，为了减少内存消耗，我们采用列表方法
        
        if len(self.tokens) < 2:
            return {}, {}
        tokens = self.tokens

        freq_deltas = defaultdict(int) # 存储频率变化量: key=pair, value=delta

        generated_tokens = []
        i = 0
        n = len(self.tokens)
        target_left, target_right = pair #target的左边界与右边界
        # 记录"上一个片段"在原序列中的右边界
        # 初始为 None，因为第一个 token 左边没有 token
        last_seg_right = None
        while i < n:
            # 检查是否匹配目标 pair
            is_target = (i < n - 1 and tokens[i] == target_left and tokens[i+1] == target_right)
            if is_target:
                # --- 发生合并 ---
                token_to_append = new_token_id
                # 当前片段在原序列中的左边界是 pair 的左边
                curr_seg_left = target_left
                # 当前片段在原序列中的右边界是 pair 的右边
                curr_seg_right = target_right
                # 目标 pair 本身被消耗掉了，频率 -1
                freq_deltas[pair] -= 1
                step = 2
            else:
                # --- 不合并 ---
                current_token = tokens[i]
                token_to_append = current_token
                # 原序列的左右边界都是它自己
                curr_seg_left = current_token
                curr_seg_right = current_token
                step = 1
            # 如果生成了新代码
            if generated_tokens:
                # 只有当接缝处发生变化时才更新频率
                # 新连接 vs 旧连接
                # 新连接: (last_token, token_to_append)
                # 旧连接: (last_seg_right, curr_seg_left)
                # 变化条件：(上一个token变了) 或 (当前token变了)
                # 也就是 (last_token != last_seg_right) or (token_to_append != curr_seg_left)
                # 注意：这里对比的是 token ID
                last_token = generated_tokens[-1]
                if (last_token != last_seg_right) or (token_to_append != curr_seg_left):
                    # 新产生的连接
                    added_pair = (last_token, token_to_append)
                    freq_deltas[added_pair] += 1
                    
                    # 消失的旧连接
                    removed_pair = (last_seg_right, curr_seg_left)
                    freq_deltas[removed_pair] -= 1
            
            generated_tokens.append(token_to_append)
            
            # 更新 last_seg_right 供下一次循环使用
            last_seg_right = curr_seg_right
            i += step
        # 更新内部 token 列表
        word_count = self.count
        self.tokens = generated_tokens
        # 将 delta 分离为 added 和 removed 字典
        removed_counts = {}
        added_counts = {}
        for p, delta in freq_deltas.items():
            if delta < 0:
                removed_counts[p] = -delta*word_count # 存正数，表示移除的次数
            elif delta > 0:
                added_counts[p] = delta*word_count

        return  added_counts,removed_counts

                











class InvertedIndex:
    """
        倒排索引 (Inverted Index)
        作用：
            解决"谁包含了这个 Pair？"的问题。
            通过 Pair (u, g) 快速找到所有包含 "ug" 的 WordObject 的 ID。
            避免了每次合并都要扫描整个词表的低效操作。
        策略：
            Lazy Add (只增不减)：当 Pair 被合并或者断裂时，我们只向索引中添加新关系，
            而不去执行昂贵的 set.remove 操作。旧关系虽然还在索引里，但在处理时
            我们会发现 WordObject 的 tokens 里已经没有这个 Pair 了，直接跳过即可。
        """
    def __init__(self):
        # 核心数据结构：字典
        # Key: Token Pair (int, int)
        # Value: 包含该 Pair 的 Word ID 集合 (Set[int])
        self.index : Dict[Tuple[int, int], Set[int]] = defaultdict(set) # 类星提示
    def add_occurrence(self,pair:Tuple[int,int],word_id:int)->None:
        # 添加新的id
        self.index[pair].add(word_id)
    def get_word_ids(self,pair:Tuple[int,int])->Set[int]:
           # 查询：获取所有可能包含该 pair 的 word_id 集合。包含lazy Delete
        return self.index.get(pair,set()) # 没有就返回默认值set()

    def clear_pair(self, pair: Tuple[int, int]) -> None:
        """
        当一个 Pair 被选为合并项时，它在语料中将彻底消失。
        直接删除整个索引条目以释放内存。
        """
        if pair in self.index:
            del self.index[pair]


class FrequencyBuckets:
    """
        频率桶 (Frequency Buckets):包括频率表与频率桶
        作用：
            解决"当前频率最高的 Pair 是谁？"的问题。
            能在 O(1) 的时间复杂度内获取最大频率的 Pair，无需对频率表进行排序。
    """
    def __init__(self):
        # 频率表 用于 O(1) 查询任意 Pair 的当前频率
        # key pair, value count
        self.stats:Dict[Tuple[int,int],int] = defaultdict(int)
        # 频率桶 将具有相同频率的 Pair 放在一起，找到最高频率的表
        # key count, value 拥有该频率的所有 Pair 的集合 (Set[Pair])
        self.buckets:Dict[int,Set[Tuple[int,int]]] = defaultdict(set)
        # 最大值真
        self.max_freq:int = 0

    def update(self, pair: Tuple[int, int], delta: int) -> None:
        """
        更新某个 Pair 的频率。 包括更新频率表，换桶（删除）
        参数:
            pair: 需要更新的字符对
            delta: 变化量（+N 表示新增引用，-N 表示移除引用）
        """
        if delta==0:
            return
        old_freq = self.stats.get(pair,0)
        # 如果原来频率桶有这个，要先再频率桶移除
        if old_freq>0:
            if old_freq in self.buckets:
                # 使用 discard 安全移除，防止因数据不同步导致的 KeyError,
                self.buckets[old_freq].discard(pair)
                #如果移除后这个桶空了，且为max_freq。不在这里更新，懒惰更新，在get_max_pair进行更新降级，这样可以分开智能
        new_freq = old_freq+delta
        if new_freq>0:
            self.buckets[new_freq].add(pair) #字典
            self.stats[pair] = new_freq
            # 如果新频率打破纪录，立即更新，因为往下兼容不能更新
            if new_freq > self.max_freq:
                self.max_freq = new_freq
        else:
            # 如果为0或者负数，从统计表中彻底删除
            if pair in self.stats:
                del self.stats[pair]

    def remove_entry(self,pair:Tuple[int,int])->None:
        """
        彻底移除一个 Pair。
        通常在某个 Pair 被选为最佳合并项并被合并后调用，因为它已经变成一个新 Token 了，
        旧的 Pair (A, B) 不应该再被统计。
        """
        if pair not in self.stats:
            return
        freq = self.stats[pair]
        del self.stats[pair]
        if freq in self.buckets:
            self.buckets[freq].discard(pair) #不删除这个key域value

    def get_max_pair(self,vocab: Dict[int, bytes])->Optional[Tuple[int,int]]:
        """
        O(1) 获取当前频率最高的 Pair。
        """
        while self.max_freq > 0 and not self.buckets[self.max_freq]: #会有空桶，但是消耗不大
            self.max_freq -= 1
        if self.max_freq==0:
            return
        current_bucket = self.buckets[self.max_freq]
        return max(current_bucket, key=lambda p: (vocab[p[0]], vocab[p[1]]))