import collections
import itertools

"""
📘 Week 7 实战示例：Tree of Thoughts (ToT) 核心算法实现演示

【学习目标】
理解 Tree of Thoughts 是如何将线性推理扩展为"树状结构"，
并通过 Generator(生成器)、Evaluator(评估器) 和 Search Algorithm(搜索算法)
协同工作，实现能够"评估前景、剪枝和回溯"的高级推理系统。

本代码对应讲义 Week 7 的 Part 4 (Tree of Thoughts——从链到树) 章节。

【核心概念对应 - Part 4: ToT】
任务场景：Game of 24 (给定 4 个数字，通过加减乘除得到 24)

1. 推理状态建模 -> `Node` 类
   - 将"当前剩下的数字和历史步骤"封装为一个节点
2. Generator (生成候选) -> `generate_next_steps` 函数
   - 对应讲义 §4.2 Step 2：在每个节点探索多个下一步候选
3. Evaluator (评估前景) -> `evaluate_state` 函数
   - 对应讲义 §4.2 Step 3：评估 `sure` / `maybe` / `impossible`
4. 搜索算法 - BFS -> `tot_bfs` 函数
   - 广度优先，逐层推进，依靠 `b` (beam_size) 截断 (对应讲义 §4.4)
5. 搜索算法 - DFS -> `tot_dfs` 函数
   - 深度优先，一扎到底，错误则回溯 (对应讲义 §4.4)

【与 CoT 的对比】
- CoT：盲目向前走，如果第一步 13-10=3 走错了，后面全错
- ToT：如果发现当前分支不可能凑出 24 (impossible)，会立刻剪枝并回溯
"""

# ==============================================================================
# 基础结构定义
# ==============================================================================

class Node:
    """定义搜索树中的一个状态节点"""
    def __init__(self, current_numbers: list, history: list = None):
        # 当前剩余可用的数字，例如 [4, 4, 10]
        self.current_numbers = sorted(current_numbers)
        # 用易于阅读的字符串记录走到这一步的操作历史，例如 ["13 - 9 = 4"]
        self.history = history or []
        # 当前状态的评估得分 ('sure', 'maybe', 'impossible')
        self.eval_score = None 

    def __repr__(self):
        return f"Node(nums={self.current_numbers}, score={self.eval_score})"
        
    def __eq__(self, other):
        # 用于防止重复状态
        return tuple(self.current_numbers) == tuple(other.current_numbers)
        
    def __hash__(self):
        return hash(tuple(self.current_numbers))


# ==============================================================================
# 模块 1: Generator (生成器) —— "列举所有可能的下一步"
# ==============================================================================

def generate_next_steps(node: Node) -> list:
    """
    ToT 的生成器模块
    
    🧮 讲义对应：§4.2 Step 2 (Thought Generation)
    在当前状态下，"生成多个不同的候选下一步"，这就是树的"分叉点"。
    
    真实场景中，这里会调用 LLM：Prompt="给定数字 [x,y,z]，请提出所有可能的第一步运算..."
    这里我们写死算法逻辑直接枚举，以确保演示的确定性。
    """
    nums = node.current_numbers
    if len(nums) <= 1:
        return []

    next_nodes = set() # 用 set 去重
    
    # 遍历当前可用数字的所有"两两组合"
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            a, b = nums[i], nums[j]
            # 剩下的数字
            remains = [nums[k] for k in range(len(nums)) if k != i and k != j]
            
            # 列举所有合法运算：+, -, *, / (禁止除0，除法只允许整除以简化问题)
            operations = [
                (a + b, f"{a} + {b} = {a+b}"),
                (a * b, f"{a} * {b} = {a*b}"),
                (abs(a - b), f"{max(a,b)} - {min(a,b)} = {abs(a-b)}"),
            ]
            if b != 0 and a % b == 0:
                operations.append((a // b, f"{a} / {b} = {a//b}"))
            elif a != 0 and b % a == 0:
                operations.append((b // a, f"{b} / {a} = {b//a}"))
                
            for new_val, op_str in operations:
                new_nums = remains + [new_val]
                new_history = node.history + [op_str]
                next_nodes.add(Node(new_nums, new_history))
                
    return list(next_nodes)


# ==============================================================================
# 模块 2: Evaluator (评估器) —— "判断进展的潜力"
# ==============================================================================

_EVAL_CACHE = {} # 避免重复打印相同状态的评估过程

def evaluate_state(node: Node, verbose: bool = False) -> str:
    """
    ToT 的评估器模块
    
    🧮 讲义对应：§4.2 Step 3 (State Evaluation) 和 评估器的核心机制
    对每个候选分支，判断这条路是否有前景。
    - sure: 确定能得到 24 (通常是只剩一个数字且是24，或者最后两个数能算出24)
    - maybe: 还有 3-4 个数字，可能可以
    - impossible: 明显走不通了 (比如剩下负数或分数，或者最后计算得不出24)
    
    真实场景中，这里是让同一个 LLM 做"裁判"。
    这里用启发式算法模拟 LLM 裁判的直觉判断。
    """
    nums = node.current_numbers
    key = tuple(nums)
    if key in _EVAL_CACHE:
        node.eval_score = _EVAL_CACHE[key]
        return _EVAL_CACHE[key]

    score = "maybe"
    
    if len(nums) == 1:
        score = "sure" if nums[0] == 24 else "impossible"
    elif len(nums) == 2:
        # 剩下两个数，如果能通过一步算出 24，就是 sure，否则 impossible
        a, b = nums[0], nums[1]
        can_24 = (a+b==24) or (a*b==24) or (abs(a-b)==24) or (b!=0 and a/b==24) or (a!=0 and b/a==24)
        score = "sure" if can_24 else "impossible"
    else:
        # 当还有 3-4 个数时，启发式判断一些"死路"
        # 例如：所有数字都非常大，或者所有数字都是 0
        if sum(nums) > 100: 
            score = "impossible"
        else:
            score = "maybe"
            
    _EVAL_CACHE[key] = score
    node.eval_score = score
    
    if verbose:
        print(f"    [评估] 评估状态 nums={nums} -> 裁判给分: {score.upper()}")
        
    return score


# ==============================================================================
# 模块 3: 搜索算法
# ==============================================================================

def tot_bfs(initial_numbers: list, beam_size: int = 5, verbose: bool = False) -> Node:
    """
    ToT 的广度优先搜索 (BFS)
    
    🧮 讲义对应：§4.4 ToT 的两种搜索算法实现 - BFS
    核心策略：每一层生成所有分支、全部评估后，只保留最优的 b 条路径，一起推进到下一层。
    
    Args:
        initial_numbers: 初始的 4 个数字
        beam_size: ToT 论文中的超参数 b，每层最多保留几个节点
        verbose: 是否打印日志
    """
    # 初始化
    root = Node(initial_numbers)
    # 起点必定是 maybe (除非直接就是24)
    evaluate_state(root) 
    candidates = [root]
    
    # 限制搜索深度，Game of 24 固定 3 步
    MAX_STEPS = 3 
    
    if verbose:
        print(f"\n🚀 开始 BFS(束宽={beam_size}) 搜索, 初始状态: {initial_numbers}")

    for step in range(MAX_STEPS):
        if verbose:
            print(f"\n  ▶ 第 {step+1} 步")
            
        next_layer_candidates = []
        
        # 1. Expand: 展开当前层所有保留节点的下一步
        for candidate in candidates:
            children = generate_next_steps(candidate)
            next_layer_candidates.extend(children)
            
        # 2. Evaluate: 对新展开的所有节点打分
        # 先去重，节省评估算力 (真实场景中调用 LLM 很贵)
        unique_candidates = list(set(next_layer_candidates))
        
        sure_nodes = []
        maybe_nodes = []
        
        for child in unique_candidates:
            score = evaluate_state(child, verbose)
            
            # 3. Prune: 剪枝 (如果是 impossible，直接抛弃)
            if score == 'sure':
                sure_nodes.append(child)
            elif score == 'maybe':
                maybe_nodes.append(child)
                
        # 4. Filter: 按照 beam_size 截断 (此处 sure 优先于 maybe)
        # 排序：先 sure 后 maybe。同一级别这里简单随机或原序，实际可让 LLM 给数值分
        all_valid = sure_nodes + maybe_nodes
        candidates = all_valid[:beam_size]
        
        if verbose:
            print(f"    [截断] 从 {len(unique_candidates)} 个候选剪枝，保留 {len(candidates)} 个")
            
        # 提前终止：如果有节点变成了 1 个数字且是 sure，提前结束
        for c in candidates:
            if len(c.current_numbers) == 1 and c.eval_score == 'sure':
                if verbose:
                    print(f"  🎉 提前找到最终解！")
                return c
                
    # 循环结束（经过 3 步），搜寻是否有只剩 1 个数字且等于 24 的节点
    for c in candidates:
        if len(c.current_numbers) == 1 and 24 in c.current_numbers:
            return c
            
    return None


def tot_dfs(node: Node, max_depth: int = 3, current_depth: int = 0, verbose: bool = False) -> Node:
    """
    ToT 的深度优先搜索 (DFS + 回溯)
    
    🧮 讲义对应：§4.4 DFS (深度优先搜索 + 回溯)
    核心策略：选一条路一头扎到底。如果到底发现不对，就回溯到岔路口。
    """
    indent = "  " * current_depth
    if verbose and current_depth == 0:
        print(f"\n🚀 开始 DFS 搜索, 初始状态: {node.current_numbers}")
        
    # 到达目标或最大层数
    if current_depth == max_depth or len(node.current_numbers) == 1:
        if len(node.current_numbers) == 1 and node.current_numbers[0] == 24:
            return node
        return None

    # 1. Expand
    children = generate_next_steps(node)
    
    for child in children:
        # 2. Evaluate
        score = evaluate_state(child)
        if verbose:
            print(f"{indent}▶ 尝试分支: {child.history[-1]:<15} -> 剩余: {child.current_numbers} [{score.upper()}]")
            
        # 3. Prune (剪枝)
        if score == "impossible":
            if verbose:
                print(f"{indent}  ✂️ 剪枝该分支")
            continue
            
        # 4. 递归深入
        result = tot_dfs(child, max_depth, current_depth + 1, verbose)
        if result is not None:
            return result # 找到了！一路抛上去
            
        # 如果走到了死胡同(result 为 None)，代码会自动 continue 尝试兄弟节点。
        # 这里发生了隐式的【回溯】(Backtracking)！
        if verbose:
            print(f"{indent}  🔙 此路不通，回溯到上一层")

    return None

# ==============================================================================
# 对比测试：只走一条路的普通 CoT (无搜索无评估)
# ==============================================================================

def simulate_standard_cot(initial_numbers: list) -> Node:
    """
    模拟普通 CoT：线性生成，没有评估机制，随机选择一步前进，无法回溯。
    用来作为 ToT 的 baseline 对比。
    """
    current_node = Node(initial_numbers)
    for _ in range(3):
        children = generate_next_steps(current_node)
        if not children:
            break
        # 盲目选择第一条合法生成的路径
        current_node = children[0] 
        
    if len(current_node.current_numbers) == 1 and current_node.current_numbers[0] == 24:
        return current_node
    return None

# ==============================================================================
# 演示主入口
# ==============================================================================

def main():
    print("=" * 70)
    print("🌳 Week 7 - Part 4: Tree of Thoughts (ToT) Game of 24 演示")
    print("=" * 70)
    print("""
    核心思想 (讲义 §4.1):
    "ToT = 将推理过程建模为一棵树 + 在树上进行有策略的搜索"
    (生成器扩展思路，评估器判断前景，搜索算法协调剪枝和回溯)
    """)
    
    # 一个经过挑选的 24 点数字集合
    puzzle = [4, 9, 10, 13]
    
    print(f"\n【题目】给出数字 {puzzle}，运用加减乘除计算出 24。")
    print(f"讲义中的经典路径: 13-9=4 -> [4,4,10] -> 10-4=6 -> [6,4] -> 6*4=24")
    
    # ------------------------------------------------------------------
    # 实验 1: ToT BFS 搜索
    # ------------------------------------------------------------------
    # 清空缓存以便观测日志
    _EVAL_CACHE.clear()
    solution_bfs = tot_bfs(puzzle, beam_size=3, verbose=True)
    
    print("\n✅ [BFS 结果]")
    if solution_bfs:
        print("  成功找到解法！路径：")
        for i, step in enumerate(solution_bfs.history, 1):
            print(f"  Step {i}: {step}")
    else:
        print("  ❌ 未找到解法。")

    # ------------------------------------------------------------------
    # 实验 2: ToT DFS 搜索（注意观察回溯）
    # ------------------------------------------------------------------
    # 准备一个能明显展示回溯的集合：[1, 2, 11, 12]
    # (如果按加法优先，它会尝试 1+2=3, 然后 3+11=14 等等，均会碰壁然后回溯，
    # 直到它找到 11+2=13, 12-1=11, 13+11=24 的正确路径)
    puzzle2 = [1, 2, 11, 12] 
    
    print(f"\n\n【换题演示回溯】给出数字 {puzzle2}")
    _EVAL_CACHE.clear()
    solution_dfs = tot_dfs(Node(puzzle2), verbose=True)
    
    print("\n✅ [DFS 结果]")
    if solution_dfs:
        print("  成功找到解法！路径：")
        for i, step in enumerate(solution_dfs.history, 1):
            print(f"  Step {i}: {step}")

    # ------------------------------------------------------------------
    # 总结说明
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📝 ToT 核心要点总结")
    print("=" * 70)
    print("""
1. 【解构问题】
   ToT 不像 CoT 那样一次性憋出最终答案，而是把问题切分为“中间步骤”（如每层运算一次）。

2. 【Evaluator 的重要性 (自我评估)】
   代码中的 `evaluate_state` 模拟了 LLM 的裁判功能。如果在剩余 3 个数字
   时 LLM 能敏锐地判断出这三个数"绝对凑不出 24" (impossible)，算法就能
   立刻剪枝，省去极其庞大的后续无效搜索。

3. 【BFS vs DFS】
   - BFS (广度优先)：每层考察多个分支，用 b 截断。稳当，不容易漏掉可能解，
     但算力开销大，对应讲义中的 `b=5` 的高成功率。(适用于答案空间相对集中的任务)
   - DFS (深度优先)：只要碰对一条路瞬间结束，如果撞南墙会自动退回上一个
     岔路（回溯）。(适用于可以利用直觉快速"一发入魂"的任务)

4. 【本质】
   Test-Time Scaling 的精髓所在 —— 推理时不怕花时间分叉和评估。
   这比用 Self-Consistency 盲目对整条链采样 100 次有效得多。
""")

if __name__ == "__main__":
    main()
