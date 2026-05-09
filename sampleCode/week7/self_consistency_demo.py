import random
from collections import Counter

"""
📘 Week 7 实战示例：Self-Consistency (多路采样 + 多数投票) 核心算法实现演示

【学习目标】
理解 Self-Consistency 如何通过"多条独立 CoT 推理链 + 多数投票"来降低
单条推理链的随机错误风险，以及 Temperature 参数在其中的关键作用。

本代码对应讲义 Week 7 的 Part 3 (Self-Consistency——多路采样的力量) 章节。

【核心概念对应 - Part 3: Self-Consistency】
1. CoT 推理链生成   -> `mock_cot_reasoning` 函数
   - 模拟 LLM 用不同推理路径解答同一问题
   - 体现「高 Temperature → 路径多样化」的关键设计
2. 答案提取         -> `extract_answer` 函数
   - 从推理链末尾提取最终数值答案
3. 多数投票         -> `majority_vote` 函数
   - 选择出现频率最高的答案作为最终输出
4. Self-Consistency 主流程 -> `self_consistency` 函数
   - 与讲义 §3.3 的伪代码一一对应

【与普通 CoT 的核心差异】
| 维度       | CoT (单路)           | Self-Consistency        |
|------------|----------------------|-------------------------|
| 采样次数   | 1 次                 | n 次 (如 5~40)          |
| Temperature| 通常较低 (0~0.3)     | 必须较高 (0.7~1.0)     |
| 投票机制   | 无                   | 多数投票 (Majority Vote)|
| 错误容忍   | 一步错则全错         | 多数正确即可            |

【关键洞察 - §3.2: 为什么有效？】
"错误是随机的，正确是一致的。"
- 不同采样路径犯的错往往是不同的（随机错误）
- 但正确的推理路径会趋同于同一个答案（一致性）

【注意】
这是一个"逻辑演示版"代码，用模拟函数代替真实 LLM 调用。
核心目标是让你理解 Self-Consistency 的算法逻辑，而非工程实现。
"""


# ==============================================================================
# Part A: 模拟 CoT 推理链的"LLM"
# ==============================================================================

# ---- 问题库 ----
# 每道题包含：题目文本、正确答案、若干条可能的推理路径（正确和错误的混合）
QUESTION_BANK = [
    {
        "question": "小明有 5 个苹果，小红给了他 3 个，他又吃掉了 2 个。请问小明现在有几个苹果？",
        "correct_answer": 6,
        "reasoning_paths": [
            # ---- 正确路径 ----
            {
                "steps": "让我一步一步来思考：\n"
                         "1. 小明最初有 5 个苹果\n"
                         "2. 小红给了他 3 个，所以 5 + 3 = 8 个\n"
                         "3. 他吃掉了 2 个，所以 8 - 2 = 6 个\n"
                         "答案是 6 个。",
                "answer": 6,
                "weight": 0.55,  # 被采样到的相对概率（正确路径占大头）
            },
            {
                "steps": "先算增加的：5 + 3 = 8\n"
                         "再算减少的：8 - 2 = 6\n"
                         "所以答案是 6。",
                "answer": 6,
                "weight": 0.20,
            },
            # ---- 错误路径 ----
            {
                "steps": "5 + 3 = 8，然后 8 - 2... 嗯我算一下... 答案是 5。",
                "answer": 5,
                "weight": 0.10,  # 中间计算出错
            },
            {
                "steps": "有 5 个苹果，给了 3 个又吃了 2 个，\n"
                         "一共减少了 3 + 2 = 5 个，5 - 5 = 0。\n"
                         "答案是 0 个。",
                "answer": 0,
                "weight": 0.10,  # 理解错误：把"给了他"理解成"给出去"
            },
            {
                "steps": "5 + 3 - 2 = 7... 不对，等等，\n"
                         "5 + 3 = 9? 9 - 2 = 7。答案是 7。",
                "answer": 7,
                "weight": 0.05,  # 加法算错
            },
        ],
    },
    {
        "question": "一个水池有 3 个进水管和 2 个出水管。每个进水管每小时进 10 升水，"
                    "每个出水管每小时出 8 升水。问 5 小时后水池增加了多少升水？",
        "correct_answer": 70,
        "reasoning_paths": [
            {
                "steps": "进水速率：3 × 10 = 30 升/小时\n"
                         "出水速率：2 × 8 = 16 升/小时\n"
                         "净进水速率：30 - 16 = 14 升/小时\n"
                         "5 小时后：14 × 5 = 70 升\n"
                         "答案是 70 升。",
                "answer": 70,
                "weight": 0.50,
            },
            {
                "steps": "每小时净增 = 3*10 - 2*8 = 30 - 16 = 14\n"
                         "5 小时 → 14 * 5 = 70\n"
                         "答案是 70。",
                "answer": 70,
                "weight": 0.20,
            },
            {
                "steps": "进水总量 = 3 * 10 * 5 = 150\n"
                         "出水总量 = 2 * 8 * 5 = 80\n"
                         "差值 = 150 - 80 = 70\n"
                         "答案是 70。",
                "answer": 70,
                "weight": 0.10,
            },
            {
                "steps": "3 个管进水 10 升 = 30，2 个管出水 8 升 = 16\n"
                         "30 - 16 = 14... 14 * 5 = 60？不对... 答案是 60。",
                "answer": 60,
                "weight": 0.10,  # 乘法算错
            },
            {
                "steps": "进水 30 升/小时，出水 16 升/小时\n"
                         "净增 = 30 + 16 = 46... 46 * 5 = 230\n"
                         "答案是 230。",
                "answer": 230,
                "weight": 0.05,  # 减号写成加号
            },
            {
                "steps": "3 * 10 = 30，2 * 8 = 16\n"
                         "嗯... 30 - 16 = 24？24 * 5 = 120\n"
                         "答案是 120。",
                "answer": 120,
                "weight": 0.05,  # 减法算错
            },
        ],
    },
]


def mock_cot_reasoning(question_data: dict, temperature: float = 0.7) -> dict:
    """
    模拟 LLM 生成一条 CoT 推理链

    🧮 讲义对应：§3.1 和 §3.3 —— "用较高的 Temperature 让模型生成多条不同的 CoT 推理链"

    【核心设计】
    - temperature 控制采样的"随机性"：
      - temperature=0 → 贪婪解码，永远选概率最高的路径（每次结果相同）
      - temperature=0.7~1.0 → 有概率走出不同的推理路径（Self-Consistency 需要这个！）

    Args:
        question_data: 问题数据（包含 reasoning_paths 列表）
        temperature: 温度参数，控制采样随机性

    Returns:
        dict: 包含 "steps"（推理过程）和 "answer"（最终答案）
    """
    paths = question_data["reasoning_paths"]
    weights = [p["weight"] for p in paths]

    if temperature == 0:
        # ------------------------------------------------------------------
        # 🔑 讲义 §3.3 [!IMPORTANT]：Temperature=0 等于 Greedy Decoding
        #    每次都选概率（权重）最高的路径 → 采 100 次也等于只采 1 次
        #    Self-Consistency 在此情况下完全无效！
        # ------------------------------------------------------------------
        best_idx = weights.index(max(weights))
        return {"steps": paths[best_idx]["steps"], "answer": paths[best_idx]["answer"]}

    # temperature > 0：按权重随机采样，模拟高 Temperature 下的多样化输出
    # 注意：temperature 越高，低权重路径被选中的概率越大 → 路径越多样
    adjusted_weights = [w ** (1.0 / temperature) for w in weights]
    total = sum(adjusted_weights)
    normalized = [w / total for w in adjusted_weights]

    chosen = random.choices(paths, weights=normalized, k=1)[0]
    return {"steps": chosen["steps"], "answer": chosen["answer"]}


# ==============================================================================
# Part B: Self-Consistency 核心实现
# ==============================================================================

def extract_answer(cot_response: dict):
    """
    从 CoT 推理链中提取最终答案

    🧮 讲义对应：§3.3 —— "从推理链末尾提取最终答案 (如 '答案是 68' → 68)"

    在真实场景中，这通常通过正则表达式或专门的 Parser 实现，
    例如匹配 "答案是 X" / "The answer is X" 这样的模式。

    这里因为是模拟数据，直接取 answer 字段。

    Args:
        cot_response: LLM 生成的 CoT 响应

    Returns:
        提取出的最终答案
    """
    return cot_response["answer"]


def majority_vote(answers: list):
    """
    多数投票：选择出现次数最多的答案

    🧮 讲义对应：§3.1 —— "多数投票 (Majority Vote)：选择出现次数最多的答案作为最终输出"

    【核心原理 - §3.2】
    "错误是随机的，正确是一致的。"
    - 多条推理链走不同的路径，犯的错误各不相同 → 错误答案分散
    - 但正确答案具有"趋同性" → 正确答案集中

    Args:
        answers: 从各条推理链中提取的答案列表

    Returns:
        出现次数最多的答案
    """
    counter = Counter(answers)
    # most_common(1) 返回 [(answer, count)] → 取第一个元素的 answer
    return counter.most_common(1)[0][0]


def self_consistency(question_data: dict, n_samples: int = 10, temperature: float = 0.7,
                     verbose: bool = False) -> dict:
    """
    🧮 讲义对应：§3.3 —— Self-Consistency 完整实现

    核心流程（与讲义伪代码一一对应）：
    1. 用 CoT 方式生成 n_samples 条独立的推理链（temperature > 0 是关键！）
    2. 从每条推理链中提取最终答案
    3. 多数投票，选出现次数最多的答案

    Args:
        question_data: 问题数据
        n_samples: 采样次数（对应讲义 §3.4 的采样数）
        temperature: 温度参数（讲义强调"必须 > 0"）
        verbose: 是否打印详细的推理过程

    Returns:
        dict: 包含最终答案、所有采样答案、投票分布等信息
    """
    # ------------------------------------------------------------------
    # Step 1: 多次独立采样 —— 每次都是独立的 CoT 推理
    # ------------------------------------------------------------------
    # 🔑 讲义 §3.3：本质就是跑 n 次独立的 CoT
    # 🔑 讲义 [!TIP]：这些采样可以完美并行（BatchSize 只受显存限制）
    all_responses = []
    answers = []

    for i in range(n_samples):
        # 调用"LLM"生成一条 CoT 推理链
        response = mock_cot_reasoning(question_data, temperature=temperature)
        all_responses.append(response)

        # Step 2: 提取每条推理链的最终答案
        answer = extract_answer(response)
        answers.append(answer)

        if verbose:
            print(f"\n{'─' * 50}")
            print(f"  采样 {i + 1}/{n_samples}:")
            print(f"  推理过程: {response['steps'][:80]}...")
            print(f"  提取答案: {answer}")

    # ------------------------------------------------------------------
    # Step 3: 多数投票
    # ------------------------------------------------------------------
    vote_counter = Counter(answers)
    final_answer = majority_vote(answers)

    if verbose:
        print(f"\n{'═' * 50}")
        print(f"  📊 投票结果:")
        for ans, count in vote_counter.most_common():
            bar = "█" * count
            print(f"     答案 {ans}: {count} 票 {bar}")
        print(f"  ✅ 最终答案: {final_answer}")

    return {
        "final_answer": final_answer,
        "all_answers": answers,
        "vote_distribution": dict(vote_counter),
        "n_samples": n_samples,
    }


# ==============================================================================
# Part C: Temperature=0 对比实验（验证讲义 §3.3 的 [!IMPORTANT]）
# ==============================================================================

def demo_temperature_effect(question_data: dict):
    """
    演示 Temperature 对 Self-Consistency 的影响

    🧮 讲义对应：§3.3 [!IMPORTANT]
    "如果 temperature=0，模型每次采样都会走完全相同的路径、
     得到完全相同的答案——采 100 次也等于只采了 1 次。"
    """
    print("\n" + "=" * 60)
    print("🔬 实验 1: Temperature 对 Self-Consistency 的影响")
    print("=" * 60)

    question = question_data["question"]
    correct = question_data["correct_answer"]
    print(f"\n📝 问题: {question}")
    print(f"📎 正确答案: {correct}")

    # ---- Temperature = 0 (Greedy) ----
    print(f"\n{'─' * 50}")
    print("❄️  Temperature = 0 (Greedy Decoding)")
    print("    讲义: '采 100 次也等于只采了 1 次'")
    print(f"{'─' * 50}")

    result_greedy = self_consistency(question_data, n_samples=10, temperature=0)
    print(f"  10 次采样的答案: {result_greedy['all_answers']}")
    print(f"  投票分布: {result_greedy['vote_distribution']}")
    print(f"  → 所有答案完全相同！Self-Consistency 毫无意义。")

    # ---- Temperature = 0.7 (推荐值) ----
    print(f"\n{'─' * 50}")
    print("🌡️  Temperature = 0.7 (推荐值)")
    print("    讲义: '让每次采样走出不同的推理路径'")
    print(f"{'─' * 50}")

    result_warm = self_consistency(question_data, n_samples=10, temperature=0.7, verbose=True)
    is_correct = result_warm["final_answer"] == correct
    print(f"\n  最终答案 {'✅ 正确' if is_correct else '❌ 错误'}!")


# ==============================================================================
# Part D: 采样数 vs 准确率（验证讲义 §3.4 的边际递减效应）
# ==============================================================================

def demo_sample_count_vs_accuracy(question_bank: list, n_trials: int = 200):
    """
    模拟不同采样数下的准确率，展示边际递减效应

    🧮 讲义对应：§3.4 的计算开销权衡表
    | 采样数 | 计算成本 | 准确率提升 |
    | 1      | 1x       | baseline  |
    | 5      | 5x       | +5-10%    |
    | 10-20  | 10-20x   | +10-15%   |
    | 40+    | 40x+     | 边际递减   |
    """
    print("\n" + "=" * 60)
    print("📊 实验 2: 采样数 vs 准确率（边际递减效应）")
    print("=" * 60)

    sample_counts = [1, 3, 5, 10, 20, 40]

    print(f"\n  对 {len(question_bank)} 道题各重复 {n_trials} 次实验\n")
    print(f"  {'采样数':>6} | {'准确率':>8} | {'计算成本':>8} | 可视化")
    print(f"  {'─' * 6} | {'─' * 8} | {'─' * 8} | {'─' * 20}")

    for n in sample_counts:
        correct_count = 0
        total_count = 0

        for question_data in question_bank:
            for _ in range(n_trials):
                result = self_consistency(question_data, n_samples=n, temperature=0.7)
                if result["final_answer"] == question_data["correct_answer"]:
                    correct_count += 1
                total_count += 1

        accuracy = correct_count / total_count
        cost = f"{n}x"
        bar = "█" * int(accuracy * 20)
        print(f"  {n:>6} | {accuracy:>7.1%} | {cost:>8} | {bar}")

    print(f"\n  💡 观察: 采样数从 1→5 提升显著，但 20→40 后收益递减")
    print(f"  📖 讲义 §3.4: '边际递减'——投入更多算力的收益逐渐减小")


# ==============================================================================
# Part E: Self-Consistency 的局限性演示
# ==============================================================================

def demo_limitations():
    """
    演示 Self-Consistency 的局限性

    🧮 讲义对应：§3.5 局限性
    1. 只适用于有"标准答案"的问题（无法对开放式问题投票）
    2. 粗粒度：只在最终答案上投票，没有利用中间步骤信息
    3. 无搜索：各条路径独立采样，没有信息共享
    """
    print("\n" + "=" * 60)
    print("⚠️  实验 3: Self-Consistency 的局限性")
    print("=" * 60)

    # ---- 局限 1: 开放式问题无法投票 ----
    print("\n  📌 局限 1: 只适用于有标准答案的问题")
    print("  ─" * 25)

    open_ended_answers = [
        "春天像一幅绿色的画卷...",
        "春天是大地苏醒的季节...",
        "春风拂面，万物复苏...",
        "冰雪消融，花开满园...",
        "阳光明媚，鸟语花香...",
    ]
    print(f"  问题: '请用一段话描述春天'")
    print(f"  5 条各不相同的回答，无法投票:")
    for i, ans in enumerate(open_ended_answers, 1):
        print(f"    采样 {i}: {ans}")
    print(f"  → 每条回答都不同，多数投票在此完全失效！")

    # ---- 局限 2: 无搜索 ----
    print(f"\n  📌 局限 2: 独立采样，无信息共享")
    print("  ─" * 25)
    print("  即使第 1 条路径发现了'死胡同'，第 2 条路径也不知道，")
    print("  可能重蹈覆辙。→ 引出 Part 4: Tree of Thoughts（树状搜索）")


# ==============================================================================
# 主程序
# ==============================================================================

def main():
    """Self-Consistency 演示主入口"""
    print("=" * 60)
    print("📘 Week 7 - Part 3: Self-Consistency 多路采样投票 演示")
    print("=" * 60)
    print("""
    核心思想 (讲义 §3.1):
    "既然一条推理链可能出错，那就让模型独立跑多条推理链，
     然后用投票来消除随机错误。"
    """)

    # 固定随机种子以便复现（教学用途）
    random.seed(42)

    # ---- 实验 1: Temperature 的影响 ----
    demo_temperature_effect(QUESTION_BANK[0])

    # ---- 实验 2: 采样数 vs 准确率 ----
    random.seed(42)
    demo_sample_count_vs_accuracy(QUESTION_BANK, n_trials=200)

    # ---- 实验 3: 局限性 ----
    demo_limitations()

    # ---- 总结 ----
    print("\n" + "=" * 60)
    print("📝 Self-Consistency 核心要点总结")
    print("=" * 60)
    print("""
1. 【核心公式】
   Final Answer = MajorityVote(CoT_1, CoT_2, ..., CoT_n)
   
   每条 CoT_i 是在高 Temperature 下独立采样的推理链。

2. 【为什么有效 —— §3.2】
   "错误是随机的，正确是一致的。"
   - 每条路径犯的错各不相同 → 错误票分散
   - 正确路径殊途同归 → 正确票集中

3. 【Temperature 的关键作用 —— §3.3】
   - T=0 → 每次采样完全相同 → Self-Consistency 无意义
   - T=0.7~1.0 → 路径多样，错误独立 → 投票有效

4. 【计算开销 —— §3.4】
   - 线性增长：n 次采样 = n 倍计算成本
   - 边际递减：1→5 提升大，20→40 收益小
   - 工程优势：所有采样可完美并行

5. 【局限性 —— §3.5】
   - 只适合有标准答案的问题（数学、代码、事实问答）
   - 只看最终答案，浪费了中间步骤的信息
   - 各路径独立，没有"学习前车之鉴"的搜索能力
   → 引出 Part 4: Tree of Thoughts（树状搜索 + 中间评估）
""")


if __name__ == "__main__":
    main()
