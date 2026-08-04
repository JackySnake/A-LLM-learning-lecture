import asyncio
import os
import re
from typing import List, Optional, Tuple
from collections import Counter

# 尝试导入 openai 库，以便实际运行；如果没有，则使用 Mock
try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️ 未安装 openai 库 ('pip install openai')，将使用 Mock 模式运行。\n")

"""
🚀 Week 7 进阶实战：基于真实 LLM API 的生产级高级推理实现

【学习目标】
前面两个的 script 是纯逻辑演示，而本脚本将展示如何在实际业务中真正落地
Self-Consistency(SC) 和 Tree of Thoughts(ToT) 策略。

【核心工程考量】
1. IO 密集型操作：真实 LLM API 请求是网络延迟主导的（几秒~几十秒），
   如果不使用异步并发 (`asyncio`)，像 SC 这样需要采样 10 次的操作将慢得不可接受。
2. 异常处理：网络抖动、频控 (Rate Limit)、上下文超限，必须实现重试机制。
3. 结构化输出：ToT 的评估器必须让 LLM 按照特定格式输出，并使用正则进行容错提取。

【对应讲义知识点】
- Part 3.4 计算开销权衡："Self-Consistency 可以完美并行"
- Part 4.2 评估器的核心机制："系统会构造一个类似下面的评估 Prompt"

如果你想执行真实的模型调用：
export OPENAI_API_KEY="sk-xxxx"
export OPENAI_BASE_URL="https://api.openai.com/v1" # 如果有代理或者使用其他兼容模型(如 DeepSeek)
"""

# ==============================================================================
# 模块 0: 生产级大模型异步客户端
# ==============================================================================

class AsyncLLMClient:
    """
    封装了错误重试、超时的异步 LLM 客户端
    真实业务中，这一层通常还会包含：容灾切换、缓存层、日志埋点等。
    """
    def __init__(self, model_name="gpt-3.5-turbo", max_retries=3):
        self.model_name = model_name
        self.max_retries = max_retries
        self.use_mock = not HAS_OPENAI or not os.getenv("OPENAI_API_KEY")
        
        if not self.use_mock:
            self.client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", None)
            )
        else:
            self.client = None
            if HAS_OPENAI:
                 print("⚠️ 未检测到 OPENAI_API_KEY 环境变量，将使用 Mock 模式。")

    async def generate(self, prompt: str, temperature: float = 0.7, stop: list = None) -> str:
        """带重试机制的通用生成接口"""
        if self.use_mock:
            return await self._mock_generate(prompt, temperature)

        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "你是一个严谨的逻辑推理助手。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    stop=stop,
                    timeout=30.0 # 生产环境必须设置超时
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"❌ API 调用失败，已达最大重试次数: {e}")
                    raise
                print(f"⚠️ API 调用异常（尝试 {attempt+1}/{self.max_retries}）: {e}，等待重试...")
                await asyncio.sleep(2 ** attempt)  # 指数退避重试

    async def _mock_generate(self, prompt: str, temperature: float) -> str:
        """用于在没有网络/Token时演示代码流程"""
        await asyncio.sleep(0.5) # 模拟网络延迟
        if "Game of 24" in prompt and "评估" in prompt:
            # 模拟 Evaluator 的输出
            if "24" in prompt: return "SURE"
            if "10, 14" in prompt or "13, 11" in prompt: return "MAYBE"
            return "IMPOSSIBLE"
        elif "Game of 24" in prompt and "列举" in prompt:
            # 模拟 Generator 输出
            return "- 12 + 2 = 14 => [1, 11, 14]\n- 11 - 1 = 10 => [2, 10, 12]"
        elif "提取答案" in prompt or "提取最终" in prompt:
            return "68"
        else:
            # 模拟普通 CoT
            return "让我一步一步来思考。\n首先... 其次...\n最终答案是 68。"


# ==============================================================================
# 模块 1: 异步并发 Self-Consistency
# ==============================================================================

async def run_self_consistency(client: AsyncLLMClient, question: str, n_samples: int = 10):
    """
    生产级 Self-Consistency 实现
    通过 asyncio.gather 实现【无阻塞并发采样】
    """
    print(f"\n🚀 开始 Self-Consistency 并发计算 (共 {n_samples} 路采样)...")
    
    # 1. 构造 CoT Prompt
    cot_prompt = f"问题：{question}\n请一步一步思考，写出详细的推理过程，并在最后用'最终答案是：XXX'的格式输出结果。"
    
    # 定义单次采样的闭包函数
    async def _sample_and_extract(idx: int) -> str:
        # a. 高 Temperature 采样推理路径
        cot_text = await client.generate(cot_prompt, temperature=0.8)
        
        # b. 提取答案 (生产级通常让 LLM 再做一次提取，或者用严格的正则)
        extract_prompt = f"请从以下推理过程中提取出最终的数值答案，只输出阿拉伯数字，不要任何其他废话：\n\n{cot_text}"
        answer = await client.generate(extract_prompt, temperature=0.0)
        
        # 简单清洗
        answer = re.sub(r'[^\d.]', '', answer)
        print(f"  ✅ 采样 {idx+1} 完成 -> 提取结果: {answer}")
        return answer

    # 2. 并发执行所有采样
    # 这是工程落地的关键：10个请求同时发出去，总耗时等于最慢的那一个请求，而不是 10 个请求耗时之和！
    tasks = [_sample_and_extract(i) for i in range(n_samples)]
    answers = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 过滤掉网络异常失败的请求
    valid_answers = [a for a in answers if isinstance(a, str) and a != ""]
    
    if not valid_answers:
        return "全部采样失败"
        
    # 3. 多数投票
    vote_counter = Counter(valid_answers)
    final_answer = vote_counter.most_common(1)[0][0]
    
    print("\n📊 并发投票完成!")
    for ans, count in vote_counter.most_common():
        print(f"  - 答案 {ans}: {count} 票")
    print(f"🎯 最终决出答案: {final_answer}")
    return final_answer


# ==============================================================================
# 模块 2: 基于真实 Prompt 的 Tree of Thoughts
# ==============================================================================

class ToTNode:
    def __init__(self, state: str, history: List[str] = None):
        self.state = state  # 当前状态的字符串描述（如 "[4, 4, 10]"）
        self.history = history or []

async def generate_thoughts(client: AsyncLLMClient, node: ToTNode) -> List[ToTNode]:
    """
    🎭 对应 ToT 生成器 (Generator)
    向模型发送 Prompt 要求其返回特定格式的子分支，然后我们进行解析。
    """
    prompt = f"""
任务：Game of 24。请运用加减乘除。
当前剩余数字：{node.state}
请列举所有可能的下一步单次运算。
要求输出格式：每行一个运算，格式为 "- 运算公式 => 剩余数字列表"。
例如：- 13 - 9 = 4 => [4, 4, 10]
提示：只生成下一步运算，不要试图直接解完！
"""
    response_text = await client.generate(prompt, temperature=0.7)
    
    # 生产级：使用正则表达式安全地解析 LLM 这种非结构化输出
    new_nodes = []
    # 匹配模式：- 4 + 4 = 8 => [8, 10]
    pattern = re.compile(r'-\s*(.*?)\s*=>\s*(\[.*?\])')
    
    for match in pattern.finditer(response_text):
        op_str, new_state_str = match.groups()
        new_nodes.append(ToTNode(
            state=new_state_str,
            history=node.history + [op_str.strip()]
        ))
        
    return new_nodes


async def evaluate_thought(client: AsyncLLMClient, node: ToTNode) -> str:
    """
    🎭 对应 ToT 评估器 (Evaluator)
    利用 Prompt Engineering 让模型成为裁判。
    """
    prompt = f"""
任务：Game of 24 评估（运用加减乘除计算出24）。
请评估以下剩余数字组合是否有希望通过进一步运算得到 24。
当前剩余数字：{node.state}

评估标准：
1. 如果确定能得到 24（例如只剩24，或者简单计算即可），请在最后一行输出 SURE
2. 如果还能走几步，有潜力但不敢肯定，请在最后一行输出 MAYBE
3. 如果出现了负数、明显无法整除的分数，或者数字完全无法组合出24，请在最后一行输出 IMPOSSIBLE

你可以先进行简单的思维推理，但必须在回答的【最后一行】且【仅用一个英文单词】输出你的最终判决标签。
"""
    response_text = await client.generate(prompt, temperature=0.0) # 裁判需要理性，temperature=0
    
    # 从最后一行或全文中提取标签
    response_upper = response_text.upper()
    if "SURE" in response_upper:
        return "SURE"
    elif "IMPOSSIBLE" in response_upper:
        return "IMPOSSIBLE"
    else:
        return "MAYBE"


async def run_tot_bfs(client: AsyncLLMClient, initial_state: str, beam_size: int = 3, max_steps: int = 3):
    """
    执行基于 API 的异步 BFS Tree of Thoughts
    """
    print(f"\n🌳 开始基于 API 的 ToT (BFS) 搜索...")
    
    candidates = [ToTNode(state=initial_state)]
    
    for step in range(max_steps):
        print(f"\n▶ 正在处理第 {step + 1} 层 (当前存活节点数: {len(candidates)})...")
        
        # 1. 异步扩展所有存活节点
        # 使用 asyncio.gather 并发调用 Generator
        expand_tasks = [generate_thoughts(client, node) for node in candidates]
        children_lists = await asyncio.gather(*expand_tasks)
        
        # 展平列表
        next_candidates = []
        for lst in children_lists:
            next_candidates.extend(lst)
            
        if not next_candidates:
            print("  ❌ 节点已无可扩展分支，搜索失败。")
            return
            
        print(f"  🌱 Generator 总共扩展了 {len(next_candidates)} 个候选节点，开始并行评估...")
        
        # 2. 异步评估所有候选节点
        eval_tasks = [evaluate_thought(client, node) for node in next_candidates]
        scores = await asyncio.gather(*eval_tasks)
        
        sure_nodes, maybe_nodes = [], []
        
        for node, score in zip(next_candidates, scores):
            if score == 'SURE':
                sure_nodes.append(node)
                # 检查是否达成最终状态 (只有一个数字且为24)
                if len(eval(node.state)) == 1 and str(eval(node.state)[0]) == '24':
                    print(f"\n🎉 成功找到解法！")
                    for s in node.history: print(f"  -> {s}")
                    return node
            elif score == 'MAYBE':
                maybe_nodes.append(node)
            else:
                pass # IMPOSSIBLE 直接在此被静默剪枝
                
        # 3. 根据 beam_size 截断 (保留 sure 优先)
        candidates = (sure_nodes + maybe_nodes)[:beam_size]
        print(f"  ✂️ 评估结束，截断保留 {len(candidates)} 个最具潜力的节点。")
        
    print("\n❌ 搜索达到最大深度，未找到明确的答案。")
    return None

# ==============================================================================
# 启动入口
# ==============================================================================

async def main():
    print("=" * 60)
    print("🏭 生产级 LLM 高级推理调用框架展示")
    print("=" * 60)
    print("""
此脚本展示了在真实业务后端中实现高级推理的必备工程手段：
- ✔️ asyncio.gather 大规模并发 API 请求（为了 Self-Consistency）
- ✔️ 带有自动超时与重试的健壮封装（为了应对网络故障/限流）
- ✔️ Prompt Engineering 指导结构化输出（为了 ToT 的中间层解析）
- ✔️ 正则表达式强制解析（从非结构化生成文本中安全提取业务需要的变量）
""")
    # 实例化我们的异步客户端
    client = AsyncLLMClient(model_name="gpt-3.5-turbo", max_retries=3)
    
    # 演示 1: 异步 Self-Consistency
    await run_self_consistency(client, question="一辆卡车装了500个苹果，路上掉了32个，又卖了100个，还剩几个？", n_samples=5)
    
    # 演示 2: 基于 LLM 裁判的 ToT
    await run_tot_bfs(client, initial_state="[2, 3, 5, 12]", beam_size=2)
    
    print("\n✅ 生产级流程度演示执行完毕。")

if __name__ == "__main__":
    # 使用 asyncio 运行异步主循环
    asyncio.run(main())
