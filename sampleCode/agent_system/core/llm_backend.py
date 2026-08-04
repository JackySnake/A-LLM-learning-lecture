"""
可插拔 LLM 后端（Week 9 地基 · 已重构为合法的 Function Calling 协议）
======================================================================

对应讲义：Week 9 · Part 3「工具调用机制——从 Prompt 到 API」

设计动机
--------
Agent 的"大脑"是一次 LLM 调用：给定 system prompt + 历史 + 可用工具，
模型决定「下一步思考什么、调用哪个工具、参数是什么」，或者「任务完成、给出答复」。

为了让这套教学代码**无需 API key 即可跑通完整流程**，我们把"大脑"抽象成
`LLMBackend` 接口，并提供多种实现：

  - MockCodingBackend     ：离线、确定性的规则后端（演示 ReAct 反馈闭环，非真 LLM）
  - AnthropicBackend      ：接真实 Claude（tool_use / tool_result 协议）
  - OpenAICompatBackend   ：接 OpenAI 兼容端点（含 aihubmix），tool_calls / tool role 协议
  - StrictProtocolBackend ：不花钱验证用——模拟真实 API 的协议校验，跑通它即证明真后端可用

关键设计：统一的"中性事件流"
-----------------------------
所有后端共享同一份对话历史（见 memory.WorkingMemory），每条事件是一个 dict：

    {"role": "user",      "content": "任务文本"}
    {"role": "assistant", "thought": "...", "tool_call": {"id","name","args"}}   # 调用工具
    {"role": "assistant", "thought": "...", "final_answer": "..."}               # 给出最终答复
    {"role": "tool",      "tool_call_id": "...", "content": "observation"}       # 工具结果

注意 tool_call 带 **唯一 id**，工具结果用同一个 id 回指——这正是 Function Calling
多轮协议的核心。各后端只负责把这条中性事件流**无损翻译**成自己的原生 API 格式
（Anthropic 的 tool_use/tool_result、OpenAI 的 tool_calls/tool role）。

这是对早期实现的关键修正：早期版本把工具调用降级成纯文本注入，等于退回了
讲义 Part 3.1 批判的"2023 年之前的脆弱做法"，换上真模型也只是绣花枕头。
"""

from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4


# ──────────────────────────────────────────────────────────────────────────
# 统一的"模型决策"数据结构
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class AgentAction:
    """一次 LLM 决策的结构化结果（对应现代 Function Calling 范式）。

    要么调用一个工具（tool_name 非空），要么给出最终答案（final_answer 非空）。
    tool_call_id：真后端从 API 响应里拿到的真实调用 id；mock/strict 后端可生成。
    """

    thought: str
    tool_name: Optional[str] = None
    tool_args: dict = field(default_factory=dict)
    tool_call_id: Optional[str] = None
    final_answer: Optional[str] = None

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None


class LLMBackend(ABC):
    """所有后端的统一接口。Agent 只依赖这个抽象，不关心背后是 mock 还是真模型。"""

    @abstractmethod
    def decide(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
    ) -> AgentAction:
        """根据 system prompt、中性事件流、工具规格列表，决定下一步动作。

        Args:
            system_prompt: 角色与规则（含记忆注入）。
            messages: 中性事件流（见模块 docstring）。
            tools: 工具的中性规格列表 [{"name","description","parameters"}]（见 tools.py）。
        """
        raise NotImplementedError

    async def adecide(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
    ) -> AgentAction:
        """`decide` 的异步版本，供 Week 10 Orchestrator 并发调度多个 Worker 时使用。

        默认实现：把同步 `decide` 丢进线程池跑（`asyncio.to_thread`）——这是"后端作者
        没有专门写异步版本"时的诚实兜底，不会阻塞事件循环，但也不是真正的异步 I/O。
        真正受益于异步的后端（如 OpenAICompatBackend）应该覆盖这个方法，改用厂商 SDK
        的异步客户端发起真实的非阻塞网络请求——这样多个 Worker 的 LLM 调用才能在网络
        等待期间真正并发，而不是排队等线程池调度。
        """
        return await asyncio.to_thread(self.decide, system_prompt, messages, tools)


# ──────────────────────────────────────────────────────────────────────────
# 协议转换工具：中性事件流 → 各厂商原生格式（被多个后端复用）
# ──────────────────────────────────────────────────────────────────────────
class ProtocolError(RuntimeError):
    """消息不满足 Function Calling 多轮协议时抛出（模拟真实 API 的 400 报错）。"""


def to_openai_messages(messages: list[dict]) -> list[dict]:
    """中性事件流 → OpenAI Chat Completions 格式（tool_calls / tool role）。"""
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "user":
            out.append({"role": "user", "content": m.get("content", "")})
        elif role == "assistant":
            if m.get("tool_call"):
                tc = m["tool_call"]
                out.append({
                    "role": "assistant",
                    "content": m.get("thought") or None,
                    "tool_calls": [{
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"], ensure_ascii=False),
                        },
                    }],
                })
            else:
                out.append({"role": "assistant", "content": m.get("final_answer") or m.get("thought", "")})
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m.get("content", "")})
    return out


def to_openai_tools(specs: list[dict]) -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": s["name"], "description": s["description"], "parameters": s["parameters"],
        }}
        for s in specs
    ]


def to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """中性事件流 → Anthropic Messages 格式（tool_use / tool_result 内容块）。"""
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "user":
            out.append({"role": "user", "content": m.get("content", "")})
        elif role == "assistant":
            blocks: list[dict] = []
            if m.get("thought"):
                blocks.append({"type": "text", "text": m["thought"]})
            if m.get("tool_call"):
                tc = m["tool_call"]
                blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["args"]})
            if m.get("final_answer"):
                blocks.append({"type": "text", "text": m["final_answer"]})
            if not blocks:
                blocks.append({"type": "text", "text": ""})
            out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            out.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m.get("content", "")}
            ]})
    return out


def to_anthropic_tools(specs: list[dict]) -> list[dict]:
    return [{"name": s["name"], "description": s["description"], "input_schema": s["parameters"]} for s in specs]


def validate_openai_protocol(api_messages: list[dict]) -> None:
    """模拟 OpenAI 的协议校验：每个 tool_call 必须有匹配 id 的 tool 响应。"""
    declared: set[str] = set()
    pending: set[str] = set()
    for msg in api_messages:
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                declared.add(tc["id"])
                pending.add(tc["id"])
        elif msg["role"] == "tool":
            tcid = msg.get("tool_call_id")
            if tcid not in declared:
                raise ProtocolError(f"tool 消息 tool_call_id={tcid} 找不到对应的 assistant.tool_calls")
            pending.discard(tcid)
    if pending:
        raise ProtocolError(f"存在未被 tool 结果响应的 tool_call：{pending}")


def validate_anthropic_protocol(api_messages: list[dict]) -> None:
    """模拟 Anthropic 的协议校验：每个 tool_use 必须有匹配 id 的 tool_result。"""
    declared: set[str] = set()
    pending: set[str] = set()
    for msg in api_messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "tool_use":
                declared.add(block["id"])
                pending.add(block["id"])
            elif block.get("type") == "tool_result":
                tcid = block.get("tool_use_id")
                if tcid not in declared:
                    raise ProtocolError(f"tool_result 的 tool_use_id={tcid} 找不到对应的 tool_use")
                pending.discard(tcid)
    if pending:
        raise ProtocolError(f"存在未被 tool_result 响应的 tool_use：{pending}")


def _last_observation(messages: list[dict]) -> Optional[str]:
    """取中性事件流里最近一次工具观察结果（mock/strict 状态机复用）。"""
    for m in reversed(messages):
        if m["role"] == "tool":
            return str(m["content"])
    return None


# ──────────────────────────────────────────────────────────────────────────
# Mock 后端用的"剧本"：把任务相关的文件名/代码抽出来，让 MockCodingBackend
# 可以复用同一套状态机逻辑演出不同的任务（Week 10 需要给两个 Worker 派不同任务）
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class MockCodingScript:
    test_file: str
    target_file: str
    buggy_code: str
    fixed_code: str
    bug_explanation: str
    final_summary: str


# 第一版实现：循环上界写成 len(nums)，末尾会产生"不完整窗口"的多余结果 → 测试失败
_MOVING_AVERAGE_BUGGY = '''\
def moving_average(nums, window):
    """计算滑动平均。"""
    result = []
    for i in range(len(nums)):              # bug：没有在末尾停下，会多算不完整的窗口
        chunk = nums[i:i + window]
        result.append(sum(chunk) / window)
    return result
'''

# 修复版：循环上界改为 len-window+1，并补上 window>len 返回空列表的边界
_MOVING_AVERAGE_FIXED = '''\
def moving_average(nums, window):
    """计算滑动平均；当 window 超过列表长度时返回空列表。"""
    if window <= 0:
        raise ValueError("window 必须为正整数")
    if window > len(nums):
        return []
    result = []
    for i in range(len(nums) - window + 1):
        chunk = nums[i:i + window]
        result.append(sum(chunk) / window)
    return result
'''

MOVING_AVERAGE_SCRIPT = MockCodingScript(
    test_file="test_math_utils.py",
    target_file="math_utils.py",
    buggy_code=_MOVING_AVERAGE_BUGGY,
    fixed_code=_MOVING_AVERAGE_FIXED,
    bug_explanation="测试失败：循环在末尾产生了不完整窗口的多余结果。我把上界改为 len-window+1，并补上 window>len 返回空列表的边界。",
    final_summary="补充了 window 超过列表长度时返回空列表的边界处理",
)

# 第一版实现：extend 只展开了一层，深层嵌套（list 里还有 list）没有递归处理 → 测试失败
_FLATTEN_BUGGY = '''\
def flatten_list(nested):
    """将嵌套列表展开为一维列表。"""
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(item)          # bug：只展开了一层，深层嵌套未递归处理
        else:
            result.append(item)
    return result
'''

# 修复版：递归调用自身展开任意深度嵌套
_FLATTEN_FIXED = '''\
def flatten_list(nested):
    """将嵌套列表递归展开为一维列表，支持任意深度嵌套。"""
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_list(item))   # 递归处理任意深度嵌套
        else:
            result.append(item)
    return result
'''

FLATTEN_SCRIPT = MockCodingScript(
    test_file="test_list_utils.py",
    target_file="list_utils.py",
    buggy_code=_FLATTEN_BUGGY,
    fixed_code=_FLATTEN_FIXED,
    bug_explanation="测试失败：extend 只展开了一层，遇到深层嵌套（列表里还有列表）没有递归处理。我改成递归调用自身。",
    final_summary="改为递归展开，支持任意深度的嵌套列表",
)

# 第一版实现：直接反转字符串比较，没有忽略大小写和空格 → 带空格/大小写混用的用例测试失败
_PALINDROME_BUGGY = '''\
def is_palindrome(s):
    """判断字符串是否为回文。"""
    return s == s[::-1]   # bug：没有忽略大小写和空格
'''

# 修复版：先归一化（只保留字母数字、转小写）再比较
_PALINDROME_FIXED = '''\
def is_palindrome(s):
    """判断字符串是否为回文，忽略大小写和非字母数字字符。"""
    cleaned = "".join(ch.lower() for ch in s if ch.isalnum())
    return cleaned == cleaned[::-1]
'''

PALINDROME_SCRIPT = MockCodingScript(
    test_file="test_string_utils.py",
    target_file="string_utils.py",
    buggy_code=_PALINDROME_BUGGY,
    fixed_code=_PALINDROME_FIXED,
    bug_explanation="测试失败：直接逐字符反转比较，没有忽略大小写和空格/标点。我先归一化（转小写、只保留字母数字）再比较。",
    final_summary="改为先归一化再比较，支持忽略大小写和标点空格",
)


# ──────────────────────────────────────────────────────────────────────────
# 离线 Mock 后端：用一个小状态机演示 ReAct 反馈闭环
# ──────────────────────────────────────────────────────────────────────────
class MockCodingBackend(LLMBackend):
    """规则驱动的离线后端（演示用，非真 LLM），扮演 Coder 角色。

    它针对 demo 任务"给某文件添加函数并通过 pytest"内置了一段决策逻辑，
    **关键在于它会读取上一步的 observation 来分支**——这正是 ReAct 区别于
    一次性 CoT 的核心：行动结果会反过来影响后续决策。

    真实场景里，下面这些 thought / 代码内容都由 LLM 动态生成；这里写死只是
    为了让没有 API key 的读者也能看到完整循环（包括"先失败、再修复"）。

    Week 10 起接受一个 `script` 参数，让同一套状态机可以演出不同的任务
    （见 MOVING_AVERAGE_SCRIPT / FLATTEN_SCRIPT），从而给多个 Worker 分派不同工作。
    """

    def __init__(self, script: MockCodingScript = MOVING_AVERAGE_SCRIPT):
        self.script = script

    def decide(self, system_prompt, messages, tools) -> AgentAction:
        s = self.script
        # 从中性事件流里统计：各工具被调用过几次、最近一次观察结果。
        tool_calls = [m["tool_call"] for m in messages
                      if m["role"] == "assistant" and m.get("tool_call")]
        called = [tc["name"] for tc in tool_calls]
        write_count = called.count("write_file")
        last_obs = _last_observation(messages)

        # ── 状态机：每个分支对应 ReAct 的一步「思考 + 行动」 ──

        if "list_dir" not in called:
            return AgentAction(
                thought="我先看看工作目录里有哪些文件，确认目标文件和测试文件的位置。",
                tool_name="list_dir", tool_args={"path": "."},
            )

        if "read_file" not in called:
            return AgentAction(
                thought=f"读取测试文件 {s.test_file}，明确目标函数需要满足哪些用例（尤其是边界条件）。",
                tool_name="read_file", tool_args={"path": s.test_file},
            )

        if write_count == 0:
            return AgentAction(
                thought="先实现一个基础版本，覆盖常规逻辑，暂不处理边界情况。",
                tool_name="write_file", tool_args={"path": s.target_file, "content": s.buggy_code},
            )

        if "run_pytest" not in called:
            return AgentAction(
                thought="实现已写入，运行 pytest 验证是否通过全部用例。",
                tool_name="run_pytest", tool_args={"path": "."},
            )

        passed = last_obs is not None and "passed" in last_obs and "failed" not in last_obs

        if not passed and write_count == 1:
            return AgentAction(
                thought=s.bug_explanation,
                tool_name="write_file", tool_args={"path": s.target_file, "content": s.fixed_code},
            )

        if not passed:
            return AgentAction(
                thought="已修复边界条件，重新运行测试确认。",
                tool_name="run_pytest", tool_args={"path": "."},
            )

        return AgentAction(
            thought="全部测试通过，任务完成。",
            final_answer=f"已为 {s.target_file} 实现并修复目标函数：{s.final_summary}，pytest 全部通过。",
        )


# ──────────────────────────────────────────────────────────────────────────
# Mock Reviewer 后端：只读 + 独立复核，不修改代码（Week 10 Worker 角色分工）
# ──────────────────────────────────────────────────────────────────────────
class MockReviewBackend(LLMBackend):
    """规则驱动的离线 Reviewer：读实现、独立跑一次 pytest、给出通过/不通过结论。

    与 MockCodingBackend 的区别体现了 Orchestrator-Worker 里的角色分工（讲义 Part 2.1）：
    Coder 负责"改到测试通过"，Reviewer 负责"独立验证结果是否可信"，两者工具集相同
    （同一个 sandbox），但决策逻辑（何时该做什么）完全不同——这正是"专精"的含义。
    """

    def __init__(self, script: MockCodingScript = MOVING_AVERAGE_SCRIPT):
        self.script = script

    def decide(self, system_prompt, messages, tools) -> AgentAction:
        called = [m["tool_call"]["name"] for m in messages
                  if m["role"] == "assistant" and m.get("tool_call")]

        if "read_file" not in called:
            return AgentAction(
                thought=f"审查 {self.script.target_file} 的实现，先看看 Coder 具体是怎么写的。",
                tool_name="read_file", tool_args={"path": self.script.target_file},
            )

        if "run_pytest" not in called:
            return AgentAction(
                thought="不直接相信 Coder 汇报的结果，独立运行一次 pytest 验证。",
                tool_name="run_pytest", tool_args={"path": "."},
            )

        last_obs = _last_observation(messages)
        passed = last_obs is not None and "passed" in last_obs and "failed" not in last_obs
        verdict = "✅ 通过" if passed else "❌ 未通过"
        return AgentAction(
            thought="复核完成，给出结论。",
            final_answer=f"[Reviewer 结论] {verdict}：{self.script.target_file} 的 pytest {'全部通过' if passed else '仍有失败用例'}。",
        )


# ──────────────────────────────────────────────────────────────────────────
# 真实 Claude 后端：Anthropic tool_use / tool_result 协议
# ──────────────────────────────────────────────────────────────────────────
class AnthropicBackend(LLMBackend):
    """接真实 Claude 的后端（需要 `pip install anthropic` 与 ANTHROPIC_API_KEY）。

    展示讲义 Part 3.2/3.3 的现代范式：工具的 JSON Schema 通过 `tools` 传给模型，
    模型用 `tool_use` 内容块返回结构化调用；工具结果用 `tool_result`（带匹配 id）回灌。
    """

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 1024):
        try:
            import anthropic  # lazy import：只有真正用真模型时才需要
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("使用 AnthropicBackend 需要：pip install anthropic") from e
        self._client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def decide(self, system_prompt, messages, tools) -> AgentAction:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            tools=to_anthropic_tools(tools),
            messages=to_anthropic_messages(messages),
        )
        thought_parts, tool_use = [], None
        for block in resp.content:
            if block.type == "text":
                thought_parts.append(block.text)
            elif block.type == "tool_use":
                tool_use = block
        thought = "\n".join(thought_parts).strip()
        if tool_use is not None:
            return AgentAction(
                thought=thought, tool_name=tool_use.name,
                tool_args=dict(tool_use.input), tool_call_id=tool_use.id,
            )
        return AgentAction(thought=thought, final_answer=thought)


# ──────────────────────────────────────────────────────────────────────────
# OpenAI 兼容后端：可指向 aihubmix 等中转端点
# ──────────────────────────────────────────────────────────────────────────
class OpenAICompatBackend(LLMBackend):
    """接 OpenAI 兼容的 Chat Completions 端点。

    项目默认指向腾讯云 lkeap（TokenHub 套餐），是项目统一的真实大模型入口：
      - base_url: https://api.lkeap.cloud.tencent.com/plan/v3
      - 默认模型: deepseek-v4-pro-202606
      - 认证: 标准 Bearer token（Authorization: Bearer {key}）
      - 已实测支持 function calling（tools 参数 + tool_calls 返回）
    文档：https://cloud.tencent.com/document/product/1823/130060

    用法：
        export LKEAP_API_KEY=sk-tp-xxx
        python demo_w9.py --backend lkeap
    """

    def __init__(
        self,
        model: str = "deepseek-v4-pro-202606",
        base_url: str = "https://api.lkeap.cloud.tencent.com/plan/v3",
        api_key: Optional[str] = None,
        api_key_env: str = "LKEAP_API_KEY",
        max_tokens: int = 1024,
    ):
        try:
            from openai import AsyncOpenAI, OpenAI  # lazy import
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("使用 OpenAICompatBackend 需要：pip install openai") from e
        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(
                f"未找到 API key：请设置环境变量 {api_key_env} 或传入 api_key 参数。"
            )
        # 同步/异步各一个客户端：demo_w9（单 Agent，sync run()）用 decide，
        # demo_w10（多 Worker 并发，async arun()）用 adecide，两者互不阻塞对方。
        self._client = OpenAI(base_url=base_url, api_key=key)
        self._async_client = AsyncOpenAI(base_url=base_url, api_key=key)
        self.model = model
        self.max_tokens = max_tokens

    def decide(self, system_prompt, messages, tools) -> AgentAction:
        resp = self._client.chat.completions.create(**self._request_kwargs(system_prompt, messages, tools))
        return self._parse_response(resp)

    async def adecide(self, system_prompt, messages, tools) -> AgentAction:
        """真正的异步网络调用（AsyncOpenAI），Week 10 多 Worker 并发调度时用。

        与默认的 `asyncio.to_thread` 兜底相比，这里发起的是原生异步 HTTP 请求：
        多个 Worker 同时调用时，等待网络响应的时间是重叠的，而不是排队占用线程池。
        """
        resp = await self._async_client.chat.completions.create(
            **self._request_kwargs(system_prompt, messages, tools)
        )
        return self._parse_response(resp)

    def _request_kwargs(self, system_prompt, messages, tools) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}] + to_openai_messages(messages),
            "tools": to_openai_tools(tools),
            "tool_choice": "auto",
            "max_tokens": self.max_tokens,
        }

    @staticmethod
    def _parse_response(resp) -> AgentAction:
        msg = resp.choices[0].message
        thought = msg.content or ""
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            return AgentAction(
                thought=thought,
                tool_name=tc.function.name,
                tool_args=json.loads(tc.function.arguments or "{}"),
                tool_call_id=tc.id,
            )
        return AgentAction(thought=thought, final_answer=thought)


# ──────────────────────────────────────────────────────────────────────────
# 协议校验后端：不花钱证明真后端可用
# ──────────────────────────────────────────────────────────────────────────
class StrictProtocolBackend(LLMBackend):
    """不调真 API，但严格模拟真实 API 的消息协议校验。

    工作方式：每次 decide 时，先把中性事件流翻译成目标厂商格式并做协议校验
    （不合法就像真 API 一样抛 ProtocolError），校验通过后再用 mock 状态机产出决策，
    并像真 API 那样为工具调用分配一个真实风格的 id。

    意义：如果 Agent 循环能跑通这个后端，就证明转换层产出的是**合法的多轮 tool 协议**，
    换上真实 Claude / aihubmix 也能跑通（因为协议一致）。这就是在没有 API key 的情况下
    证明"真后端不是绣花枕头"的办法。
    """

    def __init__(self, fmt: str = "openai", script: MockCodingScript = MOVING_AVERAGE_SCRIPT):
        self.fmt = fmt.lower()
        self._script = MockCodingBackend(script)

    def decide(self, system_prompt, messages, tools) -> AgentAction:
        if self.fmt == "openai":
            validate_openai_protocol(to_openai_messages(messages))
        elif self.fmt == "anthropic":
            validate_anthropic_protocol(to_anthropic_messages(messages))
        else:
            raise ValueError(f"未知协议格式：{self.fmt}")

        action = self._script.decide(system_prompt, messages, tools)
        if action.tool_name and not action.tool_call_id:
            action.tool_call_id = f"call_{uuid4().hex[:8]}"  # 模拟真 API 返回的调用 id
        return action


def build_backend(name: str = "mock", **kwargs) -> LLMBackend:
    """后端工厂。

    可选：mock（离线，默认） / lkeap|openai|deepseek（腾讯云 lkeap，项目默认真实后端）
          / anthropic（真 Claude） / strict（协议校验，不花钱验证真后端可用性）
    """
    name = name.lower()
    if name == "mock":
        return MockCodingBackend(**kwargs)
    if name in ("anthropic", "claude"):
        return AnthropicBackend(**kwargs)
    if name in ("openai", "lkeap", "deepseek", "tencent"):
        return OpenAICompatBackend(**kwargs)
    if name == "strict":
        return StrictProtocolBackend(**kwargs)
    raise ValueError(f"未知后端：{name}（可选 mock / lkeap / anthropic / strict）")
