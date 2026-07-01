"""
ReAct Agent 引擎（Week 9 地基的总装车间）
============================================

对应讲义：Week 9 · Part 2「ReAct——推理与行动的统一」
          Week 9 · Part 1.3「Agent 的认知循环」

这是把前面三块（LLM 后端 / 工具 / 记忆）组装成一个**单 Agent** 的地方。
它实现了 Agent 最核心的认知循环：

    ┌─────────────────────────────────────────────┐
    │  Reason  →  Act  →  Observe  →  (回到 Reason) │
    └─────────────────────────────────────────────┘

  1. Reason ：调用 LLM 后端，产出"思考 + 下一步动作"（AgentAction）
  2. Act    ：若动作是工具调用，交给 ToolRegistry 执行
  3. Observe：把工具返回（observation）写回工作记忆，进入下一轮
  4. 终止   ：模型给出 final_answer，或达到 max_steps 上限

这个循环就是讲义里"线性 CoT 不足以支撑 Agent"的答案：CoT 只推理一次，
而 ReAct 让每一步行动的真实结果反过来修正后续推理（demo 里"测试失败→修复"即此）。

Week 10 在这个单 Agent 之上长出了 Orchestrator-Worker 多智能体（arun）；
Week 11 把工具换成了 MCP（core/mcp_backend.py，execute 变成 async，见 arun 里的
inspect.iscoroutinefunction 判断）、加上了审计日志（core/audit.py）与评估 harness
（core/eval.py）。接口都保持稳定。
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Awaitable, Callable, cast

from .audit import AuditLog
from .hitl import HITLGate
from .llm_backend import AgentAction, LLMBackend
from .memory import MemorySystem
from .tools import ToolSource


@dataclass
class StepTrace:
    """一步的完整记录，用于事后审计与 trajectory 评估（Week 11 会用到）。"""

    index: int
    thought: str
    tool_name: str | None
    tool_args: dict
    observation: str | None
    tool_call_id: str | None = None


@dataclass
class AgentResult:
    task: str
    final_answer: str | None
    steps: list[StepTrace] = field(default_factory=list)
    succeeded: bool = False

    def trajectory_summary(self) -> str:
        """把轨迹压成一句话，用于写入情节记忆。"""
        actions = [s.tool_name for s in self.steps if s.tool_name]
        return f"{len(self.steps)} 步，工具序列：{' → '.join(actions) if actions else '（无）'}"


class ReActAgent:
    """单个 ReAct Agent：大脑（backend）+ 双手（tools）+ 记忆（memory）。"""

    SYSTEM_TEMPLATE = (
        "你是一个严谨的编程助手 Agent，遵循「思考→行动→观察」的循环来完成任务。\n"
        "你可以使用以下工具：\n{tools}\n\n"
        "工作规则：\n"
        "1. 每一步先思考，再选择最合适的一个工具；不要凭空假设文件内容，先读后写。\n"
        "2. 修改代码后必须运行测试验证；测试失败时，根据报错定位并修复，而不是放弃。\n"
        "3. 所有测试通过后，给出简洁的最终答复。\n"
        "{memory}"
    )

    def __init__(
        self,
        backend: LLMBackend,
        tools: ToolSource,
        memory: MemorySystem | None = None,
        max_steps: int = 12,
        verbose: bool = True,
        approval: HITLGate | None = None,
        role_instructions: str = "",
        audit: AuditLog | None = None,
    ):
        self.backend = backend
        self.tools = tools
        self.memory = memory or MemorySystem()
        self.max_steps = max_steps
        self.verbose = verbose
        self.approval = approval  # Week 10：风险工具（如 run_pytest/write_file）执行前的人工审批钩子
        # Week 10：Orchestrator-Worker 场景下，同一份 SYSTEM_TEMPLATE 需要区分角色
        # （如 Coder vs Reviewer），真实后端（lkeap 等）靠这段文字理解自己的分工；
        # mock/strict 后端不读 system_prompt，分工由各自的状态机硬编码决定。
        self.role_instructions = role_instructions
        self.audit = audit  # Week 11：Trust by Design 原则三——高风险操作要有审计轨迹

    def _build_system_prompt(self, task: str) -> str:
        memory_ctx = self.memory.build_memory_context(task)
        memory_block = f"\n相关记忆：\n{memory_ctx}\n" if memory_ctx else ""
        system_prompt = self.SYSTEM_TEMPLATE.format(
            tools=self.tools.describe_for_prompt(),
            memory=memory_block,
        )
        if self.role_instructions:
            system_prompt += f"\n你在本次任务中的角色定位：{self.role_instructions}\n"
        return system_prompt

    # ──────────────────────────────────────────────────────────────────
    def run(self, task: str) -> AgentResult:
        """执行一个任务，跑完整的 ReAct 循环。"""
        # 1) 检索长期记忆，拼进 system prompt（RAG+记忆增强）
        system_prompt = self._build_system_prompt(task)

        # 2) 任务作为工作记忆的第一条
        self.memory.working.clear()
        self.memory.working.add("user", task)

        result = AgentResult(task=task, final_answer=None)
        self._log(f"\n🎯 任务：{task}\n" + "─" * 60)

        # 3) 认知循环
        for step_idx in range(1, self.max_steps + 1):
            action: AgentAction = self.backend.decide(
                system_prompt=system_prompt,
                messages=self.memory.working.view(),
                tools=self.tools.specs(),
            )

            # —— 终止：模型给出最终答案 ——
            if action.is_final:
                self._log(f"\n💭 [{step_idx}] {action.thought}")
                self._log(f"✅ 最终答复：{action.final_answer}")
                result.final_answer = action.final_answer
                result.succeeded = True
                result.steps.append(StepTrace(step_idx, action.thought, None, {}, None))
                # 把 assistant 的收尾写入工作记忆（最终答复，无 tool_call）
                self.memory.working.add("assistant", thought=action.thought,
                                        final_answer=action.final_answer or "")
                if self.audit:
                    self.audit.record(step=step_idx, thought=action.thought,
                                       final_answer=action.final_answer)
                break

            # —— Reason + Act：记录思考、执行工具 ——
            assert action.tool_name is not None  # 契约：非 final 动作必须带 tool_name
            self._log(f"\n💭 [{step_idx}] {action.thought}")
            self._log(f"🔧 调用：{action.tool_name}({_fmt_args(action.tool_args)})")

            # 工具调用 id：真后端来自 API 响应；否则兜底生成（保证多轮协议可配对）
            call_id = action.tool_call_id or f"call_{step_idx}"

            # assistant 决策入工作记忆：结构化的 tool_call（带 id），而非纯文本注入
            self.memory.working.add(
                "assistant", thought=action.thought,
                tool_call={"id": call_id, "name": action.tool_name, "args": action.tool_args},
            )

            # —— Observe：先过 HITL 审批，通过才执行，用同一个 id 回灌观察 ——
            # run() 是同步方法，只支持同步的 ToolSource（如 ToolRegistry）；
            # 异步的 MCPToolRegistry 请配合 arun() 使用（见下方 inspect.iscoroutinefunction 判断）。
            rejection = self.approval.check(action.tool_name, action.tool_args) if self.approval else None
            observation = rejection or cast(str, self.tools.execute(action.tool_name, action.tool_args))
            self._log(f"👁  观察：{_truncate(observation)}")
            self.memory.working.add("tool", observation, tool_call_id=call_id)
            if self.audit:
                self.audit.record(step=step_idx, thought=action.thought, tool_name=action.tool_name,
                                   tool_args=action.tool_args, observation=observation,
                                   tool_call_id=call_id, approved=(rejection is None))

            result.steps.append(StepTrace(
                step_idx, action.thought, action.tool_name, action.tool_args, observation,
                tool_call_id=call_id,
            ))
        else:
            # for 循环正常结束（没 break）→ 超出步数上限
            self._log(f"\n⚠️  达到最大步数 {self.max_steps}，任务未完成。")

        # 4) 把本次任务写入情节记忆，供未来类似任务复用经验
        self.memory.episodic.record(task, result.trajectory_summary(), result.succeeded)
        return result

    # ──────────────────────────────────────────────────────────────────
    async def arun(self, task: str) -> AgentResult:
        """`run` 的异步版本：Week 10 Orchestrator 并发调度多个 Worker 时使用。

        差异只有两处：① 用 `await self.backend.adecide(...)`，让多个 Worker 的 LLM
        调用能在网络等待期间真正并发；② 工具执行——如果 `self.tools.execute` 是协程函数
        （Week 11 的 MCPToolRegistry 就是，走真实的异步 IPC），直接 `await` 它；如果是
        普通同步函数（Week 9/10 的 ToolRegistry，内部可能跑阻塞的 subprocess），丢进
        线程池 `asyncio.to_thread`，避免卡住事件循环。循环结构和 HITL 审批逻辑与 `run`
        保持一致——两个方法刻意不共享一个大函数，因为 sync/async 的控制流本质不同，
        硬拆共享会引入比重复代码更难读的抽象。
        """
        system_prompt = self._build_system_prompt(task)

        self.memory.working.clear()
        self.memory.working.add("user", task)

        result = AgentResult(task=task, final_answer=None)
        self._log(f"\n🎯 任务：{task}\n" + "─" * 60)

        for step_idx in range(1, self.max_steps + 1):
            action: AgentAction = await self.backend.adecide(
                system_prompt=system_prompt,
                messages=self.memory.working.view(),
                tools=self.tools.specs(),
            )

            if action.is_final:
                self._log(f"\n💭 [{step_idx}] {action.thought}")
                self._log(f"✅ 最终答复：{action.final_answer}")
                result.final_answer = action.final_answer
                result.succeeded = True
                result.steps.append(StepTrace(step_idx, action.thought, None, {}, None))
                self.memory.working.add("assistant", thought=action.thought,
                                        final_answer=action.final_answer or "")
                if self.audit:
                    self.audit.record(step=step_idx, thought=action.thought,
                                       final_answer=action.final_answer)
                break

            assert action.tool_name is not None
            self._log(f"\n💭 [{step_idx}] {action.thought}")
            self._log(f"🔧 调用：{action.tool_name}({_fmt_args(action.tool_args)})")

            call_id = action.tool_call_id or f"call_{step_idx}"
            self.memory.working.add(
                "assistant", thought=action.thought,
                tool_call={"id": call_id, "name": action.tool_name, "args": action.tool_args},
            )

            # 审批检查也丢进线程池：interactive_approve 内部会阻塞在 input()，
            # 如果直接同步调用会卡住整个事件循环，冻结其他正在并发执行的 Worker。
            rejection = (
                await asyncio.to_thread(self.approval.check, action.tool_name, action.tool_args)
                if self.approval else None
            )
            if rejection:
                observation = rejection
            elif inspect.iscoroutinefunction(self.tools.execute):
                coro = cast(Awaitable[str], self.tools.execute(action.tool_name, action.tool_args))
                observation = await coro
            else:
                sync_execute = cast("Callable[[str, dict], str]", self.tools.execute)
                observation = await asyncio.to_thread(sync_execute, action.tool_name, action.tool_args)
            self._log(f"👁  观察：{_truncate(observation)}")
            self.memory.working.add("tool", observation, tool_call_id=call_id)
            if self.audit:
                self.audit.record(step=step_idx, thought=action.thought, tool_name=action.tool_name,
                                   tool_args=action.tool_args, observation=observation,
                                   tool_call_id=call_id, approved=(rejection is None))

            result.steps.append(StepTrace(
                step_idx, action.thought, action.tool_name, action.tool_args, observation,
                tool_call_id=call_id,
            ))
        else:
            self._log(f"\n⚠️  达到最大步数 {self.max_steps}，任务未完成。")

        self.memory.episodic.record(task, result.trajectory_summary(), result.succeeded)
        return result

    # ──────────────────────────────────────────────────────────────────
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


def _fmt_args(args: dict) -> str:
    """把工具参数压成可读短串（长内容如文件正文做截断）。"""
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 40:
            s = s[:37] + "..."
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)


def _truncate(text: str, limit: int = 200) -> str:
    text = text.replace("\n", " ⏎ ")
    return text if len(text) <= limit else text[:limit] + " …(略)"
