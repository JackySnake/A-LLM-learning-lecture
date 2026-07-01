"""
Orchestrator-Worker 多智能体引擎（Week 10）
================================================

对应讲义：Week 10 · Part 2.1「Orchestrator-Worker 架构」
          Week 10 · Part 2.3「Agent-to-Agent 通信协议」
          Week 10 · Part 2.4「任务分解的挑战：依赖图规划」

在 Week 9 的单个 ReActAgent 之上，这里加一层"协调者"：把一份子任务列表（Subtask，
带依赖关系）按依赖图分批（wave）调度——同一 wave 内彼此没有依赖的子任务用
`asyncio.gather` + `Semaphore` 真正并发执行，不同 wave 之间按依赖顺序串行。

两个关键设计选择，呼应讲义 Part 2.3 的方案对比：

1. **消息传递而非共享状态**：下游子任务不会直接读上游 Worker 的内部状态或工作记忆，
   而是通过显式的 AgentMessage（sender/receiver/content）接收上游的 final_answer。
   这是讲义 Part 2.3 的**方案一（Message Passing）**；Week 10 附录 A.2 的教学示例
   用的是一个共享的 `results` 字典，属于**方案二（Shared State）**——两种范式在这里
   刻意做出不同选择，便于对照。

2. **任务分解本身是硬编码的**：Orchestrator 不调用 LLM 来生成 Subtask 列表（那是
   附录 A.2 展示的做法），而是由 demo_w10.py 直接写死子任务图。这保持了 Week 9 定下的
   "固定任务场景，离线可复现"的教学取舍——我们要验证的是调度/并发/HITL/消息传递机制
   本身是否正确，不是 LLM 的任务分解能力。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from .agent import AgentResult, ReActAgent


@dataclass
class Subtask:
    """一个可分配给某个 Worker 角色的子任务。"""

    id: str
    role: str
    task: str
    depends_on: list[str] = field(default_factory=list)


@dataclass
class AgentMessage:
    """Worker 间的 A2A 消息（消息传递范式，而非共享状态，见模块 docstring）。"""

    sender: str
    receiver: str
    content: str


@dataclass
class WaveResult:
    """一批（wave）并行执行的子任务与耗时，用于串行/并行耗时对比。"""

    subtask_ids: list[str]
    elapsed_seconds: float


class Orchestrator:
    """接收子任务列表，按依赖关系分波（wave）并发调度给对应角色的 Worker。"""

    def __init__(
        self,
        workers: dict[str, ReActAgent],
        max_concurrency: int = 4,
        verbose: bool = True,
    ):
        self.workers = workers
        self.max_concurrency = max_concurrency
        self.verbose = verbose

    async def run(self, subtasks: list[Subtask]) -> tuple[dict[str, AgentResult], list[WaveResult]]:
        """执行整个依赖图，返回每个子任务的结果，以及每一波（wave）的调度记录。"""
        pending = {s.id: s for s in subtasks}
        done: dict[str, AgentResult] = {}
        inbox: dict[str, list[AgentMessage]] = {s.id: [] for s in subtasks}
        waves: list[WaveResult] = []
        semaphore = asyncio.Semaphore(self.max_concurrency)

        while pending:
            # 一个 wave = 当前依赖已全部完成、可以立刻开始的子任务集合
            ready = [s for s in pending.values() if all(d in done for d in s.depends_on)]
            if not ready:
                raise RuntimeError(f"检测到无法调度的依赖（可能存在循环依赖）：{list(pending)}")

            self._log(f"\n🌊 Wave 开始：{[s.id for s in ready]}（并发上限={self.max_concurrency}）")
            start = time.perf_counter()
            results = await asyncio.gather(*(self._run_one(s, inbox, semaphore) for s in ready))
            elapsed = time.perf_counter() - start
            waves.append(WaveResult(subtask_ids=[s.id for s in ready], elapsed_seconds=elapsed))
            self._log(f"🌊 Wave 结束：耗时 {elapsed:.2f}s")

            for subtask, result in zip(ready, results):
                done[subtask.id] = result
                del pending[subtask.id]
                # 把结果作为 A2A 消息投递给依赖它的下游子任务（消息传递，非共享状态）
                for other in pending.values():
                    if subtask.id in other.depends_on:
                        inbox[other.id].append(AgentMessage(
                            sender=subtask.id,
                            receiver=other.id,
                            content=result.final_answer or "(无最终答复)",
                        ))
        return done, waves

    async def _run_one(
        self,
        subtask: Subtask,
        inbox: dict[str, list[AgentMessage]],
        semaphore: asyncio.Semaphore,
    ) -> AgentResult:
        async with semaphore:
            worker = self.workers[subtask.role]
            task_text = subtask.task
            messages = inbox.get(subtask.id) or []
            if messages:
                context = "\n".join(f"- 来自 {m.sender} 的消息：{m.content}" for m in messages)
                task_text = f"{task_text}\n\n【上游 Worker 传来的消息】\n{context}"
            self._log(f"  🚀 [{subtask.id}/{subtask.role}] 开始：{subtask.task[:50]}")
            result = await worker.arun(task_text)
            self._log(f"  🏁 [{subtask.id}/{subtask.role}] 结束（{'成功' if result.succeeded else '未完成'}）")
            return result

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
