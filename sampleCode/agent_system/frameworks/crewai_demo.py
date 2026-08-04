"""
Week 10 框架对照实现 · CrewAI 版本
========================================

运行（同样需要真实模型，见 langgraph_demo.py 文末"和 demo_w10.py 的区别"，此处同理）：
    conda activate agent_frameworks   # 见 requirements-frameworks.txt
    cd sampleCode/agent_system/frameworks
    python crewai_demo.py
    python crewai_demo.py --interactive   # HITL 改为真正等待终端确认

对应讲义：Week 10 · Part 2.1（Orchestrator-Worker）/ 2.2（HITL）/ 2.3（A2A 通信协议）
          Week 10 · Part 3.3（CrewAI：角色驱动的团队协作）
          Week 10 · 附录 A.6（sampleCode/agent_system 手搓版 Orchestrator-Worker）
          见 langgraph_demo.py（同一场景的 LangGraph 版本，可直接对照）

和 LangGraph 版本对照，三处关键差异都是 CrewAI 框架本身的设计决定的，不是我们选的：

| 机制         | LangGraph 版                              | CrewAI 版（本文件）                                            |
|--------------|---------------------------------------------|------------------------------------------------------------------|
| 依赖调度     | 图的拓扑结构 + BSP 自动并发独立分支          | 框架不负责跨任务并行——`Crew.kickoff_async()` 只管一个 Crew 内部按 `Process.sequential` 顺序执行；两个独立任务的并行完全靠我们在外层手写 `asyncio.gather`，和手搓版 Orchestrator 做法一致 |
| Coder→Reviewer 通信 | 共享 `messages` 状态（方案二） | `Task(context=[coder_task])`——显式声明"这个 Task 依赖哪个 Task 的输出"，框架只把被引用 Task 的**最终输出文本**注入，不共享中间过程（方案一：消息传递，和手搓版的 `AgentMessage` 是同一范式） |
| HITL         | 原生 `interrupt()`，工具调用粒度，可精确拦截单次 write_file/run_pytest | 原生 `human_input=True`，**任务粒度**——整个 Task 跑完、产出最终结果后才触发一次审批；拒绝时的效果也不同：不是把"拒绝"当一条 observation 塞回同一轮推理，而是**带着反馈让 Agent 重新跑一遍整个 Task** |

最后一行是本次对照里最值得记住的诚实结论：**CrewAI 没有内建"拦截单次工具调用"的机制**——它的角色化 API（Agent/Task/Crew）把粒度设计在"任务"这一层，这也解释了为什么它的 API 比 LangGraph 更简单（更少的控制点，也意味着更少的可控粒度）。工具级 HITL 如果一定要在 CrewAI 里做，得自己在 Tool 的 `_run()` 里手写拦截逻辑——等于把 LangGraph/手搓版已经做的事在框架的工具层再实现一遍，框架本身在这一点上没有提供现成支持。

角色隔离（Reviewer 拿不到 write_file 工具）在 CrewAI 里做法和 LangGraph 一致：按 Agent 只绑定允许的工具列表，是硬约束，不依赖 prompt 里的"请不要改代码"这种软约束。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import shutil
import sys
import time
from typing import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from core import FLATTEN_SCRIPT, MOVING_AVERAGE_SCRIPT, MockCodingScript, build_coding_registry  # noqa: E402
from core.hitl import auto_approve_with_log, interactive_approve  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from crewai import LLM, Agent, Crew, Process, Task
from crewai.core.providers.human_input import SyncHumanInputProvider, set_provider
from crewai.tools import tool as crewai_tool

CODER_TOOL_NAMES = ["list_dir", "read_file", "write_file", "run_pytest"]
REVIEWER_TOOL_NAMES = ["list_dir", "read_file", "run_pytest"]  # 硬约束：Reviewer 绑不到 write_file


class ApprovalHumanInputProvider(SyncHumanInputProvider):
    """把 CrewAI 原生的 human_input 流程接到我们自己的 approve_cb，
    这样非交互模式下不会卡在 CrewAI 默认的阻塞 input()。

    注意这里审批的是**整个 Task 的最终输出**（任务粒度），不是某一次工具调用——
    这正是本文件想诚实呈现的 CrewAI HITL 局限，见模块 docstring。
    """

    def __init__(self, approve_cb: Callable[[str, dict], bool]):
        self.approve_cb = approve_cb

    def handle_feedback(self, formatted_answer, context):
        output = self._get_output_string(formatted_answer)
        approved = self.approve_cb("task_output_review", {"output": output[:120]})
        context.ask_for_human_input = False
        if approved:
            return formatted_answer
        context.messages.append(context._format_feedback_message(
            "复核未通过：请重新检查实现（尤其是边界条件），确保调用测试工具验证后再给结论。"
        ))
        return context._invoke_loop()

    async def handle_feedback_async(self, formatted_answer, context):
        output = self._get_output_string(formatted_answer)
        approved = await asyncio.to_thread(self.approve_cb, "task_output_review", {"output": output[:120]})
        context.ask_for_human_input = False
        if approved:
            return formatted_answer
        context.messages.append(context._format_feedback_message(
            "复核未通过：请重新检查实现（尤其是边界条件），确保调用测试工具验证后再给结论。"
        ))
        return await context._ainvoke_loop()


def make_crewai_tools(sandbox: pathlib.Path) -> dict:
    """把 core/tools.py 的 CodingToolkit 方法包成 CrewAI 工具，按名字取用。"""
    registry, kit = build_coding_registry(sandbox)
    return {spec["name"]: crewai_tool(spec["name"])(getattr(kit, spec["name"])) for spec in registry.specs()}


def build_task_crew(llm: LLM, sandbox: pathlib.Path, script: MockCodingScript) -> Crew:
    tools = make_crewai_tools(sandbox)

    coder = Agent(
        role="Coder",
        goal=f"在 {script.target_file} 中实现目标函数，确保通过 {script.test_file} 的全部测试",
        backstory="严谨的编程助手，不凭空假设文件内容，先读后写；测试失败时根据报错定位并修复。",
        tools=[tools[n] for n in CODER_TOOL_NAMES],
        llm=llm, verbose=True,
    )
    reviewer = Agent(
        role="Reviewer",
        goal="独立复核 Coder 的实现是否真的通过测试",
        backstory="不修改任何代码（也确实拿不到 write_file 工具），不轻信上游说法，凡事自己验证一遍。",
        tools=[tools[n] for n in REVIEWER_TOOL_NAMES],
        llm=llm, verbose=True,
    )

    coder_task = Task(
        description=f"在 {script.target_file} 中实现目标函数，确保通过 {script.test_file} 的全部测试。",
        expected_output="简洁说明实现思路，以及 pytest 是否已经全部通过。",
        agent=coder,
        human_input=True,  # 任务级 HITL：整份实现产出后需要"批准"才算完成这个 Task
    )
    reviewer_task = Task(
        description=f"独立复核 {script.target_file} 的实现是否真的通过测试。"
                    "必须自己调用 run_pytest 验证一遍，不能仅凭上游消息里的说法下结论。",
        expected_output="结论必须明确写出 PASS 或 FAIL，并说明依据。",
        agent=reviewer,
        context=[coder_task],  # A2A：显式声明依赖，框架自动把 coder_task 的最终输出注入这里
    )

    return Crew(
        agents=[coder, reviewer],
        tasks=[coder_task, reviewer_task],
        process=Process.sequential,
        verbose=True,
        tracing=False,  # 避免非交互环境下弹出 tracing 偏好确认
    )


TEST_MOVING_AVERAGE = '''\
from math_utils import moving_average


def test_basic():
    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]


def test_window_equals_len():
    assert moving_average([2, 4, 6], 3) == [4.0]


def test_window_larger_than_len():
    assert moving_average([1, 2], 5) == []
'''

TEST_FLATTEN = '''\
from list_utils import flatten_list


def test_flat():
    assert flatten_list([1, 2, 3]) == [1, 2, 3]


def test_one_level_nested():
    assert flatten_list([1, [2, 3], 4]) == [1, 2, 3, 4]


def test_deeply_nested():
    assert flatten_list([1, [2, [3, [4, 5]], 6]]) == [1, 2, 3, 4, 5, 6]
'''


def prepare_sandbox(root: pathlib.Path, test_file: str, test_content: str) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / test_file).write_text(test_content, encoding="utf-8")


async def main_async(args: argparse.Namespace) -> int:
    base = pathlib.Path(__file__).resolve().parent / "workspace_crewai"
    sandbox_a, sandbox_b = base / "task_a", base / "task_b"
    prepare_sandbox(sandbox_a, MOVING_AVERAGE_SCRIPT.test_file, TEST_MOVING_AVERAGE)
    prepare_sandbox(sandbox_b, FLATTEN_SCRIPT.test_file, TEST_FLATTEN)

    llm = LLM(
        model="deepseek/deepseek-v4-pro-202606",
        base_url="https://api.lkeap.cloud.tencent.com/plan/v3",
        api_key=os.environ["LKEAP_API_KEY"],
    )
    approve_cb = interactive_approve if args.interactive else auto_approve_with_log
    set_provider(ApprovalHumanInputProvider(approve_cb))

    crew_a = build_task_crew(llm, sandbox_a, MOVING_AVERAGE_SCRIPT)
    crew_b = build_task_crew(llm, sandbox_b, FLATTEN_SCRIPT)

    print("🧭 CrewAI：2 个独立 Crew，asyncio.gather 并发跑（跨 Crew 并行是我们外层写的，非框架自带）")
    print("─" * 60)
    start = time.perf_counter()
    results = await asyncio.gather(crew_a.kickoff_async(), crew_b.kickoff_async())
    elapsed = time.perf_counter() - start

    print("\n" + "═" * 60)
    print(f"📊 两个 Crew 并发总耗时：{elapsed:.2f}s（对照 demo_w10.py / langgraph_demo.py 里同样两个任务的耗时数据）")
    for task_id, result in zip(("a", "b"), results):
        print(f"  [{task_id}] {str(result)[:300]}")
    print("═" * 60)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Week 10 框架对照：CrewAI 版 Orchestrator-Worker")
    parser.add_argument("--interactive", action="store_true", help="HITL 改为真正等待终端输入")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
