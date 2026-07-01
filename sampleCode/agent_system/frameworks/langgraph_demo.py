"""
Week 10 框架对照实现 · LangGraph 版本
========================================

运行（需要真实模型，这份对照实现没有 mock 后端——见文末"和 demo_w10.py 的区别"）：
    conda create -n agent_frameworks python=3.11 -y   # 首次，独立环境见 requirements-frameworks.txt
    conda activate agent_frameworks
    pip install -r requirements-frameworks.txt
    cd sampleCode/agent_system/frameworks
    python langgraph_demo.py
    python langgraph_demo.py --interactive   # HITL 改为真正等待终端确认

对应讲义：Week 10 · Part 2.1（Orchestrator-Worker）/ 2.2（HITL）/ 2.3（A2A 通信协议）
          Week 10 · Part 3.1（LangGraph：图状态机范式）
          Week 10 · 附录 A.1（LangGraph 的 Persistence 与 Time-Travel）
          Week 10 · 附录 A.6（sampleCode/agent_system 手搓版 Orchestrator-Worker）

和手搓版（core/orchestrator.py）解决的是同一个问题——两个独立的"改代码通过 pytest"
子任务（moving_average / flatten_list），各自先 Coder 实现，再 Reviewer 复核——但调度、
通信、审批全部换成 LangGraph 原生机制，方便直接对照：

| 机制         | 手搓版（core/orchestrator.py）             | LangGraph 版（本文件）                                    |
|--------------|---------------------------------------------|------------------------------------------------------------|
| 依赖调度     | 手写 wave 循环，反复找"依赖已完成"的子任务集合 | 图的拓扑结构本身 + BSP 并发执行（同一 superstep 内的独立分支自动并发） |
| Coder→Reviewer 通信 | 显式 `AgentMessage`（方案一：消息传递） | 共享的 `messages` 状态通道（方案二：共享状态——Reviewer 能看到 Coder 的完整过程，不只是结论） |
| HITL         | 手写 `HITLGate.check()`，同步拦截           | 原生 `interrupt()` + `Command(resume=...)` + checkpointer（附录 A.1 讨论的持久化机制） |
| 角色隔离     | 靠 prompt 里的 role_instructions（软约束）  | 靠 `bind_tools()` 按角色只绑定允许的工具（硬约束：Reviewer 的 LLM 调用里根本不存在 write_file 这个选项） |

一个诚实的观察：LangGraph 的默认习惯用法就是共享状态（一条 `messages` 列表，用
`add_messages` reducer 累加），要故意做成消息传递反而要跟框架"拧着来"——这印证了
Part 2.3 的判断："代表框架：LangGraph（图中的 State 对象）"。框架的默认路径会
悄悄替你做出架构选择，这也是选框架本身就是一次架构决策的原因。

和 demo_w10.py 的区别：这份对照实现没有 mock/strict 离线后端。LangGraph 的核心
卖点就是"框架帮你编排真实模型的决策"，硬套一个规则状态机会让对照失去意义，所以
这里必须配置 `LKEAP_API_KEY`（同 `sampleCode/agent_system/.env`）才能跑。
"""

from __future__ import annotations

import argparse
import asyncio
import operator
import pathlib
import shutil
import sys
import time
from typing import Annotated, Literal, TypedDict

# 复用 Week 9/10 已有的沙箱工具与任务剧本，保证"同样的工具、同样的任务，只换编排层"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from core import FLATTEN_SCRIPT, MOVING_AVERAGE_SCRIPT, MockCodingScript, build_coding_registry  # noqa: E402
from core.hitl import auto_approve_with_log, interactive_approve  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

GATED_TOOLS = {"write_file", "run_pytest"}
REQUIRE_EVERY_TIME = {"write_file"}

CODER_TOOL_NAMES = ["list_dir", "read_file", "write_file", "run_pytest"]
REVIEWER_TOOL_NAMES = ["list_dir", "read_file", "run_pytest"]  # 硬约束：Reviewer 绑不到 write_file


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    phase: Literal["coder", "reviewer", "done"]
    approved: Annotated[list[str], operator.add]  # 累加"已批准过的能力"（能力等级传递）


def make_tools(sandbox: pathlib.Path) -> dict[str, StructuredTool]:
    """把 core/tools.py 的 CodingToolkit 方法包成 LangChain 工具，按名字取用。"""
    registry, kit = build_coding_registry(sandbox)
    tools = {}
    for spec in registry.specs():
        tools[spec["name"]] = StructuredTool.from_function(
            func=getattr(kit, spec["name"]),
            name=spec["name"],
            description=spec["description"],
        )
    return tools


def build_task_graph(llm: ChatOpenAI, checkpointer: InMemorySaver):
    """构建一份可复用的 Coder→Reviewer 图。sandbox/script 通过 config["configurable"] 传入，
    让同一份编译好的图能同时服务两个独立任务（task_a / task_b），只是 invoke 时配置不同。
    审批回调（approve_cb）不在这里——interrupt() 只负责暂停，谁来批准是 run_task 的事。
    """
    all_tools_by_task: dict[str, dict[str, StructuredTool]] = {}

    def get_tools(task_id: str, sandbox: pathlib.Path) -> dict[str, StructuredTool]:
        if task_id not in all_tools_by_task:
            all_tools_by_task[task_id] = make_tools(sandbox)
        return all_tools_by_task[task_id]

    def system_message(phase: str, script: MockCodingScript) -> SystemMessage:
        if phase == "coder":
            text = (
                f"你是 Coder，任务：在 {script.target_file} 中实现目标函数，"
                f"确保通过 {script.test_file} 的全部测试。不要凭空假设文件内容，先读后写；"
                "测试失败时根据报错定位并修复，不要放弃。全部通过后不再调用任何工具，"
                "直接用文字给出简洁总结。"
            )
        else:
            text = (
                "你是 Reviewer，不能修改任何代码（你也确实拿不到 write_file 工具）。"
                "不能仅凭上游 Coder 的说法下结论，必须自己调用 run_pytest 独立验证一遍，"
                "再给出「通过/不通过」的结论（结论里请明确写出 PASS 或 FAIL）。"
            )
        return SystemMessage(content=text)

    def call_model(state: GraphState, config) -> dict:
        task_id = config["configurable"]["task_id"]
        sandbox = config["configurable"]["sandbox"]
        script = config["configurable"]["script"]
        tools = get_tools(task_id, sandbox)

        phase = state["phase"]
        allowed_names = CODER_TOOL_NAMES if phase == "coder" else REVIEWER_TOOL_NAMES
        bound = llm.bind_tools([tools[n] for n in allowed_names])

        messages = state["messages"]
        # 每个 phase 开始时补一条角色说明；phase 切换后旧的 system 消息仍留在历史里，
        # 但最新这条会排在最后，模型以最近的角色定位为准（LangGraph 共享状态的直接体现）。
        ai_msg: AIMessage = bound.invoke([system_message(phase, script), *messages])

        update: dict = {"messages": [ai_msg]}
        if not ai_msg.tool_calls:
            if phase == "coder":
                update["phase"] = "reviewer"
                update["messages"] = [ai_msg, HumanMessage(
                    content="（Orchestrator 提示）Coder 阶段结束，切换到 Reviewer 角色，"
                            "请独立复核上面的实现。"
                )]
            else:
                update["phase"] = "done"
        return update

    def call_tools(state: GraphState, config) -> dict:
        task_id = config["configurable"]["task_id"]
        sandbox = config["configurable"]["sandbox"]
        tools = get_tools(task_id, sandbox)

        last: AIMessage = state["messages"][-1]
        approved_now: list[str] = []
        tool_messages: list[ToolMessage] = []

        for call in last.tool_calls:
            name, args, call_id = call["name"], call["args"], call["id"]

            # 工具不存在/参数不对 → 回灌可读错误让模型自我纠正，而不是崩溃整张图
            # （同 core/tools.py ToolRegistry.execute 的可靠性工程原则，Part 3.6）。
            if name not in tools:
                tool_messages.append(ToolMessage(
                    content=f"[错误] 不存在名为 '{name}' 的工具。可用工具：{list(tools)}",
                    tool_call_id=call_id,
                ))
                continue

            if name in GATED_TOOLS:
                capability_key = f"{task_id}:{name}"
                already = capability_key in state["approved"] or capability_key in approved_now
                if name in REQUIRE_EVERY_TIME or not already:
                    # 原生 interrupt()：暂停整张图，等待 Command(resume=...) 恢复——
                    # 对应附录 A.1 的 checkpointer 持久化机制，而不是手搓版同步 check() 的写法。
                    approved = interrupt({"task": task_id, "tool": name, "args": args})
                    if not approved:
                        tool_messages.append(ToolMessage(
                            content=f"[HITL] 用户拒绝了 {name} 的执行请求（参数：{args}）。",
                            tool_call_id=call_id,
                        ))
                        continue
                    if name not in REQUIRE_EVERY_TIME:
                        approved_now.append(capability_key)

            try:
                observation = tools[name].func(**args)
            except TypeError as e:
                observation = f"[错误] 调用 {name} 的参数不正确：{e}"
            except Exception as e:  # noqa: BLE001 - 故意兜底：错误要回灌给模型
                observation = f"[错误] 执行 {name} 时出现异常：{type(e).__name__}: {e}"
            tool_messages.append(ToolMessage(content=str(observation), tool_call_id=call_id))

        return {"messages": tool_messages, "approved": approved_now}

    def route_after_agent(state: GraphState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END if state["phase"] == "done" else "agent"

    graph = StateGraph(GraphState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "agent": "agent", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


async def run_task(compiled_graph, task_id: str, sandbox: pathlib.Path, script: MockCodingScript, approve_cb) -> dict:
    config = {"configurable": {"thread_id": task_id, "task_id": task_id, "sandbox": sandbox, "script": script}}
    initial: GraphState = {"messages": [], "phase": "coder", "approved": []}

    result = await compiled_graph.ainvoke(initial, config)
    while "__interrupt__" in result:
        req = result["__interrupt__"][0].value
        print(f"    🔒 [HITL/{task_id}] 检测到风险操作 {req['tool']}（参数：{_brief(req['args'])}）")
        approved = approve_cb(req["tool"], req["args"])
        result = await compiled_graph.ainvoke(Command(resume=approved), config)

    final_texts = [m.content for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls]
    return {"task_id": task_id, "final_answer": final_texts[-1] if final_texts else "", "n_messages": len(result["messages"])}


def _brief(args: dict) -> str:
    parts = []
    for k, v in args.items():
        s = str(v)
        parts.append(f"{k}={s[:30]!r}" if len(s) > 30 else f"{k}={s!r}")
    return ", ".join(parts)


def prepare_sandbox(root: pathlib.Path, test_file: str, test_content: str) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / test_file).write_text(test_content, encoding="utf-8")


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


async def main_async(args: argparse.Namespace) -> int:
    import os
    base = pathlib.Path(__file__).resolve().parent / "workspace_langgraph"
    sandbox_a, sandbox_b = base / "task_a", base / "task_b"
    prepare_sandbox(sandbox_a, MOVING_AVERAGE_SCRIPT.test_file, TEST_MOVING_AVERAGE)
    prepare_sandbox(sandbox_b, FLATTEN_SCRIPT.test_file, TEST_FLATTEN)

    llm = ChatOpenAI(
        model="deepseek-v4-pro-202606",
        base_url="https://api.lkeap.cloud.tencent.com/plan/v3",
        api_key=os.environ["LKEAP_API_KEY"],
    )
    approve_cb = interactive_approve if args.interactive else auto_approve_with_log
    checkpointer = InMemorySaver()
    compiled = build_task_graph(llm, checkpointer)

    print("🧭 LangGraph Orchestrator 启动：2 个独立任务图，asyncio.gather 并发跑")
    print("─" * 60)
    start = time.perf_counter()
    results = await asyncio.gather(
        run_task(compiled, "a", sandbox_a, MOVING_AVERAGE_SCRIPT, approve_cb),
        run_task(compiled, "b", sandbox_b, FLATTEN_SCRIPT, approve_cb),
    )
    elapsed = time.perf_counter() - start

    print("\n" + "═" * 60)
    print(f"📊 两个任务并发总耗时：{elapsed:.2f}s（对照 demo_w10.py 里同样两个任务的耗时数据）")
    for r in results:
        print(f"  [{r['task_id']}] {r['n_messages']} 条消息 | 结论：{r['final_answer'][:200]}")
    print("═" * 60)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Week 10 框架对照：LangGraph 版 Orchestrator-Worker")
    parser.add_argument("--interactive", action="store_true", help="HITL 改为真正等待终端输入")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
