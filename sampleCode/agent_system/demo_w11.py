"""
Week 11 演示：MCP 封装 + 评估 harness
========================================

运行（默认离线 mock，无需任何 API key）：
    source /Users/jackymm/miniforge3/bin/activate claude_p13
    cd sampleCode/agent_system
    python demo_w11.py

切换到真实大模型：
    python demo_w11.py --backend lkeap      # 腾讯云 lkeap / deepseek-v4-pro（需 .env 配置 LKEAP_API_KEY）
    python demo_w11.py --backend anthropic  # 真实 Claude

这个 demo 在 Week 9 单 Agent 的基础上长出三个新能力：
    1. MCP 工具封装（mcp_server.py + core/mcp_backend.py）：Agent 通过真实的 MCP
       stdio 子进程调用工具，而不是进程内函数调用——ReActAgent 完全不用改代码，
       Week 10 给 `arun()` 加的 `inspect.iscoroutinefunction` 判断已经把这条路铺好了。
    2. 审计日志（core/audit.py）：每一步都落盘成 JSONL（见 audit_logs/ 目录），
       对应 Trust by Design 原则三——高风险操作要有可追溯的审计轨迹。
    3. 评估 harness（core/eval.py）：对 3 个独立任务分别跑
       Outcome Eval（独立重新跑一次 pytest，不信任 Agent 的自我报告）+
       Trajectory Eval（先读后写？验证再回答？有没有连续重复调用？），
       汇总成类 SWE-Bench 的 Resolve Rate 报告。

任务场景：3 个相互独立的"改代码通过 pytest"任务（moving_average / flatten_list /
is_palindrome），每个任务各自起一个独立的 MCP Server 子进程（独立沙箱），顺序执行、
独立评估——这次不追求并发（Week 10 已经证明过 asyncio.gather 的并发能力），
重点在 MCP 协议往返是否可靠、评估结论是否可信。
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import shutil
import sys

# 让脚本能直接运行：把 agent_system 目录加入 import 路径
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from core import (  # noqa: E402
    AuditLog,
    EvalRecord,
    EvalSummary,
    FLATTEN_SCRIPT,
    MOVING_AVERAGE_SCRIPT,
    MemorySystem,
    MockCodingBackend,
    PALINDROME_SCRIPT,
    ReActAgent,
    build_backend,
    evaluate_outcome,
    evaluate_trajectory,
)
from core.mcp_backend import MCPToolSession  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).resolve().parent / ".env")  # 加载 LKEAP_API_KEY 等
except ImportError:
    pass

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

TEST_PALINDROME = '''\
from string_utils import is_palindrome


def test_simple():
    assert is_palindrome("level")


def test_case_and_space():
    assert is_palindrome("A man a plan a canal Panama")


def test_not_palindrome():
    assert not is_palindrome("hello")
'''

TASKS = {
    "moving_average": (MOVING_AVERAGE_SCRIPT, TEST_MOVING_AVERAGE),
    "flatten_list": (FLATTEN_SCRIPT, TEST_FLATTEN),
    "is_palindrome": (PALINDROME_SCRIPT, TEST_PALINDROME),
}


def prepare_sandbox(root: pathlib.Path, test_file: str, test_content: str) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / test_file).write_text(test_content, encoding="utf-8")


async def run_one_task(
    task_id: str, script, test_content: str, workspace_root: pathlib.Path,
    audit_dir: pathlib.Path, backend_name: str,
) -> EvalRecord:
    sandbox = workspace_root / task_id
    prepare_sandbox(sandbox, script.test_file, test_content)
    audit = AuditLog(audit_dir / f"{task_id}.jsonl")

    async with MCPToolSession(sandbox) as registry:
        if backend_name == "mock":
            backend = MockCodingBackend(script)
        elif backend_name == "strict":
            backend = build_backend("strict", script=script)
        else:
            backend = build_backend(backend_name)

        memory = MemorySystem()
        memory.semantic.store("测试框架", "本项目使用 pytest 进行测试，实现需通过全部用例。")
        memory.semantic.store("边界要求", "数值/容器/字符串类函数需考虑空输入、越界参数、大小写等边界情况。")

        agent = ReActAgent(
            backend=backend, tools=registry, memory=memory,
            audit=audit, max_steps=12, verbose=True,
        )
        task_text = f"请在 {script.target_file} 中实现目标函数，确保通过 {script.test_file} 的全部测试。"
        result = await agent.arun(task_text)

    # Outcome Eval：不信任 result.final_answer，独立重新跑一次 pytest 才算数
    resolved = evaluate_outcome(sandbox)
    trajectory = evaluate_trajectory(result)
    return EvalRecord(task_id=task_id, resolved=resolved, trajectory=trajectory)


async def main_async(args: argparse.Namespace) -> int:
    workspace_root = pathlib.Path(__file__).resolve().parent / "workspace_w11"
    audit_dir = pathlib.Path(__file__).resolve().parent / "audit_logs"

    records = []
    for task_id, (script, test_content) in TASKS.items():
        print(f"\n{'═' * 60}\n▶ 任务：{task_id}\n{'═' * 60}")
        record = await run_one_task(task_id, script, test_content, workspace_root, audit_dir, args.backend)
        records.append(record)

    summary = EvalSummary(records)
    print("\n" + "═" * 60)
    print("📊 评估报告（Outcome Eval 独立验证 + Trajectory Eval 启发式检查）")
    print(summary.report())
    print(f"\n审计日志：{audit_dir}/*.jsonl")
    print("═" * 60)
    return 0 if summary.resolved == summary.total else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Week 11 演示：MCP 封装 + 评估 harness")
    parser.add_argument("--backend", default="mock", choices=["mock", "anthropic", "lkeap", "strict"],
                        help="LLM 后端：mock（离线，默认）/ lkeap（腾讯云 deepseek-v4-pro，真实）/ "
                             "anthropic（真实 Claude）/ strict（协议校验，不调 API）")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
