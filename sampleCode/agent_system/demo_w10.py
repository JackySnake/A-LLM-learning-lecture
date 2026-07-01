"""
Week 10 演示：Orchestrator-Worker 多智能体协作（异步并发 + HITL + A2A 消息传递）
====================================================================================

运行（默认离线 mock，无需任何 API key）：
    source /Users/jackymm/miniforge3/bin/activate claude_p13
    cd sampleCode/agent_system
    python demo_w10.py

切换到真实大模型：
    python demo_w10.py --backend lkeap                 # 腾讯云 lkeap / deepseek-v4-pro（需 .env 配置 LKEAP_API_KEY）
    python demo_w10.py --backend lkeap --interactive   # HITL 改为真正等待终端确认，而非自动批准

对比串行 vs 并行耗时（只有真实网络后端才有意义；mock 是离线状态机，近乎瞬时，看不出差异）：
    python demo_w10.py --backend lkeap --max-concurrency 1   # 强制串行：两个 Coder 排队跑
    python demo_w10.py --backend lkeap --max-concurrency 4   # 默认并行：两个 Coder 同时跑

这个 demo 在 Week 9 单 Agent 的基础上长出四个新能力：
    1. Orchestrator-Worker 架构（core/orchestrator.py）：4 个子任务，按依赖分 2 波调度
    2. asyncio.gather + Semaphore 真并发（同一波内的 Worker 并行执行，core/agent.py 的 arun）
    3. HITL 审批钩子（core/hitl.py）：write_file 每次、run_pytest 首次都需要"批准"
    4. A2A 消息传递（core/orchestrator.py 的 inbox）：Reviewer 通过显式 AgentMessage
       收到 Coder 的结论，而不是共享 Coder 的工作记忆或读它的私有状态

Reviewer 的工具集用 ToolRegistry.scoped() 拿掉了 write_file（Week 11 补的最小权限硬约束，
见 core/tools.py）——Reviewer 的 LLM 调用里根本不存在这个工具，不是靠 prompt 里"请不要
改代码"这句话自觉遵守。

任务场景：两个相互独立的"改代码通过 pytest"子任务（moving_average / flatten_list，
各自在自己的沙箱子目录里），各自先由 Coder 实现+自我修复，再由 Reviewer 独立复核。
Coder 与 Reviewer 在 mock/strict 模式下用不同的规则状态机（MockCodingBackend vs
MockReviewBackend）；在真实后端（lkeap/anthropic）下复用同一个 LLMBackend 类，
靠 role_instructions 区分分工——这是 Week 9 memory.py 里"接口稳定、实现可替换"
取舍的又一次应用。
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
    FLATTEN_SCRIPT,
    HITLGate,
    MOVING_AVERAGE_SCRIPT,
    MemorySystem,
    MockCodingBackend,
    MockCodingScript,
    MockReviewBackend,
    Orchestrator,
    ReActAgent,
    Subtask,
    auto_approve_with_log,
    build_backend,
    build_coding_registry,
    interactive_approve,
)

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


def prepare_sandbox(root: pathlib.Path, test_file: str, test_content: str) -> None:
    """整目录清空重建（同 demo_w9 的做法：避免 __pycache__ 里的旧字节码干扰重复运行）。"""
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / test_file).write_text(test_content, encoding="utf-8")


def build_worker(
    *,
    role: str,
    script: MockCodingScript,
    sandbox: pathlib.Path,
    backend_name: str,
    approval: HITLGate,
) -> ReActAgent:
    """按角色（coder/reviewer）+ 后端名装配一个 Worker。"""
    registry, _ = build_coding_registry(sandbox)
    if role == "reviewer" and backend_name != "strict":
        # 最小权限硬约束（Week 11）：Reviewer 拿不到 write_file。
        # strict 模式例外——它的 Reviewer 复用 Coder 剧本（见下方分支的注释），
        # 剧本本身会调用 write_file，这里收窄反而会打断它的既定流程。
        registry = registry.scoped(["list_dir", "read_file", "run_pytest"])

    if backend_name == "mock":
        backend = MockCodingBackend(script) if role == "coder" else MockReviewBackend(script)
    elif backend_name == "strict":
        # StrictProtocolBackend 的价值在协议校验，不区分角色：Reviewer 在这个模式下
        # 也会跑一遍 Coder 剧本（先写 buggy、再修复），行为上会重复读写 Coder 已经改好
        # 的文件，但最终仍收敛到"测试全部通过"，不影响 demo 的正确性。
        backend = build_backend("strict", script=script)
    else:
        backend = build_backend(backend_name)

    role_instructions = (
        "你是 Coder：实现或修复代码，直到 pytest 全部通过。"
        if role == "coder"
        else "你是 Reviewer：不要修改任何代码。不能仅凭上游 Coder 消息里的说法下结论——"
             "必须调用 run_pytest 亲自验证一遍，再给出「通过/不通过」的结论（结论里请明确写出 PASS 或 FAIL）。"
    )

    memory = MemorySystem()
    memory.semantic.store("测试框架", "本项目使用 pytest 进行测试，实现需通过全部用例。")
    memory.semantic.store("边界要求", "数值/容器类函数需考虑空输入、越界参数、深层嵌套等边界情况。")

    return ReActAgent(
        backend=backend, tools=registry, memory=memory,
        max_steps=12, verbose=True,
        approval=approval, role_instructions=role_instructions,
    )


async def main_async(args: argparse.Namespace) -> int:
    base = pathlib.Path(__file__).resolve().parent / "workspace"
    sandbox_a = base / "task_a"
    sandbox_b = base / "task_b"
    prepare_sandbox(sandbox_a, MOVING_AVERAGE_SCRIPT.test_file, TEST_MOVING_AVERAGE)
    prepare_sandbox(sandbox_b, FLATTEN_SCRIPT.test_file, TEST_FLATTEN)

    approve_cb = interactive_approve if args.interactive else auto_approve_with_log

    def new_gate() -> HITLGate:
        # 每个 Worker 一份独立的 HITLGate：批准是"这个 Worker 自己的一次性批准"，
        # 不是全局共享开关——授权应该有边界，而不是一次批准就对所有 Worker 生效。
        return HITLGate(
            gated_tools={"write_file", "run_pytest"},
            approve=approve_cb,
            require_every_time={"write_file"},  # run_pytest 首次批准后放行（能力等级传递）
        )

    workers = {
        "coder_a": build_worker(role="coder", script=MOVING_AVERAGE_SCRIPT, sandbox=sandbox_a,
                                 backend_name=args.backend, approval=new_gate()),
        "reviewer_a": build_worker(role="reviewer", script=MOVING_AVERAGE_SCRIPT, sandbox=sandbox_a,
                                    backend_name=args.backend, approval=new_gate()),
        "coder_b": build_worker(role="coder", script=FLATTEN_SCRIPT, sandbox=sandbox_b,
                                 backend_name=args.backend, approval=new_gate()),
        "reviewer_b": build_worker(role="reviewer", script=FLATTEN_SCRIPT, sandbox=sandbox_b,
                                    backend_name=args.backend, approval=new_gate()),
    }

    subtasks = [
        Subtask(id="a1", role="coder_a", depends_on=[],
                task="请在 math_utils.py 中实现 moving_average(nums, window)，确保通过 test_math_utils.py 的全部测试。"),
        Subtask(id="b1", role="coder_b", depends_on=[],
                task="请在 list_utils.py 中实现 flatten_list(nested)，确保通过 test_list_utils.py 的全部测试。"),
        Subtask(id="a2", role="reviewer_a", depends_on=["a1"],
                task="独立复核 math_utils.py 的实现是否真的通过测试。"),
        Subtask(id="b2", role="reviewer_b", depends_on=["b1"],
                task="独立复核 list_utils.py 的实现是否真的通过测试。"),
    ]

    orchestrator = Orchestrator(workers=workers, max_concurrency=args.max_concurrency)
    print(f"\n🧭 Orchestrator 启动：4 个子任务，2 波依赖调度，并发上限={args.max_concurrency}")
    print("─" * 60)

    results, waves = await orchestrator.run(subtasks)

    print("\n" + "═" * 60)
    print("📊 各 wave 耗时（mock 后端近乎瞬时，看不出串并行差异；--backend lkeap 才有意义）：")
    total = 0.0
    for w in waves:
        print(f"  wave {w.subtask_ids}：{w.elapsed_seconds:.3f}s")
        total += w.elapsed_seconds
    print(f"  总耗时：{total:.3f}s")

    all_ok = all(r.succeeded for r in results.values())
    print(f"\n任务{'全部成功 ✅' if all_ok else '存在未完成 ⚠️'}")
    for sid, r in results.items():
        print(f"  [{sid}] {'✅' if r.succeeded else '⚠️'} {r.final_answer}")
    print("═" * 60)
    return 0 if all_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Week 10 Orchestrator-Worker 多智能体演示")
    parser.add_argument("--backend", default="mock", choices=["mock", "anthropic", "lkeap", "strict"],
                        help="LLM 后端：mock（离线，默认）/ lkeap（腾讯云 deepseek-v4-pro，真实）/ "
                             "anthropic（真实 Claude）/ strict（协议校验，不调 API）")
    parser.add_argument("--max-concurrency", type=int, default=4,
                        help="同一 wave 内最多并发执行几个 Worker（默认 4，设为 1 可对比串行耗时）")
    parser.add_argument("--interactive", action="store_true",
                        help="HITL 审批改为真正等待终端输入（默认自动批准并打印日志）")
    args = parser.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
