"""
Week 9 演示：一个能"改代码 + 跑测试 + 自我修复"的单 Agent
==============================================================

运行（默认离线 mock，无需任何 API key）：
    source /Users/jackymm/miniforge3/bin/activate claude_p13
    cd sampleCode/agent_system
    python demo_w9.py

切换到真实大模型：
    python demo_w9.py --backend lkeap      # 腾讯云 lkeap / deepseek-v4-pro（需 .env 配置 LKEAP_API_KEY，见 README）
    python demo_w9.py --backend anthropic  # 真实 Claude（需 pip install anthropic 且配置 ANTHROPIC_API_KEY）
    python demo_w9.py --backend strict     # 不调 API，仅校验消息协议是否合法

这个 demo 串起了 Week 9 的四个核心概念：
    ReAct 循环（agent.py） + 工具调用/JSON Schema（tools.py）
    + 四层记忆（memory.py） + 可插拔 LLM 后端（llm_backend.py）

观察重点：Agent 写出的第一版代码会让 pytest 失败，它会**读到失败信息后再修复**，
这正是 ReAct 区别于一次性 CoT 的地方——行动的真实结果会反过来修正后续推理。
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

# 让脚本能直接运行：把 agent_system 目录加入 import 路径
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from core import MemorySystem, ReActAgent, build_backend, build_coding_registry  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).resolve().parent / ".env")  # 加载 LKEAP_API_KEY 等
except ImportError:
    pass  # 未安装 python-dotenv 时，仍可通过系统环境变量传入 key

# 预置的测试文件：定义了 moving_average 的验收标准（含边界用例）
TEST_FILE = '''\
from math_utils import moving_average


def test_basic():
    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]


def test_window_equals_len():
    assert moving_average([2, 4, 6], 3) == [4.0]


def test_window_larger_than_len():
    # 边界：窗口比列表还长时，应返回空列表而不是报错或多算
    assert moving_average([1, 2], 5) == []
'''


def prepare_sandbox(root: pathlib.Path) -> None:
    """初始化沙箱：整目录清空重建（含 __pycache__），再放入测试文件，保证可重复运行。"""
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "test_math_utils.py").write_text(TEST_FILE, encoding="utf-8")


def seed_memory(memory: MemorySystem) -> None:
    """注入长期记忆，演示"记忆如何影响 Agent 行为"。"""
    # 语义记忆：项目约定（类 CLAUDE.md），会被检索并注入 system prompt
    memory.semantic.store("测试框架", "本项目使用 pytest 进行测试，实现需通过全部用例。")
    memory.semantic.store("命名规范", "函数与变量统一使用 snake_case 命名。")
    memory.semantic.store("边界要求", "数值函数需考虑空输入与越界参数等边界情况。")

    # 程序记忆：修 bug 的标准流程（SOP）
    memory.procedural.add_skill(
        "修复失败的测试",
        ["阅读报错定位失败用例", "分析根因", "最小改动修复", "重新运行测试确认"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Week 9 单 Agent 演示")
    parser.add_argument("--backend", default="mock", choices=["mock", "anthropic", "lkeap", "strict"],
                        help="LLM 后端：mock（离线，默认）/ lkeap（腾讯云 deepseek-v4-pro，真实）/ "
                             "anthropic（真实 Claude）/ strict（协议校验，不调 API）")
    args = parser.parse_args()

    sandbox = pathlib.Path(__file__).resolve().parent / "workspace"
    prepare_sandbox(sandbox)

    # 组装：工具注册表（约束在 sandbox 内） + 四层记忆 + 可插拔后端
    registry, _ = build_coding_registry(sandbox)
    memory = MemorySystem()
    seed_memory(memory)
    backend = build_backend(args.backend)

    agent = ReActAgent(backend=backend, tools=registry, memory=memory, max_steps=12)

    task = "请在 math_utils.py 中实现 moving_average(nums, window) 函数，并确保通过 test_math_utils.py 的全部测试。"
    result = agent.run(task)

    # 收尾：打印轨迹摘要 + 验证最终产物
    print("\n" + "═" * 60)
    print(f"任务{'成功 ✅' if result.succeeded else '未完成 ⚠️'}  |  共 {len(result.steps)} 步")
    print(f"轨迹：{result.trajectory_summary()}")
    print(f"沙箱产物：{sandbox / 'math_utils.py'}")
    print("═" * 60)
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
