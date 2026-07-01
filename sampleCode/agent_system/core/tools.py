"""
工具注册表与编程领域工具集（Week 9 地基）
==============================================

对应讲义：Week 9 · Part 3.3「JSON Schema：约束工具调用的可靠性」
          Week 9 · Part 3.6「工具调用的可靠性工程」

核心概念
--------
1. 每个工具 = 一个普通 Python 函数 + 一份 JSON Schema（描述名字/用途/入参）。
   JSON Schema 是给 LLM 看的"说明书"，也是 Function Calling 的约束契约。
2. ToolRegistry 负责：注册工具、产出给 LLM 的 schema 列表、按名字安全地执行。
3. 所有文件/命令操作都被限制在一个 **sandbox 工作目录** 内——这是最基础的
   安全边界。更完整的权限审批（HITL）和命令白名单留到 Week 10/11 演进。

注意：这里的"沙箱"只是路径限制（防止越界读写），不是 OS 级隔离。
真实生产环境应使用容器/seccomp 等（见 Week 11 讲义 Part 2「Agent 安全治理」）。
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Protocol, Union, runtime_checkable


@runtime_checkable
class ToolSource(Protocol):
    """`ReActAgent` 实际依赖的最小接口——不是"必须是 ToolRegistry"，而是"长得像它"。

    Week 11 的 `MCPToolRegistry`（core/mcp_backend.py）满足同样的接口但 `execute`
    是协程；用 Protocol 显式声明这个契约，而不是让 Pyright 只能通过具体类型判断，
    这是"接口稳定、实现可替换"原则第一次写成正式类型（此前 memory.py/frameworks/
    的同类取舍都只停留在文档注释里）。
    """

    def specs(self) -> list[dict]: ...
    def describe_for_prompt(self) -> str: ...
    def execute(self, name: str, args: dict) -> Union[str, Awaitable[str]]: ...


@dataclass
class Tool:
    """一个工具：可调用对象 + 给 LLM 的 JSON Schema 描述。"""

    name: str
    description: str
    parameters: dict          # JSON Schema（type/properties/required）
    func: Callable[..., str]  # 实际执行体，返回字符串形式的 observation

    def to_spec(self) -> dict:
        """厂商中性的工具规格。各 LLM 后端再把它转成自己的原生格式
        （Anthropic 的 input_schema / OpenAI 的 function.parameters，见 llm_backend.py）。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,  # JSON Schema（type/properties/required）
        }


class ToolRegistry:
    """工具注册表：集中管理工具，对外暴露 schema、对内安全执行。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def scoped(self, names: list[str]) -> "ToolRegistry":
        """返回一个只包含指定工具子集的新注册表——最小权限原则的**硬约束**实现
        （Week 11 · Part 2.3「Trust by Design」原则二：最小权限，按需授权）。

        对比 Week 10 demo_w10.py 早期做法：Reviewer 靠 system prompt 里的文字
        （"你是 Reviewer，不要修改任何代码"）约束自己不用 write_file——这是软约束，
        模型理论上仍能调用；scoped() 从注册表层面直接拿掉这个工具，Reviewer 的
        LLM 调用里根本不存在 write_file 这个选项，是真正拿不到，不是"选择不用"。
        """
        sub = ToolRegistry()
        for name in names:
            if name in self._tools:
                sub.register(self._tools[name])
        return sub

    def specs(self) -> list[dict]:
        """返回所有工具的中性规格，供 LLM 后端转换后注入 API 的 tools 参数。"""
        return [t.to_spec() for t in self._tools.values()]

    def describe_for_prompt(self) -> str:
        """生成可读的工具说明，用于注入 system prompt（mock 后端 / 纯文本 ReAct 时用）。"""
        lines = []
        for t in self._tools.values():
            params = ", ".join(t.parameters.get("properties", {}).keys())
            lines.append(f"- {t.name}({params}): {t.description}")
        return "\n".join(lines)

    def execute(self, name: str, args: dict) -> str:
        """按名字执行工具。这是工具调用可靠性工程的关键一环：

        - 工具不存在 / 参数错误，都要转成可读的 observation 返回给 LLM，
          而不是抛异常中断 Agent——让模型有机会"看到错误并自我纠正"。
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"[错误] 不存在名为 '{name}' 的工具。可用工具：{list(self._tools)}"
        try:
            return tool.func(**args)
        except TypeError as e:
            return f"[错误] 调用 {name} 的参数不正确：{e}"
        except Exception as e:  # noqa: BLE001 - 故意兜底：错误要回灌给模型
            return f"[错误] 执行 {name} 时出现异常：{type(e).__name__}: {e}"


# ──────────────────────────────────────────────────────────────────────────
# 编程领域工具集：全部约束在 sandbox 目录内
# ──────────────────────────────────────────────────────────────────────────
class CodingToolkit:
    """文件读写 / 列目录 / 运行 pytest，均限制在 sandbox 根目录内。"""

    def __init__(self, sandbox_root: str | Path):
        self.root = Path(sandbox_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # —— 路径安全：防止 ../ 越界到沙箱之外 ——
    def _resolve(self, rel_path: str) -> Path:
        target = (self.root / rel_path).resolve()
        if self.root not in target.parents and target != self.root:
            raise PermissionError(f"路径 '{rel_path}' 越出了沙箱目录，已拒绝。")
        return target

    def list_dir(self, path: str = ".") -> str:
        """列出工作目录下的文件与子目录，用于了解项目结构。"""
        target = self._resolve(path)
        if not target.exists():
            return f"目录不存在：{path}"
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        return f"{path} 下的内容：" + (", ".join(entries) if entries else "（空）")

    def read_file(self, path: str) -> str:
        """读取指定文件的全部内容。"""
        target = self._resolve(path)
        if not target.exists():
            return f"文件不存在：{path}"
        text = target.read_text(encoding="utf-8")
        return f"--- {path} 内容开始 ---\n{text}\n--- {path} 内容结束 ---"

    def write_file(self, path: str, content: str) -> str:
        """把内容写入指定文件（覆盖式）。用于创建或修改代码。"""
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"已写入 {path}（{len(content)} 字符）。"

    def run_pytest(self, path: str = ".") -> str:
        """在沙箱内运行 pytest，返回精简后的结果（最后若干行 + 结论）。"""
        target = self._resolve(path)
        # 禁用字节码缓存：避免上一版（buggy）实现的 .pyc 被 pytest 复用，
        # 导致修复后的代码不生效——这是 Agent 反复改代码场景里的真实陷阱。
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", str(target),
                 "-q", "--no-header", "-p", "no:cacheprovider"],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
        except FileNotFoundError:
            return self._run_pytest_inprocess(target)
        except subprocess.TimeoutExpired:
            return "[错误] pytest 运行超时（>60s）。"
        out = (proc.stdout or "") + (proc.stderr or "")
        # 只回传尾部摘要，避免把整页输出塞进上下文（呼应 Context Rot：观察要精简）
        tail = "\n".join(out.strip().splitlines()[-8:])
        return f"pytest 退出码={proc.returncode}\n{tail}"

    def _run_pytest_inprocess(self, target: Path) -> str:
        """环境中没有 pytest CLI 时的兜底：进程内调用（保证 demo 可跑）。"""
        try:
            import pytest  # noqa: F401
        except ImportError:
            return "[提示] 环境未安装 pytest，无法运行测试。请 pip install pytest。"
        buf = io.StringIO()
        with redirect_stdout(buf):
            import pytest as _pytest

            code = _pytest.main([str(target), "-q", "--no-header"])
        tail = "\n".join(buf.getvalue().strip().splitlines()[-8:])
        return f"pytest 退出码={int(code)}\n{tail}"


def build_coding_registry(sandbox_root: str | Path) -> tuple[ToolRegistry, CodingToolkit]:
    """组装编程助手的工具注册表，并附上每个工具的 JSON Schema。"""
    kit = CodingToolkit(sandbox_root)
    reg = ToolRegistry()

    reg.register(Tool(
        name="list_dir",
        description="列出工作目录下的文件与子目录，用于了解项目结构。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对沙箱根目录的路径，默认当前目录。"}
            },
            "required": [],
        },
        func=kit.list_dir,
    ))

    reg.register(Tool(
        name="read_file",
        description="读取指定文件的全部内容。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要读取的文件路径（相对沙箱根目录）。"}
            },
            "required": ["path"],
        },
        func=kit.read_file,
    ))

    reg.register(Tool(
        name="write_file",
        description="把内容写入指定文件（覆盖式）。用于创建或修改代码。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要写入的文件路径（相对沙箱根目录）。"},
                "content": {"type": "string", "description": "要写入的完整文件内容。"},
            },
            "required": ["path", "content"],
        },
        func=kit.write_file,
    ))

    reg.register(Tool(
        name="run_pytest",
        description="在沙箱内运行 pytest 测试，返回测试结果摘要。用于验证代码是否正确。",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "测试路径，默认整个沙箱目录。"}
            },
            "required": [],
        },
        func=kit.run_pytest,
    ))

    return reg, kit
