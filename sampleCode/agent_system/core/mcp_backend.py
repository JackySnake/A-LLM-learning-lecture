"""
MCP 客户端封装（Week 11）
==============================

对应讲义：Week 11 · Part 1（MCP——工具生态的统一语言），尤其 1.3（三大原语——这里只用 Tool）、
          1.7（MCP 实际接入）

`MCPToolRegistry` 实现了和 `core.tools.ToolRegistry` 相同的 `ToolSource` 接口
（`specs()` / `describe_for_prompt()` / `execute()`），但工具执行不再是进程内函数调用，
而是通过 MCP 协议（stdio 子进程）转发给 `mcp_server.py`。

这是"接口稳定、实现可替换"原则的第三次应用（前两次：memory.py 的向量库替换、
frameworks/ 的编排层替换）——`ReActAgent` 完全不需要知道自己面对的是本地函数还是
一个通过标准输入输出通信的独立子进程；唯一的差别是 `execute` 变成了协程，
这也是 Week 10 给 `ReActAgent.arun` 加 `inspect.iscoroutinefunction` 判断的原因
（见 core/agent.py）。
"""

from __future__ import annotations

import os
import pathlib
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPToolRegistry:
    """通过 MCP 协议暴露的工具注册表，接口对齐 core.tools.ToolRegistry（ToolSource 协议）。"""

    def __init__(self, session: ClientSession, specs: list[dict]):
        self._session = session
        self._specs = specs

    def specs(self) -> list[dict]:
        return list(self._specs)

    def describe_for_prompt(self) -> str:
        lines = []
        for s in self._specs:
            params = ", ".join(s["parameters"].get("properties", {}).keys())
            lines.append(f"- {s['name']}({params}): {s['description']}")
        return "\n".join(lines)

    async def execute(self, name: str, args: dict) -> str:
        """转发一次 MCP tool call。异常/协议错误也转成可读 observation，
        而不是让异常直接冒出来中断 Agent——和 ToolRegistry.execute 的可靠性原则一致，
        只是这里多了一层"跨进程通信可能失败"的现实。
        """
        try:
            result = await self._session.call_tool(name, args)
        except Exception as e:  # noqa: BLE001 - 故意兜底：MCP 调用失败也要回灌给模型
            return f"[错误] 调用 MCP 工具 {name} 时出现异常：{type(e).__name__}: {e}"
        texts = [block.text for block in result.content if hasattr(block, "text")]
        return "\n".join(texts) if texts else "(无输出)"


class MCPToolSession:
    """管理 MCP Server 子进程连接的生命周期（async context manager）。

    用法：
        async with MCPToolSession(sandbox) as registry:
            agent = ReActAgent(backend=..., tools=registry, ...)
            result = await agent.arun(task)
    """

    def __init__(self, sandbox: pathlib.Path, server_script: pathlib.Path | None = None):
        self.sandbox = pathlib.Path(sandbox)
        self.server_script = server_script or (pathlib.Path(__file__).resolve().parent.parent / "mcp_server.py")
        self._stack: AsyncExitStack | None = None
        self.registry: MCPToolRegistry | None = None

    async def __aenter__(self) -> MCPToolRegistry:
        self._stack = AsyncExitStack()
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(self.server_script)],
            env={**os.environ, "MCP_SANDBOX_ROOT": str(self.sandbox)},
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        tools_result = await session.list_tools()
        specs = [
            {"name": t.name, "description": t.description, "parameters": t.inputSchema}
            for t in tools_result.tools
        ]
        self.registry = MCPToolRegistry(session, specs)
        return self.registry

    async def __aexit__(self, *_exc_info) -> None:
        if self._stack is not None:
            await self._stack.aclose()
