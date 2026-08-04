"""
Week 11：把 core/tools.py 的 CodingToolkit 封装成 MCP Server
================================================================

对应讲义：Week 11 · Part 1.7「MCP 实际接入」、附录 A.1「完整的本地 MCP Server 接入示例」

用官方 mcp SDK 的底层 `Server` API（和讲义 A.1 的写法一致），把 Week 9 就有的 4 个工具
原样暴露成一个真正的 MCP stdio server：只做协议适配，不改工具本身的行为/沙箱边界——
这正是 MCP"统一语言"的价值：工具的实现不用重写，只是换了一层通讯协议（Part 1.2）。

这个文件通常不会被直接运行，而是作为子进程被 core/mcp_backend.py 的 MCPToolSession
拉起（stdio 传输：MCP Server 与 Client 之间通过标准输入输出通信，这是 MCP 最简单、
最常见的本地传输方式）。

沙箱根目录通过环境变量 MCP_SANDBOX_ROOT 传入——MCP Server 作为独立子进程启动，
配置只能走启动参数/环境变量，不能像普通函数调用那样直接传参（这也是 MCP Server
配置一定会长得像 Week 11 附录 A.1 里 Claude Desktop 配置文件那样的原因）。
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from mcp.server import Server  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402
from mcp.types import TextContent, Tool  # noqa: E402

from core.tools import build_coding_registry  # noqa: E402

SANDBOX_ROOT = pathlib.Path(
    os.environ.get("MCP_SANDBOX_ROOT", str(pathlib.Path(__file__).resolve().parent / "workspace_mcp"))
)
registry, _ = build_coding_registry(SANDBOX_ROOT)

app = Server("agent-system-coding-tools")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """声明本 Server 提供的工具——直接复用 ToolRegistry.specs()，不重复定义 JSON Schema。"""
    return [
        Tool(name=s["name"], description=s["description"], inputSchema=s["parameters"])
        for s in registry.specs()
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """处理工具调用请求——转发给 ToolRegistry.execute，白名单/沙箱边界完全复用，不重新实现。"""
    observation = registry.execute(name, arguments)
    return [TextContent(type="text", text=observation)]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
