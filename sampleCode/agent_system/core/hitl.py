"""
HITL 审批钩子（Week 10）
============================

对应讲义：Week 10 · Part 2.2「Supervisor 模式与人在回路（HITL）」

设计对应讲义里的**能力等级传递（Capability Propagation）**原则：
run_pytest 本质是代码执行能力，无论这次具体跑什么命令，都应该在首次调用时
被当成高风险操作对待——这正是 Claude Code 里 Bash 工具"首次调用必须授权"的
设计（讲义 Part 2.2 解法三）。write_file 则每次都需要确认，因为每次改的内容不同。

HITLGate 不关心"批准"具体怎么做——脚本化 demo 用自动批准并打印日志，
真实产品会换成向用户弹窗 / Slack 审批消息，接口不变。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class HITLGate:
    """在风险工具执行前拦截，交给 `approve` 回调决定是否放行。

    gated_tools       ：需要过审批的工具名集合。
    approve           ：审批回调 (tool_name, tool_args) -> 是否批准。
    require_every_time：这些工具每次调用都要重新审批；不在此集合但在 gated_tools 里的工具，
                         首次批准后本次任务内自动放行（对应"能力等级传递"：批准一次即代表
                         批准了这个能力本身，而不是逐次重复确认）。
    """

    gated_tools: set[str]
    approve: Callable[[str, dict], bool]
    require_every_time: set[str] = field(default_factory=set)
    _approved_once: set[str] = field(default_factory=set, repr=False)

    def check(self, tool_name: str, tool_args: dict) -> Optional[str]:
        """返回 None 表示放行；返回非空字符串表示拒绝，该字符串会被当作 observation 回灌给模型。"""
        if tool_name not in self.gated_tools:
            return None
        if tool_name not in self.require_every_time and tool_name in self._approved_once:
            return None

        approved = self.approve(tool_name, tool_args)
        if approved:
            if tool_name not in self.require_every_time:
                self._approved_once.add(tool_name)
            return None
        return f"[HITL] 用户拒绝了 {tool_name} 的执行请求（参数：{tool_args}），请改用其他方式或停止。"


def auto_approve_with_log(tool_name: str, tool_args: dict) -> bool:
    """脚本化 demo 用：自动批准，但把审批事件打印出来，让 HITL 拦截点可见。"""
    print(f"    🔒 [HITL] 检测到风险操作 {tool_name}（参数：{_brief(tool_args)}）—— 自动批准（演示模式）")
    return True


def interactive_approve(tool_name: str, tool_args: dict) -> bool:
    """真正需要人工确认时用：阻塞等待终端输入。"""
    answer = input(f"    🔒 [HITL] Agent 请求执行 {tool_name}（参数：{_brief(tool_args)}）。批准？[y/N] ")
    return answer.strip().lower() in ("y", "yes")


def _brief(args: dict) -> str:
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 30:
            s = s[:27] + "..."
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)
