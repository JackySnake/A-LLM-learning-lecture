"""
评估 harness（Week 11）
==========================

对应讲义：Week 11 · Part 3「Agent 评估体系」，尤其 3.2「两种评估范式：Outcome vs Trajectory」
          附录 A.3「SWE-Bench 评测流程详解」

Outcome Eval 对应 SWE-Bench 的 Resolve Rate 思路：**不解析 Agent 自己怎么说，独立重新
跑一次 pytest 才算数**——附录 A.3 里 FAIL_TO_PASS/PASS_TO_PASS 测试也是评测时独立验证，
而不是信任 Agent 的自我报告。这一点在本项目里格外有必要：我们已经实测过，真实模型有
时会在没有调用 run_pytest 的情况下就给出"PASS"结论（见 frameworks/README.md 的诚实笔记），
如果 Outcome Eval 直接读 Agent 的 final_answer，会把这种情况误判为成功。

Trajectory Eval 检查 Week 9 讲义定的行为规范（"先读后写""改完必须验证"）是否被真正遵守，
用的是 core/agent.py 产出的 StepTrace 轨迹。这是本教学场景的简化版本：真实 SWE-Bench 级别
的轨迹评估通常需要参考轨迹或 LLM-as-judge（讲义 Part 3.2 提到"标注成本高"），这里先做
不需要参考答案的规则启发式版本，能抓住最常见的几类问题，但不是全面的轨迹质量评分。
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

from .agent import AgentResult
from .tools import CodingToolkit


@dataclass
class TrajectoryReport:
    """一次任务执行的轨迹质量报告。`issues` 为空即视为"干净"。"""

    step_count: int
    read_before_write: bool
    verified_before_final: bool
    redundant_calls: int
    issues: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.issues


@dataclass
class EvalRecord:
    """一个任务的完整评估结果：结果对不对 + 过程好不好。"""

    task_id: str
    resolved: bool
    trajectory: TrajectoryReport


@dataclass
class EvalSummary:
    records: list[EvalRecord]

    @property
    def total(self) -> int:
        return len(self.records)

    @property
    def resolved(self) -> int:
        return sum(1 for r in self.records if r.resolved)

    @property
    def resolve_rate(self) -> float:
        """对应 SWE-Bench 的 Resolve Rate：成功解决的实例数 / 总实例数。"""
        return self.resolved / self.total if self.total else 0.0

    def report(self) -> str:
        lines = [f"Resolve Rate：{self.resolved}/{self.total}（{self.resolve_rate:.0%}）"]
        for r in self.records:
            verdict = "✅ 通过" if r.resolved else "❌ 未通过"
            traj = r.trajectory
            issue_text = "；".join(traj.issues) if traj.issues else "无异常"
            lines.append(
                f"  [{r.task_id}] {verdict} | {traj.step_count} 步 | 轨迹：{issue_text}"
            )
        return "\n".join(lines)


def evaluate_outcome(sandbox: pathlib.Path) -> bool:
    """Outcome Eval：独立、客观地验证——不看 Agent 怎么说，在沙箱里重新跑一次 pytest。"""
    kit = CodingToolkit(sandbox)
    observation = kit.run_pytest()
    return "passed" in observation and "failed" not in observation


def evaluate_trajectory(result: AgentResult) -> TrajectoryReport:
    """Trajectory Eval：基于规则的启发式检查（不需要参考轨迹）。

    目前检查三类最常见的过程问题：
      1. 先探索后写：第一次 write_file 之前，是否先用 read_file/list_dir 探索过环境
      2. 验证再回答：final_answer 之前是否至少调用过一次 run_pytest
      3. 连续重复调用：相邻两步是否用完全相同的工具+参数重复调用（浪费步数）
    """
    steps = result.steps
    issues: list[str] = []

    # 注意：这里检查的是"写之前是否探索过环境"，而不是"写之前是否读过同一路径"——
    # 本任务场景是从零创建目标文件，第一次 write_file 时那个路径根本还不存在，
    # 要求"读过同一路径"在这个领域里没有意义（曾经这样实现过，实测跑出来发现
    # 每个任务都被误判违规，才改成检查全局顺序：探索是否发生在第一次写之前）。
    first_write_idx = next((i for i, s in enumerate(steps) if s.tool_name == "write_file"), None)
    first_explore_idx = next((i for i, s in enumerate(steps) if s.tool_name in ("read_file", "list_dir")), None)
    read_before_write = first_write_idx is None or (
        first_explore_idx is not None and first_explore_idx < first_write_idx
    )
    if not read_before_write:
        issues.append("没有先 read_file/list_dir 探索环境就直接 write_file")

    ran_pytest = any(s.tool_name == "run_pytest" for s in steps)
    if not ran_pytest:
        issues.append("全程没有调用 run_pytest，final_answer 缺乏验证依据")

    redundant = sum(
        1 for prev, cur in zip(steps, steps[1:])
        if prev.tool_name and prev.tool_name == cur.tool_name and prev.tool_args == cur.tool_args
    )
    if redundant:
        issues.append(f"检测到 {redundant} 次连续重复的工具调用（相同工具 + 相同参数）")

    return TrajectoryReport(
        step_count=len(steps),
        read_before_write=read_before_write,
        verified_before_final=ran_pytest,
        redundant_calls=redundant,
        issues=issues,
    )
