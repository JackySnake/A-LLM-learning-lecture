"""
审计日志（Week 11 · Trust by Design 原则三）
================================================

对应讲义：Week 11 · Part 2.3「Trust by Design：安全作为架构一等公民」
          原则三——"高风险操作必须有审计轨迹"：调用时间、触发原因（Agent 的推理链）、
          参数、执行结果都要记录，既用于事后审查，也用于模型改进。

`ReActAgent` 一直有 `StepTrace`（见 core/agent.py），但只存在内存里，进程一结束就没了。
`AuditLog` 把同样的信息落盘成 JSONL——一行一个事件，可追加、可流式读取，是审计日志最朴素
的落地形式（真实生产系统通常会换成结构化日志系统/时序数据库，但接口不变：`record(**event)`）。
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass


@dataclass
class AuditLog:
    """按 JSONL 追加写入的审计日志。每条记录自动打时间戳。"""

    path: pathlib.Path

    def __post_init__(self) -> None:
        self.path = pathlib.Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, **event) -> None:
        entry = {"timestamp": time.time(), **event}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict]:
        """读回全部记录，主要给评估 harness / 事后审查用。"""
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
