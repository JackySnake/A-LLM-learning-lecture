"""Week 9-10 核心组件：LLM 后端 / 工具 / 记忆 / ReAct 引擎 / Orchestrator-Worker / HITL。"""

from .agent import AgentResult, ReActAgent, StepTrace
from .audit import AuditLog
from .hitl import HITLGate, auto_approve_with_log, interactive_approve
from .llm_backend import (
    AgentAction,
    FLATTEN_SCRIPT,
    LLMBackend,
    MOVING_AVERAGE_SCRIPT,
    MockCodingBackend,
    MockCodingScript,
    MockReviewBackend,
    PALINDROME_SCRIPT,
    build_backend,
)
from .eval import EvalRecord, EvalSummary, TrajectoryReport, evaluate_outcome, evaluate_trajectory
from .mcp_backend import MCPToolRegistry, MCPToolSession
from .memory import MemorySystem
from .orchestrator import AgentMessage, Orchestrator, Subtask, WaveResult
from .tools import CodingToolkit, Tool, ToolRegistry, ToolSource, build_coding_registry

__all__ = [
    "AgentAction",
    "AgentMessage",
    "AgentResult",
    "AuditLog",
    "CodingToolkit",
    "EvalRecord",
    "EvalSummary",
    "FLATTEN_SCRIPT",
    "HITLGate",
    "LLMBackend",
    "MCPToolRegistry",
    "MCPToolSession",
    "MOVING_AVERAGE_SCRIPT",
    "MemorySystem",
    "MockCodingBackend",
    "MockCodingScript",
    "MockReviewBackend",
    "Orchestrator",
    "PALINDROME_SCRIPT",
    "ReActAgent",
    "StepTrace",
    "Subtask",
    "ToolRegistry",
    "Tool",
    "ToolSource",
    "TrajectoryReport",
    "WaveResult",
    "auto_approve_with_log",
    "build_backend",
    "build_coding_registry",
    "evaluate_outcome",
    "evaluate_trajectory",
    "interactive_approve",
]
