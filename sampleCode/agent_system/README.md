# Agent System：贯穿 Week 9-11 的小型编程助手 Agent

配合 `Week9讲义.md` / `Week10讲义.md` / `Week11讲义.md` 使用的**可运行代码**。设计原则：麻雀虽小、五脏俱全——用同一个"改代码通过测试"的编程助手场景，逐周长出新能力，而不是三份互不相关的示例代码。

## 任务场景

Agent 的任务始终是同一类："在沙箱目录里实现一个函数，并让 `pytest` 全部通过"（默认任务：`moving_average` 滑动平均，含边界条件）。这个场景选择是为了让 Week 11 的评估（outcome + trajectory）有天然、可自动化的判定标准。

## 快速开始

```bash
source /Users/jackymm/miniforge3/bin/activate claude_p13
cd sampleCode/agent_system
pip install -r requirements.txt   # 首次运行需要（pytest / openai / python-dotenv / mcp，anthropic 可选）
python demo_w9.py                 # Week 9：单 Agent，默认离线 mock 后端，无需任何 API key
python demo_w10.py                # Week 10：Orchestrator-Worker 多 Agent，同样默认离线 mock
python demo_w11.py                # Week 11：MCP 封装 + 评估 harness，同样默认离线 mock
```

## LLM 后端

通过 `--backend` 切换，Agent 循环代码完全不变（后端只是"大脑"，接口见 `core/llm_backend.py`）：

| backend | 说明 | 需要的配置 |
|---|---|---|
| `mock`（默认） | 规则驱动的离线状态机，非真 LLM。用于零成本演示 ReAct 反馈闭环（先失败→读报错→修复） | 无 |
| `lkeap` | 真实大模型，腾讯云 lkeap（deepseek-v4-pro-202606），OpenAI 兼容协议 | `.env` 中配置 `LKEAP_API_KEY`（见下） |
| `anthropic` | 真实 Claude | `pip install anthropic` + 环境变量 `ANTHROPIC_API_KEY` |
| `strict` | 不调用任何 API，只用 mock 状态机 + 严格模拟真实 API 的协议校验（tool_call_id 配对等）。用于免费验证"消息格式是否合法" | 无 |

**配置 `.env`**（项目目录下已有 `.env`，含 `LKEAP_API_KEY`，已被 `.gitignore` 排除，不会进入 Git）：

```
LKEAP_API_KEY=sk-tp-xxxxxxxx
```

`demo_w9.py`/`demo_w10.py`/`demo_w11.py` 启动时都会自动通过 `python-dotenv` 加载同目录下的 `.env`。lkeap 文档：https://cloud.tencent.com/document/product/1823/130060

## 架构

```
agent_system/
├── core/
│   ├── llm_backend.py   # 可插拔 LLM 后端 + 厂商中性事件流协议转换（含 adecide 异步版本）
│   ├── tools.py         # 工具注册表（JSON Schema）+ 编程工具集（沙箱内文件读写/pytest）+ scoped()
│   ├── memory.py        # 四层记忆（工作/情节/语义/程序）
│   ├── agent.py         # ReAct 循环引擎：run（同步，Week 9）+ arun（异步，Week 10/11）
│   ├── hitl.py           # HITL 审批钩子（Week 10：Supervisor 模式 / 能力等级传递）
│   ├── orchestrator.py   # Orchestrator-Worker 调度：依赖分波 + asyncio 并发 + A2A 消息传递（Week 10）
│   ├── mcp_backend.py     # MCP 客户端封装：MCPToolRegistry/MCPToolSession（Week 11）
│   ├── audit.py           # 审计日志 AuditLog，JSONL 落盘（Week 11：Trust by Design 原则三）
│   └── eval.py            # 评估 harness：evaluate_outcome / evaluate_trajectory（Week 11）
├── mcp_server.py          # 把 CodingToolkit 封装成 MCP stdio server（Week 11）
├── demo_w9.py             # Week 9 入口：单 Agent 演示
├── demo_w10.py            # Week 10 入口：Orchestrator-Worker 多 Agent 演示
├── demo_w11.py            # Week 11 入口：MCP + 审计日志 + 评估 harness 演示
├── frameworks/            # 同一场景的 LangGraph / CrewAI 对照实现（独立 conda 环境，见其 README）
├── .env                   # 真实 API key（gitignored）
├── workspace/             # Week 9 运行时沙箱目录
├── workspace_mcp/         # mcp_server.py 独立运行时的默认沙箱（demo_w11 会覆盖为每任务一个子目录）
├── workspace_w11/         # demo_w11.py 每个任务各一个子沙箱
└── audit_logs/            # demo_w11.py 产出的 JSONL 审计日志
```

### 核心设计：厂商中性事件流

Agent 循环（`core/agent.py`）不直接和某家 API 的消息格式打交道，而是维护一份中性事件流：

```python
{"role": "user",      "content": "任务文本"}
{"role": "assistant", "thought": "...", "tool_call": {"id","name","args"}}   # 调用工具
{"role": "assistant", "thought": "...", "final_answer": "..."}               # 给出最终答复
{"role": "tool",      "tool_call_id": "...", "content": "observation"}       # 工具结果
```

每个 `LLMBackend` 实现只负责把这份中性事件流**无损翻译**成自己的原生协议（Anthropic 的 tool_use/tool_result、OpenAI 的 tool_calls/tool role），且工具调用带唯一 `tool_call_id`，与结果强配对——这是 Function Calling 多轮协议的硬性要求，`llm_backend.py` 里的 `validate_openai_protocol` / `validate_anthropic_protocol` 就是在校验这件事（`strict` 后端会强制跑这个校验）。

### 沙箱安全边界

`core/tools.py` 的 `CodingToolkit` 把所有文件读写/pytest 执行限制在 `workspace/` 目录内（路径校验拒绝 `../` 越界）。这是最基础的安全边界，权限审批（HITL）与命令白名单是 Week 10/11 要加的更完整机制。

### Week 10：Orchestrator-Worker + 异步并发 + HITL + A2A

`demo_w10.py` 在 Week 9 单 Agent 之上派生出两组 Coder/Reviewer Worker，各自独立完成一个"改代码通过 pytest"子任务（`moving_average` / `flatten_list`），体现四个新机制：

- **依赖分波调度**（`core/orchestrator.py`）：4 个 `Subtask` 按 `depends_on` 分成两波——两个 Coder 先并行跑，两个 Reviewer 等各自的 Coder 完成后再并行跑。
- **真异步并发**：`ReActAgent.arun` 用 `await backend.adecide(...)`，`OpenAICompatBackend` 为此提供了基于 `AsyncOpenAI` 的真实非阻塞实现（而不是简单丢进线程池）；同一波内的 Worker 通过 `asyncio.gather` + `Semaphore(max_concurrency)` 真正并发执行。实测（`--backend lkeap`）：并发上限 4 时总耗时约 25s，强制串行（`--max-concurrency 1`）约 46s。
- **HITL 审批**（`core/hitl.py`）：`write_file` 每次、`run_pytest` 首次调用都要经过 `HITLGate.check`，体现讲义 Part 2.2"能力等级传递"原则——默认自动批准并打印日志，`--interactive` 可切换成真正阻塞等待终端确认。
- **A2A 消息传递**：Reviewer 不会读 Coder 的工作记忆，而是通过 `Orchestrator` 投递的显式 `AgentMessage`（sender/receiver/content）收到 Coder 的 `final_answer`——对应讲义 Part 2.3 的**消息传递范式**，区别于附录 A.2 教学示例用的**共享状态**（一个 `results` 字典）范式。

### Week 11：MCP 封装 + 白名单升级 + 审计日志 + 评估 harness

`demo_w11.py` 对 3 个独立任务（`moving_average` / `flatten_list` / `is_palindrome`）各自跑一遍单 Agent（Week 9 的 `ReActAgent`），但工具调用改走真实的 MCP 协议，跑完后接一套评估 harness：

- **MCP 封装**：`mcp_server.py` 把 `core/tools.py` 的 4 个工具原样暴露成 stdio server；`core/mcp_backend.py` 的 `MCPToolRegistry` 在 Agent 侧接住，接口和 `ToolRegistry` 完全一致，只是 `execute()` 变成协程——`ReActAgent.arun()` 早在 Week 10 就加了 `inspect.iscoroutinefunction` 判断，这次直接复用，Agent 侧零改动。
- **白名单/沙箱升级**：MCP 协议本身就是比字典查找更强的一层边界（只能调用 `list_tools()` 声明过的工具）；`ToolRegistry.scoped(names)` 把最小权限从 prompt 软约束升级成注册表硬约束，`demo_w10.py` 的 Reviewer 已经改用这个方法拿掉了 `write_file`。
- **审计日志**：`core/audit.py` 的 `AuditLog` 把每一步（时间戳/思考/工具/参数/结果/是否批准）落盘成 JSONL——对应 Trust by Design 原则三，也顺手清掉了本清单里"StepTrace 没有落盘"的旧债。
- **评估 harness**：`core/eval.py` 的 `evaluate_outcome()` 独立重新跑一次 pytest（不信任 Agent 自述），`evaluate_trajectory()` 检查"先探索后写""验证再回答""连续重复调用"三条启发式规则，汇总成 Resolve Rate 报告。

## 与讲义章节的对应关系

| 文件 | 对应讲义章节 |
|------|------|
| `core/llm_backend.py` | Week 9 Part 3.2-3.3（Function Calling 协议、JSON Schema） |
| `core/tools.py` | Week 9 Part 3.3、3.6（工具注册、可靠性工程） |
| `core/memory.py` | Week 9 Part 4.2（四层记忆架构），附录 A.3（语义记忆与向量库的接口关系） |
| `core/agent.py` | Week 9 Part 2.2-2.4（ReAct 循环、停止条件、最大步数） |
| `core/orchestrator.py` | Week 10 Part 2.1（Orchestrator-Worker 架构）、2.3（A2A 通信协议）、2.4（依赖图规划） |
| `core/hitl.py` | Week 10 Part 2.2（Supervisor 模式与人在回路、能力等级传递原则） |
| `mcp_server.py` / `core/mcp_backend.py` | Week 11 Part 1.7（MCP 实际接入）、附录 A.1 |
| `core/tools.py`（`scoped()`） | Week 11 Part 2.3（Trust by Design 原则一/二） |
| `core/audit.py` | Week 11 Part 2.3（Trust by Design 原则三：审计轨迹） |
| `core/eval.py` | Week 11 Part 3.2（Outcome vs Trajectory）、附录 A.3（SWE-Bench 评测流程） |

Week 9 讲义附录 A.2 直接引用本目录作为"可运行的完整实现"；Week 10 讲义附录 A.6、Week 11 讲义附录 A.5 同样直接引用本目录，分别覆盖 Orchestrator-Worker 调度/异步并发/HITL/A2A，以及 MCP 封装/白名单升级/审计日志/评估 harness。

## 演进计划（Week 9 → 11）

> 这份清单是本项目的 single source of truth。每次接手时先读这里，而不是依赖对话历史——过往曾出现"声称完成但实际未落盘"的情况。

### Week 9（单 Agent 地基）—— ✅ 已完成

- [x] 可插拔 LLM 后端：mock / lkeap / anthropic / strict
- [x] 工具注册表 + JSON Schema + 沙箱安全边界
- [x] 四层记忆（工作/情节/语义/程序）
- [x] ReAct 循环引擎，厂商中性事件流，`tool_call_id` 多轮协议
- [x] `.env` 配置 + `python-dotenv` 自动加载
- [x] 真实模型验证：`--backend lkeap`（deepseek-v4-pro-202606）端到端跑通
- [x] Week9讲义.md 附录 A.2/A.3 引用本项目

### Week 10（多 Agent）—— ✅ 已完成

- [x] `LLMBackend.adecide` 异步版本：`OpenAICompatBackend` 用 `AsyncOpenAI` 做真实异步网络调用，
      其余后端默认走 `asyncio.to_thread` 兜底（`core/llm_backend.py`）
- [x] Orchestrator-Worker 架构：coder / reviewer 角色分工，两个独立任务各一对
      （`core/orchestrator.py` + `demo_w10.py`）
- [x] HITL 审批钩子：`write_file` 每次、`run_pytest` 首次调用需要"批准"，体现"能力等级传递"
      原则（`core/hitl.py`，接入 `ReActAgent.run`/`arun`）
- [x] A2A 消息传递：Reviewer 通过显式 `AgentMessage`（sender/receiver/content）收到 Coder 的
      `final_answer`，不共享工作记忆（`core/orchestrator.py` 的 inbox 机制）
- [x] `asyncio.gather` + `Semaphore` 并行执行 + 串行对比：`--max-concurrency` 可调，
      实测 lkeap 真实网络下并行（4）总耗时 ~25s vs 串行（1）~46s，约 1.8x
- [x] `demo_w10.py`：mock / strict / lkeap 三种后端均已实测跑通
- [x] LangGraph/CrewAI 重写同一流程的对照实现（`frameworks/`，独立 conda 环境
      `agent_frameworks` + `requirements-frameworks.txt`，两者均已用真实 lkeap 模型实测跑通；
      对比结论见 `frameworks/README.md`）

### Week 11（生态接入 + 评估）—— ✅ 已完成

- [x] 把 `core/tools.py` 的工具封装成 MCP server（官方 SDK，`mcp_server.py`），`core/mcp_backend.py`
      提供对齐 `ToolRegistry` 接口的 `MCPToolRegistry`，`ReActAgent.arun()` 零改动直接兼容
- [x] 白名单/沙箱升级（Trust by Design）：MCP 协议本身的工具白名单边界 + `ToolRegistry.scoped()`
      的注册表级最小权限硬约束（已回填进 `demo_w10.py` 的 Reviewer 构造逻辑）
- [x] 审计日志：`core/audit.py` 的 `AuditLog`，JSONL 落盘，接入 `ReActAgent` 的 `audit` 参数
- [x] 评估 harness：`core/eval.py` 的 `evaluate_outcome`（独立重跑 pytest）+
      `evaluate_trajectory`（先探索后写/验证再回答/连续重复调用三条启发式规则）
- [x] `demo_w11.py`：3 个独立任务（moving_average/flatten_list/is_palindrome），
      mock 和 lkeap 后端均已实测跑通，Resolve Rate 3/3

### 已知的零碎债（随时清理，非阻塞）

- [x] ~~`agent.py` 的 `StepTrace`/`AgentResult` 目前没有落盘到文件~~ —— 已在 Week 11 用
      `core/audit.py` 的 `AuditLog` 解决（JSONL 落盘，见 `audit_logs/`）
- [x] ~~`OpenAICompatBackend` 目前是同步调用~~ —— 已在 Week 10 加 `adecide`（`AsyncOpenAI`），`decide`/`adecide` 各自独立客户端
- [ ] `Orchestrator` 暂无 Worker 失败重试/降级逻辑（讲义附录 A.3 讨论的三种策略）——`_run_one` 抛出的异常会通过 `asyncio.gather` 直接向上传播、中断整个 `run()`；Phase 5（Week 9-11）到此收官，这条留给未来如果重新拾起这个项目时再补，不阻塞任何已规划的工作
- [ ] LangGraph/CrewAI 对照实现（`frameworks/`）目前没有覆盖 AutoGen（讲义 Part 3.2 讨论的第三个框架），也没有演示 LangGraph 的 Time-Travel 回溯调试（附录 A.1）——按需补充，不阻塞
- [ ] **"透明化"功能（`--inspect` flag / DebugBackend）尚未实现**：Week 9 阶段验证真实后端（aihubmix）跑通后曾承诺要做——把 `to_openai_messages()`/`to_anthropic_messages()` 转换后、真正发给 API 的原始请求 payload 打印/落盘出来，让"中性抽象 vs 真实协议"这组对比对用户可见（现有 `verbose=True` 只打印中性事件流的思考/工具调用/观察，不是转换后的原始 payload）。当时被 aihubmix key 泄露、切换 lkeap 的插曲岔开，一直没做。不紧急、不阻塞，但如果要做，建议范围收在 `demo_w9.py`（最贴近最初提出它的单 Agent + 真实后端场景），不用牵扯 Week 10/11 已完工的部分。

## 常见问题

**Q: 为什么第一次跑 mock 后端时，Agent 会先写出一个有 bug 的实现？**
这是刻意设计（`MockCodingBackend` 里的 `BUGGY_CODE`），用来演示 ReAct 区别于一次性 CoT 的核心：行动的真实结果（pytest 失败）会反过来修正后续推理，而不是模型一次性把所有步骤想完。换真实模型（`--backend lkeap`）后，模型可能一次就写对——这恰好说明 mock 是"剧本"而真模型是真推理。

**Q: `demo_w10.py` 里 Reviewer 会不会偷懒，直接相信 Coder 的说法就下结论？**
真实模型第一版实测确实出现过这个情况——Reviewer 读完代码就直接给出空结论，没有独立跑 pytest。这是 role_instructions 措辞不够强导致的真实观察（不是靠猜测），修复方式是在提示词里明确要求"不能仅凭上游消息下结论，必须调用 run_pytest 验证"，修正后 Reviewer 稳定地会独立验证再给出 PASS/FAIL 结论。这也是提示词工程的一个真实案例：角色分工不能只靠 system prompt 里的一句话身份声明，关键约束要显式写出来。

**Q: pytest 结果时好时坏？**
已修复：`run_pytest` 强制禁用字节码缓存（`PYTHONDONTWRITEBYTECODE=1` + `-p no:cacheprovider`），并且 `demo_w9.py` 每次运行前用 `shutil.rmtree` 整个清空沙箱目录（含 `__pycache__`），避免 Agent 反复改代码时读到过期的 `.pyc`。

**Q: `evaluate_trajectory` 一开始为什么把每个任务都判定为"违反先读后写"？**
真实踩过的坑：最初的实现检查"write_file 之前是否读过*同一路径*"，但本项目的任务场景是从零创建目标文件——第一次 write_file 时那个文件根本不存在，要求"读过它"没有意义。改成检查全局顺序（第一次探索 list_dir/read_file 是否发生在第一次 write_file 之前）后，mock 和 lkeap 的正常轨迹都能正确判定为"无异常"。这是先跑通再看真实结果、而不是凭直觉写评估规则的一个具体例子。

**Q: MCP Server 对未注册的工具名会怎么处理？**
客户端调用一个不存在的工具名时，mcp SDK 会先打一条警告（"Tool 'xxx' not listed, no validation will be performed"），然后请求仍会转发到 `call_tool` 处理函数——真正的拒绝逻辑在 `registry.execute()` 里（返回"不存在名为 'xxx' 的工具"这样的可读错误，而不是抛异常）。也就是说 MCP 协议层的白名单主要体现在 `list_tools()` 只声明真实存在的工具，`call_tool` 本身仍然依赖应用层的校验——这也是为什么 `ToolRegistry.execute()` 的容错设计在 MCP 封装之后依然重要，不能假设协议层会完全兜底。
