# Week 10 框架对照实现：LangGraph vs CrewAI vs 手搓版

`sampleCode/agent_system/core/orchestrator.py`（见上级目录 README、Week 10 讲义附录 A.6）用纯 Python + asyncio 手搓了一个 Orchestrator-Worker 系统。这个目录用**同样的任务场景**（两个独立的"改代码通过 pytest"子任务：`moving_average` / `flatten_list`，各自先 Coder 实现，再 Reviewer 复核）分别用 LangGraph 和 CrewAI 重新实现一遍，方便直接对照——同一个问题，三种不同的编排方式。

对应讲义：Week 10 Part 3.1（LangGraph）/ 3.3（CrewAI）/ 3.4（框架对比与选型指南），以及附录 A.6。

## 环境搭建（独立于 claude_p13）

`langgraph`/`crewai` 各自锁定 `langchain-core`/`pydantic`/`litellm` 的版本范围，容易和主环境的其他依赖冲突，所以单独建一个 conda 环境：

```bash
conda create -n agent_frameworks python=3.11 -y
conda activate agent_frameworks
cd sampleCode/agent_system/frameworks
pip install -r requirements-frameworks.txt
```

两份对照实现都需要真实模型（复用 `sampleCode/agent_system/.env` 里的 `LKEAP_API_KEY`）——这一点和 `demo_w9.py`/`demo_w10.py` 不同：那两个有 mock/strict 离线后端，这里没有。原因很直接：框架的核心卖点就是"帮你编排真实模型的决策"，硬套一个规则状态机会让对照失去意义。

```bash
python langgraph_demo.py
python crewai_demo.py
python langgraph_demo.py --interactive   # HITL 改为真正等待终端确认，两个脚本都支持
```

## 三版本对照表

| 维度 | 手搓版（`core/orchestrator.py`） | LangGraph（`langgraph_demo.py`） | CrewAI（`crewai_demo.py`） |
|---|---|---|---|
| 依赖调度 | 手写 wave 循环，反复找"依赖已完成"的子任务集合 | 图的拓扑结构本身 + BSP 并发执行，同一 superstep 内独立分支自动并发 | 框架不管跨任务并行——`Process.sequential` 只管一个 Crew 内部顺序执行；跨 Crew 的并行要自己在外层写 `asyncio.gather`，做法和手搓版一致 |
| Coder→Reviewer 通信 | 显式 `AgentMessage`（方案一：消息传递） | 共享的 `messages` 状态通道（方案二：共享状态，Reviewer 能看到 Coder 的完整过程） | `Task(context=[coder_task])`——显式声明依赖，框架自动把被引用 Task 的**最终输出文本**注入（方案一：消息传递，和手搓版同一范式） |
| HITL 粒度 | 手写 `HITLGate.check()`，工具调用级，同步拦截 | 原生 `interrupt()` + `Command(resume=...)` + checkpointer，工具调用级 | 原生 `human_input=True`，**任务级**——整个 Task 跑完才审批一次；拒绝时是带反馈重跑整个 Task，不是把拒绝原因当一条 observation 塞回同一轮推理 |
| 角色隔离 | 靠 prompt 里的文字说明（软约束） | 按角色 `bind_tools()` 只绑定允许的工具（硬约束） | 按 Agent 只绑定允许的工具（硬约束，和 LangGraph 一致） |
| 是否需要真实 API Key | 否（mock/strict 离线可跑） | 是 | 是 |
| 代码量（不含公共 core/ 工具） | ~280 行（`orchestrator.py` + `hitl.py`） | ~230 行 | ~200 行 |

## 实测耗时对照（`--backend lkeap` / 真实模型，两个独立任务并发）

| 实现 | 总耗时 |
|---|---|
| 手搓版（`demo_w10.py`，并发上限 4） | 25.4s |
| LangGraph 版 | 24.9s |
| CrewAI 版 | 34.1s |

三者耗时接近，CrewAI 略慢——推测和它的 Agent Executor 自带更多提示词包装（role/goal/backstory 拼接）、以及 verbose 模式下的 rich 终端渲染开销有关，没有专门做归因实验，这里只报告观察到的数字，不下确定性结论。

## 诚实的框架局限笔记

**LangGraph 的默认习惯用法是共享状态，不是消息传递。** 要故意做成消息传递（每个节点只拿到"别人想让它看到的"内容）反而要跟框架"拧着来"。这印证了讲义 Part 2.3 的判断——"代表框架：LangGraph（图中的 State 对象）"。框架的默认路径会悄悄替你做出架构选择，选框架本身就是一次架构决策。

**CrewAI 没有内建"拦截单次工具调用"的机制。** 它的角色化 API（Agent/Task/Crew）把控制粒度设计在"任务"这一层——这也解释了为什么它的 API 比 LangGraph 更简单：更少的控制点，也意味着更少的可控粒度。如果一定要在 CrewAI 里做工具级 HITL，得自己在 Tool 的 `_run()` 里手写拦截逻辑，等于把 LangGraph/手搓版已经做的事在框架的工具层再实现一遍——框架本身没有提供现成支持。`crewai_demo.py` 里的 `ApprovalHumanInputProvider` 用的是 CrewAI 较新的 `crewai.core.providers.human_input` 可插拔 provider 接口，只把默认的阻塞式终端确认换成我们自己的 `approve_cb`，粒度本身没变。

**两个框架都支持"按角色限定工具集"这个硬约束，比手搓版的软约束（prompt 里写"不要改代码"）更可靠。** 这是本次对照里少数两个框架都比手搓版更强的地方——Reviewer 的 LLM 调用里根本不存在 `write_file` 这个选项，模型想违反角色分工都做不到，不需要依赖它"听话"。

**这次对照没有覆盖的**：LangGraph 的 Time-Travel/回溯调试（讲义附录 A.1 讨论过，这里的 checkpointer 只用来支持 `interrupt()`/`resume`，没有演示回到某个历史 checkpoint 重新分支执行）；CrewAI 的 Flow（更底层、事件驱动的编排原语，Crew 只是它的高层封装之一）；AutoGen（讲义 Part 3.2 讨论的第三个框架，这次对照只做了 LangGraph/CrewAI 两个）。
