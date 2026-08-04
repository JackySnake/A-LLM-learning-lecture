# Harness 工程与上下文工程：模型之外的另一半 Agent

*补充来源：Week 9–11 复盘时发现，讲义只在 Meta-Harness（Week 11 3.4，把 Harness 当自动优化对象）处触及 Harness，而"Harness 本身怎么设计"这门 2025–2026 年的显学没有展开；Anthropic Agent Skills（渐进式披露）也未覆盖（2026-06）*

> **前置知识**：Week 9（ReAct、工具调用、Context Rot）、Week 10（Claude Code 案例、Hermes Skill Synthesis）、Week 11（Meta-Harness）。与 [AgenticRL 专题](AgenticRL与Agent训练.md) 互为表里：那边讲模型怎么训练，这边讲模型外面那层壳怎么造。

---

## 一、Harness：被名字耽误的核心概念

### 1.1 定义与组成

$$\text{Agent} = \text{Model} + \text{Harness}$$

**Harness（运行挽具）是包裹在模型外面的全部循环代码**——模型只负责"给定上下文，输出下一段 token"，其余一切都是 Harness 的职责：

```
┌─────────────────────── Harness ───────────────────────┐
│                                                        │
│  System Prompt 工程     工具集定义与描述（ACI）          │
│  Agent Loop 驱动        工具执行与结果回填               │
│  上下文管理（压缩/卸载）  权限与沙箱                      │
│  Subagent 调度          错误处理 / 重试 / 终止判断       │
│  Skills / 记忆加载       Hooks 与可观测性                │
│                                                        │
│              ┌──────────────────┐                      │
│              │      Model       │  ← 只做一件事：        │
│              │  (context→token) │    续写上下文          │
│              └──────────────────┘                      │
└────────────────────────────────────────────────────────┘
```

回头看 Week 9–11，会发现讲过的大部分内容其实都是 Harness 的某个部件：ReAct 循环驱动（W9 Part 2）、工具可靠性工程（W9 3.6）、记忆系统（W9 Part 4）、Subagent 与权限沙箱（W10 4.1）、工具检索层（W11 1.6）。**本专题做的事是把这些散件装回一台整机，并补上讲义没讲的两个部件：上下文工程方法论与 Skills。**

### 1.2 Harness 有多重要：同一个模型，分数差出十几个点

Harness 不是"胶水代码"，它直接决定 Agent 性能。几个标志性证据：

- **SWE-agent 论文（2024）的核心贡献不是模型而是接口**。它提出 **ACI（Agent-Computer Interface）**概念：人类用 IDE 高效工作，但 IDE 的交互方式（滚动、点击、大量视觉信息）对 LLM 是灾难。为 LLM 重新设计的文件查看器（每次只显示 100 行 + 行号）、带语法检查反馈的编辑命令，让**同一个 GPT-4** 在 SWE-Bench 上的成绩翻了数倍。
- **mini-SWE-agent 现象（2025）**：SWE-agent 团队用 ~100 行 Python（只给模型一个 bash 工具）就让强模型达到接近完整 Harness 的 SWE-Bench 成绩——但换成弱模型，简陋 Harness 的成绩就崩塌。结论很微妙：**模型越强，越能自己兜住 Harness 的简陋；模型越弱，越依赖 Harness 的精心设计**。
- **榜单可比性危机**：同一模型配不同 Harness，SWE-Bench/Terminal-Bench 分数可以差 10–20 个点。所以 2025 年起各榜单开始强制标注 Harness（这也是 Week 11 Meta-Harness 工作的出发点——既然 Harness 是关键变量，干脆把它变成优化对象）。

### 1.3 ACI 设计原则：把模型当"没有眼睛和肌肉记忆的工程师"

SWE-agent 与 Anthropic 工具设计博文的经验可以浓缩为四条：

1. **反馈要短而信息密**：工具返回 10 万 token 的原始 HTML 是在谋杀上下文——返回提炼后的文本 + "完整内容已存至 xx 文件"；
2. **错误要可操作**：`Error: invalid input` 是废话；`参数 date 格式应为 YYYY-MM-DD，收到的是 "6月1日"` 让模型一步改对（呼应 W9 3.6 可靠性工程）；
3. **防呆设计（guardrails）**：模型会犯人类不会犯的错——编辑命令自带语法检查并拒绝写入坏代码，比事后报错好得多；
4. **工具数量克制、职责正交**：20 个边界模糊的工具不如 7 个边界清晰的（工具多到必须检索时才上 W11 1.6 的工具检索层）。

一个值得内化的心智模型：**写工具描述和错误信息时，想象接收者是一位能力极强、但失忆、且看不到屏幕的新同事**——他只能依赖你写下的文字行动。

### 1.4 Harness 与模型的协同演化

Harness 不是模型无关的：模型在特定 Harness 格式下做过 agentic 训练（见 [AgenticRL 专题](AgenticRL与Agent训练.md) 3.3 节"问题三"），换一种上下文组织方式就是分布偏移。实践推论：

- 用厂商模型时，**优先沿用厂商的原生格式**（如 Claude 的 tool use 块、interleaved thinking 的保留规则），别自创 XML 格式再正则解析；
- 这也是 Claude Code → Claude Agent SDK 这条产品线的逻辑：Anthropic 直接把自家训练时对齐过的 Harness 产品化卖给你。

---

## 二、上下文工程：管理最稀缺的资源

### 2.1 从 Prompt Engineering 到 Context Engineering

2025 年中起，"context engineering" 取代 "prompt engineering" 成为 Agent 工程的中心词。区别在对象：

- **Prompt 工程**：写好一段静态指令（一次性的）；
- **上下文工程**：管理一个**随轮次动态增长的 token 流**——什么进来、什么留下、什么被压缩、什么被丢弃。

为什么是核心问题？Week 9 4.4 讲过 Context Rot：上下文不是越长越好，注意力是会被稀释的预算（attention budget）。一个跑 50 轮的 Agent 如果不加管理，工具返回的原始输出会把窗口塞爆，而且**信噪比持续恶化**——第 3 轮那次失败的网页抓取的完整 HTML，到第 40 轮还躺在上下文里消耗注意力。

业界方法论可以总结为**三板斧：压缩、卸载、隔离**。

### 2.2 第一板斧：压缩（Compaction）

**做法**：当上下文逼近阈值（如窗口的 80%），把历史轨迹交给模型做一次结构化摘要，用摘要替换原文，继续干活。Claude Code 的 auto-compact 是标准实现。

**关键不在"压"而在"保"**——压缩是有损操作，错保了就是永久失忆。优先保留：

| 必须保留 | 可以丢弃 |
|---|---|
| 任务目标与当前进度 | 旧轮次工具的原始输出 |
| 已做出的决策及理由 | 已修复问题的报错全文 |
| 关键文件路径 / ID / URL | 中间探索的死胡同细节 |
| 未解决的问题清单 | 历史轮次的思考过程 |

**工程细节**：压缩点的选择影响 KV cache（见 2.5）；摘要 prompt 本身需要迭代调优——"丢了什么"往往要在长任务失败后才暴露，这是上下文工程最难调试的部分。

### 2.3 第二板斧：卸载（Offloading）

**核心思想（Manus 博客的著名论断）：文件系统是终极上下文**——上下文窗口是昂贵的工作记忆，文件系统是无限的外部记忆。

具体模式：

1. **工具结果引用化**：大体积结果（网页全文、查询结果集）写入文件，上下文里只留路径和一行摘要；需要时再读回来。**可恢复性是关键**——压缩是不可逆的，卸载是可逆的；
2. **计划外置 + 复述**：把任务计划写进 `todo.md`，每完成一步就更新并重读。这不只是记录——把目标反复"复述"到上下文**末尾**，是对抗 lost-in-the-middle / 长任务目标漂移最便宜的手段（Manus 称之为 "recitation"）；
3. **跨会话持久化**：笔记文件、CLAUDE.md 类项目记忆——这把卸载从"本次任务的草稿纸"升级为 Week 9 记忆分层里的情节/语义记忆载体。

### 2.4 第三板斧：隔离（Isolation）

**做法**：把上下文消耗大户（如"读 50 个文件找一个答案"）派给 **Subagent**——它在自己独立的上下文窗口里翻完所有文件，只把三句话结论返回主线程。主 Agent 的上下文只增加三句话，而不是 50 个文件。

这揭示了一个 Week 10 没有点破的视角：**多智能体架构的第一性价值往往不是"分工协作"，而是上下文隔离**。Claude Code 的 Subagent、Anthropic 多智能体研究系统的并行检索 worker，本质上都是"用独立上下文窗口换主线程的注意力预算"。（单 vs 多 Agent 的完整决策框架见 Week 10 附录 A.5）

三板斧的选择逻辑：**先卸载（无损、便宜）→ 不够再隔离（适合读密集子任务）→ 最后才压缩（有损、兜底）**。

### 2.5 KV Cache 友好设计：上下文工程与推理优化的交汇

这是 Manus 博客中最"工程"也最容易被忽视的一节，把 Week 6 的 prefix caching 知识接到了 Agent 场景：

Agent 的上下文是典型的"长前缀 + 增量追加"模式，prefix cache 命中与否带来约 **10 倍**的首 token 成本/延迟差异。三条纪律：

1. **前缀保持稳定**：不要在 system prompt 开头放时间戳/随机 ID——开头变一个 token，整条 cache 作废；
2. **只追加，不回改**：不要回头修改或重排历史消息（这也约束了压缩的实现：compaction 必然全量作废 cache，所以要低频、批量地做）；
3. **工具集不要动态增删**：每次增删工具都会改动靠前的工具定义区 → cache 全灭。需要限制可用工具时，**用约束解码 / logits mask 屏蔽**（Week 9 3.3 的 Constrained Decoding 在此复用），而不是改工具列表。

---

## 三、Skills：渐进式披露的程序性知识

### 3.1 Agent Skills 是什么

Anthropic 于 2025-10 提出 **Agent Skills**：把某项任务的操作知识打包成一个文件夹——

```
pdf-processing/
├── SKILL.md          # 必需：frontmatter（name + description）+ 操作指南正文
├── reference.md      # 可选：深度参考资料
└── scripts/
    └── fill_form.py  # 可选：可执行脚本（确定性操作不必让模型现写代码）
```

精髓在**渐进式披露（progressive disclosure）**的三级加载：

| 层级 | 加载时机 | 上下文成本 |
|---|---|---|
| ① name + description | 启动时全部预载 | 每个 Skill 仅几十 token |
| ② SKILL.md 正文 | 模型判断任务相关时才读取 | 千 token 级 |
| ③ 附带文件 / 脚本 | 正文指引下按需读取/执行 | 用多少付多少 |

**这本质上是一种上下文工程机制**：装 100 个 Skills，闲置成本只有 100 条元数据；模型像人翻手册一样——先看目录，需要哪章翻哪章。对比"把所有操作指南全塞进 system prompt"，是又一次"按需加载替代全量预载"（与 Week 11 1.6 工具检索层解决的是同构问题，只是这里让**模型自己**做检索决策，而非外挂检索器）。

### 3.2 Skill 与工具的区别：能力 vs 知识

容易混淆，一句话区分：

> **工具（Tool/MCP）给 Agent 接上"手"——能做什么；Skill 给 Agent 一本"操作手册"——该怎么做。**

| | 工具 / MCP | Skill |
|---|---|---|
| 本体 | API 接口 + JSON Schema | Markdown 说明书（可附脚本） |
| 解决的问题 | 能力边界（连接外部世界） | 程序性知识（流程、规范、经验） |
| 典型内容 | `query_database(sql)` | "本公司周报的标准生成流程：先查 X 表，注意 Y 坑，格式参照 Z" |
| 与对方的关系 | 被 Skill 指导着使用 | 常常编排多个工具的使用顺序 |

两者互补：MCP 接入数据库工具，Skill 教模型"我们团队的库表惯例和查询规范"。

### 3.3 与 Hermes Skill Synthesis 的对照：同一记忆层的两种哲学

Week 10 4.2 讲的 Hermes 技能合成与 Agent Skills 同名不同物，但恰好构成一组漂亮的对照——两者都落在 Week 9 记忆分层的**程序记忆**层：

| | Anthropic Agent Skills | Hermes Skill Synthesis |
|---|---|---|
| 来源 | **人工编写**（专家知识下沉） | **Agent 自动合成**（任务经验上浮） |
| 形态 | Markdown 指南 + 可选脚本 | 可执行代码（Python 函数） |
| 加载机制 | 渐进式披露（模型自主决定） | 向量检索（外挂相似度匹配） |
| 质量保障 | 人工审核，质量可控 | 自动验证，存在老化/去重问题（W10 A.4） |
| 哲学 | 知识管理：把组织的 know-how 结构化交给 Agent | 持续学习：让 Agent 从自己的经验中积累 |

成熟系统会两者并用：人工 Skills 托底（合规流程、领域规范这类不容出错的知识），自动合成扩展长尾。这组对照也呼应了 AgenticRL 专题第八节的"参数化 vs 非参数化自我改进"：RL 改权重、Hermes 存代码、Skills 存文档——三种把经验固化下来的位置不同的方案。

---

## 四、实践启示：Agent 表现差，先审 Harness

模型能力短期内你改变不了，Harness 今天下午就能改。诊断顺序建议：

1. **看一条完整轨迹的原始上下文**（不是日志摘要，是模型实际看到的 token 流）——90% 的问题在这一步现形：工具返回了几万 token 垃圾？错误信息不可操作？目标淹没在中段？
2. **查工具描述**：拿给一个不了解系统的人看，他能正确选择和调用吗？不能，模型也不能；
3. **查上下文增长曲线**：第 30 轮时窗口里还剩多少比例是"当前有用"的信息？
4. **查格式对齐**：是否在用模型训练时见过的工具调用格式？
5. 以上都没问题，再考虑换模型 / 微调（进入 AgenticRL 专题的领域）。

新系统设计 checklist：工具反馈短而密 ✦ 错误可操作 ✦ 大结果落盘引用化 ✦ 计划外置并复述 ✦ 读密集子任务隔离给 subagent ✦ 前缀稳定不动工具集 ✦ 操作知识写成 Skills 而非塞 system prompt。

---

## 五、前沿动态（2025–2026）

- **Anthropic**：这条线上输出最系统的一家——《Building Effective Agents》《Effective Context Engineering for AI Agents》《Writing Tools for Agents》系列博文 + Claude Agent SDK（把 Claude Code 的 Harness 整体产品化：自动 compaction、subagent、hooks、Skills 全内置）+ Skills 开源仓库。其策略与训练侧保密形成互补：**模型配方不公开，Harness 方法论全公开**。
- **Manus**：《Context Engineering for AI Agents: Lessons from Building Manus》是 2025 年被引用最多的工程博文之一，KV cache 纪律、文件系统即上下文、recitation、"保留错误轨迹让模型自己看到失败"等论点大多已成行业共识。
- **OpenAI**：Agents SDK / AgentKit 走同样的"官方 Harness"路线；Responses API 把工具执行（搜索、代码解释器）收进 API 内部——**Harness 边界向模型侧迁移**的信号。
- **争论：Harness 是护城河还是过渡产物？** Bitter Lesson 派认为模型会逐步"吃掉" Harness：上下文变长 → compaction 不再必要；模型原生多轮训练 → 外置计划复述成为冗余。反方观察：过去两年模型每变强一代，Harness 不是变薄而是变厚（Claude Code 的代码量持续增长），因为强模型解锁了更长的任务，而更长的任务需要更多管理。目前的共识性判断：**具体技巧会过时（为模型缺陷打的补丁最先死），但"管理注意力预算"这个问题与任务长度同增长，不会消失**。
- **评测侧**：Harness 标准化（榜单强制披露 Harness 配置）与 Week 11 Meta-Harness 的自动优化是同一问题的两个回应：前者控制变量，后者干脆把变量变成优化对象。

---

## 六、与正式讲义的连接

| 本专题内容 | 连接讲义 | 关系 |
|---|---|---|
| Harness 组成清单 | Week 9 Part 2/3、Week 10 4.1 | 把散讲的部件装回整机 |
| ACI 设计原则 | Week 9 3.6 工具可靠性 | 从"可靠性"上升到"接口设计哲学" |
| Compaction / Context Rot | Week 9 4.4 | Context Rot 是问题，三板斧是解法体系 |
| 隔离 = 多 Agent 第一性价值 | Week 10 Part 2、附录 A.5 | 重新解读 Orchestrator-Worker 的动机 |
| KV cache 纪律 | Week 6 prefix caching | 推理优化知识在 Agent 场景的复用 |
| Skills 渐进式披露 | Week 11 1.6 工具检索层 | 同构问题：按需加载；区别：谁做检索决策 |
| Skills vs Hermes 对照 | Week 9 4.2 程序记忆、Week 10 4.2 | 程序记忆层的人工 vs 自动两条路线 |
| Harness-模型协同演化 | AgenticRL 专题 3.3 | 训练-推理一致性的 Harness 侧视角 |

### 检查点

- [ ] 能列出 Harness 的主要组成部件，并解释 "Model + Harness = Agent"
- [ ] 能用 ACI 的视角解释：为什么同一模型换 Harness 在 SWE-Bench 上分差巨大
- [ ] 能说出上下文工程三板斧（压缩/卸载/隔离）各自的适用场景与代价，及正确的使用顺序
- [ ] 能解释为什么动态增删工具集会摧毁 KV cache，正确做法是什么
- [ ] 能区分 Tool / MCP / Skill 各自解决什么问题
- [ ] 能对比 Agent Skills 与 Hermes Skill Synthesis 的设计哲学
- [ ] 拿到一个表现差的 Agent，知道按什么顺序诊断 Harness
