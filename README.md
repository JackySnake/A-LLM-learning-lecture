# 大语言模型系统学习讲义

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## 这是什么

这是我在系统学习大语言模型过程中整理的一套笔记讲义，从 2025 年初开始，按 15 周的计划推进，目前已完成 Phase 0 + Week 1–10，覆盖从 Transformer 基础架构到多智能体系统的完整技术栈。

讲义的撰写方式有点特别：我以自己的知识背景为基础，大量借助 AI（主要是 Claude）来帮我整理内容、推导细节、补充前沿动态，中间也穿插了很多轮交互讨论，让知识点逐渐丰富起来。所以这套讲义既不是纯人工写作，也不是简单的 AI 生成——更像是一个持续的**人机协作学习过程**的记录。

我选择把它发出来，一方面是希望自己的学习经历能对同样在摸索这条路的人有些参考价值；另一方面，**也真诚地希望大家帮我查缺补漏**。这套内容不可避免会有疏漏、表述不严谨甚至错误的地方——无论是概念有误、推导跳步，还是遗漏了某个重要知识点，都欢迎[提 Issue](../../issues) 告诉我，我会认真查看并修订。

---

## 讲义目录

| 章节 | 主题 | 核心内容 | 下载 |
|------|------|----------|------|
| Phase 0 | 大模型完整生命周期导引 | 预训练 → SFT → RLHF → 推理部署，全局架构鸟瞰 | [📄 PDF](pdfs/Phase0讲义.pdf) |
| Week 1 | 现代大模型架构原理 | Transformer 精讲、RoPE、RMSNorm、SiLU/SwiGLU | [📄 PDF](pdfs/Week1讲义.pdf) |
| Week 2 | 进阶架构与大规模训练工程 | MoE、GQA/MQA/MLA、3D 并行、DeepSpeed/FSDP | [📄 PDF](pdfs/Week2讲义.pdf) |
| Week 3 | 参数高效微调与 SFT 实战 | LoRA、QLoRA、DoRA、Prefix Tuning、Adapter | [📄 PDF](pdfs/Week3讲义.pdf) |
| Week 4 | RLHF 完整体系与对齐算法 | PPO、RLHF pipeline、DPO、DeepSeek GRPO | [📄 PDF](pdfs/Week4讲义.pdf) |
| Week 5 | 持续学习与模型融合 | 灾难性遗忘、EWC、Task Arithmetic、TIES、DARE | [📄 PDF](pdfs/Week5讲义.pdf) |
| Week 6 | 推理优化与框架生态 | PagedAttention、Continuous Batching、Speculative Decoding、vLLM | [📄 PDF](pdfs/Week6讲义.pdf) |
| Week 7 | Test-Time Scaling 与高级推理策略 | 解码策略、树搜索（MCTS）、过程奖励模型（PRM） | [📄 PDF](pdfs/Week7讲义.pdf) |
| Week 8 | 多模态与长序列处理 | Qwen-VL、M-RoPE、YaRN、StreamingLLM、LongRoPE | [📄 PDF](pdfs/Week8讲义.pdf) |
| Week 9 | Agent 基础范式与记忆系统 | ReAct、Tool Use、分层记忆架构、Context Rot | [📄 PDF](pdfs/Week9讲义.pdf) |
| Week 10 | 多智能体系统与工程案例 | LangGraph、AutoGen、Claude Code 架构、Hermes | [📄 PDF](pdfs/Week10讲义.pdf) |

后续的 Week 11（MCP 与 Agent 评估）、Week 12（DeepSeek 深度解析）、Week 13（Qwen 全家桶）等整理完成后会持续更新。

---

## 配套代码

`sampleCode/` 目录下有部分讲义的配套示例代码，目前覆盖 Week 1、4、7：

| 目录 | 文件 | 说明 |
|------|------|------|
| `week1/` | `rope_demo.py` | RoPE（旋转位置编码）的 PyTorch 核心实现，对应 Week 1 |
| `week4/` | `ppo_implementation_demo.py` | PPO 算法简化实现示意，对应 Week 4 |
| `week4/` | `dpo_implementation_demo.py` | DPO 训练流程实现示意，对应 Week 4 |
| `week4/` | `grpo_implementation_demo.py` | DeepSeek GRPO 算法实现示意，对应 Week 4 |
| `week7/` | `self_consistency_demo.py` | Self-Consistency 解码策略演示，对应 Week 7 |
| `week7/` | `tree_of_thoughts_demo.py` | Tree of Thoughts 搜索过程演示，对应 Week 7 |
| `week7/` | `production_llm_reasoning.py` | 生产级推理策略综合示例，对应 Week 7 |

代码以**演示和理解为主**，不追求生产级完整性，建议配合对应章节的 PDF 一起阅读。

---

## 适合谁看

- 有 Python 和深度学习基础，想系统了解 LLM 技术栈的工程师或研究者
- 正在准备大模型相关岗位面试的同学
- 对 LLM 工程化（微调、对齐、推理部署、Agent）有实践需求的人

不适合完全零基础的读者——讲义假设你对神经网络和反向传播有基本概念，但不假设你对 LLM 有任何了解，Phase 0 会从头带你建立整体认知。

---

## 参考资料

讲义内容主要基于以下核心论文，并结合业界实践补充：

- **Transformer**: Vaswani et al., 2017 — *Attention is All You Need*
- **RoPE**: Su et al., 2021 — *RoFormer: Enhanced Transformer with Rotary Position Embedding*
- **Scaling Law**: Hoffmann et al., 2022 — *Training Compute-Optimal Large Language Models* (Chinchilla)
- **Flash Attention**: Dao et al., 2022 — *FlashAttention: Fast and Memory-Efficient Exact Attention*
- **LoRA**: Hu et al., 2021 — *LoRA: Low-Rank Adaptation of Large Language Models*
- **DPO**: Rafailov et al., 2023 — *Direct Preference Optimization*
- **DeepSeek-V3**: DeepSeek-AI, 2024 — *DeepSeek-V3 Technical Report*
- **Qwen2-VL / Qwen3**: Qwen Team, 2024/2025 — Technical Reports

---

## 许可证

本作品采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可。

简单说：欢迎非商业目的的转载和改编，但请注明出处，且派生内容须沿用相同许可证。商业用途请先联系我。

---

<div align="center">

如果这份笔记对你有帮助，欢迎点个 Star ⭐<br>
发现错误或有改进建议，欢迎 <a href="../../issues">提 Issue</a>，真的很欢迎。

</div>
