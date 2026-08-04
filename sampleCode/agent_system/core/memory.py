"""
四层记忆系统（Week 9 地基）
================================

对应讲义：Week 9 · Part 4「记忆系统设计——让 Agent 不再失忆」
          Week 9 · Part 4.2「四层记忆分层架构」
          Week 9 · Part 4.4「Context Rot（上下文腐烂）问题」

四层记忆速查
------------
| 记忆类型 | 存什么                       | 生命周期      | 本文件实现            |
|----------|------------------------------|---------------|-----------------------|
| 工作记忆 | 当前任务的 ReAct 轨迹        | 单次任务      | WorkingMemory         |
| 情节记忆 | 历史任务的完整轨迹           | 跨任务        | EpisodicMemory        |
| 语义记忆 | 项目事实/约定（类 CLAUDE.md）| 长期          | SemanticMemory        |
| 程序记忆 | 可复用的 SOP/技能            | 长期          | ProceduralMemory      |

工程取舍
--------
语义记忆在真实系统里用 **向量数据库 + embedding** 做相似度检索（见 Week 9 附录 A.3）。
为了让本教学代码零外部依赖、可离线运行，这里用一个**关键词重叠打分**当作
"穷人版 embedding"。检索接口（store / retrieve）与真实系统一致，未来把打分函数
换成向量余弦相似度即可——这就是讲义强调的"接口稳定、实现可替换"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


def _tokenize(text: str) -> set[str]:
    """极简分词：抽取英文单词与中文字符，转小写。仅用于离线相似度演示。"""
    text = text.lower()
    words = set(re.findall(r"[a-z_][a-z0-9_]+", text))
    chars = set(re.findall(r"[一-鿿]", text))
    return words | chars


def _overlap_score(query: str, doc: str) -> float:
    """关键词重叠相似度（Jaccard 的简化版），代替向量余弦相似度。"""
    q, d = _tokenize(query), _tokenize(doc)
    if not q or not d:
        return 0.0
    return len(q & d) / len(q | d)


# ──────────────────────────────────────────────────────────────────────────
# 1) 工作记忆：当前任务的 ReAct 轨迹（短期、易溢出 → 需要管理）
# ──────────────────────────────────────────────────────────────────────────
class WorkingMemory:
    """保存当前任务正在进行的对话/轨迹。

    这是最易触发 Context Rot 的地方：轨迹越长，无关历史越多，模型越容易迷失。
    这里用一个简单的"保留 system 提示 + 最近 N 条"的截断策略来演示对抗手段
    （真实系统还会做摘要压缩、工具结果裁剪等，见 Harness 专题）。
    """

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.messages: list[dict] = []

    def add(self, role: str, content: str = "", **extra) -> None:
        self.messages.append({"role": role, "content": content, **extra})

    def view(self) -> list[dict]:
        """返回喂给 LLM 的轨迹；超长时只保留最近 max_turns 条（防止上下文腐烂）。"""
        if len(self.messages) <= self.max_turns:
            return list(self.messages)
        return self.messages[-self.max_turns:]

    def clear(self) -> None:
        self.messages.clear()


# ──────────────────────────────────────────────────────────────────────────
# 2) 情节记忆：跨任务的历史轨迹，可按相似度召回"上次怎么做的"
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Episode:
    task: str
    trajectory_summary: str
    success: bool


class EpisodicMemory:
    def __init__(self):
        self._episodes: list[Episode] = []

    def record(self, task: str, trajectory_summary: str, success: bool) -> None:
        self._episodes.append(Episode(task, trajectory_summary, success))

    def retrieve(self, query: str, top_k: int = 2) -> list[Episode]:
        """召回与当前任务最相似的历史 episode，用于"经验复用"。"""
        scored = sorted(
            self._episodes,
            key=lambda e: _overlap_score(query, e.task + " " + e.trajectory_summary),
            reverse=True,
        )
        return [e for e in scored[:top_k] if _overlap_score(query, e.task) > 0]


# ──────────────────────────────────────────────────────────────────────────
# 3) 语义记忆：项目事实/约定（类 CLAUDE.md），长期稳定
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Fact:
    key: str
    content: str


class SemanticMemory:
    """存放关于项目的事实性知识，检索后注入 system prompt。

    在 Claude Code 里，CLAUDE.md 就是语义记忆的工程化形态（见 Week 10 案例）。
    """

    def __init__(self):
        self._facts: list[Fact] = []

    def store(self, key: str, content: str) -> None:
        self._facts.append(Fact(key, content))

    def retrieve(self, query: str, top_k: int = 3) -> list[Fact]:
        scored = sorted(
            self._facts,
            key=lambda f: _overlap_score(query, f.key + " " + f.content),
            reverse=True,
        )
        return [f for f in scored[:top_k] if _overlap_score(query, f.key + " " + f.content) > 0]


# ──────────────────────────────────────────────────────────────────────────
# 4) 程序记忆：可复用的标准作业流程（SOP）/技能
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Skill:
    name: str
    steps: list[str]


class ProceduralMemory:
    """存放"怎么做某类任务"的流程。Week 10 的 Hermes 案例会让 Agent 自动合成技能；

    这里先做成手工注册 + 按需取用的静态版本。
    """

    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def add_skill(self, name: str, steps: list[str]) -> None:
        self._skills[name] = Skill(name, steps)

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def all_skills(self) -> list[Skill]:
        return list(self._skills.values())


# ──────────────────────────────────────────────────────────────────────────
# 统一封装：一个 Agent 持有一套四层记忆
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class MemorySystem:
    working: WorkingMemory = field(default_factory=WorkingMemory)
    episodic: EpisodicMemory = field(default_factory=EpisodicMemory)
    semantic: SemanticMemory = field(default_factory=SemanticMemory)
    procedural: ProceduralMemory = field(default_factory=ProceduralMemory)

    def build_memory_context(self, task: str) -> str:
        """把长期记忆（语义/情节/程序）拼成一段文本，注入 system prompt。

        这一步是 RAG+记忆增强的核心（Week 9 Part 4.3）：检索相关记忆 → 拼接进上下文。
        """
        blocks: list[str] = []

        facts = self.semantic.retrieve(task)
        if facts:
            blocks.append("【项目约定（语义记忆）】\n" + "\n".join(f"- {f.content}" for f in facts))

        episodes = self.episodic.retrieve(task)
        if episodes:
            lines = [
                f"- 任务「{e.task}」{'✅成功' if e.success else '❌失败'}：{e.trajectory_summary}"
                for e in episodes
            ]
            blocks.append("【相似历史经验（情节记忆）】\n" + "\n".join(lines))

        skills = self.procedural.all_skills()
        if skills:
            lines = [f"- {s.name}：{' → '.join(s.steps)}" for s in skills]
            blocks.append("【可复用流程（程序记忆）】\n" + "\n".join(lines))

        return "\n\n".join(blocks)
