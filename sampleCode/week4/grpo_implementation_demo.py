import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

"""
📘 Week 4 实战示例：GRPO (Group Relative Policy Optimization) 核心算法实现演示

【学习目标】
理解 DeepSeek 的 GRPO 算法是如何在保持 RL 能力的同时去掉 Critic Model 的。
本代码对应讲义 Week 4 的 Part 3.2 (从 DPO 到 GRPO) 章节。

【核心概念对应 - Part 3.2: GRPO 的革命】
1. Group Sampling (组采样) -> `group_sampling` 函数
   - 对同一个 Prompt，生成一组（如 G=8 个）不同的回答
2. Group Relative Advantage (组内相对优势) -> `compute_group_advantage` 函数
   - 不用 Critic 预测基线，而是用组内平均分作为基线
   - 公式：A_i = (r_i - mean(R_group)) / std(R_group)
3. 3模型架构 (Actor, RM, Ref) -> 初始化部分
   - 去掉了 Critic Model，但保留了 Reward Model
4. PPO 风格的更新 -> `train_grpo_step` 函数
   - 仍然使用 Importance Ratio 和 Clipping

【GRPO vs PPO vs DPO 的核心差异】
| 维度 | PPO | DPO | GRPO |
|------|-----|-----|------|
| 模型数量 | 4个 | 2个 | 3个 |
| Critic | ✅ 需要 | ❌ 不需要 | ❌ 不需要 |
| Reward Model | ✅ 需要 | ❌ 不需要 | ✅ 需要 (或规则) |
| 训练范式 | RL (在线) | SL (离线) | RL (在线) |
| Advantage 来源 | Critic + GAE | 隐式在 Loss 里 | 组内相对排名 |

【为什么 DeepSeek 选择 GRPO？】
1. 省显存：去掉了和 Actor 一样大的 Critic
2. 适合推理任务：数学题有确定性答案，可以用规则打分
3. 保留探索能力：仍然是 RL 范式，能涌现新解法

【注意】
这是一个"逻辑演示版"代码，为了可读性简化了工程细节。
"""


class GRPOConfig:
    """GRPO 超参数配置"""
    def __init__(self):
        self.lr = 1e-5                  # 学习率
        self.clip_range = 0.2           # PPO 裁剪范围
        self.kl_coef = 0.1              # KL 散度惩罚系数
        self.group_size = 8             # 每个 Prompt 采样的回答数量 G
        self.batch_size = 4             # Prompt 批次大小
        self.ppo_epochs = 4             # PPO 更新轮数


class MockTransformer(nn.Module):
    """模拟一个简单的 Transformer 模型接口"""
    def __init__(self, vocab_size=1000, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layer = nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        """前向传播，返回 logits"""
        x = self.embedding(input_ids)
        x = F.relu(self.layer(x))
        logits = self.lm_head(x)
        return logits

    def generate(self, prompt_ids, max_new_tokens=10, temperature=1.0):
        """
        简化版生成函数（用于演示）

        实际中会使用更复杂的采样策略（Top-p, Top-k 等）
        """
        self.eval()
        generated = prompt_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature

                # 多项式采样（引入随机性）
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=-1)

        return generated


class MockRewardModel(nn.Module):
    """
    模拟 Reward Model

    在 GRPO 中，Reward Model 可以是：
    1. 训练好的神经网络 RM
    2. 确定性规则（如数学题的正误判断）
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.scorer = nn.Linear(hidden_dim, 1)
        self.embedding = nn.Embedding(1000, hidden_dim)

    def forward(self, input_ids):
        """返回序列的奖励分数"""
        x = self.embedding(input_ids)
        x = x.mean(dim=1)  # 简单平均池化
        score = self.scorer(x).squeeze(-1)
        return score


# ==============================================================================
# 核心功能模块
# ==============================================================================

def compute_log_probs_per_token(model, input_ids):
    """
    计算每个 token 的对数概率

    Args:
        model: 语言模型
        input_ids: [batch_size, seq_len]

    Returns:
        log_probs: [batch_size, seq_len-1] 每个位置的 log_prob
    """
    logits = model(input_ids)  # [batch, seq_len, vocab]
    log_softmax = F.log_softmax(logits, dim=-1)

    # 自回归：位置 t 的 logits 预测位置 t+1 的 token
    log_probs = log_softmax[:, :-1, :].gather(
        dim=-1,
        index=input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)

    return log_probs  # [batch, seq_len-1]


def group_sampling(actor_model, prompt_ids, group_size, max_new_tokens=10):
    """
    🧮 讲义对应：Group Sampling (组采样)

    对同一个 Prompt，生成 G 个不同的回答。
    这是 GRPO 的第一步：构建"竞争组"。

    Args:
        actor_model: Actor 模型
        prompt_ids: [batch_size, prompt_len] Prompt 序列
        group_size: 每个 Prompt 采样的回答数量 G
        max_new_tokens: 最大生成长度

    Returns:
        all_responses: [batch_size * group_size, total_len] 所有生成的回答
        prompt_lengths: 每个样本的 prompt 长度（用于后续只更新生成部分）
    """
    batch_size = prompt_ids.shape[0]
    all_responses = []

    for _ in range(group_size):
        # 每次采样都会因为 temperature 和多项式采样产生不同结果
        responses = actor_model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.0  # temperature > 0 引入随机性
        )
        all_responses.append(responses)

    # 拼接所有采样结果: [batch * group_size, seq_len]
    all_responses = torch.cat(all_responses, dim=0)

    # 记录 prompt 长度
    prompt_length = prompt_ids.shape[1]

    return all_responses, prompt_length


def compute_group_advantage(rewards, group_size):
    """
    🧮 讲义对应：Group Relative Advantage (组内相对优势)

    核心公式：A_i = (r_i - mean(R_group)) / std(R_group)

    【直觉理解】
    - 不靠 Critic 预测基线
    - 直接用这组回答的平均分作为基线
    - 如果你在这组里排在前 10%，优势就是正的
    - 如果你是垫底的，优势就是负的

    【与 PPO 的核心区别】
    - PPO: Advantage = TD Target - V(s)，需要 Critic 模型
    - GRPO: Advantage = (r_i - mean) / std，只需要组内统计

    Args:
        rewards: [batch_size * group_size] 所有回答的奖励分数
        group_size: 组大小 G

    Returns:
        advantages: [batch_size * group_size] 组内相对优势
    """
    # 将 rewards reshape 成 [batch_size, group_size]
    batch_size = rewards.shape[0] // group_size
    rewards_grouped = rewards.view(batch_size, group_size)

    # 计算每组的均值和标准差
    mean = rewards_grouped.mean(dim=1, keepdim=True)  # [batch, 1]
    std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8  # [batch, 1] 加小量防止除零

    # 标准化得到组内相对优势
    advantages_grouped = (rewards_grouped - mean) / std  # [batch, group_size]

    # 展平回 [batch_size * group_size]
    advantages = advantages_grouped.view(-1)

    return advantages


def compute_kl_penalty(actor_log_probs, ref_log_probs, kl_coef):
    """
    计算 KL 散度惩罚 (Token-level)

    🧮 讲义对应：Part 2.3 中的 KL 约束

    KL(π_θ || π_ref) ≈ log(π_θ) - log(π_ref)

    返回的是惩罚项（负值），会被加到 reward 里
    """
    # 简化的 KL 计算：log(π_θ) - log(π_ref)
    kl = actor_log_probs - ref_log_probs
    kl_penalty = -kl_coef * kl  # 惩罚项是负的
    return kl_penalty


def train_grpo_step(config, actor, ref_model, reward_model, optimizer, prompt_ids):
    """
    🧮 执行一步 GRPO 更新

    【完整流程】
    1. Group Sampling: 对每个 Prompt 采样 G 个回答
    2. Scoring: 用 Reward Model（或规则）给每个回答打分
    3. Group Advantage: 计算组内相对优势
    4. PPO Update: 用 Importance Ratio + Clipping 更新 Actor

    【与 PPO 的区别】
    - 不需要 Critic 模型
    - Advantage 来自组内相对排名，而非 GAE

    【与 DPO 的区别】
    - 仍然是 RL 范式（在线探索）
    - 需要 Reward Model
    - 保留了 Token-level 的更新差异（通过 Importance Ratio）
    """
    actor.train()

    # ------------------------------------------------------------------
    # Phase 1: Group Sampling (组采样)
    # ------------------------------------------------------------------
    # 对每个 Prompt 生成 G 个不同的回答
    with torch.no_grad():
        all_responses, prompt_length = group_sampling(
            actor, prompt_ids, config.group_size
        )

    batch_size = prompt_ids.shape[0]
    total_samples = batch_size * config.group_size

    # ------------------------------------------------------------------
    # Phase 2: Scoring (打分)
    # ------------------------------------------------------------------
    with torch.no_grad():
        # 用 Reward Model 给每个回答打分
        # 在数学推理任务中，这可以换成确定性规则（对=1，错=0）
        rewards = reward_model(all_responses)  # [batch * group_size]

        # 计算 Reference Model 的 log_probs (用于 KL 约束)
        ref_log_probs = compute_log_probs_per_token(ref_model, all_responses)

        # 计算 Actor 的 old log_probs (采样时的概率，后续计算 Ratio 用)
        old_log_probs = compute_log_probs_per_token(actor, all_responses)

    # ------------------------------------------------------------------
    # Phase 3: Compute Group Advantage (组内相对优势)
    # ------------------------------------------------------------------
    # 🔑 GRPO 的核心：不用 Critic，用组内统计
    advantages = compute_group_advantage(rewards, config.group_size)

    # 将序列级优势扩展到每个 token
    # 注意：GRPO 的优势是序列级的，但更新是 Token 级的
    seq_len = all_responses.shape[1] - 1
    advantages_expanded = advantages.unsqueeze(1).expand(-1, seq_len)

    # ------------------------------------------------------------------
    # Phase 4: PPO-style Update (多轮优化)
    # ------------------------------------------------------------------
    total_loss = 0
    total_policy_loss = 0
    total_kl_loss = 0

    for epoch in range(config.ppo_epochs):
        # 计算当前策略的 log_probs
        new_log_probs = compute_log_probs_per_token(actor, all_responses)

        # --- 组件 1: Importance Ratio (重要性比率) ---
        # ratio = π_new / π_old = exp(log_π_new - log_π_old)
        ratio = torch.exp(new_log_probs - old_log_probs)

        # --- 组件 2: Clipping (截断) ---
        # PPO 的核心魔法：限制更新步幅
        surr1 = ratio * advantages_expanded
        surr2 = torch.clamp(
            ratio,
            1.0 - config.clip_range,
            1.0 + config.clip_range
        ) * advantages_expanded

        # Policy Loss: 取两者的最小值，然后取负（因为我们要最大化优势）
        policy_loss = -torch.min(surr1, surr2).mean()

        # --- 组件 3: KL Penalty (KL 惩罚) ---
        # 防止策略偏离 Reference 太远
        kl_loss = (new_log_probs - ref_log_probs).mean() * config.kl_coef

        # 总 Loss
        loss = policy_loss + kl_loss

        # 反向传播与更新
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_kl_loss += kl_loss.item()

    # 计算指标
    metrics = {
        "total_loss": total_loss / config.ppo_epochs,
        "policy_loss": total_policy_loss / config.ppo_epochs,
        "kl_loss": total_kl_loss / config.ppo_epochs,
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item(),
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std().item(),
    }

    return metrics


# ==============================================================================
# 完整训练流程演示
# ==============================================================================

def main():
    """GRPO 训练演示"""
    print("=" * 70)
    print("📘 GRPO (Group Relative Policy Optimization) 实现演示")
    print("   —— DeepSeek-R1 的秘密武器")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Phase 1: 初始化 (3模型架构)
    # ------------------------------------------------------------------
    print("\n🚀 Phase 1: 初始化模型...")

    config = GRPOConfig()

    # 1️⃣ Actor Model (正在训练)
    base_model = MockTransformer()
    actor = deepcopy(base_model)

    # 2️⃣ Reference Model (冻结)
    ref_model = deepcopy(base_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # 3️⃣ Reward Model (冻结)
    # 在数学推理中，这可以换成确定性规则
    reward_model = MockRewardModel()
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False

    print("   ✅ Actor Model: 正在训练")
    print("   ✅ Reference Model: 冻结 (用于 KL 约束)")
    print("   ✅ Reward Model: 冻结 (用于打分)")
    print("   ❌ 不需要 Critic Model! (这是 GRPO 的革命)")

    # 优化器
    optimizer = torch.optim.AdamW(actor.parameters(), lr=config.lr)

    # ------------------------------------------------------------------
    # Phase 2: 准备 Prompt 数据
    # ------------------------------------------------------------------
    print("\n📊 Phase 2: 准备数据...")
    prompt_ids = torch.randint(0, 1000, (config.batch_size, 5))
    print(f"   Prompt 批次大小: {config.batch_size}")
    print(f"   每个 Prompt 采样数: {config.group_size}")
    print(f"   总样本数: {config.batch_size * config.group_size}")

    # ------------------------------------------------------------------
    # Phase 3: 训练循环
    # ------------------------------------------------------------------
    print("\n🔄 Phase 3: 开始 GRPO 训练...")

    num_steps = 3
    for step in range(num_steps):
        metrics = train_grpo_step(
            config, actor, ref_model, reward_model, optimizer, prompt_ids
        )

        print(f"\n   Step {step + 1}/{num_steps}:")
        print(f"   - Total Loss: {metrics['total_loss']:.4f}")
        print(f"   - Policy Loss: {metrics['policy_loss']:.4f}")
        print(f"   - KL Loss: {metrics['kl_loss']:.4f}")
        print(f"   - Reward (mean ± std): {metrics['reward_mean']:.4f} ± {metrics['reward_std']:.4f}")
        print(f"   - Advantage (mean ± std): {metrics['advantage_mean']:.4f} ± {metrics['advantage_std']:.4f}")

    # ------------------------------------------------------------------
    # 总结
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📝 GRPO 核心要点总结")
    print("=" * 70)
    print("""
1. 【Group Sampling - 组采样】
   对同一个 Prompt 生成 G 个不同的回答（如 G=64）。
   通过 temperature > 0 的采样引入多样性。

2. 【Group Relative Advantage - 组内相对优势】
   核心公式：A_i = (r_i - mean(R_group)) / std(R_group)

   直觉：
   - 用组内平均分作为基线，而非 Critic 的预测
   - 如果你在组内排名靠前，优势为正；垫底则为负
   - 这是"自我博弈"的思想：只要比平均水平好就受奖励

3. 【为什么能去掉 Critic？】
   - PPO 的 Critic: 预测"当前状态的未来预期分数" V(s)
   - GRPO 的替代: 直接用"这组回答的平均分"作为基线
   - 代价: 失去了 Token 级的精细功劳分配
   - 补偿: 通过 Importance Ratio 保持 Token 级更新差异

4. 【与 PPO/DPO 对比】
   | 维度 | PPO | DPO | GRPO |
   |------|-----|-----|------|
   | 模型数 | 4个 | 2个 | 3个 |
   | Critic | ✅ | ❌ | ❌ |
   | RM | ✅ | ❌ | ✅ (或规则) |
   | 探索能力 | 强 | 弱 | 强 |
   | 显存 | 极大 | 最小 | 中等 |

5. 【为什么 DeepSeek 选择 GRPO？】
   - 省显存：去掉了和 Actor 一样大的 Critic
   - 适合推理：数学题有确定性答案，可以用规则打分
   - 保留探索：仍是 RL 范式，能在采样中涌现新解法
   - 这就是 DeepSeek-R1 能涌现出复杂推理链的关键！

6. 【关键洞察 - 序列级优势 vs Token 级更新】
   虽然 GRPO 的 Advantage 是序列级的（整句共享），
   但通过 Importance Ratio 机制，更新幅度仍有差异：
   - Ratio 衡量每个 Token 的"信心变动"
   - 概率变动大的 Token 获得更大的梯度贡献
""")


if __name__ == "__main__":
    main()
