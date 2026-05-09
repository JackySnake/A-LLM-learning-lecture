import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

"""
📘 Week 4 实战示例：DPO (Direct Preference Optimization) 核心算法实现演示

【学习目标】
理解 DPO 是如何把 RLHF 简化成一个"分类问题"的，以及它为什么不需要 Reward Model 和 Critic。
本代码对应讲义 Week 4 的 Part 3.1 (从 PPO 到 DPO) 和附录 E (DPO 数学推导) 章节。

【核心概念对应 - Part 3.1: DPO 的革命】
1. 隐式 Reward 建模 -> `compute_log_ratios` 函数
   - 核心公式：r*(x,y) = β * log(π_θ(y|x) / π_ref(y|x)) + β*log(Z(x))
   - 意义：Reward 本质上就是"信心提升度"
2. 2模型架构 (Policy + Reference) -> 初始化部分
   - 去掉了 Reward Model 和 Critic Model
3. Bradley-Terry 偏好模型 -> `compute_dpo_loss` 函数
   - P(y_w > y_l) = σ(r(y_w) - r(y_l))
4. DPO Loss 计算 -> `compute_dpo_loss` 函数
   - 公式：L = -log σ(β * [log(π_θ(y_w)/π_ref(y_w)) - log(π_θ(y_l)/π_ref(y_l))])

【与 PPO 的核心差异】
| 维度 | PPO | DPO |
|------|-----|-----|
| 模型数量 | 4个 (Actor, Critic, RM, Ref) | 2个 (Policy, Ref) |
| 训练范式 | 强化学习 (在线探索) | 监督学习 (离线分类) |
| 数据需求 | 奖励信号 | 成对偏好数据 (A > B) |
| 优势计算 | Critic + GAE | 不需要 (隐式在 Loss 里) |

【注意】
这是一个"逻辑演示版"代码，为了可读性简化了工程细节。
工业界实战推荐使用 HuggingFace TRL 库的 DPOTrainer。
"""


class DPOConfig:
    """DPO 超参数配置"""
    def __init__(self):
        self.beta = 0.1             # KL 约束系数 (控制与 Reference 的偏离程度)
        self.lr = 1e-6              # 学习率 (DPO 通常用更小的学习率)
        self.batch_size = 4         # 批次大小
        self.label_smoothing = 0.0  # 标签平滑 (可选，防止过拟合)


class MockTransformer(nn.Module):
    """模拟一个简单的 Transformer 模型接口"""
    def __init__(self, vocab_size=1000, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layer = nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        """
        前向传播，返回 logits

        Args:
            input_ids: [batch_size, seq_len]
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        x = self.embedding(input_ids)
        x = F.relu(self.layer(x))
        logits = self.lm_head(x)
        return logits


# ==============================================================================
# 核心功能模块
# ==============================================================================

def compute_log_probs(model, input_ids):
    """
    计算模型对给定序列的对数概率

    🧮 讲义对应：DPO 需要计算 π(y|x) 的对数概率

    Args:
        model: 语言模型
        input_ids: [batch_size, seq_len] 输入序列

    Returns:
        log_probs: [batch_size] 每个序列的总对数概率
    """
    with torch.set_grad_enabled(model.training):
        logits = model(input_ids)  # [batch, seq_len, vocab_size]

        # 计算每个位置的 log_softmax
        log_softmax = F.log_softmax(logits, dim=-1)

        # 获取实际 token 的 log_prob (shift by 1 for autoregressive)
        # 对于位置 t，我们用位置 t 的 logits 预测位置 t+1 的 token
        # 所以 logits[:, :-1] 预测 input_ids[:, 1:]
        log_probs_per_token = log_softmax[:, :-1, :].gather(
            dim=-1,
            index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)  # [batch, seq_len-1]

        # 对序列求和得到总的 log_prob
        # 注意：实际中需要考虑 padding mask
        total_log_probs = log_probs_per_token.sum(dim=-1)  # [batch]

    return total_log_probs


def compute_log_ratios(policy_model, ref_model, input_ids):
    """
    🧮 讲义对应：计算"信心提升度" log(π_θ(y|x) / π_ref(y|x))

    这是 DPO 的核心！根据讲义 Part 3.1：
    r*(x,y) = β * log(π_θ(y|x) / π_ref(y|x)) + β*log(Z(x))

    由于 Z(x) 在比较 winner 和 loser 时会被抵消，
    我们只需要计算 log ratio 即可。

    Args:
        policy_model: 正在训练的策略模型
        ref_model: 冻结的参考模型
        input_ids: 输入序列

    Returns:
        log_ratio: log(π_θ / π_ref)，即"信心提升度"
    """
    # Policy Model 的 log_prob (需要梯度)
    policy_log_probs = compute_log_probs(policy_model, input_ids)

    # Reference Model 的 log_prob (不需要梯度，冻结状态)
    with torch.no_grad():
        ref_log_probs = compute_log_probs(ref_model, input_ids)

    # 计算 log ratio = log(π_θ / π_ref) = log(π_θ) - log(π_ref)
    log_ratios = policy_log_probs - ref_log_probs

    return log_ratios


def compute_dpo_loss(config, policy_model, ref_model, batch):
    """
    🧮 讲义对应：DPO Loss 计算 (Part 3.1 核心公式)

    完整公式：
    L_DPO = -log σ(β * [log(π_θ(y_w)/π_ref(y_w)) - log(π_θ(y_l)/π_ref(y_l))])

    直觉理解：
    1. Winner 的信心提升度 - Loser 的信心提升度 = Gap
    2. 我们希望 Gap 越大越好 (模型更自信地选择好答案)
    3. 通过 sigmoid + log 转化为分类问题

    Args:
        config: DPO 配置
        policy_model: 正在训练的策略模型
        ref_model: 冻结的参考模型 (原始 SFT 模型)
        batch: 包含 chosen_ids 和 rejected_ids 的数据批次

    Returns:
        loss: DPO 损失值
        metrics: 用于监控的指标
    """
    chosen_ids = batch["chosen_ids"]    # Winner 序列 (y_w)
    rejected_ids = batch["rejected_ids"]  # Loser 序列 (y_l)

    # ------------------------------------------------------------------
    # Step 1: 计算 Winner 和 Loser 的"信心提升度" (Log Ratio)
    # ------------------------------------------------------------------
    # 🔑 核心洞察：Reward 本质上就是 log(π_θ / π_ref)

    # Winner 的 log ratio: log(π_θ(y_w) / π_ref(y_w))
    chosen_log_ratios = compute_log_ratios(policy_model, ref_model, chosen_ids)

    # Loser 的 log ratio: log(π_θ(y_l) / π_ref(y_l))
    rejected_log_ratios = compute_log_ratios(policy_model, ref_model, rejected_ids)

    # ------------------------------------------------------------------
    # Step 2: 计算 Gap (Winner 与 Loser 的差距)
    # ------------------------------------------------------------------
    # Gap = β * (chosen_ratio - rejected_ratio)
    # 直觉：我们希望模型"更倾向于生成 Winner，而不是 Loser"

    logits = config.beta * (chosen_log_ratios - rejected_log_ratios)

    # ------------------------------------------------------------------
    # Step 3: 转化为分类 Loss (Bradley-Terry 偏好模型)
    # ------------------------------------------------------------------
    # P(y_w > y_l) = σ(r(y_w) - r(y_l)) = σ(logits)
    # Loss = -log(P(y_w > y_l)) = -log(σ(logits))
    #
    # 用 F.logsigmoid 直接计算 log(σ(x))，数值更稳定

    if config.label_smoothing > 0:
        # 可选：标签平滑，防止过拟合
        # 相当于说"我们对偏好判断不是 100% 确定"
        losses = (
            -F.logsigmoid(logits) * (1 - config.label_smoothing)
            -F.logsigmoid(-logits) * config.label_smoothing
        )
    else:
        losses = -F.logsigmoid(logits)

    loss = losses.mean()

    # ------------------------------------------------------------------
    # Step 4: 计算监控指标
    # ------------------------------------------------------------------
    with torch.no_grad():
        # 准确率：模型是否正确地给 Winner 更高的"信心"
        accuracy = (logits > 0).float().mean()

        # 隐式 Reward 的绝对值 (用于监控训练稳定性)
        chosen_rewards = config.beta * chosen_log_ratios
        rejected_rewards = config.beta * rejected_log_ratios
        reward_margin = (chosen_rewards - rejected_rewards).mean()

    metrics = {
        "loss": loss.item(),
        "accuracy": accuracy.item(),
        "reward_margin": reward_margin.item(),
        "chosen_rewards": chosen_rewards.mean().item(),
        "rejected_rewards": rejected_rewards.mean().item(),
    }

    return loss, metrics


# ==============================================================================
# DPO 训练主循环
# ==============================================================================

def train_dpo_step(config, policy_model, ref_model, optimizer, batch):
    """
    执行一步 DPO 更新

    与 PPO 的关键区别：
    - 没有 Critic 和 GAE 计算
    - 没有 Reward Model
    - 没有 Importance Ratio 和 Clipping
    - 就是一个简单的监督学习循环！
    """
    policy_model.train()

    # 1. 计算 DPO Loss
    loss, metrics = compute_dpo_loss(config, policy_model, ref_model, batch)

    # 2. 反向传播
    optimizer.zero_grad()
    loss.backward()

    # 3. 梯度裁剪 (可选，但推荐)
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)

    # 4. 参数更新
    optimizer.step()

    return metrics


# ==============================================================================
# 完整训练流程演示
# ==============================================================================

def prepare_preference_data(num_samples=4, seq_len=10, vocab_size=1000):
    """
    准备偏好数据 (模拟)

    🧮 讲义对应：DPO 需要三元组数据 (Prompt, Winner, Loser)

    在实际场景中，这些数据来自人工标注或 AI 辅助标注。
    例如：对于同一个 Prompt，让模型生成多个回答，
    然后由人类选择哪个更好。
    """
    # 模拟数据：chosen 比 rejected "更好"
    # 在真实场景中，这需要人工标注
    data = {
        "prompt_ids": torch.randint(0, vocab_size, (num_samples, seq_len)),
        "chosen_ids": torch.randint(0, vocab_size, (num_samples, seq_len)),
        "rejected_ids": torch.randint(0, vocab_size, (num_samples, seq_len)),
    }
    return data


def main():
    """DPO 训练演示"""
    print("=" * 60)
    print("📘 DPO (Direct Preference Optimization) 实现演示")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Phase 1: 初始化 (2模型架构，比 PPO 的 4模型简单多了!)
    # ------------------------------------------------------------------
    print("\n🚀 Phase 1: 初始化模型...")

    config = DPOConfig()

    # 1️⃣ Policy Model (正在训练)
    # 通常从 SFT 模型初始化
    base_model = MockTransformer()
    policy_model = deepcopy(base_model)

    # 2️⃣ Reference Model (冻结)
    # 就是原始的 SFT 模型，用于计算 KL 约束
    ref_model = deepcopy(base_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    print("   ✅ Policy Model: 正在训练")
    print("   ✅ Reference Model: 冻结 (eval mode)")
    print("   ❌ 不需要 Reward Model!")
    print("   ❌ 不需要 Critic Model!")

    # 优化器
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.lr)

    # ------------------------------------------------------------------
    # Phase 2: 准备偏好数据
    # ------------------------------------------------------------------
    print("\n📊 Phase 2: 准备偏好数据...")
    batch = prepare_preference_data()
    print(f"   数据格式: (Prompt, Winner, Loser)")
    print(f"   批次大小: {batch['chosen_ids'].shape[0]}")
    print(f"   序列长度: {batch['chosen_ids'].shape[1]}")

    # ------------------------------------------------------------------
    # Phase 3: 训练循环
    # ------------------------------------------------------------------
    print("\n🔄 Phase 3: 开始 DPO 训练...")

    num_epochs = 3
    for epoch in range(num_epochs):
        metrics = train_dpo_step(config, policy_model, ref_model, optimizer, batch)

        print(f"\n   Epoch {epoch + 1}/{num_epochs}:")
        print(f"   - Loss: {metrics['loss']:.4f}")
        print(f"   - Accuracy: {metrics['accuracy']:.2%}")
        print(f"   - Reward Margin (Winner - Loser): {metrics['reward_margin']:.4f}")
        print(f"   - Chosen Reward: {metrics['chosen_rewards']:.4f}")
        print(f"   - Rejected Reward: {metrics['rejected_rewards']:.4f}")

    # ------------------------------------------------------------------
    # 总结
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📝 DPO 核心要点总结")
    print("=" * 60)
    print("""
1. 【隐式 Reward】
   DPO 不需要显式的 Reward Model！
   它利用公式 r(y) = β * log(π_θ(y) / π_ref(y)) 直接从策略概率中推导奖励。

2. 【分类问题】
   DPO 把 RL 问题转化为分类问题：
   "让模型更倾向于生成 Winner，而不是 Loser"

3. 【Loss 公式】
   L = -log σ(β * [log(π_θ(y_w)/π_ref(y_w)) - log(π_θ(y_l)/π_ref(y_l))])

   拆解：
   - β: 控制与 Reference 的偏离程度
   - log(π_θ/π_ref): "信心提升度"
   - 差值: Winner 的提升度应该大于 Loser

4. 【与 PPO 对比】
   | 维度 | PPO | DPO |
   |------|-----|-----|
   | 模型数量 | 4个 | 2个 |
   | 训练范式 | RL (在线) | SL (离线) |
   | 稳定性 | 需要精细调参 | 相对稳定 |

5. 【代价】
   DPO 放弃了 Token 级别的精准功劳分配。
   对于需要细粒度推理的任务（如数学），可能不如 PPO/GRPO。
""")


if __name__ == "__main__":
    main()