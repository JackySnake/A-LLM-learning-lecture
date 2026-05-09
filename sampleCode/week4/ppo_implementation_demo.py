import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

"""
📘 Week 4 实战示例：PPO (Proximal Policy Optimization) 核心算法实现演示

【学习目标】
理解 RLHF 中最复杂的 PPO 更新循环是如何一步步代码实现的。
本代码对应讲义 Week 4 的 Part 1 (RL 基础) 和 Part 2.3.1 (PPO 更新机制) 章节。

【核心概念对应 - Part 1: RL 基础与 Credit Assignment】
1. Credit Assignment (功劳分配) -> `compute_gae` 函数
   - 核心问题：Reward 只在最后给出，如何判断每个 token 的功过？
   - 解决方案：通过 Critic 估计每个位置的价值，计算 Advantage
2. TD Error (时序差分误差) -> `compute_gae` 中的 delta 变量
   - 公式：delta = r_t + γ*V(s_{t+1}) - V(s_t)
   - 含义：现实与预期的差距，用于更新 Critic 和计算 Advantage
3. Actor-Critic 架构 -> 4模型中的 Actor + Critic
   - Actor：决策者，输出 token 概率分布
   - Critic：评估者，估计状态价值 V(s)，为 Actor 提供基线

【核心概念对应 - Part 2: PPO 更新机制】
4. 4模型架构 (Actor, Critic, Reference, Reward) -> 初始化部分
5. KL 散度惩罚 -> `compute_rewards_with_kl` 函数
6. 优势函数 (Advantage / GAE) -> `compute_gae` 函数
7. 重要性比率 (Importance Ratio) -> `train_ppo_step` 中的 ratio 变量
8. 截断机制 (Clipping) -> `train_ppo_step` 中的 surr1/surr2 计算
9. 策略更新 (Policy Loss) + 价值更新 (Value Loss) -> `train_ppo_step` 函数

【注意】
这是一个"逻辑演示版"代码，为了可读性简化了分布式训练、显存优化等工程细节。
工业界实战推荐使用 HuggingFace TRL 库。
"""

class PPOConfig:
    """超参数配置"""
    def __init__(self):
        self.lr = 1e-5              # 学习率
        self.gamma = 0.99           # 折扣因子 (“未来“的奖励打几折，即下一步)
        self.lam = 0.95             # GAE 参数 (权衡方差和偏差)
        self.clip_range = 0.2       # PPO 裁剪范围 (防止更新步幅过大)
        self.kl_coef = 0.1          # KL 散度惩罚系数 (beta)
        self.value_loss_coef = 0.5  # Value Loss 的权重
        self.batch_size = 4         # 批次大小

# 模拟一个简单的 Transformer 模型接口
class MockTransformer(nn.Module):
    def __init__(self, vocab_size=1000, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layer = nn.Linear(hidden_dim, hidden_dim)
        # Actor Head: 输出词表概率
        self.policy_head = nn.Linear(hidden_dim, vocab_size)
        # Critic Head: 输出价值标量 (仅 Critic 模型使用)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = F.relu(self.layer(x))
        # 实际模型会更复杂，这里仅演示
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

# ==============================================================================
# 核心功能模块
# ==============================================================================

def compute_rewards_with_kl(rewards, kl_coef, log_probs, ref_log_probs):
    """
    🧮 讲义对应：Step 2 计算综合 Reward (加入 KL 惩罚)

    公式：R_total = R_model - beta * KL(pi_theta || pi_ref)
    代码简化版：R_total = R_model - beta * (log_p - log_ref_p)
    """
    # 计算每个 token 的 KL 散度近似值
    # log(p/q) = log(p) - log(q)
    kl = log_probs - ref_log_probs

    # 将 KL 惩罚作为负奖励加进去
    # 注意：通常只在非 padding 区域计算，这里假设全有效
    rewards_with_kl = rewards - kl_coef * kl

    return rewards_with_kl

def compute_gae(rewards, values, next_value, gamma, lam):
    """
    🧮 讲义对应：
       - Part 1.3 Credit Assignment 解决方案
       - Part 1.4 Actor-Critic 中的 Critic 训练机制
       - 附录 B: GAE 算法原理

    【核心作用 - 解决 Credit Assignment（功劳分配）问题】
    虽然 Reward Model 只给整个序列一个总分（稀疏奖励），但通过：
    1. Critic 估计每个位置的价值 V(s_t)
    2. 计算 TD Error（现实与预期的差距）
    3. 用 GAE 平滑累积得到每个 token 的 Advantage
    我们就能知道"哪个 token 该奖励，哪个该惩罚"，实现"精准打击"。

    【关键变量对应讲义公式】
    - values[t]: V(s_t)，Critic 对第 t 步状态的价值预测
    - delta: TD Error = r_t + γ*V(s_{t+1}) - V(s_t)
            > 0 表示"惊喜"（实际比预期好）
            < 0 表示"失望"（实际比预期差）
    - advantages[t]: 第 t 个 token 的功劳/过失值
            > 0 表示该 token 贡献为正，应增加其概率
            < 0 表示该 token 贡献为负，应降低其概率

    【GAE 的作用 - 平衡偏差与方差】(详见附录 B)
    - λ=0: 等同于 TD-0，低方差但高偏差（完全信任 Critic）
    - λ=1: 等同于 Monte Carlo，无偏差但高方差（完全信任真实回报）
    - λ=0.95 (推荐): 折中方案，兼顾稳定性和准确性
    """
    advantages = []
    gae = 0

    # 从后往前倒推计算（因为 GAE 需要用到未来的 Advantage）
    for t in reversed(range(len(rewards))):
        # ------------------------------------------------------------------
        # Step 1: 计算 TD Error (delta) —— Credit Assignment 的核心
        # ------------------------------------------------------------------
        # 公式：delta = r_t + γ * V(s_{t+1}) - V(s_t)
        # 含义：这一步的"真实价值"(r_t + γ*V_next) 与 Critic "预期"(V_curr) 的差距
        if t == len(rewards) - 1:
            next_val = next_value  # 序列结束后的价值（通常为 0）
        else:
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val - values[t]
        # 💡 这个 delta 就是讲义 Part 1.3 中的 TD Error！
        # 它同时用于：1) 训练 Critic (让预测更准) 2) 计算 Advantage (指导 Actor)

        # ------------------------------------------------------------------
        # Step 2: 递归累积得到 GAE
        # ------------------------------------------------------------------
        # 公式：A_t = delta_t + (γλ) * A_{t+1}
        # 含义：当前的优势 = 当前的 TD Error + 折扣后的未来优势
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)

    return torch.tensor(advantages)

# ==============================================================================
# PPO 训练主循环模拟
# ==============================================================================

def train_ppo_step(config, actor, critic, ref_model, reward_model, batch):
    """
    执行一步 PPO 更新
    """
    input_ids = batch["input_ids"]

    # ----------------------------------------------------------------------
    # Phase 1: 采样 (Rollout) - 生成数据
    # ----------------------------------------------------------------------
    with torch.no_grad():
        # 1. Actor 生成文本 (假设 input_ids 已经是生成好的完整序列)
        # 在真实场景中，这里会调用 actor.generate(prompt)
        actor_logits, _ = actor(input_ids)
        actor_probs = F.softmax(actor_logits, dim=-1)
        # 获取当前策略下每个 token 的对数概率
        # gather 操作：只取实际生成的那个 token 的概率
        action_log_probs = torch.log(actor_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1))

        # 2. Reference Model 推理 (计算 KL 用)
        ref_logits, _ = ref_model(input_ids)
        ref_probs = F.softmax(ref_logits, dim=-1)
        ref_log_probs = torch.log(ref_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1))

        # 3. Reward Model 打分
        # 注意：真实 RM 通常只给最后一个 token 打分，这里简化为每步都有分，或者最后一步分摊
        # 假设 reward_model 返回一个标量分
        raw_score = reward_model(input_ids) # 假设返回 [batch, seq_len]

        # 4. Critic 估计价值 (Value)
        # 注意：Critic 也要对当前序列进行评估，作为基线
        _, old_values = critic(input_ids)
        old_values = old_values.squeeze(-1)

    # ----------------------------------------------------------------------
    # Phase 2: 计算信号 (Signal Calculation)
    # ----------------------------------------------------------------------

    # 1. 计算包含 KL 惩罚的综合奖励
    rewards = compute_rewards_with_kl(raw_score, config.kl_coef, action_log_probs, ref_log_probs)

    # 2. 计算优势函数 (Advantage)
    # 这里为了演示简单，只算单条样本
    advantages = compute_gae(rewards[0], old_values[0], 0, config.gamma, config.lam)

    # 计算 Returns (用于训练 Critic) -> Returns = Advantage + Value
    # Critic 的目标是预测真实回报，而真实回报的最佳估计就是：当前优势 + 状态基线
    returns = advantages + old_values[0]

    # ----------------------------------------------------------------------
    # Phase 3: PPO 更新 (Optimization Loop)
    # ----------------------------------------------------------------------

    # ⚠️ PPO 核心特征：使用同一批数据进行多次迭代更新 (Epochs)
    # 这里模拟一个简单的 Loop，假设更新 4 次
    ppo_epochs = 4
    total_loss_val = 0

    for _ in range(ppo_epochs):
        # 重新前向传播 (这次需要梯度) - 计算当前的 Log Probs 和 Values
        # 注意：这里的 actor 参数在每一轮 optimizer.step() 后都会变化
        current_logits, current_values = actor(input_ids)
        current_values = current_values.squeeze(-1)

        current_probs = F.softmax(current_logits, dim=-1)
        # 获取当前策略下生成对应 token 的 log probability (New Log Probs)
        new_log_probs = torch.log(current_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1))

        # 1. Actor Loss (对应讲义“三大组件”中的 2 和 3)
        # --- 组件 2: 重要性比率 (Importance Ratio) ---
        # 计算 r_t(theta) = pi_new / pi_old。pi_old 是采样时锁死的，pi_new 随训练迭代变化
        ratio = torch.exp(new_log_probs[0] - action_log_probs[0])

        # --- 组件 3: 截断 (Clipping) ---
        # PPO 的核心魔法：限制 ratio 的波动范围，防止模型跑飞
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 2. Critic Loss (对应讲义“三大组件”中的 1: 优势的基石)
        # Critic 的目标是让价值预测越来越准，从而提供更可靠的 Advantage
        value_loss = F.mse_loss(current_values[0], returns)

        # 3. 总损失
        total_loss = policy_loss + config.value_loss_coef * value_loss

        # 4. 反向传播与参数更新
        # 这一步会让 actor 和 critic 的参数发生微小变化
        # 从而导致下一轮循环中 computed 的 new_log_probs 发生变化
        optimizer.zero_grad() # 假设有一个外部 optimizer
        total_loss.backward(retain_graph=True) # 模拟更新
        # optimizer.step()

        total_loss_val += total_loss.item()

    return total_loss_val / ppo_epochs

# ==============================================================================
# 运行演示
# ==============================================================================
if __name__ == "__main__":
    print("🚀 正在初始化 PPO 4模型架构...")
    config = PPOConfig()

    # 1. 初始化模型
    # 真实场景中，actor 和 ref 是 LLM，critic 和 reward model 也是 LLM
    base_model = MockTransformer()

    actor = deepcopy(base_model)      # 1️⃣ Actor (训练中)
    critic = deepcopy(base_model)     # 2️⃣ Critic (训练中)
    ref_model = deepcopy(base_model)  # 3️⃣ Reference (冻结)
    reward_model = deepcopy(base_model)# 4️⃣ Reward Model (冻结)

    # 模拟数据输入 (Batch size=1, Seq len=10)
    mock_batch = {"input_ids": torch.randint(0, 1000, (1, 10))}

    # 模拟 Reward Model 的输出 (随机分)
    # 这里临时 mock 一下 reward_model 的行为
    reward_model.forward = lambda x: torch.randn(1, 10)

    print("🔄 开始执行 PPO 更新步...")
    loss = train_ppo_step(config, actor, critic, ref_model, reward_model, mock_batch)

    print(f"✅ PPO Step 完成! Total Loss: {loss:.4f}")
    print("\n💡 观察重点：")
    print("1. compute_rewards_with_kl 中 KL 散度是如何被减去的。")
    print("2. compute_gae 如何利用 Critic 的 old_values 来计算优势。")
    print("3. PPO Clip 机制如何限制 ratio 的更新幅度。")
