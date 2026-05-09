import torch

"""
RoPE (Rotary Position Embedding) 核心实现示例
对应讲义: Week1讲义.md — Part 4.3 节 "RoPE 的 PyTorch 实现"

========================================================
【主实现】相邻配对风格（LLaMA / Meta 官方，当前主流）
  配对方式: (x0, x1), (x2, x3), ...（相邻两个元素为一对，直接对应复数实部/虚部）
  核心 API: torch.view_as_complex + 复数乘法
  代表模型: LLaMA 1/2/3, Mistral, Qwen, DeepSeek

【补充实现】分半旋转风格（GPT-NeoX 风格，见文件末尾）
  配对方式: (x0, x_{d/2}), (x1, x_{d/2+1}), ...（前后两半交叉配对）
  核心 API: torch.cat([x1*cos - x2*sin, x1*sin + x2*cos])
  代表模型: GPT-NeoX，早期 HuggingFace 移植版

  ⚠️  两者数学完全等价，但维度排列不同，混用会导致 KV Cache 错位。
========================================================

核心原理（对应讲义附录 A.1、A.2）:
  - 每个位置 m 施加旋转矩阵 R_{Θ,m}
  - 点积 q'_m · k'_n = q_m^T · R_{Θ,n-m} · k_n，只取决于相对位置 (n-m)
  - 旋转矩阵是正交矩阵，保持向量模长不变

运行方式:
  source /Users/jackymm/miniforge3/bin/activate claude_p13
  python sampleCode/week1/rope_demo.py
"""

# ============================================================
# 【主实现】相邻配对风格（LLaMA 风格）
# ============================================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算旋转频率，返回复数张量 freqs_cis [seq_len, dim/2]

    对应讲义公式: θ_k = 10000^{-2k/d}，k = 0, 1, ..., dim/2 - 1
    每个元素 freqs_cis[m, k] = e^{i·m·θ_k}（欧拉公式）

    Args:
        dim:   head_dim，必须是偶数（对应讲义中的 d_k）
        end:   预计算的最大序列长度
        theta: 基频，默认 10000.0；长上下文模型（如 LLaMA-3-70B）会调大到 500000.0
    """
    # θ_k = 10000^{-2k/d}，共 dim/2 个频率，对应讲义中 d/2 个二维对
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    t = torch.arange(end, device=freqs.device).float()   # 位置 m = 0, 1, ..., end-1

    # 外积：freqs[m, k] = m · θ_k，shape [seq_len, dim/2]
    freqs = torch.outer(t, freqs)

    # torch.polar(r, φ) = r·e^{iφ}；模长为 1（正交变换不改变模长），辐角为 m·θ_k
    # 对应讲义：利用欧拉公式，复数乘法与旋转矩阵运算完全等价
    return torch.polar(torch.ones_like(freqs), freqs)    # 复数张量 [seq_len, dim/2]


def apply_rotary_emb_llama(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    对 Q、K 施加 RoPE 旋转（LLaMA 风格，V 不旋转）

    xq, xk:    [batch, seq_len, n_heads, head_dim]
    freqs_cis: [seq_len, head_dim/2]，复数张量（由 precompute_freqs_cis 生成）
    """
    # 步骤 1：将相邻两个实数视为一个复数
    # [x0, x1, x2, x3, ...] → [x0 + i·x1,  x2 + i·x3, ...]
    # 对应讲义：将二维分量对 (q_{2i}, q_{2i+1}) 视为复平面上的点 z = q_{2i} + i·q_{2i+1}
    # reshape: [..., head_dim] → [..., head_dim/2, 2]，再转为复数 [..., head_dim/2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 步骤 2：复数乘法 = 旋转（欧拉公式）
    # z' = z · e^{i·m·θ}，对应讲义：z' = z · e^{im\theta}
    # freqs_cis [seq_len, head_dim/2] 广播到 [1, seq_len, 1, head_dim/2]
    ndim = xq_.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(xq_.shape)]
    freqs_cis = freqs_cis.view(*shape)

    # view_as_real：复数 [..., head_dim/2] → 实数 [..., head_dim/2, 2]，再 flatten 还原 head_dim
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# ============================================================
# 【补充实现】分半旋转风格（GPT-NeoX / 早期 HuggingFace 风格）
# 与 LLaMA 风格数学等价，但维度配对方式不同，不可混用
# ============================================================

def precompute_freqs_half(seq_len: int, head_dim: int, theta: float = 10000.0):
    """
    预计算旋转频率（分半风格），返回 (cos, sin) 各 [seq_len, head_dim]

    与 LLaMA 风格的区别：
    - LLaMA：返回复数张量，每个元素对应相邻配对的旋转因子
    - 此处：返回实数 cos/sin 矩阵，前后两半各复制一份以匹配 head_dim
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, freqs)                          # [seq_len, head_dim/2]
    # 复制一份：[cos(mθ_0),...,cos(mθ_{d/2-1}), cos(mθ_0),...] 共 head_dim 个
    freqs = torch.cat([freqs, freqs], dim=-1)              # [seq_len, head_dim]
    return freqs.cos(), freqs.sin()


def apply_rotary_emb_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    对单个张量施加 RoPE 旋转（GPT-NeoX 风格）

    x:        [batch, seq_len, n_heads, head_dim]
    cos, sin: [seq_len, head_dim]（由 precompute_freqs_half 生成）

    配对方式：前 d/2 维与后 d/2 维交叉配对
      (x0, x_{d/2}), (x1, x_{d/2+1}), ...
    旋转公式展开：
      x'_i       = x_i · cos(mθ_i)  - x_{i+d/2} · sin(mθ_i)   (前半段)
      x'_{i+d/2} = x_i · sin(mθ_i)  + x_{i+d/2} · cos(mθ_i)   (后半段)
    """
    x1 = x[..., : x.shape[-1] // 2]   # 前 d/2 维
    x2 = x[..., x.shape[-1] // 2 :]   # 后 d/2 维

    # cos/sin 广播：[seq_len, head_dim] → [1, seq_len, 1, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # 分别对前后两半应用旋转，再拼回
    return torch.cat([
        x1 * cos[..., : x.shape[-1] // 2] - x2 * sin[..., : x.shape[-1] // 2],
        x1 * sin[..., x.shape[-1] // 2 :] + x2 * cos[..., x.shape[-1] // 2 :],
    ], dim=-1)


# ============================================================
# 验证实验
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 1
    seq_len    = 5
    n_heads    = 4
    head_dim   = 16    # 对应讲义中的 d_k，必须是偶数

    print("=" * 55)
    print("RoPE 实现验证")
    print(f"配置: seq_len={seq_len}, n_heads={n_heads}, head_dim={head_dim}")
    print("=" * 55)

    xq = torch.randn(batch_size, seq_len, n_heads, head_dim)
    xk = torch.randn(batch_size, seq_len, n_heads, head_dim)

    # --------------------------------------------------
    # 实验 1：验证相对位置性质
    # 理论：对于内容相同、仅位置不同的向量序列，
    #       score[i, j] 只取决于相对距离 |i-j|
    # --------------------------------------------------
    print("\n【实验 1】验证相对位置编码性质（LLaMA 风格）")
    print("讲义结论: 点积只取决于相对位置 (n-m)，见附录 A.1")

    freqs_cis = precompute_freqs_cis(head_dim, seq_len)

    # 构造全同序列：所有位置内容相同，仅位置不同
    const_vec = torch.randn(1, 1, 1, head_dim)
    x_const = const_vec.expand(1, seq_len, 1, head_dim)
    x_rot, _ = apply_rotary_emb_llama(x_const, x_const, freqs_cis)

    vecs = x_rot[0, :, 0, :]          # [seq_len, head_dim]
    scores = vecs @ vecs.T             # [seq_len, seq_len]

    print("\n全同输入向量（内容相同，仅位置不同）的注意力分数：")
    for i in range(seq_len - 1):
        print(f"  Score(pos {i}, pos {i+1})  [相对距离 1]: {scores[i, i+1].item():.6f}")

    diffs = [abs(scores[i, i+1] - scores[0, 1]).item() for i in range(1, seq_len - 1)]
    max_diff = max(diffs)
    print(f"\n与 Score(0,1) 的最大偏差: {max_diff:.2e}")
    if max_diff < 1e-5:
        print("✅ 验证通过：相同相对距离的分数一致。")
    else:
        print("❌ 存在误差，请检查实现。")

    # --------------------------------------------------
    # 实验 2：验证 GPT-NeoX 风格同样满足相对位置性质
    # 说明：两种风格维度配对方式不同，对同一输入的数值输出不同，
    #       但各自内部都正确编码了相对位置信息。
    # --------------------------------------------------
    print("\n【实验 2】验证 GPT-NeoX 风格同样满足相对位置性质")
    print("（两种风格数值不同，但都是合法的 RoPE 实现）")

    cos_neox, sin_neox = precompute_freqs_half(seq_len, head_dim)

    # 同样用全同序列验证相对位置性质
    x_const2 = const_vec.expand(1, seq_len, 1, head_dim)
    x_rot_neox = apply_rotary_emb_neox(x_const2, cos_neox, sin_neox)

    vecs_neox = x_rot_neox[0, :, 0, :]
    scores_neox = vecs_neox @ vecs_neox.T

    print("\nGPT-NeoX 风格，全同输入向量的注意力分数：")
    for i in range(seq_len - 1):
        print(f"  Score(pos {i}, pos {i+1})  [相对距离 1]: {scores_neox[i, i+1].item():.6f}")

    diffs_neox = [abs(scores_neox[i, i+1] - scores_neox[0, 1]).item() for i in range(1, seq_len - 1)]
    if max(diffs_neox) < 1e-5:
        print("✅ GPT-NeoX 风格也满足相对位置性质。")

    # 两种风格对同一输入的数值确实不同（这是正常的）
    xq_llama, _ = apply_rotary_emb_llama(xq, xk, freqs_cis)
    xq_neox_rand = apply_rotary_emb_neox(xq, cos_neox, sin_neox)
    val_diff = (xq_llama - xq_neox_rand).abs().max().item()
    print(f"\n两种风格对随机输入的旋转结果最大差异: {val_diff:.4f}（预期不为 0，属正常）")
    print("→ 两者不是数值等价，而是各自内部都正确实现了 RoPE 的相对位置性质。")
    print("→ 混用（Q 用一种、K 用另一种）会破坏位置信息，需严格统一风格。")
