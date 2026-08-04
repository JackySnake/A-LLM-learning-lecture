# MLA 专题讲义：Multi-head Latent Attention 深度解析

> 本讲义为独立专题，聚焦 DeepSeek-V2/V3 中提出的 MLA（Multi-head Latent Attention）机制。
> 后续整合到 Week 12 DeepSeek 深度解析讲义时，可直接引用本文档。

---

## 🎯 核心目标

学完本讲义，你应该能够：

1. **理解问题**：清晰阐述 MHA KV Cache 的内存瓶颈，以及 MQA/GQA 的权衡
2. **理解机制**：从数学推导角度理解 MLA 的低秩压缩思路
3. **理解技巧**：解释"解耦 RoPE"为何必须存在，以及矩阵吸收（Matrix Absorption）trick 的原理
4. **面试输出**：能够用对比表格说清楚 MHA → GQA → MLA 的演进逻辑和各自代价
5. **工程直觉**：知道 MLA 实际节省了多少 KV Cache，代价是什么

---

## 🗺️ 知识图谱

```
KV Cache 内存问题
    ├── MHA：每层 2 × n_heads × d_head × seq_len 的缓存
    ├── MQA：共享一个 KV head，压缩比高但质量损失大
    ├── GQA：分组共享，折中方案（LLaMA 3、Qwen 使用）
    └── MLA（DeepSeek-V2/V3）
           ├── 核心思想：低秩联合压缩 KV
           ├── 压缩向量 c^KV 替代完整 KV 进行缓存
           ├── 解耦 RoPE：解决位置编码与矩阵吸收的冲突
           └── 矩阵吸收（Absorption）：推理时消除中间矩阵

相关背景知识
    ├── Multi-Head Attention（MHA）基础
    ├── RoPE 位置编码（见 Week 1 讲义）
    └── KV Cache 机制（见 Week 6 讲义）
```

---

## Part 0：问题背景——KV Cache 的内存危机

### 0.1 KV Cache 是什么，为什么必要

在自回归推理中，每次生成一个新 token，模型都需要计算该 token 对所有历史 token 的 Attention。如果重新计算所有历史 token 的 K、V，时间复杂度是 $O(n^2)$，极为低效。

**KV Cache** 的做法是：将每层已计算出的 K、V 矩阵缓存下来，新 token 只需计算自己的 Q、K、V，然后用缓存的历史 K、V 做 Attention 即可。这将推理变为 $O(n)$ 的增量计算。

这是 LLM 工程部署的基础优化（见 Week 6 讲义）。

### 0.2 KV Cache 有多大？

标准 MHA（Multi-head Attention）中，对于每一层，每个 token 需要缓存：

$$\text{KV Cache per token per layer} = 2 \times n_h \times d_h \times \text{dtype\_bytes}$$

其中 $n_h$ 是 head 数，$d_h$ 是每个 head 的维度，通常 $d_h = d_{model} / n_h$，所以：

$$= 2 \times d_{model} \times \text{dtype\_bytes}$$

以 DeepSeek-V2 的参数规模为例（$d_{model} = 5120$，128 层，BF16）：

$$\text{per token} = 128 \times 2 \times 5120 \times 2 \text{ bytes} = 2.62 \text{ MB/token}$$

对于 32K 的上下文长度：

$$32768 \times 2.62 \text{ MB} \approx 85 \text{ GB}$$

这比模型权重本身还大——这就是 **KV Cache 内存危机**。长序列下，KV Cache 成为部署的主要瓶颈。

### 0.3 已有方案：MQA 与 GQA

在 MLA 之前，业界主要有两种思路：

**MQA（Multi-Query Attention，Shazeer 2019）**

所有 Query head 共享同一组 K、V head（即只有 1 个 KV head）：

$$\text{KV Cache per token} = 2 \times 1 \times d_h \times L \text{ bytes}$$

节省比例是 $n_h$ 倍，但质量损失明显，实际工程中较少在大模型上单独使用。

**GQA（Grouped Query Attention，Ainslie et al. 2023）**

将 $n_h$ 个 Q head 分成 $g$ 组，每组共享一对 KV head：

$$\text{KV Cache per token} = 2 \times g \times d_h \times L \text{ bytes}$$

当 $g = n_h$ 时退化为 MHA，当 $g = 1$ 时退化为 MQA。GQA 是目前主流的折中方案，被 LLaMA 3、Qwen2.5 等广泛采用。

**但 GQA 本质上还是在"粗粒度地减少 KV head 数量"，没有改变 KV 向量本身的表示方式。**

---

## Part 1：MLA 的核心思想——低秩联合压缩

### 1.1 直觉：压缩信息而非删减头

GQA 的做法是"少存几个头"——代价是强制多个 Q head 使用同一份 K、V，损失了表达多样性。

MLA 的出发点不同：**能否把所有 KV head 的信息联合压缩成一个低维向量，推理时再解压？**

这背后的直觉是：K 和 V 矩阵虽然维度很高（$n_h \times d_h$），但其信息可能存在冗余——即它们的有效信息维度远低于名义维度，可以用低秩矩阵近似。

这与 LoRA 的思路（见 Week 3 讲义）异曲同工：LoRA 压缩权重矩阵的更新量，MLA 压缩 KV Cache 的存储量。

### 1.2 MLA 的数学定义

设模型隐层状态为 $h_t \in \mathbb{R}^d$，$d$ 为 model dimension。

**传统 MHA 的做法：**

$$K_t = W^K h_t, \quad V_t = W^V h_t$$

每层需缓存 $K_t$ 和 $V_t$，维度各为 $n_h \times d_h$。

**MLA 的做法：先降维，再升维**

第一步，**联合下投影**（down-projection），得到压缩潜向量：

$$c_t^{KV} = W^{DKV} h_t$$

其中 $W^{DKV} \in \mathbb{R}^{d_c \times d}$，$d_c \ll n_h \times d_h$（压缩维度远小于全展开维度）。

第二步，**上投影**（up-projection），从压缩向量还原出 K 和 V：

$$k_t^C = W^{UK} c_t^{KV}, \quad v_t = W^{UV} c_t^{KV}$$

其中 $W^{UK} \in \mathbb{R}^{(n_h \times d_h) \times d_c}$，$W^{UV} \in \mathbb{R}^{(n_h \times d_h) \times d_c}$。

**关键洞察：推理时只需缓存 $c_t^{KV}$，而非 $k_t^C$ 和 $v_t$。** 需要时现场解压即可。

类似地，Query 也做低秩压缩（这主要是为了减少激活内存，对 KV Cache 无直接影响，但统一了框架）：

$$c_t^Q = W^{DQ} h_t, \quad q_t^C = W^{UQ} c_t^Q$$

其中 $c_t^Q \in \mathbb{R}^{d_c'}$（Query 压缩维度可以与 KV 不同）。

### 1.3 DeepSeek-V2 的具体维度

| 参数 | 维度 | 说明 |
|------|------|------|
| $d_{model}$ | 5120 | 模型隐层维度 |
| $n_h$ | 128 | Query head 数 |
| $d_h$ | 128 | 每个 head 维度 |
| $n_h \times d_h$ | 16384 | 展开后全 KV 维度 |
| $d_c$（KV 压缩维度） | **512** | MLA 实际缓存维度 |
| $d_c'$（Q 压缩维度） | 1536 | Query 压缩维度 |

**KV Cache 压缩比：$16384 / 512 = 32$**，即相比 MHA，每个 token 的 KV Cache 缩小到原来的 1/32。

与 GQA 的对比：若用 GQA 达到等效压缩比，需要将 128 个 KV head 减少到 4 个，这会严重损害模型表达能力；而 MLA 通过低秩投影保留了所有 128 个 head 的完整表达力。

---

## Part 2：解耦 RoPE——一个不得不解决的冲突

### 2.1 RoPE 为什么会造成问题

在现代 LLM 中，位置信息通过 RoPE（Rotary Position Embedding）注入到 K 和 Q 中（见 Week 1 讲义）。RoPE 的操作形如：

$$\text{RoPE}(x, m) = R_m x$$

其中 $R_m$ 是一个依赖于位置 $m$ 的旋转矩阵。RoPE 是**位置相关**的变换。

**MLA 的矩阵吸收（Matrix Absorption）需要 K 和 Q 的投影矩阵是位置无关的常数**，才能在推理时把上投影矩阵"预乘"进去，消除中间展开的 K 矩阵（见 Part 3）。

如果 K 上面叠加了 RoPE，这个"预乘"就无法进行：

$$\text{RoPE}(W^{UK} c_t^{KV}, m) \neq W^{UK} \cdot \text{RoPE}(c_t^{KV}, m)$$

因为 $R_m$ 不能穿过 $W^{UK}$ 提到外面（旋转矩阵和线性映射不满足交换律）。

### 2.2 解耦 RoPE 的设计

DeepSeek-V2 的解决方案是**将位置信息和内容信息分开处理**，通过一对独立的"RoPE 专用向量"携带位置信息：

对于 **Key**，将其拆成两部分拼接：

$$k_t = \left[ k_t^C \;;\; k_t^R \right]$$

- $k_t^C = W^{UK} c_t^{KV}$：**内容键**，从压缩向量解压，不带 RoPE
- $k_t^R = \text{RoPE}(W^{KR} h_t, t)$：**位置键**，直接从 $h_t$ 投影并施加 RoPE，$W^{KR} \in \mathbb{R}^{d_h^R \times d}$

对于 **Query**，同样拆成两部分：

$$q_t = \left[ q_t^C \;;\; q_t^R \right]$$

- $q_t^C = W^{UQ} c_t^Q$：内容查询
- $q_t^R = \text{RoPE}(W^{QR} c_t^Q, t)$：位置查询

**Attention 分数的计算**（对应位置的向量做内积）：

$$\text{score}(q_t, k_s) = (q_t^C)^\top k_s^C + (q_t^R)^\top k_s^R$$

前半部分 $(q_t^C)^\top k_s^C$ 是纯内容相关的注意力，后半部分 $(q_t^R)^\top k_s^R$ 携带了相对位置信息（因为 RoPE 使内积只依赖于 $t - s$）。

### 2.3 解耦 RoPE 的代价

由于 $k_t^R$ 是从 $h_t$ 直接投影的（绕过了压缩），它**必须也被缓存**，否则推理时无法计算位置注意力。

因此，实际缓存内容为：

$$\text{缓存} = \left( c_t^{KV},\; k_t^R \right)$$

缓存维度 = $d_c + n_h \times d_h^R$，其中 $d_h^R$ 是 RoPE head 的维度（通常远小于 $d_h$，DeepSeek-V2 中为 64）。

以 DeepSeek-V2 为例：$512 + 128 \times 64 = 512 + 8192 = 8704$，相比完整 MHA 的 $16384 \times 2 = 32768$ 仍然节省约 **3.75倍**。

---

## Part 3：矩阵吸收（Matrix Absorption）——推理加速的关键 Trick

### 3.1 为什么要做矩阵吸收

如果按照 Part 1 的定义直接实现，推理时的计算流程是：

```
c^KV = W^{DKV} * h          # 存入 KV Cache
k^C = W^{UK} * c^KV         # 每次 attention 时解压
v   = W^{UV} * c^KV         # 每次 attention 时解压
q^C = W^{UQ} * c^Q          # 同样解压
score = q^C @ k^C.T + q^R @ k^R.T
out = score * v
out = W^O * reshape(out)
```

这里 $W^{UK}$ 和 $W^{UV}$ 是上投影矩阵，在每次 Attention 计算时都需要用到。能否把它们"合并"到其他矩阵里，避免额外的矩阵乘法？

### 3.2 内容键的矩阵吸收

注意到内容部分的 Attention 分数：

$$s^C = (q_t^C)^\top k_s^C = (W^{UQ} c_t^Q)^\top (W^{UK} c_s^{KV})$$

$$= (c_t^Q)^\top \underbrace{(W^{UQ})^\top W^{UK}}_{\tilde{W}^{QK}} c_s^{KV}$$

如果定义 $\tilde{W}^{QK} = (W^{UQ})^\top W^{UK}$，则：

$$s^C = (c_t^Q)^\top \tilde{W}^{QK} c_s^{KV}$$

**直接用压缩向量 $c_t^Q$ 和 $c_s^{KV}$ 计算 Attention 分数**，完全不需要展开成完整的 $q_t^C$ 和 $k_s^C$！

$\tilde{W}^{QK}$ 的维度为 $d_c' \times d_c$（比如 $1536 \times 512$），可以在推理前预计算好。

### 3.3 Value 的矩阵吸收

类似地，对输出的计算：

$$\text{out}_t = \sum_s \alpha_{ts} v_s = \sum_s \alpha_{ts} W^{UV} c_s^{KV}$$

$$= W^{UV} \underbrace{\sum_s \alpha_{ts} c_s^{KV}}_{\text{加权和}}$$

先对 $c_s^{KV}$ 做加权求和，再用 $W^{UV}$ 上投影。而 $W^{UV}$ 可以和输出投影矩阵 $W^O$ 合并：

$$\tilde{W}^{VO} = W^O \cdot W^{UV}$$

这样整个注意力输出的投影只需一次矩阵乘法（从 $d_c$ 直接映射到最终输出维度），省去了显式展开 V 的步骤。

### 3.4 矩阵吸收的约束条件

**矩阵吸收只在 KV head 不带 RoPE 时才成立。** 如果 $k^C$ 带有 RoPE，则 $k_s^C = \text{RoPE}(W^{UK} c_s^{KV}, s)$，位置相关的旋转矩阵 $R_s$ 无法提到 $W^{UK}$ 外面，矩阵吸收失效。

这正是为什么 MLA 必须采用"解耦 RoPE"的设计：内容部分不带 RoPE，才能做矩阵吸收；位置信息由独立的 $k^R$/$q^R$ 携带。

### 3.5 小结：两种推理模式的对比

| 推理模式 | 缓存内容 | 计算方式 | 适用场景 |
|----------|----------|----------|----------|
| **标准模式**（不用吸收）| $c^{KV}$, $k^R$ | 先解压 K/V，再做 Attention | KV Cache 内存有限 |
| **矩阵吸收模式** | $c^{KV}$, $k^R$ | 直接用压缩向量计算，省去展开 K/V | 计算密集型场景 |

两种模式缓存的内容**完全一样**，只是计算路径不同。矩阵吸收减少了矩阵乘法次数，但要求 $\tilde{W}^{QK}$ 和 $\tilde{W}^{VO}$ 这两个预计算矩阵能够放入显存。

---

## Part 4：完整 MLA 计算流程总结

将前三个 Part 整合，MLA 的完整前向计算如下：

**训练时（标准展开）：**

```
1. 下投影：
   c^Q  = W^{DQ} * h          # [d_c']
   c^KV = W^{DKV} * h         # [d_c]

2. 内容部分上投影：
   q^C = W^{UQ} * c^Q         # [n_h * d_h]
   k^C = W^{UK} * c^KV        # [n_h * d_h]
   v   = W^{UV} * c^KV        # [n_h * d_h]

3. 位置部分（解耦 RoPE）：
   q^R = RoPE(W^{QR} * c^Q, pos)   # [n_h * d_h^R]
   k^R = RoPE(W^{KR} * h,  pos)    # [n_h * d_h^R]

4. 拼接：
   q = [q^C; q^R]             # [n_h * (d_h + d_h^R)]
   k = [k^C; k^R]             # [n_h * (d_h + d_h^R)]

5. Attention + 输出投影：
   out = softmax(q @ k.T / sqrt(d_h + d_h^R)) @ v
   y   = W^O * out
```

**推理时（KV Cache 模式）：**

- 缓存：每个历史 token 存储 $(c_t^{KV}, k_t^R)$，维度 $d_c + n_h \times d_h^R$
- 新 token 到来时，计算 Q（包括 $q^C$ 和 $q^R$），与缓存做 Attention
- 可选：用矩阵吸收 trick，直接在压缩空间计算内容 Attention 分数

---

## Part 5：MHA → GQA → MLA 横向对比

| 维度 | MHA | GQA | MLA |
|------|-----|-----|-----|
| **核心思路** | 每个 Q head 独立 KV | 多个 Q head 共享 KV head | 联合低秩压缩 KV |
| **KV Cache 大小**（相对 MHA） | 1× | $g/n_h$（如 1/8 → 8x 减少） | $d_c/(n_h \times d_h)$（如 1/32） |
| **模型表达力** | 最强 | 中等（KV head 共享损失） | 强（只压缩冗余，非截断 head） |
| **位置编码** | 标准 RoPE | 标准 RoPE | 解耦 RoPE（内容/位置分离） |
| **推理计算量** | 基准 | 略低（KV head 少） | 略高（上投影/吸收有额外算术） |
| **额外参数量** | 无 | 无 | 有（$W^{DKV}$, $W^{UK}$, $W^{UV}$, $W^{KR}$ 等） |
| **代表模型** | GPT 系列、早期 LLaMA | LLaMA 3、Qwen2.5、Mistral | DeepSeek-V2、DeepSeek-V3 |
| **复杂度** | 低 | 低-中 | 高（工程实现复杂） |

**核心权衡总结：**

- GQA：以牺牲少量模型质量换 KV Cache 节省，工程简单
- MLA：以增加参数量和工程复杂度换取**更大的 KV Cache 节省 + 更好的模型质量**

---

## Part 6：工程实践与实际效果

### 6.1 DeepSeek-V2 的实测数据

DeepSeek-V2（236B MoE 模型）论文中报告：

- 相比 MHA 基准，KV Cache 减少约 **93.3%**（即缓存大小约为 MHA 的 6.7%）
- 在保持相近模型质量的前提下，支持的 batch size 更大，推理吞吐量显著提升
- 相比同规模 MHA 模型，推理成本降低约 **5.76×**

注：上述 6.7% 的数字来自对解耦 RoPE 的完整统计（$d_c + n_h \times d_h^R$ vs $2 \times n_h \times d_h$）。

### 6.2 工程实现的挑战

1. **FlashAttention 兼容性**：MLA 的非对称 Q/K 维度（Q 带完整 RoPE，K 只带一半）不能直接套用标准 FlashAttention 核，需要定制 CUDA kernel

2. **矩阵吸收的内存-计算权衡**：吸收后的 $\tilde{W}^{QK}$ 矩阵维度为 $d_c' \times d_c = 1536 \times 512 = 786432$ 个参数，相比展开路径增加了参数量，但节省了运行时的内存带宽

3. **训练稳定性**：多个下投影/上投影矩阵叠加可能造成梯度流问题，DeepSeek 在实现中使用了一些初始化技巧（如 $W^{UK}$ 不使用 bias）

4. **与 MoE 的结合**：DeepSeek-V2/V3 同时使用 MLA 和 MoE，两者独立设计，注意力部分使用 MLA，FFN 部分使用 MoE

### 6.3 MLA 与 KV Cache 量化的关系

由于 MLA 缓存的是压缩后的 $c^{KV}$（而非解压后的 K、V），**量化时的误差特性也不同**。压缩向量在值域分布上可能比 K、V 矩阵更均匀，有利于量化。但目前公开的详细研究较少。

---

## 实践启示

### 面试场景下的回答框架

**问：解释一下 MLA 是什么？**

> "MLA 是 DeepSeek-V2 提出的一种注意力机制，核心解决的是 KV Cache 内存过大的问题。传统 MHA 每个 token 需要缓存完整的 K、V 矩阵；MLA 改为通过一个低秩下投影矩阵将 K、V 联合压缩成一个低维潜向量 $c^{KV}$ 缓存，需要时再用上投影矩阵还原。这样 KV Cache 从 $n_h \times d_h$ 的规模缩到了 $d_c$（约缩小 32 倍）。同时，由于 K、V 从同一个压缩向量解压，保留了 MHA 的完整表达能力，优于 GQA 的截断方式。"

**问：为什么需要解耦 RoPE？**

> "MLA 有个重要的推理优化叫矩阵吸收——把 K 的上投影矩阵预乘进 Q 的上投影矩阵，这样直接用压缩向量计算 Attention 分数，省去展开 K 的步骤。但这要求 K 的计算是线性的、位置无关的，而 RoPE 是位置相关的旋转变换，不能和线性映射换序。所以 MLA 把位置信息分离出来，用独立的 $k^R$/$q^R$ 向量（直接从原始隐层投影并施加 RoPE）携带位置信息，内容向量 $k^C$/$q^C$ 不带 RoPE，这样矩阵吸收只作用于内容部分，RoPE 问题得到规避。"

**问：MLA 的代价是什么？**

> "主要有三点代价：一是额外参数量，需要额外的下投影/上投影矩阵（$W^{DKV}$, $W^{UK}$, $W^{UV}$, $W^{KR}$）；二是训练和推理的工程复杂度更高，不能直接用标准 FlashAttention，需要定制 CUDA kernel；三是位置信息的 $k^R$ 仍需单独缓存，缓存并非零开销，不过相比 MHA 仍有大幅节省。"

### 技术选型直觉

- 如果是**资源受限的小模型**（小于 7B）：GQA 是更好的选择，MLA 的工程复杂度不值得
- 如果是**超大规模生产模型**（100B+），特别是长序列推理场景：MLA 能显著减少 KV Cache，提升并发容量
- 如果是**科研原型**：需要评估是否有定制 FlashAttention 的能力

---

## 前沿发展动态

### MLA 在 DeepSeek-V3 中的演进

DeepSeek-V3（2024 年 12 月）在 MLA 架构上基本继承了 V2 的设计，但做了以下工程优化：

- 使用 **FP8 训练**：在低精度下 MLA 的矩阵乘法精度影响需要仔细处理
- 与 **Multi-head Coordinate Descent (mHC)** 结合，进一步优化 MoE 路由
- KV Cache 量化和压缩与 MLA 压缩结合，形成两层压缩

### 学术界的跟进研究

**MLA 的理论分析（2024-2025）**

- Weng et al. (2024) 从低秩近似的角度分析了 MLA 的表达能力上界，指出当 $d_c < n_h \times d_h / 2$ 时存在不可忽略的表达损失，但实验上损失很小
- 有研究探索将 MLA 思路扩展到 Attention Score 矩阵本身（即不只压缩 KV，也压缩 Q-K 内积结果），但工程难度更大

**其他厂商的 KV Cache 压缩方向**

- **Apple MLX / Mistral**：主要仍用 GQA，工程简单优先
- **Google（Gemini 系列）**：Multi-Query Attention 变体，另有 Linear Attention 探索
- **百川 / 阿里 Qwen**：Qwen2.5 使用 GQA，目前公开信息中尚未迁移到 MLA；Qwen3 部分模型据悉在探索类 MLA 架构
- **Meta LLaMA**：GQA 路线，LLaMA 4 中有 Mixture-of-Experts 但 Attention 仍为 GQA
- **Kimi（Moonshot）**：内部架构未完全公开，据报道在长文本场景中有类似的 KV 压缩机制

**MLA vs Linear Attention 的争论**

Linear Attention（如 Mamba、RWKV、RetNet）从根本上消除了 KV Cache 的序列长度依赖（固定大小的状态），与 MLA 的路线形成竞争。目前主流观点是：
- 对于精确召回（RAG、代码、长文档理解），标准 Attention（包括 MLA）仍有优势
- Linear Attention 在超长序列（>100K token）时的效率优势显著，但实现质量参差不齐

**MLA-in-Transformer 标准化趋势（2025）**

业界开始讨论是否将 MLA 作为大模型的"默认注意力"替代 GQA。Anthropic 和 OpenAI 的相关架构尚未公开，但大量开源复现工作（如 miniMLA、MLA-from-scratch 等）表明 MLA 的工程实现正在走向成熟。

---

## 📎 附录

### A.1 从低秩分解角度理解 MLA

> 关联正文：**Part 1.1 直觉：压缩信息而非删减头**

MLA 的压缩投影 $W^{DKV} \in \mathbb{R}^{d_c \times d}$ 和上投影 $W^{UK} \in \mathbb{R}^{n_h d_h \times d_c}$ 合起来等价于：

$$K_t = W^{UK} W^{DKV} h_t = \tilde{W}^K h_t$$

其中 $\tilde{W}^K = W^{UK} W^{DKV}$ 是一个秩不超过 $d_c$ 的矩阵。

这与 LoRA 的结构 $W = W_0 + BA$（其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$）完全同构。MLA 就是在说：**假设 Key 和 Value 投影矩阵的有效秩不超过 $d_c$**，因此可以用低秩分解表示。

不同的是：
- LoRA 是在已有的全秩权重 $W_0$ 上做低秩增量（微调时节省参数）
- MLA 是直接用低秩矩阵替代完整投影（预训练时节省缓存）

两者都是"低秩假设在 LLM 中普遍成立"这一经验观察的工程化利用。

---

### A.2 矩阵吸收的完整推导

> 关联正文：**Part 3 矩阵吸收**

**内容 Attention 分数的推导：**

$$s_{ts}^C = (q_t^C)^\top k_s^C$$

展开 $q_t^C = W^{UQ} c_t^Q$ 和 $k_s^C = W^{UK} c_s^{KV}$：

$$s_{ts}^C = (c_t^Q)^\top (W^{UQ})^\top W^{UK} c_s^{KV}$$

定义 $\tilde{W} = (W^{UQ})^\top W^{UK} \in \mathbb{R}^{d_c' \times d_c}$，则：

$$s_{ts}^C = (c_t^Q)^\top \tilde{W} \, c_s^{KV}$$

这是一个维度为 $d_c' \times d_c$ 的矩阵乘法，完全在压缩空间内完成。

**输出的推导：**

$$o_t = W^O \sum_s \alpha_{ts} v_s = W^O \sum_s \alpha_{ts} W^{UV} c_s^{KV} = \underbrace{W^O W^{UV}}_{\tilde{W}^{VO}} \sum_s \alpha_{ts} c_s^{KV}$$

先对缓存向量做加权求和（维度 $d_c$），再通过 $\tilde{W}^{VO} \in \mathbb{R}^{d_{out} \times d_c}$ 一步到位，省去显式展开 $v_s$ 的步骤。

**条件：** 上述推导要求 $W^{UK}$ 和 $W^{UV}$ 是固定矩阵（与位置无关），不能包含 RoPE 变换。

---

### A.3 解耦 RoPE 的内积分解验证

> 关联正文：**Part 2 解耦 RoPE**

完整的 Attention 分数（内容 + 位置）：

$$\text{score}(t, s) = (q_t^C)^\top k_s^C + (q_t^R)^\top k_s^R$$

由于 RoPE 的性质，$(q_t^R)^\top k_s^R$ 只依赖于相对位置 $t - s$（而非绝对位置 $t$, $s$），即：

$$(q_t^R)^\top k_s^R = f(t - s, q_t^R, k_s^R)$$

这保留了 RoPE 的相对位置建模能力。同时，$k_s^C$ 不带 RoPE，使得矩阵吸收可以在内容部分单独进行。

两部分相加后，整体注意力分数同时具备内容相关性和相对位置感知，兼顾了两方面的能力。

---

### A.4 DeepSeek-V2 MLA 参数表

> 关联正文：**Part 1.3 具体维度**

| 矩阵 | 形状 | 对应操作 |
|------|------|----------|
| $W^{DQ}$ | $d_c' \times d = 1536 \times 5120$ | Q 下投影 |
| $W^{UQ}$ | $n_h d_h \times d_c' = 16384 \times 1536$ | Q 内容上投影 |
| $W^{QR}$ | $n_h d_h^R \times d_c' = 8192 \times 1536$ | Q RoPE 投影 |
| $W^{DKV}$ | $d_c \times d = 512 \times 5120$ | KV 联合下投影 |
| $W^{UK}$ | $n_h d_h \times d_c = 16384 \times 512$ | K 内容上投影 |
| $W^{UV}$ | $n_h d_h \times d_c = 16384 \times 512$ | V 上投影 |
| $W^{KR}$ | $n_h d_h^R \times d = 8192 \times 5120$ | K RoPE 投影 |

**KV Cache 每个 token 每层的维度对比：**

| 方案 | 缓存维度 |
|------|----------|
| MHA | $2 \times 16384 = 32768$ |
| GQA (g=8) | $2 \times 8 \times 128 = 2048$ |
| MLA | $512 + 8192 = 8704$ |

注：MLA 的实际缓存量约为 MHA 的 26.5%，是 GQA(g=8) 的约 4.25 倍——但 MLA 保留了 128 个完整 head 的表达能力，GQA(g=8) 只有 8 个 KV head。

---

*本讲义撰写于 2026-04-17，后续整合至 Week 12 DeepSeek 深度解析讲义时可直接引用。*
