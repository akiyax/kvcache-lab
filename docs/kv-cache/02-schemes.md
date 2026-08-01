# 注意力方案对比

各方案的差异直接决定 KV Cache 的大小，进而决定卸载方案的设计空间。
计算公式见 [03-sizing.md](03-sizing.md)。

---

## 1. 速查

| 缩写 | 全称 | 中文 | KV 头数 | 相对体积 | 代表模型 |
|------|------|------|---------|---------|---------|
| **MHA** | Multi-**H**ead Attention | 多头注意力 | = Q 头数 | 基准 100% | GPT-2、早期 Llama |
| **MQA** | Multi-**Q**uery Attention | 多查询注意力 | 1 | ~3% | PaLM、Falcon |
| **GQA** | **G**rouped-Query Attention | 分组查询注意力 | 中间值 | 12%~25% | Qwen、Llama-3、GLM-4 |
| **MLA** | Multi-head **L**atent Attention | 多头潜在注意力 | 低秩压缩 | 另一套公式 | DeepSeek-V2/V3 |

> 记忆线索看中间的字母：**H**ead（每头一份）→ **Q**uery（全部共享）→
> **G**rouped（分组共享）→ **L**atent（压缩成潜在向量）。

---

## 2. MHA / MQA / GQA：调整保留几组 K/V

三者在同一条轴上，MQA 与 MHA 是 GQA 的两个极端：

```
假设 32 个 Q 头：

MHA   Q Q Q Q Q Q Q Q ... (32个)
      K K K K K K K K ... (32个)   ← 一对一，KV 最肥

MQA   Q Q Q Q Q Q Q Q ... (32个)
      K                            ← 全部共享 1 组，最省但质量下降

GQA   Q Q Q Q | Q Q Q Q | ...      ← 分成若干组
      K       | K       | ...      ← 每组共享 1 组，折中
```

**GQA 是当前的事实标准。** 实际取值差异很大：

| 模型 | Q 头 | KV 头 | 几个 Q 共用一组 |
|------|------|-------|----------------|
| GLM-4-9B | 32 | 2 | 16 |
| Qwen2.5-7B | 28 | 4 | 7 |
| Llama-3.1-8B | 32 | 8 | 4 |

共用越多越省显存。质量损失有限的原因见 [01-attention.md](01-attention.md) 第 3 节。

---

## 3. MLA：不减头数，改为低秩压缩

**MLA 不在上面那条轴上。** 它不减少头数，而是把 K、V **联合压缩成一个低秩潜在
向量**存储，使用时再上投影还原。

DeepSeek-V2-Lite 的实际参数：

| 字段 | 值 | 是否进缓存 |
|------|-----|-----------|
| `kv_lora_rank` | 512 | ✅ 潜在向量维度 |
| `qk_rope_head_dim` | 64 | ✅ 解耦 RoPE 部分 |
| `qk_nope_head_dim` | 128 | ❌ 走压缩路径，不单独存 |
| `v_head_dim` | 128 | ❌ 从潜在向量还原 |
| `num_key_value_heads` | 16 | ❌ **残留字段，MLA 不使用** |

三个关键点：

- **不乘 2**：K 与 V 被压缩进同一个潜在向量，一份数据同时还原两者
- **不乘头数**：该潜在向量为所有头共享
- **额外的 `+ qk_rope_head_dim`**：RoPE 位置编码无法作用于压缩表示，故 key 被拆成
  两部分——128 维走压缩路径，64 维单独保留并施加 RoPE（称为解耦 RoPE），
  这 64 维同样全头共享

压缩效果：若按 GQA 方式存储，每层每 token 需 `16 × (192 + 128) = 5,120` 个元素；
MLA 实际只存 `512 + 64 = 576` 个，**压缩约 8.9 倍**。

存储量低于 MQA，质量接近 MHA，代价是计算更复杂、对推理框架的支持要求更高。

---

## 4. SWA 是另一条正交的轴

**SWA**（**S**liding **W**indow Attention，滑动窗口注意力）常与上面几个并列出现，
但它回答的是完全不同的问题：

```
每个 token 存多少   ──→  MHA / MQA / GQA / MLA    （纵向压缩）
存多少个 token      ──→  全部 / 只保留最近 N 个     （横向截断）
```

**两者正交，可以叠加。** GPT-OSS-20B 即为典型：其 `num_key_value_heads: 8` 是标准
GQA，同时 24 层中有 12 层使用窗口为 128 的滑动窗口。准确描述是「GQA + SWA」，
而非第五种独立方案。（`benchmarks/kv_size.py` 输出中的 `SWA` 标签是简写，
指「含滑窗层的模型」。）

滑窗层只保留最近 N 个 token 的 KV，超出部分直接丢弃，因此其占用**不随上下文增长**，
在长上下文场景下极为省显存——代价是这些层无法直接看到远处的信息，需依赖全注意力
层与层间传递来弥补，故实践中多为二者交替排列。

---

## 5. 架构演进与当前格局

> 本节信息检索于 2026 年 8 月，该领域迭代很快，引用前建议复核。

**主流开源模型的绝大多数仍是 GQA**（Llama-3、Mistral、Qwen2.5、GLM-4 这一代），
它是效率与质量的最优折中。但 2026 年榜单前列的模型基本都已走出纯 GQA：

| 模型 | 方案 |
|------|------|
| GLM-5.2、Kimi K2.7 | **MLA** |
| DeepSeek-V4 | **稀疏注意力**（CSA / HCA） |
| MiniMax-M3 | GQA 骨架 + **稀疏块选择**（MSA） |
| Qwen3.5 | **门控线性注意力混合** |

值得注意的是 **GLM 从 GQA 转向了 MLA**——本实验台矩阵中的 GLM-4-9B 属于「极限 GQA」，
而新一代 GLM-5.2 已改用 MLA。

### 第三条轴：稀疏注意力

DeepSeek-V4、MiniMax-M3 走的方向不是压缩每份 KV，而是**让每次注意力只看一部分
token**：

```
每 token 存多少   →  MHA / MQA / GQA / MLA      （纵向压缩）
存多少个 token    →  SWA 滑动窗口               （横向截断）
每次看多少个      →  稀疏注意力 CSA / MSA / HCA  （动态选择）
```

滑窗是「固定只看最近 N 个」，稀疏是「动态挑选看哪些块」——更灵活，长上下文效果更好。

### 第四条轴：线性注意力混合

Qwen3.5、Kimi Linear、MiniMax-01 采用混合架构：大部分层用线性注意力
（Gated DeltaNet 等变体），少部分层保留标准 softmax 注意力。

**这一方向对 KV 卸载的影响最大**：线性注意力的「KV」是一个**固定大小的状态**，
不随上下文增长。混合架构下，会膨胀的 KV Cache 只剩那少数几层。

### 对本实验台的意义

现有模型矩阵（GQA × 3 + MLA + SWA）作为学习基础依然成立——GQA 是存量主流，
MLA 是增量趋势，且前沿模型的规模远超单卡可跑范围。但需知道 KV 优化已有四条路线，
各自对存储层的压力模式不同：

| 路线 | 做法 | 对卸载的影响 |
|------|------|-------------|
| 压缩（MLA） | 每 token 存得更少 | 直接减小搬运量 |
| 截断（SWA） | 只保留最近 N 个 | 部分层占用变常数 |
| 稀疏 | 每次只读一部分 | 减少读带宽，但仍需全量驻留 |
| 线性混合 | 部分层用固定状态 | 大幅减少需卸载的层数 |

---

## 参考

- [Recent Developments in LLM Architectures: KV Sharing, mHC, and Compressed Attention](https://magazine.sebastianraschka.com/p/recent-developments-in-llm-architectures)
- [KV Cache Optimization for LLMs 2026: Engineering Guide](https://www.digitalapplied.com/blog/kv-cache-optimization-techniques-2026-engineering-guide)
- [HySparse: A Hybrid Sparse Attention Architecture](https://arxiv.org/pdf/2602.03560)
- [GQLA: Group-Query Latent Attention for Hardware-Adaptive LLM Decoding](https://arxiv.org/pdf/2605.15250)
