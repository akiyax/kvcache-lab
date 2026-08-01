# KV 体积计算

本文是 [`benchmarks/kv_size.py`](../../benchmarks/kv_size.py) 的设计依据。
所有数值由 [`experiments/configs/`](../../experiments/configs/) 中的 `config.json`
快照计算得出，可复现。

---

## 1. 公式

### GQA

```
单 token = 层数 × 2 × KV头数 × head_dim × 每元素字节数
                   ↑ K 和 V 各一份
```

### MLA

```
单 token = 层数 × (kv_lora_rank + qk_rope_head_dim) × 每元素字节数
```

- **无 ×2**：K 与 V 被联合压缩进同一个潜在向量，一份数据同时还原 K 和 V
- **无 ×头数**：该潜在向量为所有头共享
- **+ qk_rope_head_dim**：RoPE 无法作用于压缩表示，故位置编码部分（解耦 RoPE）
  单独保留，同样全头共享

### 滑动窗口

```
线性部分 = 全注意力层数 × 2 × KV头数 × head_dim × 字节数    （随上下文增长）
常数部分 = 滑窗层数 × 2 × KV头数 × head_dim × 字节数 × 窗口大小   （封顶）
```

上下文短于窗口时，滑窗部分仍按实际长度线性增长；超过窗口后封顶为常数。

### 关于精度

`quantization_config` 描述的是**权重量化，与 KV 精度无关**。AWQ 4-bit 模型的 KV
默认仍是 fp16/bf16（2 字节）。vLLM 可通过 `--kv-cache-dtype fp8` 将 KV 减半，
这是 Phase 3 的实验变量之一。

---

## 2. 实测数据

上下文 32K、单序列、KV 精度 2 字节：

| 模型 | 方案 | 每层存储量 | 层数 | 单 token | @32K |
|------|------|-----------|------|---------|------|
| GPT-OSS-20B | GQA + 滑窗(128) | 8×2×64 = 1,024 | 12 全 + 12 滑 | 24.0 KiB † | 0.75 GiB |
| DeepSeek-V2-Lite | MLA | 512+64 = 576 | 27 | 30.4 KiB | 0.95 GiB |
| GLM-4-9B-Chat | GQA(2) | 2×2×128 = 512 | 40 | 40.0 KiB | 1.25 GiB |
| Qwen2.5-7B-AWQ | GQA(4) | 4×2×128 = 1,024 | 28 | 56.0 KiB | 1.75 GiB |
| Llama-3.1-8B-AWQ | GQA(8) | 8×2×128 = 2,048 | 32 | 128.0 KiB | 4.00 GiB |

† GPT-OSS 为两段式：12 层全注意力线性增长，12 层滑窗在 128 token 处封顶（3 MiB 常数）。

**「每层存储量 × 层数」是理解差异的正确视角**：GLM 层数比 Qwen 多（40 vs 28），但
每层只有一半的 KV 头，净结果仍更省。

作为对照，若 Qwen 退回 MHA（28 组 KV），单 token 将达 **392 KiB**，是 GQA 方案的 7 倍。

四个主线模型首尾相差 **4.2 倍**：

```
DeepSeek MLA 30.4 KiB  <  GLM 40.0  <  Qwen 56.0  <  Llama 128.0 KiB
```

---

## 3. config.json 字段与陷阱

五份配置使用了**四套互不兼容的命名体系**：

| 架构 | `model_type` | 关键字段 |
|------|--------------|---------|
| GQA（标准） | `qwen2` / `llama` | `num_hidden_layers`、`num_key_value_heads`、`hidden_size` |
| GQA（ChatGLM） | `chatglm` | `num_layers`、`multi_query_group_num`、`kv_channels` |
| MLA | `deepseek_v2` | `num_hidden_layers`、`kv_lora_rank`、`qk_rope_head_dim` |
| 滑动窗口 | `gpt_oss` | `layer_types`、`sliding_window`、`head_dim` |

已确认的陷阱：

| # | 陷阱 | 出现在 | 后果 |
|---|------|--------|------|
| 1 | `head_dim` 缺失，需由 `hidden_size / num_attention_heads` 推导 | Qwen、Llama、GLM | 需 fallback 逻辑 |
| 2 | `head_dim` 显式存在且 **≠** 推导值（64 vs 2880/64=45） | **GPT-OSS** | 盲目推导**低估 42%** |
| 3 | `num_key_value_heads: 16` 为残留字段，MLA 不使用 | **DeepSeek** | 误用 GQA 公式**高估 8.9 倍** |
| 4 | `sliding_window: 131072` 但 `use_sliding_window: false` | **Qwen** | 误走滑窗公式 |
| 5 | 整套字段名不同，标准 key 不存在 | **GLM** | 直接 `KeyError` |
| 6 | 无 `torch_dtype` 字段 | **GPT-OSS** | 字节数无从判断，需默认值 |
| 7 | `use_cache: false` | Llama AWQ | 不影响 KV 计算，起服务时留意 |

**结论：KV 体积计算必须先读 `model_type` 分派公式，`head_dim` 优先取显式值，
滑窗需检查开关，缺失必需字段应明确报错而非静默使用默认值。**

各配置文件的来源与 commit 见
[`experiments/configs/README.md`](../../experiments/configs/README.md)。

---

## 4. 脚本用法

零依赖（仅 stdlib），Windows / WSL 均可直接运行。

```bash
# 五模型对照表
python benchmarks/kv_size.py --all

# 单模型，含推导过程（显示每个字段取自哪里、走的哪条公式）
python benchmarks/kv_size.py deepseek-v2-lite-chat -v

# 估算并发路数
python benchmarks/kv_size.py qwen2.5-7b-instruct-awq --ctx 32768 --gpu 16 --weights 5.5
#   → 显存预算 8.90 GiB，可并发约 5 路

# fp8 KV 的效果
python benchmarks/kv_size.py qwen2.5-7b-instruct-awq --kv-dtype fp8 --gpu 16 --weights 5.5
#   → 体积减半，可并发约 10 路

# 测试（18 个用例，每个钉住一个上表中的陷阱）
python -m unittest discover -s benchmarks
```

新增模型时需在 `_HANDLERS` 中注册 `model_type`。**未知架构应报错而非套用现有分支**
——不同方案的 KV 存储方式差异极大，猜测的代价是数量级的误差。

---

## 5. 对卸载方案的影响

单 token KV 体积直接决定卸载的可行性——同样的存储后端带宽，MLA 方案（30.4 KiB）
比 Llama 的 GQA（128 KiB）宽裕 4.2 倍。

这是 MLA 对分布式 KV 存储场景格外重要的原因，也是本实验台把 DeepSeek-V2-Lite
列为主线模型的理由。带宽约束的详细分析见 [04-serving.md](04-serving.md)。

---

## 待验证

- [x] vLLM 中 MLA 的实际存储布局是否与理论公式一致（部分 kernel 可能解压后再存）
      → **一致**。TRITON_MLA 后端按 `kv_lora_rank + qk_rope_head_dim` 存潜在向量，
      未解压。两个 utilization 下各验证一次，误差 < 0.1%。
      见 [`../mla-verification.md`](../mla-verification.md)。
- [ ] 实际显存占用与理论计算的差距（分页开销、碎片、预留块）
- [ ] `--kv-cache-dtype fp8` 的实际减半效果与质量影响
- [ ] GPT-OSS 滑动窗口在 vLLM 中的实现方式与理论模型的一致性
