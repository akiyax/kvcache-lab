# KV Cache 笔记

| 文档 | 内容 | 什么时候看 |
|------|------|-----------|
| [01-attention.md](01-attention.md) | QKV 机制、为什么缓存 K/V、多头、层与维度 | 建立基本理解 |
| [02-schemes.md](02-schemes.md) | MHA / MQA / GQA / MLA / SWA、稀疏与线性混合、2026 格局 | 搞清各方案差异 |
| [03-sizing.md](03-sizing.md) | 计算公式、config 字段陷阱、实测数据 | 算 KV 体积、写脚本 |
| [04-serving.md](04-serving.md) | Prefill/Decode、分块、前缀复用、卸载介入点 | 理解卸载工作本身 |

以下为速查，回来查东西看这一页即可。

---

## 公式

```
GQA    单 token = 层数 × 2 × KV头数 × head_dim × 字节数

MLA    单 token = 层数 × (kv_lora_rank + qk_rope_head_dim) × 字节数
                  ↑ 不乘 2（K/V 联合压缩），不乘头数（全头共享）

滑窗    线性部分 = 全注意力层数 × 2 × KV头数 × head_dim × 字节数
       常数部分 = 滑窗层数 × 2 × KV头数 × head_dim × 字节数 × 窗口大小
```

字节数默认 2（fp16/bf16）。**权重量化（AWQ 4-bit 等）不影响 KV 精度。**

## 缩写

| 缩写 | 全称 | KV 头数 | 代表模型 |
|------|------|---------|---------|
| **MHA** | Multi-**H**ead Attention | = Q 头数 | GPT-2、早期 Llama |
| **MQA** | Multi-**Q**uery Attention | 1 | PaLM、Falcon |
| **GQA** | **G**rouped-Query Attention | 中间值 | Qwen、Llama-3、GLM-4 |
| **MLA** | Multi-head **L**atent Attention | 低秩压缩 | DeepSeek-V2/V3 |
| **SWA** | **S**liding **W**indow Attention | 正交概念，可与上叠加 | Mistral、Gemma、GPT-OSS |

> 记忆线索看中间字母：**H**ead → **Q**uery → **G**rouped → **L**atent。
> SWA 不是第五种方案，详见 [02-schemes.md](02-schemes.md)。

## 实测数据

上下文 32K、单序列、KV 精度 2 字节：

| 模型 | 方案 | 单 token | @32K |
|------|------|---------|------|
| GPT-OSS-20B | GQA + 滑窗(128) | 24.0 KiB † | 0.75 GiB |
| DeepSeek-V2-Lite | MLA | 30.4 KiB | 0.95 GiB |
| GLM-4-9B-Chat | GQA(2) | 40.0 KiB | 1.25 GiB |
| Qwen2.5-7B-AWQ | GQA(4) | 56.0 KiB | 1.75 GiB |
| Llama-3.1-8B-AWQ | GQA(8) | 128.0 KiB | 4.00 GiB |

† 仅线性部分，滑窗层另有 3 MiB 常数。

## config.json 陷阱

| # | 陷阱 | 出现在 | 后果 |
|---|------|--------|------|
| 1 | `head_dim` 缺失需推导 | Qwen、Llama、GLM | 需 fallback |
| 2 | `head_dim` 显式值 ≠ 推导值（64 vs 45） | **GPT-OSS** | 低估 42% |
| 3 | `num_key_value_heads` 为 MLA 残留字段 | **DeepSeek** | 高估 8.9 倍 |
| 4 | `sliding_window` 有值但 `use_sliding_window: false` | **Qwen** | 误走滑窗公式 |
| 5 | 整套字段名不同 | **GLM** | `KeyError` |
| 6 | 无 `torch_dtype` | **GPT-OSS** | 需默认值 |
| 7 | `use_cache: false` | Llama AWQ | 不影响计算 |

**必须先读 `model_type` 分派公式，`head_dim` 优先取显式值。** 详见 [03-sizing.md](03-sizing.md)。

## 脚本

```bash
python benchmarks/kv_size.py --all                        # 对照表
python benchmarks/kv_size.py qwen2.5-7b-instruct-awq -v   # 推导过程
python benchmarks/kv_size.py qwen2.5-7b-instruct-awq --gpu 16 --weights 5.5
python benchmarks/kv_size.py qwen2.5-7b-instruct-awq --kv-dtype fp8

python -m unittest discover -s benchmarks                 # 18 个测试
```

## 一句话结论

- 缓存 K、V 而不缓存 Q，因为前者被反复读取、后者用完即弃
- 存储是 **O(N)** 不是 O(N²)：每个 token 只写一次，反复复用
- 每层各存一份且不可替代，是「N 个抽象层次的理解快照」而非副本
- **Decode 是带宽瓶颈**：每生成 1 token 要读完整个 KV Cache
- 批处理能均摊权重，**不能均摊 KV** —— 并发越高 KV 越主导带宽
- 活跃请求的 KV 必须常驻显存；卸载的是**不活跃请求**与**待复用前缀**
- 最高明的搬运是**不用搬**：前缀命中直接跳过 prefill
