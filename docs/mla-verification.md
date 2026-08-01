# MLA 实机验证：DeepSeek-V2-Lite

**日期**：2026-08-02　**硬件**：RTX 4080 16 GiB / WSL2　**vLLM**：0.26.0

MLA 是国产模型最重要的差异化 KV 方案，也最贴合「KV 卸载到分布式存储」这个方向——
单 token KV 越小，卸载的带宽压力越低。`docs/notes/02-schemes.md` 里的 MLA 公式此前
只有理论推导，本次做实机核对。

同时这是一次**前置探雷**：`experiments/models.yaml` 早先标注 DeepSeek「在 vLLM 上的
支持可能不稳」。若 MLA 跑不通，Phase 3/4/5 的模型选择需要整体重排，越早知道越好。

**结论：跑得通，公式准，但需要打一个上游未合并的补丁。**

---

## 1. 模型选择

DeepSeek-V2-Lite 是 15.7B MoE（激活 2.4B），BF16 权重约 31 GB——显存和 WSL 内存
（15 GiB）都放不下，必须用量化版。筛下来只有一个可用：

| 候选 | 体积 | 结论 |
|---|---|---|
| `deepseek-ai/DeepSeek-V2-Lite-Chat` | ~31 GB | 装不下 |
| `gaunernst/DeepSeek-V2-Lite-Chat-FP8` | ~15.7 GB | 装得下但没有 KV 的余量 |
| **`TechxGenus/DeepSeek-V2-Lite-Chat-AWQ`** | **8.46 GiB** | **采用** |
| 各 GGUF 版本 | — | vLLM 不适用 |

AWQ / MLA / MoE 是三个互不相干的维度，可以叠加：

| | 全称 | 改造对象 | 效果 |
|---|---|---|---|
| AWQ | Activation-aware Weight Quantization | **权重** | 31 GB → 8.5 GB |
| MLA | Multi-head Latent Attention | **注意力层的 KV 存储** | 单 token KV 变小 |
| MoE | Mixture of Experts | **前馈层** | 15.7B 参数只激活 2.4B |

**AWQ 不影响 KV Cache**——它只压权重，KV 精度由 `--kv-cache-dtype` 单独控制。
因此量化版测出的 KV 数据与原版完全等价，这是本次能用量化版替代的前提。

该仓库 config.json 的 KV 相关字段与 `experiments/configs/deepseek-v2-lite-chat.json`
快照完全一致（`kv_lora_rank: 512`、`qk_rope_head_dim: 64`、`num_hidden_layers: 27`）。

---

## 2. MLA 路径确认

启动日志中三个后端全部按预期选中：

```
Using TRITON_MLA attention backend out of potential backends: ['TRITON_MLA'].
Using FLASH_ATTN MLA prefill backend.
Using 'MARLIN' WNA16 MoE backend.
```

源码侧也对得上。`vllm/config/model.py`：

```python
@property
def use_mla(self) -> bool:
    return self.is_deepseek_mla and not envs.VLLM_MLA_DISABLE
```

**没有量化相关的门槛**——AWQ 不会让模型退回非 MLA 路径。而
`vllm/model_executor/models/deepseek_v2.py:512` 里 KV 投影的输出维度写的正是
`self.kv_lora_rank + self.qk_rope_head_dim`，与理论公式的 576 一致。

---

## 3. KV 体积公式验证 ✅

```
27 层 × (512 + 64) × 2 字节 = 31,104 字节/token = 30.38 KiB
```

不乘 2（K/V 联合压缩成一个潜在向量），不乘头数（潜在向量全头共享），
`qk_rope_head_dim` 是解耦 RoPE 部分，单独存。

两个不同的 `gpu_memory_utilization` 下各验证一次：

| util | vLLM 报告的 KV 显存 | vLLM 报告的 token 数 | 反推 = token 数 × 31,104 | 误差 |
|---|---|---|---|---|
| 0.85 | 5.07 GiB | 175,120 | 5,446,332,480 B = **5.0723 GiB** | < 0.1% |
| 0.80 | 3.46 GiB | 119,328 | 3,711,578,112 B = **3.4567 GiB** | < 0.1% |

**MLA 公式实测确认。** 两次独立测量、四位有效数字吻合，排除巧合。

### 与 GQA 的对照

| | Qwen2.5-7B-AWQ (GQA) | DeepSeek-V2-Lite-AWQ (MLA) |
|---|---|---|
| 单 token KV | 57,344 B（56.0 KiB） | 31,104 B（30.4 KiB） |
| KV 显存 | 8.0 GiB | 5.07 GiB @util 0.85 |
| 可容纳 token | 149,712 | **175,120** |
| ITL p50 | 9.40 ms | **4.90 ms** |

MLA **用更少的显存装下了更多的 token**（少 37% 显存、多 17% token）。
ITL 快近一倍则是 MoE 的功劳（激活 2.4B vs 7B），与 MLA 无关——两个独立的收益不要混淆。

对卸载场景的意义：单 token 30.4 KiB vs 56.0 KiB，意味着同样的存储带宽下
MLA 能支撑约 1.84 倍的 KV 搬运量。

---

## 4. 踩到的三个坑

### 4.1 `nvidia-smi` 在 WSL2 下的显存读数不可信

杀掉上一个 vLLM 服务后：

```
$ nvidia-smi --query-gpu=memory.used,memory.free --format=csv
15852 MiB, 199 MiB          ← 声称只剩 199 MiB
```

而 vLLM 启动时报告：

```
Free memory on device cuda:0 (13.88/15.99 GiB) on startup ...
```

**实际空闲 13.88 GiB，nvidia-smi 报的 199 MiB 是假的。** WDDM 驱动模型下 WSL 里的
`nvidia-smi` 既看不到 Windows 侧进程，报的 `memory.used` 也不反映真实占用。

> **判断显存够不够，只能以 vLLM 自己的启动检查为准。**

### 4.2 杀服务会留下孤儿 EngineCore

`fuser -k 8000/tcp` 只杀掉 API server，子进程 `VLLM::EngineCore` 会存活并继续占用
2.4 GB 内存。必须补一刀：

```bash
fuser -k 8000/tcp; sleep 5; pkill -9 -f "VLLM::EngineCore"
```

（注意不要用 `pkill -f "vllm serve"`——如果你的命令行里含有同样的字符串，会连自己一起杀掉。）

### 4.3 显存贴边时崩在 CUDA graph 捕获

`--gpu-memory-utilization 0.85` 下，权重 8.63 GiB + KV 5.07 GiB = 13.7 GiB，
已经贴着 13.88 GiB 的空闲上限，最后一步捕获计算图时溢出：

```
Capturing CUDA graphs (PIECEWISE):   0%|          | 0/51
torch.AcceleratorError: CUDA error: unknown error
```

注意它**不是干净的 OOM**，而是 `cudaErrorUnknown`，很容易误判成驱动问题。
降到 `0.80` 即可（KV 让出 1.6 GiB 给计算图）。

没有采用 `--enforce-eager`：那样会关掉 CUDA graph，性能数据将无法与 Qwen 基线对照。

---

## 5. 上游 Bug：MLA + 量化在前缀缓存命中时崩溃

服务起来、短请求也正常，但一跑真实压测（3,058 token/请求）就全军覆没：

```
成功 0   失败 8
HTTP 500: EngineCore encountered an issue.
```

根因在 `vllm/model_executor/layers/attention/mla_attention.py:2223`：

```
AttributeError: 'ColumnParallelLinear' object has no attribute 'weight'.
Did you mean: 'qweight'?
```

上方 2213–2216 行**已经为量化层算好了防御变量**，2223 行却没用上。
详细分析、上游 PR 列表与应用方式见 [`patches/README.md`](../patches/README.md)。

打上 `patches/vllm-0.26.0-mla-quantized-dtype.patch` 后：

```
成功 8   失败 0
TTFT p50    54.2 ms       ITL p50   4.90 ms
输入 24,470 token（均 3,058/请求）   输出 473 token
前缀缓存    查询 24,470   命中 21,168   命中率 86.5%
```

**为什么短 prompt 测不出来**：出错的 `_compute_prefill_context()` 只在「有已算好的
上下文需要注意」时才调用——即前缀缓存命中或分块 prefill。16 token 的冒烟请求
一次 prefill 完成，走不到这条路径。

> 这条经验值得单独记：**验证一个部署是否可用，冒烟测试必须覆盖前缀缓存命中路径**，
> 否则你会得到一个「启动成功、单请求正常、上线即崩」的结论。

---

## 6. 结论与后续影响

| 问题 | 结论 |
|---|---|
| MLA 能否在 vLLM 跑通？ | **能**，需打一行补丁 |
| MLA 公式是否与实际存储一致？ | **一致**，两次验证误差 < 0.1% |
| AWQ 是否影响 KV？ | **不影响**，只压权重 |
| 是否需要重排 Phase 3/4/5 的模型选择？ | **不需要** |

**对 Phase 3 的直接影响**：上游 PR [#47564](https://github.com/vllm-project/vllm/pull/47564)
的评论区中，有人用 MLA 模型配合 `LMCacheConnectorV1` 复现了同一个崩溃——经 KV
connector 取回的前缀同样会走 `_compute_prefill_context`。**这个补丁是 Phase 3 的前置条件，
不是可选项。**

### 未做的部分

- 未与 Qwen 在**相同 utilization** 下对照（本次 0.80 vs 基线 0.90，因显存不足）。
  性能数字仅供量级参考，不作为正式对比数据。
- 未测长上下文（8K / 32K）下 MLA 的 ITL 变化。
- TTFT p90 出现 11.3 s 的离群值（p50 仅 54.2 ms），疑似 torch.compile 的重编译，未深究。
