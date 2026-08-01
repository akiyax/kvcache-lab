# Phase 1 实验记录：vLLM 基线

日期：2026-08-02

目标：在 WSL2 中部署 vLLM，建立 TTFT / 吞吐基线，并用实测数据验证
[`benchmarks/kv_size.py`](../benchmarks/kv_size.py) 的 KV 体积计算。

---

## 1. 环境与版本

| 项 | 值 |
|----|-----|
| GPU | RTX 4080 16376 MiB，驱动 591.86，CUDA 13.1 |
| 系统 | Windows 11 + WSL2 Ubuntu 24.04.1，内核 6.18.33.2 |
| Python | 3.12.3（venv 位于 `~/venvs/vllm`） |
| **vLLM** | **0.26.0** |
| PyTorch | 2.11.0+cu130 |
| transformers | 5.14.1 |
| 模型 | Qwen2.5-7B-Instruct-AWQ，commit `b25037543e93`，本地 `~/models/` |

WSL 可用内存 15 GiB（宿主 32 GB 的一半）。Phase 3 做 CPU offload 时需通过
`.wslconfig` 调大。

---

## 2. WSL2 上的五个坑

按遇到的顺序记录。**每个都会导致启动直接失败**，且报错信息大多不指向真正的解法。

### ① `RuntimeError: UVA is not available`

vLLM 0.26 的新 GPU model runner 用 `UvaBuffer` 管理 staged write，强依赖 pinned memory。
而 `vllm/platforms/cuda.py` 中：

```python
if in_wsl():
    version = _get_wsl_kernel_version()
    if version is None or version < (4, 19, 121):
        return False
    # On compatible WSL2 kernels, pinned memory is supported but
    # disabled by default. Enable it via VLLM_WSL2_ENABLE_PIN_MEMORY=1.
    return envs.VLLM_WSL2_ENABLE_PIN_MEMORY   # 默认 False
```

**内核版本是够的**（6.18.33 ≫ 4.19.121），但 vLLM 出于保守在 WSL2 上默认关闭 pinned
memory。报错信息完全没提这个环境变量，需读源码才能定位。

**解法**：`VLLM_WSL2_ENABLE_PIN_MEMORY=1`

### ② `fatal error: Python.h: No such file or directory`

Triton 运行时需编译 `cuda_utils.c`，要求 Python 开发头文件。Ubuntu 默认只装运行时。

**解法**：`sudo apt install -y python3.12-dev build-essential`

### ③ `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`

flashinfer 需 JIT 编译，要找 nvcc。pip 装的 `nvidia-cuda-nvcc` 在 venv 内，
不在系统默认路径。

**解法**：`export CUDA_HOME=$VENV/lib/python3.12/site-packages/nvidia/cu13`
（该目录是完整的 toolkit 布局：`bin/` `include/` `lib/` `nvvm/`）

### ④ `FileNotFoundError: 'ninja'`

pip 装了 `ninja`，但用绝对路径调用 `vllm` 时 venv 的 `bin/` 不在 PATH 上。

**解法**：`export PATH=$VENV/bin:$PATH`（或直接激活 venv）

### ⑤ `CUDA compiler and CUDA toolkit headers are incompatible`

flashinfer 0.6.14 自带的 CCCL 头文件与 CUDA 13.3 的 nvcc 不兼容——**上游版本矩阵问题，
非配置错误**。触发点是 flashinfer 的 sampling 模块 JIT 编译。

**解法**：`VLLM_USE_FLASHINFER_SAMPLER=0`，回退到 vLLM 原生采样。
仅影响采样算子，注意力后端不受影响。

> 未采用 `--enforce-eager` 绕过编译问题：它会同时关闭 torch.compile 与 CUDA graph，
> 测出的基线不能代表 vLLM 真实性能，而后续所有 Phase 的对比都建立在此基线上。

---

## 3. 可用的启动命令

```bash
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME=$HOME/venvs/vllm/lib/python3.12/site-packages/nvidia/cu13
export PATH=$HOME/venvs/vllm/bin:$CUDA_HOME/bin:$PATH

vllm serve ~/models/Qwen2.5-7B-Instruct-AWQ \
  --served-model-name qwen2.5-7b-awq \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

启动耗时约 100 秒（权重 6 秒 + torch.compile 22 秒 + CUDA graph 9 秒 + 其余）。
`~/.cache/vllm/torch_compile_cache/` 有缓存，二次启动更快。

**其他环境事实**：

- PyPI 必须只配清华源，**不能加 `extra-index-url = https://pypi.org/simple`**。
  两个源都有的包 pip 会选到 pypi.org，实测 230 KB/s vs 清华 35 MB/s，差 150 倍。
- HuggingFace 直连不通，走 `hf-mirror.com`，按 commit 固定 revision 下载。

---

## 4. KV 体积公式验证 ✅

vLLM 启动日志：

```
Available KV cache memory: 8.0 GiB
GPU KV cache size: 149,712 tokens
Maximum concurrency for 32,768 tokens per request: 4.57x
```

对账 `kv_size.py` 的计算：

```
149,712 tokens × 57,344 字节/token = 8,585,564,928 = 7.996 GiB
                                     ≈ vLLM 报告的 8.0 GiB   ✅
```

**单 token 56.0 KiB 得到实测确认，GQA 公式正确。**

### 修正：`gpu_memory_utilization` 的语义

初次预测（6.22 GiB / 约 3 路并发）偏低，原因是把 utilization 理解成了「按空闲显存计算」。
vLLM 的实际口径是**按显存总量**：

```
14.39 GiB  = 15.99 总量 × 0.9        ← 不是 14.69 空闲 × 0.9
-  5.29    模型权重
-  1.04    峰值激活
-  0.48    CUDAGraph
-  0.06    non-torch
=  8.0 GiB 实际给 KV
```

已据此修正 `kv_size.py` 的 `--gpu` 语义并新增 `--overhead` 参数（默认 1.6 GiB，
取自上述实测）。修正后预测 7.50 GiB / 140,453 tokens / 4.3 路，与实测
8.0 GiB / 149,712 tokens / 4.57 路 **相差约 6%**。

残余偏差的原因：vLLM 基于启动时的**真实空闲显存**（14.69 GiB）做 profiling，
而非严格按 utilization 目标（14.39 GiB），故实际分配略高于请求值。日志中也提示了
`--kv-cache-memory=7911991092`（7.37 GiB）才是「恰好符合请求」的值。

---

## 5. 基线数据

负载为「长系统提示 + 长文档 + 不同问题」，素材见
[`benchmarks/prompts/`](../benchmarks/prompts/)，输入约 2,900 token/请求。

```bash
# 基线：前缀完全不命中
python benchmarks/bench_serve.py --model qwen2.5-7b-awq -n 32 -c 8 \
  --doc-chars 4000 --output-len 128

# 对照：共享 system + doc，仅问题不同
python benchmarks/bench_serve.py --model qwen2.5-7b-awq -n 32 -c 8 \
  --doc-chars 4000 --output-len 128 --shared
```

输入约 2,930 token，**输出 512 token**（接近真实问答场景，见下文「输出长度的影响」）。
命中率取自 vLLM `/metrics` 的 `prefix_cache_*` 计数器差值。

| 并发 | 前缀 | TTFT p50 | ITL p50 | 吞吐 (tok/s) | 命中率 |
|------|------|---------|---------|-------------|-------|
| 1 | 不命中 | 541.6 | 9.40 | 95.7 | **0.0%** |
| 1 | 命中 | **26.3** | 9.44 | 105.8 | 99.7% |
| 8 | 不命中 | 1465.9 | — | 398.7 | **0.0%** |
| 8 | 命中 | **67.1** | — | **740.1** | 99.7% |

前缀缓存的收益：

| | 并发 1 | 并发 8 |
|---|-------|-------|
| TTFT | −95.1% | **−95.4%** |
| 吞吐 | +10.6% | **+85.6%** |

分析：

- **命中率 0.0% 印证了基线构造正确**——唯一标记置于首块使前缀从第一个块起即不匹配，
  vLLM 自身的计数器确认零命中。命中侧 99.7%，未达 100% 是因末尾问题所在的部分块不同。
- **TTFT 降幅约 95%**：命中后 99.7% 的 token 无需 prefill，仅剩末尾问题需要计算。
- **并发 1 的吞吐提升可由算术完全解释**：

  ```
  不命中  541.6 ms + 512 × 9.40 ms = 5,354 ms/请求  →  95.6 tok/s（实测 95.7）
  命中     26.3 ms + 512 × 9.44 ms = 4,860 ms/请求  → 105.4 tok/s（实测 105.8）
                                            提升 10.2%（实测 10.6%）
  ```

  即**并发 1 时缓存对吞吐的贡献，仅是省下的 TTFT 摊薄到整段生成时间上**。
  ITL 完全不变（9.40 → 9.44），因为 decode 阶段不受前缀缓存影响。

- **并发 8 能达到 +85.6%，多出的部分来自消除 prefill 争抢。** vLLM 默认开启
  chunked prefill，prefill 分块与 decode 交错争抢 GPU；命中后 prefill 工作量骤减，
  decode 得以连续执行。这正是 [04-serving.md](notes/04-serving.md) 判断的实证：
  **缓存直接受益的是 TTFT，对吞吐的改善是间接的**，且并发越高、prefill 争抢越严重，
  间接收益越大。

### 输出长度的影响

输出长度决定输入输出比，而该比值直接影响吞吐收益的幅度。并发 8 下实测：

| output-len | 输入:输出 | 吞吐（不命中 → 命中） | 提升 | TTFT 降幅 |
|-----------|---------|---------------------|------|----------|
| 128 | 22.7 : 1 | 174.1 → 641.8 | **+268.6%** | −94.5% |
| **512** | **5.7 : 1** | 398.7 → 740.1 | **+85.6%** | −95.4% |

**仅把输出改为真实长度，吞吐收益就从 +269% 降到 +86%**，相差约 3 倍。而 TTFT 降幅
几乎不变——**TTFT 只与 prefill 相关，与输出长度无关**。

早期采用 128 的上限，且 `system.txt` 中还写有「默认用三到五句话回答」，两者叠加把
输出压到极短，人为放大了输入输出比。现已修正：`system.txt` 对回答长度保持中立，
长度仅由 `--output-len` 单一变量控制，便于干净地扫描该维度。

> **已废弃的早期数据**：
> - 「TTFT −52.8%、吞吐 +97.0%」：从 Windows 侧压测，被约 2.1 秒的客户端伪影淹没
>   （见第 7 节）。
> - 「吞吐 +298.4%」：输出限制为 128 token 且系统提示要求简短，输入输出比虚高。
> - 「249.9 tok/s」：基于「两句话重复十余次」的合成填充文本，输入仅 676 token。

---

## 6. 与业界数据的对照

本节的目的是给上面的数字一个参照系。**+298% 的吞吐提升是特定负载下的上界，
不是通用结论**，直接引用会产生误导。

### 解码速度：正常

| | 值 |
|---|---|
| 本次实测（Qwen2.5-7B-AWQ @ RTX 4080） | 单流 ITL 9.4 ms ≈ **106 tok/s** |
| 公开数据（8B Q4 @ RTX 4090） | 90 ~ 140 tok/s |
| 理论上限（权重 5.29 GiB ÷ 716 GB/s） | 135 tok/s |

4080 带宽低于 4090（716 vs 1008 GB/s），但本模型也小于 8B，两项因素大致抵消，
106 tok/s 落在公开区间中段，达到理论上限的 78%。**解码性能无异常。**

### 缓存收益：显著高于业界典型值

| 指标 | 业界典型（聊天负载） | 本次实测 |
|------|---------------------|---------|
| 前缀命中率 | 70 ~ 90% | **99.7%** |
| TTFT 降幅 | −77%（480 → 110 ms） | **−95%** |
| 吞吐提升 | +30 ~ 50% | **+86%** |

修正输出长度后（见上节），吞吐提升从 +298% 降至 +86%，与业界区间已属同一量级。
**剩余差距主要由命中率解释，而非模型大小。**

**命中率接近满值。** 本负载 4,772 字符中 4,742 字符共享（99.4%），仅末尾问题不同。
真实聊天场景中用户消息不断累积，共享比例低得多。按比例推算：命中 80% 则 prefill
剩余 20%，TTFT 应降约 80%——与业界报告的 −77% 吻合。**两组数据是同一条曲线上的
不同点，并不矛盾。**

输入输出比亦仍高于典型聊天（5.7 : 1 vs 约 1.7 : 1），进一步放大了收益。

### 为何与模型大小无关

模型规模在比值中会约掉：

```
prefill 耗时 ∝ 输入 token × 参数量 ÷ 算力
decode  耗时 ∝ 输出 token × 参数量 ÷ 带宽
                          ↑ 参数量在比值中约去

比值 ≈ (输入/输出) × (带宽/算力)
```

即 prefill 与 decode 的耗时比只取决于**负载形状**与**硬件的算力带宽比**。换用更大
模型，两侧同步变慢，缓存收益的百分比基本不变。

### 结论

本次数字可信，但**适用范围是「高共享前缀 + 长输入短输出」这类负载**。这正是
`design.md` 选择长文档问答的原因：它是 KV 复用收益最明显的场景，也是生产环境中
真正需要 KV 卸载的场景。后续若要给出通用结论，需补测不同共享比例与输入输出比的
组合。

参考：
[SqueezeBits: vLLM vs TensorRT-LLM #12](https://blog.squeezebits.com/vllm-vs-tensorrtllm-12-automatic-prefix-caching-38189)、
[vLLM Automatic Prefix Caching 设计文档](https://docs.vllm.ai/en/stable/design/prefix_caching/)、
[8B 模型在单卡 4090 上的 vLLM 实测](https://ermolushka.github.io/posts/vllm-benchmark-4090/)

---

## 7. 压测方法：必须在 WSL 内运行

**压测客户端与被测服务必须同侧。** 从 Windows 压测 WSL 内的服务会引入两项与被测
系统完全无关的干扰，且量级足以淹没真实信号。

### ① `localhost` 的 IPv6 回退：约 2,100 ms

Windows 上 `localhost` 优先解析为 IPv6 `::1`，而 WSL2 的端口转发只监听 IPv4，
连接需等超时后回退。仅改主机名、其余不变：

| URL | TTFT p50 |
|-----|---------|
| `http://localhost:8000` | 2,106.9 ms |
| `http://127.0.0.1:8000` | 43.9 ms |

已将脚本默认值改为 `127.0.0.1`。

### ② WSL2 端口转发的固定延迟：约 42 ms

即便用 IPv4，Windows→WSL 路径仍存在恒定延迟，**与 prompt 大小无关**：

| doc-chars | Windows | WSL 内 | 差值 |
|-----------|---------|--------|------|
| 100 | 61.7 | 19.4 | 42.3 |
| 1,000 | 61.7 | 18.8 | 42.9 |
| 4,000 | 67.5 | 24.4 | 43.1 |
| 8,000 | 74.7 | 32.2 | 42.5 |

差值恒定约 42.7 ms 而不随数据量变化，符合 Nagle 算法与延迟 ACK 相互作用的特征
（客户端分多次写入请求头与 body，Nagle 压住后续小包等待 ACK）。脚本已设置
`TCP_NODELAY`，但**未继续深究**——这条路径上的行为不是本项目要测量的对象，
在 WSL 内运行即可完全规避。

### 为何影响巨大

这些伪影是**每请求的固定开销**。当缓存命中把 TTFT 压到 25 ms 量级时，2,100 ms 的
伪影会使真实信号完全不可见：

```
真实（WSL 内）      542.1 → 25.4 ms    改善 21 倍
被污染（Windows）  4632.3 → 2185.9 ms  改善 2 倍
```

**测量系统的开销必须远小于被测量的对象**，否则优化效果越好，测量误差占比越大。

### ③ 连接复用

脚本改为每个 worker 持有一条长连接。新建 TCP 连接经 WSL2 转发约需 14 ms，
在 WSL 内部仅 0.07 ms。真实压测客户端亦均复用连接。

---

## 8. 待办

- [ ] 补测长上下文（8K / 32K）基线，观察 KV 增大后 ITL 的变化
- [ ] 扫描不同共享比例（如 30% / 50% / 80% / 99%），画出命中率与 TTFT 降幅的关系曲线，
      验证本次的 99.7% 与业界的 70~90% 确在同一条曲线上
- [ ] 扫描不同输入输出比（22:1 / 5:1 / 1.7:1），量化负载形状对缓存收益的影响
- [ ] `--kv-cache-dtype fp8` 的实际减半效果与质量影响
- [ ] 其余四个模型的部署验证，尤其 DeepSeek-V2-Lite 的 MLA 在 vLLM 中的
      实际显存布局是否与理论公式一致
