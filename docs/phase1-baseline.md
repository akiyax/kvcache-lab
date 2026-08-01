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

```bash
python benchmarks/bench_serve.py --model qwen2.5-7b-awq \
  -n 32 -c 8 --input-len 1024 --output-len 128
```

前缀缓存**未命中**（每请求首块含唯一标记），即纯 prefill 基线。

| 指标 | p50 | p90 | p99 | 均值 |
|------|-----|-----|-----|------|
| TTFT (ms) | 2882.8 | 2985.1 | 3026.7 | 2752.5 |
| ITL (ms) | 11.37 | 11.56 | 14.00 | 10.56 |

| | |
|---|---|
| 请求数 / 并发 | 32 / 8（32 成功，0 失败） |
| 输入 token | 21,654（均 676/请求） |
| 输出 token | 4,096（均 128/请求） |
| **吞吐** | **249.9 output tok/s，1.95 req/s** |
| 总耗时 | 16.39 s |

说明：

- TTFT 接近 2.9 秒是**并发 8 一次性打满**导致的排队，非单请求延迟。后续需补测
  并发 1 的纯延迟基线作为对照。
- ITL 11.37 ms ≈ 88 tok/s 单流。此时上下文仅约 676 token，KV 读取量可忽略，
  瓶颈主要在权重读取——符合 [04-serving.md](notes/04-serving.md) 的带宽模型。
- 压测脚本零依赖，本次从 **Windows 侧**直接压 WSL 内的服务。

---

## 6. 待办

- [ ] 补测并发 1 的纯延迟基线（分离排队延迟与单请求延迟）
- [ ] 补测长上下文（8K / 32K）基线，观察 KV 增大后 ITL 的变化
- [ ] 用 `--shared-prefix-len` 测前缀缓存命中的收益（Phase 2 正题）
- [ ] `--kv-cache-dtype fp8` 的实际减半效果与质量影响
- [ ] 其余四个模型的部署验证，尤其 DeepSeek-V2-Lite 的 MLA 在 vLLM 中的
      实际显存布局是否与理论公式一致
