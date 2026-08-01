# kvcache-lab

> 在单卡消费级 GPU 上搭建 KV Cache 分层卸载实验台：从 GPU 显存 → CPU 内存 → 本地磁盘 → 分布式存储，
> 基于 vLLM (LMCache / KV Connector) 与 SGLang (HiCache) 做逐层卸载实验与量化对比。

A hands-on lab for LLM KV cache tiering & offloading on a single consumer GPU,
covering vLLM (LMCache / KV Connector) and SGLang (HiCache).

## 动机

大模型推理服务中，KV Cache 是显存的最大消耗方之一。生产环境的主流做法是把 KV Cache
分层卸载：GPU 显存放热数据，CPU 内存 / 本地盘 / 分布式存储（Mooncake Store、3FS、
Redis、S3 等）放温冷数据，跨请求、跨实例复用前缀缓存以降低 TTFT。

本仓库的目标是在一台 RTX 4080（16GB）+ WSL2 的开发机上，把这条链路完整走一遍，
并留下可复现的脚本与数据。

## 实验环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX 4080 16GB (Ada, SM 8.9) |
| 内存 | 32GB |
| 系统 | Windows 11 + WSL2 Ubuntu |
| 容器 | Docker Desktop（GPU passthrough） |

## 模型矩阵

实验台是模型无关的：模型在 `experiments/models.yaml` 中注册，压测与实验脚本统一用
`--model <name>` 切换。选型原则：**每个模型代表一条不同的 KV Cache 技术路线**（侧重
国产主流），同一组卸载实验跑下来即可横向对比不同方案对存储层的影响。

| 模型 | KV 方案 | 单 token KV | @32K 上下文 | 定位 |
|------|---------|------------|------------|------|
| Qwen2.5-7B-Instruct-AWQ | GQA（4 KV heads） | 56.0 KiB | 1.75 GiB | 国产第一主流，基线 |
| GLM-4-9B-Chat（GPTQ/AWQ） | 极限 GQA（仅 2 KV heads） | 40.0 KiB | 1.25 GiB | GQA 压缩的极端案例 |
| DeepSeek-V2-Lite-Chat（4-bit） | MLA + MoE，KV 只存压缩潜在向量 | 30.4 KiB | 0.95 GiB | 国产最重要的差异化方案 |
| Llama-3.1-8B-Instruct-AWQ | GQA（8 KV heads，KV 最肥） | 128.0 KiB | 4.00 GiB | 国际对照组 |
| GPT-OSS-20B（MXFP4，可选） | 滑动窗口（128）+ attention sink | 24.0 KiB † | 0.75 GiB | 有余力再跑，不占主线 |

四个主线模型在「单 token KV 体积」上构成完整光谱，首尾相差 **4.2 倍**：

```
DeepSeek MLA 30.4 KiB  <  GLM 40.0  <  Qwen 56.0  <  Llama 128.0 KiB
```

上表数值由 `experiments/configs/` 中的 `config.json` 快照计算得出（假设 KV 为
fp16/bf16，即 2 字节/元素，vLLM 默认；开启 `--kv-cache-dtype fp8` 可再减半）。
Phase 1 的计算脚本会把这一步固化为一条可复现的命令。

† GPT-OSS 为两段式：24 层中 12 层全注意力随上下文线性增长（24.0 KiB/token），
另 12 层滑动窗口在 128 token 处封顶（合计 3 MiB 常数），因此长上下文下反而最省。

⚠️ DeepSeek-V2-Lite 的 config 同时含有 `num_key_value_heads: 16` 等 GQA 风格字段，
误用 GQA 公式会算出 270 KiB/token——是 MLA 真实值的 **8.9 倍**。KV 体积计算必须先按
`model_type` 分支选择公式，详见 [`experiments/configs/README.md`](experiments/configs/README.md)。

## 路线图

- [ ] **Phase 1 — 基线**：WSL2 中部署 vLLM 服务，编写压测脚本，建立 TTFT / 吞吐基线；
      编写「`config.json` → 单 token KV 体积」计算脚本，验证模型矩阵中的 KV 差异
- [ ] **Phase 2 — 前缀缓存**：开启 prefix caching，用多轮长文档问答负载测量 KV 复用收益
- [ ] **Phase 3 — vLLM 侧卸载**：接入 LMCache，依次实验 CPU offload → 本地磁盘 → Redis
      （Docker 容器模拟分布式存储），测量各层命中率与 TTFT 变化
- [ ] **Phase 4 — SGLang 侧卸载**：HiCache 分层缓存（L1 GPU / L2 CPU / L3 存储后端），
      与 vLLM 方案做同负载对比，分析 PagedAttention+Connector 与 RadixTree 分层的设计差异
- [ ] **Phase 5 — 自定义存储后端**：实现一个极简的 vLLM KV Connector（或 LMCache 存储
      后端），把 KV 块写入 MinIO/S3，亲手打通「KV Cache → 分布式存储」全链路

每个 Phase 独立成立，产出对应的脚本（`benchmarks/`、`experiments/`）与实验记录（`docs/`）。

## 目录结构

```
kvcache-lab/
├── benchmarks/          # 压测与测量脚本（TTFT、吞吐、缓存命中率）
├── experiments/         # 各 Phase 的部署配置与实验脚本
│   └── configs/         # 各模型 config.json 快照（仅架构参数，不含权重）
├── docs/                # 设计文档与实验记录
└── README.md
```

## License

[MIT](LICENSE)
