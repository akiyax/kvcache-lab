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
| 模型 | Qwen2.5-7B-Instruct-AWQ（约 5.5GB 显存，给 KV Cache 留足空间） |

## 路线图

- [ ] **Phase 1 — 基线**：WSL2 中部署 vLLM 服务，编写压测脚本，建立 TTFT / 吞吐基线
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
├── benchmarks/    # 压测与测量脚本（TTFT、吞吐、缓存命中率）
├── experiments/   # 各 Phase 的部署配置与实验脚本
├── docs/          # 设计文档与实验记录
└── README.md
```

## License

[MIT](LICENSE)
