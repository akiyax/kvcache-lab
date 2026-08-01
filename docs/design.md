# 设计文档：KV Cache 分层卸载实验台

日期：2026-08-01

## 背景与目标

作者即将从事「KV Cache 卸载到分布式存储」方向的推理系统工作，需要通过动手项目
建立对 vLLM 与 SGLang 两个主流推理框架 KV 缓存体系的工程理解。

目标：在单机（RTX 4080 16GB + WSL2 Ubuntu + Docker）上复现生产环境的 KV Cache
分层结构，量化每一层卸载的收益与开销，最终亲手实现一个对接对象存储的极简后端。

非目标（YAGNI）：

- 不做多卡 / 多节点部署（硬件不支持）
- 不做 RDMA / NIXL 传输层实验（家用网络环境无意义）
- 不追求生产级代码质量，实验脚本以可复现为准

## 总体架构

```
                    ┌────────────────────────────┐
  压测客户端 ──────▶│  vLLM / SGLang (WSL2)      │
  benchmarks/       │  Qwen2.5-7B-Instruct-AWQ   │
                    └──────────┬─────────────────┘
                               │ KV Cache 分层
             ┌─────────────────┼──────────────────┐
             ▼                 ▼                  ▼
        GPU 显存 (L1)     CPU 内存 (L2)      存储后端 (L3)
        PagedAttention/   LMCache CPU /      本地磁盘 / Redis /
        RadixCache        HiCache host mem   MinIO (Docker 容器)
```

## 实验负载

统一使用「多轮长文档问答」负载模拟真实场景：长系统提示 + 文档上下文（共享前缀）
+ 多个不同问题。这种负载能体现前缀缓存 / KV 复用的收益，也是 KV 卸载在生产中的
典型受益场景。

核心指标：

- **TTFT**（Time To First Token）：前缀缓存命中与否的最直接体现
- **吞吐**（tokens/s，requests/s）
- **缓存命中率**（框架侧 metrics）
- **显存 / 内存占用**

## 分阶段计划

| Phase | 内容 | 关键产出 |
|-------|------|----------|
| 1 | vLLM 基线部署 + 压测脚本 | `benchmarks/` 压测工具，基线数据 |
| 2 | prefix caching 实验 | KV 复用收益数据与分析 |
| 3 | vLLM + LMCache：CPU → 磁盘 → Redis | 各层 TTFT / 命中率对比 |
| 4 | SGLang HiCache：L2 CPU / L3 文件后端 | 与 Phase 3 的同负载横向对比 |
| 5 | 自定义 KV Connector / 存储后端 → MinIO | 可运行的最小实现 + 代码解读笔记 |

每个 Phase 在 `experiments/` 下有独立目录（部署配置 + 运行脚本），实验记录写入
`docs/`。Phase 之间只有弱依赖（复用压测脚本），任一 Phase 完成即可独立成文。

## 技术选型说明

- **模型**：Qwen2.5-7B-Instruct-AWQ。4-bit 量化后约 5.5GB 权重，16GB 显存下可留
  约 9GB 给 KV Cache，足以制造「缓存放不下需要卸载」的实验条件（配合调低
  `gpu-memory-utilization` 可进一步压缩 L1，放大卸载效果）。
- **分布式存储替身**：Redis 与 MinIO 以 Docker 容器运行。单机容器无法体现网络
  延迟的真实量级，但接口语义（远端 KV 存取、序列化、命中判断）与生产一致。
- **运行环境**：一律在 WSL2 Ubuntu 内运行（vLLM / SGLang 不支持 Windows 原生）。

## 风险与备选

- LMCache / HiCache 版本迭代快，文档滞后 → 以框架源码与官方示例为准，实验前
  固定版本号并记录在各实验目录。
- 7B AWQ 若在某些组合下不受支持 → 备选 Qwen2.5-3B-Instruct（FP16 约 6GB）。
- WSL2 内存上限影响 CPU offload 实验 → 通过 `.wslconfig` 调大 WSL 可用内存。
