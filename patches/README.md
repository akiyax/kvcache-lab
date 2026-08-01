# 补丁

实验过程中需要打在依赖上的补丁。每个补丁必须记录**为什么需要**、**上游状态**、
以及**去掉它会怎样**——否则后来者无法判断该不该继续带着它。

## `vllm-0.26.0-mla-quantized-dtype.patch`

| | |
|---|---|
| 影响版本 | vLLM 0.26.0（更早版本同样存在，至少可追溯到 0.24.0） |
| 触发条件 | MLA 模型 + pack 量化（AWQ / GPTQ）+ 前缀缓存命中或分块 prefill |
| 症状 | `AttributeError: 'ColumnParallelLinear' object has no attribute 'weight'` |
| 上游状态 | 已有多个 PR，**均未合并**（见下） |

### 症状

首个请求正常返回，**第二个共享前缀的请求直接打死 EngineCore**，此后所有请求
返回 HTTP 500。短 prompt 的冒烟测试发现不了——16 token 的请求一次 prefill 完成，
走不到出问题的代码路径。

### 成因

`_compute_prefill_context()` 里已经为量化层写好了防御逻辑：

```python
# For quantized layers (AWQ/GPTQ) that lack a .weight attribute,
# use params_dtype which is the expected input dtype.
_kv_b_proj_w_dtype = (
    self.kv_b_proj.weight.dtype
    if hasattr(self.kv_b_proj, "weight")
    else self.kv_b_proj.params_dtype
)
```

但紧接着的类型转换那一行忘了用这个变量，仍然直接取 `self.kv_b_proj.weight.dtype`。
AWQ/GPTQ 把权重打包成 `.qweight`，`.weight` 属性不存在，于是抛异常。

补丁只改这一行。下一行的 `self.kv_b_proj(kv_c_normed)` 本身对量化层工作正常，
**所以 MLA 与量化并非根本不兼容**——这纯粹是个遗漏。

### 为什么上游没修

不是没人发现。截至 2026-08，至少有 6 个 issue/PR 指向同一处：

| PR | 状态 |
|---|---|
| [#43889](https://github.com/vllm-project/vllm/pull/43889) | open |
| [#46218](https://github.com/vllm-project/vllm/pull/46218) | open |
| [#47564](https://github.com/vllm-project/vllm/pull/47564) | open |
| [#47795](https://github.com/vllm-project/vllm/pull/47795) | closed |
| [#43264](https://github.com/vllm-project/vllm/pull/43264) | closed |
| [#35576](https://github.com/vllm-project/vllm/pull/35576) | closed |

真实原因是**这个组合在 CI 里没有覆盖**：生产环境的 MLA 模型（DeepSeek-V2/V3/R1）
都在 H100/H200 上跑 BF16 或 FP8，而 **FP8 层是有 `.weight` 属性的**，官方测试
永远碰不到这条路径。只有「消费级显卡 + 4-bit 量化 + MLA」才会撞上。

### 应用方式

```bash
cd "$HOME/venvs/vllm/lib/python3.12/site-packages"
patch -p1 < /path/to/kvcache-lab/patches/vllm-0.26.0-mla-quantized-dtype.patch
```

### 对本项目的意义

这不是一个可以绕过的边角问题。[#47564](https://github.com/vllm-project/vllm/pull/47564)
的评论区里，有人用 `GLM-4.7-Flash-AWQ-4bit`（同样是 MLA）**配合 `LMCacheConnectorV1`**
复现了它——那正是 Phase 3 要用的组件。任何经 KV connector 取回前缀的请求都会走
`_compute_prefill_context`，也就是说：**不打这个补丁，Phase 3 的 MLA + LMCache
实验根本跑不起来。**
