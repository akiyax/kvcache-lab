# 模型 config.json 快照

这里存放模型矩阵中各模型的 `config.json` 快照（仅几 KB 的架构描述文件，**不含权重**）。

## 为什么要存进仓库

- 单 token KV 体积完全由 `config.json` 里的架构参数决定，算 KV 不需要下载权重。
- 快照进 git 后，KV 计算脚本离线可跑，任何人 clone 仓库都能复现出同样的数字。
- 模型仓库的参数会随版本变动，快照锁定了「当初算的是哪一版」。

## 来源

抓取日期：2026-08-02，均取自各仓库 `main` 分支（国内经 `hf-mirror.com` 镜像）。

| 文件 | HuggingFace 仓库 | commit |
|------|------------------|--------|
| `qwen2.5-7b-instruct-awq.json` | `Qwen/Qwen2.5-7B-Instruct-AWQ` | `b25037543e93` |
| `glm-4-9b-chat.json` | `zai-org/glm-4-9b-chat` | `bd8234fe5e0c` |
| `deepseek-v2-lite-chat.json` | `deepseek-ai/DeepSeek-V2-Lite-Chat` | `85864749cd61` |
| `llama-3.1-8b-instruct-awq.json` | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | `db1f81ad4b8c` |
| `gpt-oss-20b.json` | `openai/gpt-oss-20b` | `6cee5e81ee83` |

说明：

- **GLM-4-9B 取的是未量化版本**。量化只压缩权重，不改变 KV cache 的形状，因此用于 KV 计算是等价的；实际部署时需另选 GPTQ/AWQ 仓库。原仓库 `THUDM/glm-4-9b-chat` 已更名为 `zai-org/glm-4-9b-chat`，旧路径仍可重定向。
- **Llama-3.1 官方仓库为 gated**，故采用社区 AWQ 量化版 `hugging-quants/...`，架构参数与官方一致。

## 重新抓取

```bash
# 直连（境外网络）
BASE=https://huggingface.co
# 或走镜像（国内）
BASE=https://hf-mirror.com

curl -L -o qwen2.5-7b-instruct-awq.json  $BASE/Qwen/Qwen2.5-7B-Instruct-AWQ/resolve/main/config.json
curl -L -o glm-4-9b-chat.json            $BASE/zai-org/glm-4-9b-chat/resolve/main/config.json
curl -L -o deepseek-v2-lite-chat.json    $BASE/deepseek-ai/DeepSeek-V2-Lite-Chat/resolve/main/config.json
curl -L -o llama-3.1-8b-instruct-awq.json $BASE/hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4/resolve/main/config.json
curl -L -o gpt-oss-20b.json              $BASE/openai/gpt-oss-20b/resolve/main/config.json
```

## 字段差异提示

不同注意力方案的字段名与计算公式都不同，KV 计算脚本必须先看 `model_type` 再决定公式：

| 架构 | `model_type` | KV 相关字段 |
|------|--------------|-------------|
| GQA | `qwen2` / `llama` | `num_hidden_layers`、`num_key_value_heads`、`hidden_size` / `num_attention_heads` |
| GQA（ChatGLM 命名） | `chatglm` | `num_layers`、`multi_query_group_num`、`kv_channels` |
| MLA | `deepseek_v2` | `num_hidden_layers`、`kv_lora_rank`、`qk_rope_head_dim` |
| 滑动窗口 | `gpt_oss` | `layer_types`、`sliding_window`、`num_key_value_heads`、`head_dim` |

⚠️ **DeepSeek-V2-Lite 的坑**：它的 config 同时含有 `num_key_value_heads: 16`、`qk_nope_head_dim`、`v_head_dim` 等 GQA 风格字段。若误用 GQA 公式会算出约 270 KiB/token，是 MLA 真实值的 8.9 倍。MLA 只缓存压缩后的潜在向量，公式为 `层数 × (kv_lora_rank + qk_rope_head_dim) × 字节数`，既不乘 2 也不乘头数。
