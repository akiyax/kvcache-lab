#!/usr/bin/env bash
#
# Phase 3 卸载实验的服务端。
#
#   LMCACHE=off bash experiments/serve-phase3.sh    # 对照组
#   LMCACHE=cpu bash experiments/serve-phase3.sh    # CPU 内存层
#   LMCACHE=disk bash experiments/serve-phase3.sh   # 本地磁盘层
#   LMCACHE=redis bash experiments/serve-phase3.sh  # Redis 层
#
# ── 为什么要人为把 GPU 缓存卡小 ──────────────────────────────────
# 卸载只在「前缀会被复用 **且** 总量超过 GPU 容量」时才有收益。默认配置下
# GPU KV 有 11 万 token，而单个会话前缀仅约 2,900 token，永远装得下，
# 卸载层的命中率恒为零。故用 --kv-cache-memory 显式压到 0.5 GiB
# （9,360 token ≈ 3.2 个会话），配合 bench_serve.py --sessions 12
# 制造约 4 倍超配，保证轮转一圈后必被驱逐。
#
# --max-model-len 也必须同步下调：vLLM 要求 KV 至少装得下一个满长度请求，
# 32768 需 1.75 GiB，会直接拒绝启动。4096 覆盖「2,933 输入 + 512 输出」。
#
# ⚠ 因此本阶段的绝对性能数字与 Phase 1/2 基线不可直接比较，
#   有意义的只有本阶段内部 off / cpu / disk / redis 四组之间的对照。
set -uo pipefail
source "$HOME/venvs/vllm/bin/activate"

# —— Phase 1 记录的 WSL 必需项，详见 docs/phase1-baseline.md 第 2 节 ——
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export CUDA_HOME="$HOME/venvs/vllm/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$HOME/venvs/vllm/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL="${MODEL:-$HOME/models/Qwen2.5-7B-Instruct-AWQ}"
KV_BYTES="${KV_BYTES:-536870912}"        # 0.5 GiB ≈ 9,360 token（Qwen 57,344 B/token）

ARGS=(
  "$MODEL"
  --served-model-name "${SERVED_NAME:-qwen2.5-7b-awq}"
  --max-model-len 4096
  --gpu-memory-utilization 0.80
  --kv-cache-memory "$KV_BYTES"
  --port 8000
)

TIER="${LMCACHE:-off}"

if [ "$TIER" != "off" ]; then
  # Python 内置 hash() 对字符串每进程随机加种。单机自存自取无碍，但跨进程/跨机器
  # 共享时同样内容会算出不同的键，命中率**静默归零**——不报错、不崩溃，只是全部
  # 走重算。Redis 与 MinIO 层必须固定种子。详见 docs/phase3-offload.md 第 5 节。
  #
  # 注：CPU 层的已记录结果是在加入本行**之前**测得的。它不改变单进程内的命中行为，
  # 故不影响那批数据的可复现性。
  export PYTHONHASHSEED=0

  export LMCACHE_CHUNK_SIZE=256          # 每块 256 token ≈ 14.7 MiB（Qwen）
  export LMCACHE_LOCAL_CPU=True
  export LMCACHE_MAX_LOCAL_CPU_SIZE=4.0  # GB；WSL 共 15 GiB，留足余量

  case "$TIER" in
    cpu)   ;;                            # 仅 CPU 内存
    disk)
      export LMCACHE_LOCAL_DISK="file:///$HOME/lmcache-disk/"
      export LMCACHE_MAX_LOCAL_DISK_SIZE=8.0
      mkdir -p "$HOME/lmcache-disk"
      ;;
    redis)
      export LMCACHE_REMOTE_URL="${REDIS_URL:-redis://localhost:6379}"
      export LMCACHE_REMOTE_SERDE=naive
      ;;
    *) echo "未知的 LMCACHE 取值: $TIER" >&2; exit 2 ;;
  esac

  ARGS+=(--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}')
fi

echo ">>> 卸载层: $TIER"
exec vllm serve "${ARGS[@]}"
