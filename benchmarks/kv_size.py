#!/usr/bin/env python3
"""从 config.json 计算单 token 的 KV Cache 体积。

不同注意力方案的字段名与公式互不兼容，必须先按 model_type 分派：

    GQA（标准命名）  层数 × 2 × KV头数 × head_dim × 字节数
    GQA（ChatGLM）   同上，但字段名为 num_layers / multi_query_group_num / kv_channels
    MLA              层数 × (kv_lora_rank + qk_rope_head_dim) × 字节数
                     不乘 2（K/V 联合压缩），不乘头数（全头共享）
    滑动窗口          全注意力层线性增长 + 滑窗层在 window 处封顶

公式推导与已知字段陷阱见 docs/notes/03-sizing.md。

用法：
    python benchmarks/kv_size.py --all
    python benchmarks/kv_size.py qwen2.5-7b-instruct-awq -v
    python benchmarks/kv_size.py qwen2.5-7b-instruct-awq --ctx 32768 --gpu 16 --weights 5.5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "experiments" / "configs"

# README 模型矩阵的展示顺序
PREFERRED_ORDER = [
    "qwen2.5-7b-instruct-awq",
    "glm-4-9b-chat",
    "deepseek-v2-lite-chat",
    "llama-3.1-8b-instruct-awq",
    "gpt-oss-20b",
]

# KV cache 的元素字节数。注意：quantization_config 描述的是权重量化，与此无关。
DTYPE_BYTES = {
    "float32": 4, "float16": 2, "bfloat16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1, "fp8": 1,
}
DEFAULT_DTYPE_BYTES = 2


class ConfigFieldError(KeyError):
    """config.json 缺少该架构必需的字段。"""


class UnsupportedArchitecture(ValueError):
    """未知的 model_type，公式无从确定。"""


@dataclass
class KVSpec:
    name: str
    scheme: str
    bytes_per_token: int                  # 随上下文线性增长的部分
    sliding_bytes_per_token: int = 0      # 滑窗层每 token（受 window 封顶）
    window: int = 0
    detail: dict = field(default_factory=dict)

    @property
    def constant_bytes(self) -> int:
        """滑窗部分的封顶值；无滑窗时为 0。"""
        return self.sliding_bytes_per_token * self.window


def total_bytes(spec: KVSpec, n_tokens: int) -> int:
    """给定上下文长度下的 KV 总量。"""
    total = spec.bytes_per_token * n_tokens
    if spec.window:
        total += spec.sliding_bytes_per_token * min(n_tokens, spec.window)
    return total


# --------------------------------------------------------------------------- #
# 字段读取
# --------------------------------------------------------------------------- #

def _require(cfg: dict, key: str, arch: str) -> int:
    if key not in cfg:
        raise ConfigFieldError(
            f"{arch} 架构需要字段 '{key}'，但 config.json 中不存在。"
            f" 现有字段：{sorted(cfg)[:12]}..."
        )
    return cfg[key]


def _resolve_head_dim(cfg: dict, arch: str) -> tuple[int, str]:
    """优先取显式 head_dim。

    陷阱：GPT-OSS 的 head_dim=64，而 hidden_size/num_attention_heads=45，
    盲目推导会低估 42%。因此显式值必须优先。
    """
    if "head_dim" in cfg:
        return cfg["head_dim"], "显式字段"
    hidden = _require(cfg, "hidden_size", arch)
    heads = _require(cfg, "num_attention_heads", arch)
    return hidden // heads, f"推导 {hidden}/{heads}"


def _resolve_dtype(cfg: dict, override: int | None) -> tuple[int, str]:
    if override is not None:
        return override, "命令行指定"
    dtype = cfg.get("torch_dtype")
    if dtype in DTYPE_BYTES:
        return DTYPE_BYTES[dtype], f"torch_dtype={dtype}"
    if dtype is not None:
        return DEFAULT_DTYPE_BYTES, f"torch_dtype={dtype} 未知，默认 {DEFAULT_DTYPE_BYTES} 字节"
    return DEFAULT_DTYPE_BYTES, f"config 无 torch_dtype，默认 {DEFAULT_DTYPE_BYTES} 字节"


# --------------------------------------------------------------------------- #
# 各架构的公式
# --------------------------------------------------------------------------- #

def _gqa(cfg: dict, nbytes: int, dtype_src: str, *, chatglm: bool) -> KVSpec:
    arch = "GQA(ChatGLM 命名)" if chatglm else "GQA"

    if chatglm:
        layers = _require(cfg, "num_layers", arch)
        kv_heads = _require(cfg, "multi_query_group_num", arch)
        head_dim, hd_src = _require(cfg, "kv_channels", arch), "kv_channels"
    else:
        layers = _require(cfg, "num_hidden_layers", arch)
        kv_heads = _require(cfg, "num_key_value_heads", arch)
        head_dim, hd_src = _resolve_head_dim(cfg, arch)

    per_token = layers * 2 * kv_heads * head_dim * nbytes

    detail = {
        "layers": layers, "kv_heads": kv_heads,
        "head_dim": head_dim, "head_dim_source": hd_src,
        "dtype_bytes": nbytes, "dtype_source": dtype_src,
        "formula": f"{layers} × 2 × {kv_heads} × {head_dim} × {nbytes}",
    }
    # 陷阱：Qwen 有 sliding_window 字段但 use_sliding_window 为 false，按全注意力处理。
    if cfg.get("use_sliding_window"):
        detail["note"] = (
            f"该模型启用了滑动窗口（sliding_window={cfg.get('sliding_window')}），"
            "当前按全注意力计算，结果为高估上界。"
        )
    return KVSpec("", "GQA", per_token, detail=detail)


def _mla(cfg: dict, nbytes: int, dtype_src: str) -> KVSpec:
    """MLA：K/V 联合压缩为潜在向量，全头共享，故不乘 2 也不乘头数。

    陷阱：config 中的 num_key_value_heads / qk_nope_head_dim / v_head_dim
    是 GQA 风格的残留字段，MLA 计算中一律不使用。
    """
    arch = "MLA"
    layers = _require(cfg, "num_hidden_layers", arch)
    lora_rank = _require(cfg, "kv_lora_rank", arch)
    rope_dim = _require(cfg, "qk_rope_head_dim", arch)

    per_token = layers * (lora_rank + rope_dim) * nbytes

    return KVSpec("", "MLA", per_token, detail={
        "layers": layers,
        "kv_lora_rank": lora_rank,
        "qk_rope_head_dim": rope_dim,
        "dtype_bytes": nbytes, "dtype_source": dtype_src,
        "formula": f"{layers} × ({lora_rank} + {rope_dim}) × {nbytes}",
        "note": "不乘 2（K/V 联合压缩），不乘头数（潜在向量全头共享）",
    })


def _sliding_window(cfg: dict, nbytes: int, dtype_src: str) -> KVSpec:
    """滑动窗口：全注意力层线性增长，滑窗层在 window 处封顶。"""
    arch = "SWA"
    layer_types = _require(cfg, "layer_types", arch)
    kv_heads = _require(cfg, "num_key_value_heads", arch)
    window = _require(cfg, "sliding_window", arch)
    head_dim, hd_src = _resolve_head_dim(cfg, arch)

    n_full = layer_types.count("full_attention")
    n_slide = layer_types.count("sliding_attention")
    per_layer = 2 * kv_heads * head_dim * nbytes

    return KVSpec("", "SWA", n_full * per_layer,
                  sliding_bytes_per_token=n_slide * per_layer,
                  window=window,
                  detail={
                      "layers": len(layer_types),
                      "full_layers": n_full, "sliding_layers": n_slide,
                      "window": window, "kv_heads": kv_heads,
                      "head_dim": head_dim, "head_dim_source": hd_src,
                      "dtype_bytes": nbytes, "dtype_source": dtype_src,
                      "formula": (f"线性 {n_full} × 2 × {kv_heads} × {head_dim} × {nbytes}"
                                  f"  +  封顶 {n_slide} 层 × {window} token"),
                  })


# model_type → 公式。同族模型字段命名一致，一并支持。
_HANDLERS = {
    "qwen2": ("gqa", {}), "qwen3": ("gqa", {}),
    "llama": ("gqa", {}), "mistral": ("gqa", {}),
    "chatglm": ("gqa", {"chatglm": True}),
    "deepseek_v2": ("mla", {}), "deepseek_v3": ("mla", {}),
    "gpt_oss": ("swa", {}),
}


def kv_spec(cfg: dict, dtype_bytes: int | None = None) -> KVSpec:
    """计算 KV 规格。dtype_bytes 可覆盖自动判定（如 fp8 KV 传 1）。"""
    model_type = cfg.get("model_type")
    if model_type not in _HANDLERS:
        raise UnsupportedArchitecture(
            f"未知 model_type={model_type!r}。已支持：{sorted(_HANDLERS)}。"
            " 新架构需确认其 KV 存储方式后再添加公式，勿套用现有分支。"
        )

    nbytes, dtype_src = _resolve_dtype(cfg, dtype_bytes)
    kind, kwargs = _HANDLERS[model_type]

    if kind == "gqa":
        spec = _gqa(cfg, nbytes, dtype_src, chatglm=kwargs.get("chatglm", False))
    elif kind == "mla":
        spec = _mla(cfg, nbytes, dtype_src)
    else:
        spec = _sliding_window(cfg, nbytes, dtype_src)

    spec.detail["model_type"] = model_type
    return spec


def load_config(name: str) -> dict:
    """按文件名（不含扩展名）读取 experiments/configs/ 下的快照。"""
    path = CONFIG_DIR / f"{name}.json"
    if not path.exists():
        available = sorted(p.stem for p in CONFIG_DIR.glob("*.json"))
        matches = [a for a in available if a.startswith(name)]
        if len(matches) == 1:
            path = CONFIG_DIR / f"{matches[0]}.json"
        else:
            raise FileNotFoundError(f"未找到 {name}.json。可用：{available}")
    return json.loads(path.read_text(encoding="utf-8"))


def available_models() -> list[str]:
    found = {p.stem for p in CONFIG_DIR.glob("*.json")}
    ordered = [m for m in PREFERRED_ORDER if m in found]
    return ordered + sorted(found - set(ordered))


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #

def human(n: float) -> str:
    for unit, size in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= size:
            return f"{n / size:.2f} {unit}"
    return f"{n:.0f} B"


def print_table(models: list[str], dtype_bytes: int | None, contexts: list[int]) -> None:
    specs: list[tuple[str, KVSpec]] = []
    for name in models:
        try:
            specs.append((name, kv_spec(load_config(name), dtype_bytes)))
        except (ConfigFieldError, UnsupportedArchitecture) as exc:
            print(f"{name:<26} 跳过：{exc}")

    header = (f"{'模型':<24} {'架构':<4} {'单 token':>10}"
              + "".join(f"{f'@{c // 1024}K':>11}" for c in contexts))
    print(header)
    print("-" * 74)

    for name, spec in specs:
        row = f"{name:<26} {spec.scheme:<5} {human(spec.bytes_per_token):>10}"
        row += "".join(f"{human(total_bytes(spec, c)):>11}" for c in contexts)
        print(row + ("  †" if spec.window else ""))

    if any(spec.window for _, spec in specs):
        print("\n† 滑动窗口模型：单 token 列仅为线性部分，"
              "滑窗层另有封顶常数，已计入各上下文列。")


def print_detail(name: str, spec: KVSpec, ctx: int) -> None:
    print(f"模型: {name}")
    print(f"架构: {spec.scheme}  (model_type={spec.detail['model_type']})\n")

    skip = {"formula", "note", "model_type"}
    labels = {
        "layers": "层数", "kv_heads": "KV 头数", "head_dim": "head_dim",
        "head_dim_source": "  └ 来源", "dtype_bytes": "KV 字节数",
        "dtype_source": "  └ 来源", "kv_lora_rank": "kv_lora_rank",
        "qk_rope_head_dim": "qk_rope_head_dim", "full_layers": "全注意力层",
        "sliding_layers": "滑窗层", "window": "窗口大小",
    }
    for key, value in spec.detail.items():
        if key not in skip:
            print(f"  {labels.get(key, key):<18} = {value}")

    print(f"\n  {spec.detail['formula']}")
    print(f"  = {spec.bytes_per_token:,} 字节/token = {human(spec.bytes_per_token)}")
    if spec.window:
        print(f"  + 滑窗封顶常数 {human(spec.constant_bytes)}")
    if "note" in spec.detail:
        print(f"\n  注意：{spec.detail['note']}")

    print(f"\n@{ctx:,} tokens 单序列: {human(total_bytes(spec, ctx))}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="从 config.json 计算单 token KV Cache 体积",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"可用模型: {', '.join(available_models())}")
    p.add_argument("model", nargs="?", help="模型名（experiments/configs/ 下的文件名）")
    p.add_argument("--all", action="store_true", help="列出全部模型对照表")
    p.add_argument("--ctx", type=int, default=32768, help="上下文长度（默认 32768）")
    p.add_argument("--kv-dtype", choices=["fp16", "fp8"],
                   help="覆盖 KV 精度（fp8 即每元素 1 字节）")
    p.add_argument("-v", "--verbose", action="store_true", help="显示推导过程")
    p.add_argument("--gpu", type=float, help="显存容量 GiB，用于估算可并发路数")
    p.add_argument("--weights", type=float, default=0.0, help="模型权重占用 GiB")
    p.add_argument("--util", type=float, default=0.9,
                   help="gpu-memory-utilization（默认 0.9）")
    args = p.parse_args(argv)

    dtype_bytes = {"fp16": 2, "fp8": 1}.get(args.kv_dtype)

    if args.all or not args.model:
        print_table(available_models(), dtype_bytes, [8192, 32768, 131072])
        return 0

    try:
        cfg = load_config(args.model)
        spec = kv_spec(cfg, dtype_bytes)
    except (FileNotFoundError, ConfigFieldError, UnsupportedArchitecture) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    if args.verbose:
        print_detail(args.model, spec, args.ctx)
    else:
        print(f"{args.model}  [{spec.scheme}]  {human(spec.bytes_per_token)}/token"
              f"  @{args.ctx:,} = {human(total_bytes(spec, args.ctx))}")

    if args.gpu:
        budget = args.gpu * args.util - args.weights
        per_req = total_bytes(spec, args.ctx) / (1 << 30)
        print(f"\n显存预算: {args.gpu} GiB × {args.util} - 权重 {args.weights} GiB"
              f" = {budget:.2f} GiB 可用于 KV")
        print(f"单请求 @{args.ctx:,} tokens: {per_req:.2f} GiB")
        if budget <= 0:
            print("→ 权重已超出预算，无法容纳 KV")
        else:
            print(f"→ 可并发约 {int(budget / per_req)} 路")

    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
