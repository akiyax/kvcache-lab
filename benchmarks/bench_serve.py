#!/usr/bin/env python3
"""对 OpenAI 兼容端点做 TTFT / 吞吐压测。

零依赖（仅 stdlib），可从 Windows 直接压 WSL 内的服务。

两类指标对应两个阶段的瓶颈（见 docs/notes/04-serving.md）：

    TTFT   首 token 延迟 —— 反映 prefill，受算力限制，也是前缀缓存收益的直接体现
    ITL    token 间延迟 —— 反映 decode，受显存带宽限制

前缀控制（Phase 2 会用到）：

    默认每个请求的**开头**插入唯一标记，确保前缀缓存完全不命中，用于测干净基线。
    传 --shared-prefix-len 则构造共享前缀 + 各自不同的问题，用于测缓存收益。
    可变内容必须放在最后，否则前缀从第一块起就不同，命中率归零。

用法：
    python benchmarks/bench_serve.py --model qwen --num-requests 32 --concurrency 8
    python benchmarks/bench_serve.py --model qwen --shared-prefix-len 2000 -n 16 -c 4
"""

from __future__ import annotations

import argparse
import http.client
import json
import statistics
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlparse

FILLER = (
    "推理系统需要在有限的显存中同时服务多个请求，缓存的管理策略直接决定了整体吞吐。"
    "分层存储将热数据保留在高速介质中，冷数据下沉到容量更大但延迟更高的层级。"
)


@dataclass
class Result:
    ok: bool
    ttft: float = 0.0            # 秒
    latency: float = 0.0         # 秒，端到端
    prompt_tokens: int = 0
    output_tokens: int = 0
    error: str = ""

    @property
    def itl(self) -> float:
        """token 间平均延迟（毫秒）。"""
        if self.output_tokens < 2:
            return 0.0
        return (self.latency - self.ttft) / (self.output_tokens - 1) * 1000


def build_prompt(index: int, input_len: int, shared_prefix_len: int) -> str:
    """构造指定长度的提示词。

    input_len 为字符数近似值——真实 token 数以服务端返回的 usage 为准，
    因此近似不影响测量准确性。
    """
    if shared_prefix_len:
        # 共享前缀在前，可变部分在后 —— 前缀缓存可命中
        prefix = (FILLER * (shared_prefix_len // len(FILLER) + 1))[:shared_prefix_len]
        return f"{prefix}\n\n问题 {index}：请概括上文要点。"

    # 唯一标记置于**开头**，确保前缀缓存完全不命中（干净基线）
    unique = f"[请求 {index} 会话 {time.time_ns()}] "
    body = (FILLER * (input_len // len(FILLER) + 1))[:max(0, input_len - len(unique))]
    return unique + body + "\n\n请简要回答上述内容涉及的主题。"


def one_request(url: str, model: str, prompt: str, output_len: int,
                timeout: float) -> Result:
    """发一个流式请求，测量 TTFT 与端到端延迟。"""
    parsed = urlparse(url)
    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parsed.hostname, parsed.port, timeout=timeout)

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    })

    start = time.perf_counter()
    ttft = 0.0
    prompt_tokens = output_tokens = 0

    try:
        conn.request("POST", "/v1/completions", body=payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            return Result(False, error=f"HTTP {resp.status}: {resp.read()[:200]!r}")

        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break

            chunk = json.loads(data)
            # usage 只在最后一个 chunk 中出现（include_usage）
            if chunk.get("usage"):
                prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                output_tokens = chunk["usage"].get("completion_tokens", 0)
            if not ttft and chunk.get("choices") and chunk["choices"][0].get("text"):
                ttft = time.perf_counter() - start

        return Result(True, ttft, time.perf_counter() - start,
                      prompt_tokens, output_tokens)
    except Exception as exc:
        return Result(False, error=f"{type(exc).__name__}: {exc}")
    finally:
        conn.close()


def run(args) -> list[Result]:
    """以固定并发数发送 num_requests 个请求。"""
    results: list[Result] = []
    lock = threading.Lock()
    counter = threading.Lock()
    next_index = [0]

    def worker():
        while True:
            with counter:
                i = next_index[0]
                if i >= args.num_requests:
                    return
                next_index[0] += 1
            prompt = build_prompt(i, args.input_len, args.shared_prefix_len)
            r = one_request(args.url, args.model, prompt, args.output_len, args.timeout)
            with lock:
                results.append(r)
                done = len(results)
            print(f"\r  {done}/{args.num_requests}", end="", flush=True)

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print()
    return results


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * p), len(ordered) - 1)]


def report(results: list[Result], wall: float, args) -> None:
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    print(f"\n{'=' * 58}")
    print(f"模型      {args.model}")
    print(f"并发      {args.concurrency}   请求数 {args.num_requests}"
          f"   成功 {len(ok)}   失败 {len(failed)}")
    if args.shared_prefix_len:
        print(f"共享前缀  {args.shared_prefix_len} 字符（前缀缓存可命中）")
    else:
        print("共享前缀  无（每请求首块唯一，确保不命中）")
    print("=" * 58)

    if not ok:
        for r in failed[:3]:
            print(f"  错误: {r.error}")
        return

    ttfts = [r.ttft * 1000 for r in ok]
    itls = [r.itl for r in ok if r.itl]
    total_out = sum(r.output_tokens for r in ok)
    total_in = sum(r.prompt_tokens for r in ok)

    print(f"\nTTFT (ms)      p50 {pct(ttfts, .5):8.1f}   p90 {pct(ttfts, .9):8.1f}"
          f"   p99 {pct(ttfts, .99):8.1f}   均值 {statistics.mean(ttfts):8.1f}")
    if itls:
        print(f"ITL  (ms)      p50 {pct(itls, .5):8.2f}   p90 {pct(itls, .9):8.2f}"
              f"   p99 {pct(itls, .99):8.2f}   均值 {statistics.mean(itls):8.2f}")

    print(f"\n输入 token     {total_in:,}  (均 {total_in // len(ok):,}/请求)")
    print(f"输出 token     {total_out:,}  (均 {total_out // len(ok):,}/请求)")
    print(f"\n吞吐           {total_out / wall:8.1f} output tok/s"
          f"   {len(ok) / wall:6.2f} req/s")
    print(f"耗时           {wall:.2f} s")

    if failed:
        print(f"\n失败样例: {failed[0].error}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="TTFT / 吞吐压测（OpenAI 兼容端点）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000", help="服务地址")
    p.add_argument("--model", required=True, help="服务端注册的模型名")
    p.add_argument("-n", "--num-requests", type=int, default=32)
    p.add_argument("-c", "--concurrency", type=int, default=8)
    p.add_argument("--input-len", type=int, default=1024, help="提示词字符数（近似）")
    p.add_argument("--output-len", type=int, default=128, help="max_tokens")
    p.add_argument("--shared-prefix-len", type=int, default=0,
                   help="共享前缀字符数；>0 时构造可命中前缀缓存的负载")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--warmup", type=int, default=1, help="正式测量前的预热请求数")
    args = p.parse_args()

    if args.warmup:
        print(f"预热 {args.warmup} 个请求...")
        for i in range(args.warmup):
            r = one_request(args.url, args.model,
                            build_prompt(-i - 1, args.input_len, 0),
                            args.output_len, args.timeout)
            if not r.ok:
                print(f"预热失败: {r.error}")
                return 1

    print(f"压测中（并发 {args.concurrency}）...")
    start = time.perf_counter()
    results = run(args)
    report(results, time.perf_counter() - start, args)
    return 0 if any(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
