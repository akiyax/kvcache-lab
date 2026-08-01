#!/usr/bin/env python3
"""对 OpenAI 兼容端点做 TTFT / 吞吐压测。

零依赖（仅 stdlib）。

⚠️ **必须在 WSL 内运行**（与 vLLM 服务同侧）。从 Windows 侧压测会引入两项与
被测系统无关的干扰，实测数据见 docs/phase1-baseline.md 第 6 节：

  - 用 localhost 时先解析 IPv6 ::1，等超时回退 IPv4 约 2.1 秒
  - 经 WSL2 端口转发存在约 42 ms 的固定延迟，与 prompt 大小无关

    cd /mnt/c/.../kvcache-lab && python3 benchmarks/bench_serve.py ...

两类指标对应两个阶段的瓶颈（见 docs/notes/04-serving.md）：

    TTFT   首 token 延迟 —— 反映 prefill，受算力限制，也是前缀缓存收益的直接体现
    ITL    token 间延迟 —— 反映 decode，受显存带宽限制

负载为「长系统提示 + 长文档 + 不同问题」，素材见 benchmarks/prompts/。

前缀控制：

    默认在**最开头**插入唯一标记，前缀缓存完全不命中，测纯 prefill 基线。
    传 --shared 则所有请求共用同一段 system + doc，仅问题不同，可命中缓存。
    可变内容必须放在末尾，否则前缀从第一块起就不同，命中率归零。

    两次运行的差值即前缀缓存的收益。

用法：
    python benchmarks/bench_serve.py --model qwen -n 32 -c 8
    python benchmarks/bench_serve.py --model qwen -n 32 -c 8 --shared
"""

from __future__ import annotations

import argparse
import http.client
import json
import socket
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


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


class Corpus:
    """从 benchmarks/prompts/ 载入的压测素材。"""

    def __init__(self, directory: Path):
        def read(name: str) -> str:
            path = directory / name
            if not path.exists():
                raise FileNotFoundError(f"缺少语料文件: {path}")
            return path.read_text(encoding="utf-8").strip()

        self.system = read("system.txt")
        self.doc = read("doc-kvcache.txt")
        self.questions = [
            line.strip()
            for line in read("questions.txt").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not self.questions:
            raise ValueError("questions.txt 中没有有效问题")


def build_prompt(index: int, corpus: Corpus, doc_chars: int, shared: bool) -> str:
    """构造「长系统提示 + 文档 + 问题」的提示词。

    问题按 index 轮转，确保各请求的可变部分不同（否则整条 prompt 完全一致，
    连不该命中的基线模式也会命中）。
    """
    question = corpus.questions[index % len(corpus.questions)]
    context = corpus.doc[:doc_chars] if doc_chars else corpus.doc
    body = f"{corpus.system}\n\n{context}\n\n问题：{question}\n回答："

    if shared:
        # system + doc 对所有请求完全相同，仅末尾问题不同 —— 前缀缓存可命中
        return body

    # 唯一标记置于**最开头**，前缀从第一块起即不同，确保完全不命中
    return f"[会话 {index}-{time.time_ns()}]\n{body}"


# vLLM /metrics 中的前缀缓存计数器（累计值，压测前后取差得到本次数据）。
# external_* 对应 KV Connector / LMCache 等外部缓存层，Phase 3 起才非零。
_METRIC_KEYS = {
    "vllm:prefix_cache_queries_total": "queries",
    "vllm:prefix_cache_hits_total": "hits",
    "vllm:external_prefix_cache_queries_total": "ext_queries",
    "vllm:external_prefix_cache_hits_total": "ext_hits",
}


def read_metrics(url: str, timeout: float = 10.0) -> dict[str, float]:
    """读取前缀缓存计数器；端点不可用时返回空字典，不影响压测。"""
    parsed = urlparse(url)
    conn_cls = (http.client.HTTPSConnection if parsed.scheme == "https"
                else http.client.HTTPConnection)
    conn = conn_cls(parsed.hostname, parsed.port, timeout=timeout)
    try:
        conn.request("GET", "/metrics")
        resp = conn.getresponse()
        if resp.status != 200:
            return {}
        out: dict[str, float] = {}
        for line in resp.read().decode("utf-8").splitlines():
            if line.startswith("#") or " " not in line:
                continue
            key = _METRIC_KEYS.get(line.split("{")[0].split(" ")[0])
            if key:
                try:
                    out[key] = float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
        return out
    except Exception:
        return {}
    finally:
        conn.close()


def new_connection(url: str, timeout: float) -> http.client.HTTPConnection:
    """建立连接并关闭 Nagle 算法。

    必须设置 TCP_NODELAY：http.client 分多次写入请求头与 body，Nagle 会压住
    后续小包等待前一包的 ACK，与对端的延迟 ACK 定时器叠加，产生约 40 ms 的
    固定停顿。实测 Windows→WSL 路径上该停顿使 TTFT 恒定虚高约 42 ms，
    且不随 prompt 大小变化。
    """
    parsed = urlparse(url)
    conn_cls = (http.client.HTTPSConnection if parsed.scheme == "https"
                else http.client.HTTPConnection)
    conn = conn_cls(parsed.hostname, parsed.port, timeout=timeout)
    conn.connect()
    conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return conn


def one_request(conn: http.client.HTTPConnection, model: str, prompt: str,
                output_len: int) -> Result:
    """在已有连接上发一个流式请求，测量 TTFT 与端到端延迟。

    连接复用是必须的：新建 TCP 连接经 WSL2 端口转发约需 14 ms，
    而 WSL 内部 loopback 仅 0.07 ms。若每请求新建连接，这段开销会直接
    计入 TTFT，在缓存命中场景下甚至超过 TTFT 本身。
    """
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

        # 排空剩余字节，保持连接可复用（keep-alive）
        resp.read()
        return Result(True, ttft, time.perf_counter() - start,
                      prompt_tokens, output_tokens)
    except Exception as exc:
        return Result(False, error=f"{type(exc).__name__}: {exc}")


def run(args, corpus: Corpus) -> list[Result]:
    """以固定并发数发送 num_requests 个请求。"""
    results: list[Result] = []
    lock = threading.Lock()
    counter = threading.Lock()
    next_index = [0]

    def worker():
        conn = new_connection(args.url, args.timeout)  # 每个 worker 一条长连接
        try:
            while True:
                with counter:
                    i = next_index[0]
                    if i >= args.num_requests:
                        return
                    next_index[0] += 1
                prompt = build_prompt(i, corpus, args.doc_chars, args.shared)
                r = one_request(conn, args.model, prompt, args.output_len)
                if not r.ok:
                    # 连接状态可能已损坏，重建后继续
                    conn.close()
                    conn = new_connection(args.url, args.timeout)
                with lock:
                    results.append(r)
                    done = len(results)
                print(f"\r  {done}/{args.num_requests}", end="", flush=True)
        finally:
            conn.close()

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


def report_cache(before: dict[str, float], after: dict[str, float]) -> None:
    """打印本次压测期间的前缀缓存命中情况（取计数器差值）。"""
    if not before or not after:
        print("\n前缀缓存     metrics 端点不可用，跳过")
        return

    q = after.get("queries", 0) - before.get("queries", 0)
    h = after.get("hits", 0) - before.get("hits", 0)
    if q <= 0:
        print("\n前缀缓存     本次无查询记录")
        return

    print(f"\n前缀缓存     查询 {q:,.0f} token   命中 {h:,.0f} token"
          f"   命中率 {h / q * 100:.1f}%")

    eq = after.get("ext_queries", 0) - before.get("ext_queries", 0)
    if eq > 0:
        eh = after.get("ext_hits", 0) - before.get("ext_hits", 0)
        print(f"外部缓存层   查询 {eq:,.0f}   命中 {eh:,.0f}"
              f"   命中率 {eh / eq * 100:.1f}%")


def report(results: list[Result], wall: float, args) -> None:
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    print(f"\n{'=' * 58}")
    print(f"模型      {args.model}")
    print(f"并发      {args.concurrency}   请求数 {args.num_requests}"
          f"   成功 {len(ok)}   失败 {len(failed)}")
    if args.shared:
        print("前缀模式  共享 system + doc，仅问题不同（缓存可命中）")
    else:
        print("前缀模式  首块含唯一标记（确保完全不命中）")
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
    # 必须用 127.0.0.1 而非 localhost：Windows 上 localhost 先解析为 IPv6 ::1，
    # 而 WSL2 的端口转发只监听 IPv4，等超时回退需约 2.1 秒，会完全淹没真实 TTFT。
    p.add_argument("--url", default="http://127.0.0.1:8000", help="服务地址")
    p.add_argument("--model", required=True, help="服务端注册的模型名")
    p.add_argument("-n", "--num-requests", type=int, default=32)
    p.add_argument("-c", "--concurrency", type=int, default=8)
    # 输出长度直接决定输入输出比，而该比值强烈影响前缀缓存的收益幅度。
    # 512 接近真实问答场景；system.txt 对长度保持中立，故长度仅由本参数控制。
    p.add_argument("--output-len", type=int, default=512, help="max_tokens")
    p.add_argument("--doc-chars", type=int, default=4000,
                   help="取文档前 N 字符作为上下文（0 表示全文）")
    p.add_argument("--shared", action="store_true",
                   help="共享 system + doc 前缀，仅问题不同（测缓存命中收益）")
    p.add_argument("--prompts", type=Path, default=PROMPT_DIR, help="语料目录")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--warmup", type=int, default=1, help="正式测量前的预热请求数")
    args = p.parse_args()

    try:
        corpus = Corpus(args.prompts)
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误: {exc}")
        return 1

    if args.warmup:
        print(f"预热 {args.warmup} 个请求...")
        conn = new_connection(args.url, args.timeout)
        try:
            for i in range(args.warmup):
                # 预热一律用不命中模式，避免污染后续的共享前缀测量
                r = one_request(conn, args.model,
                                build_prompt(-i - 1, corpus, args.doc_chars, False),
                                args.output_len)
                if not r.ok:
                    print(f"预热失败: {r.error}")
                    return 1
        finally:
            conn.close()

    print(f"压测中（并发 {args.concurrency}）...")
    metrics_before = read_metrics(args.url)
    start = time.perf_counter()
    results = run(args, corpus)
    wall = time.perf_counter() - start
    metrics_after = read_metrics(args.url)

    report(results, wall, args)
    report_cache(metrics_before, metrics_after)
    return 0 if any(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
