#!/usr/bin/env python3
"""Simple local HTTP load generator without external dependencies."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    fraction = rank - lower
    return lower_value + (upper_value - lower_value) * fraction


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight local load tests.")
    parser.add_argument("--url", required=True, help="Target URL.")
    parser.add_argument("--requests", type=int, default=200, help="Total requests to send.")
    parser.add_argument("--concurrency", type=int, default=20, help="Concurrent workers.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-request timeout seconds.")
    parser.add_argument("--token", default="", help="Bearer token (optional).")
    parser.add_argument(
        "--mode",
        choices=["healthz", "chat-stream"],
        default="chat-stream",
        help="Predefined request payload mode.",
    )
    parser.add_argument(
        "--prompt",
        default="find ai labs in germany",
        help="Prompt used in chat-stream mode.",
    )
    parser.add_argument(
        "--user-id",
        default="load-user",
        help="user_id used in chat-stream mode.",
    )
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=0.05,
        help="Exit non-zero if error rate exceeds this threshold.",
    )
    return parser.parse_args()


def _request_once(args: argparse.Namespace, request_index: int) -> tuple[int | None, float, str]:
    headers: dict[str, str] = {}
    body: bytes | None = None
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    if args.mode == "chat-stream":
        headers["Content-Type"] = "application/json"
        payload = {
            "user_id": args.user_id,
            "session_id": f"load-session-{request_index % max(1, args.concurrency)}",
            "prompt": args.prompt,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    started_at = time.perf_counter()
    try:
        request = urllib.request.Request(
            args.url,
            data=body,
            headers=headers,
            method="POST" if body is not None else "GET",
        )
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            _ = response.read()
            latency_ms = (time.perf_counter() - started_at) * 1000
            return int(response.getcode()), latency_ms, ""
    except urllib.error.HTTPError as exc:
        latency_ms = (time.perf_counter() - started_at) * 1000
        return int(exc.code), latency_ms, f"http_error:{exc.code}"
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - started_at) * 1000
        return None, latency_ms, str(exc)


def main() -> int:
    args = _parse_args()
    if args.requests <= 0:
        print("requests must be > 0", file=sys.stderr)
        return 2
    if args.concurrency <= 0:
        print("concurrency must be > 0", file=sys.stderr)
        return 2

    latencies: list[float] = []
    status_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()

    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(_request_once, args, i) for i in range(args.requests)]
        for future in futures:
            status_code, latency_ms, error = future.result()
            latencies.append(latency_ms)
            if status_code is None:
                status_counts["connection_error"] += 1
            else:
                status_counts[str(status_code)] += 1
            if error:
                error_counts[error] += 1
    total_seconds = max(1e-9, time.perf_counter() - started_at)

    success_count = sum(
        count for code, count in status_counts.items() if code.isdigit() and code.startswith("2")
    )
    error_count = args.requests - success_count
    error_rate = error_count / args.requests

    print("=== Load Test Summary ===")
    print(f"url:              {args.url}")
    print(f"mode:             {args.mode}")
    print(f"requests:         {args.requests}")
    print(f"concurrency:      {args.concurrency}")
    print(f"duration_sec:     {total_seconds:.2f}")
    print(f"throughput_rps:   {args.requests / total_seconds:.2f}")
    print(f"success_count:    {success_count}")
    print(f"error_count:      {error_count}")
    print(f"error_rate:       {error_rate:.4f}")
    print(f"latency_p50_ms:   {_percentile(latencies, 50):.2f}")
    print(f"latency_p95_ms:   {_percentile(latencies, 95):.2f}")
    print(f"latency_p99_ms:   {_percentile(latencies, 99):.2f}")
    print(f"latency_max_ms:   {max(latencies) if latencies else 0.0:.2f}")
    print(f"status_counts:    {dict(status_counts)}")
    if error_counts:
        print(f"errors:           {dict(error_counts)}")

    if error_rate > args.max_error_rate:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
