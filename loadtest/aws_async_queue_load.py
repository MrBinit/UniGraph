#!/usr/bin/env python3
"""Load test async /chat queue flow using YAML config.

This runner enqueues jobs through /api/v1/chat and polls /api/v1/chat/{job_id}
until completion, so SQS + worker + DynamoDB paths are exercised.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import yaml

try:
    from loadtest.common_stats import percentile
except ModuleNotFoundError:
    from common_stats import percentile


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YAML-driven async queue load tests.")
    parser.add_argument(
        "--config",
        default="loadtest/aws_full_stack_load.yaml",
        help="YAML config path.",
    )
    parser.add_argument("--token", required=True, help="Bearer token for API calls.")
    return parser.parse_args()


def _read_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config must be a YAML mapping.")
    return payload


def _cfg(payload: dict, dotted_path: str, default=None):
    node = payload
    for part in dotted_path.split("."):
        if not isinstance(node, dict):
            return default
        node = node.get(part)
    return default if node is None else node


def _http_json(
    *,
    method: str,
    url: str,
    token: str,
    timeout_seconds: float,
    payload: dict | None = None,
) -> tuple[int | None, dict, str]:
    headers = {"Authorization": f"Bearer {token}"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request = urllib.request.Request(
        url=url,
        data=body,
        method=method,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw) if raw.strip() else {}
            if not isinstance(parsed, dict):
                parsed = {}
            return int(response.getcode()), parsed, ""
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        return int(exc.code), parsed, f"http_error:{exc.code}"
    except Exception as exc:  # noqa: BLE001
        return None, {}, str(exc)


def _enqueue_one(
    *,
    base_url: str,
    token: str,
    timeout_seconds: float,
    user_id: str,
    session_id: str,
    prompt: str,
) -> dict:
    started_at = time.perf_counter()
    status_code, payload, error = _http_json(
        method="POST",
        url=f"{base_url}/api/v1/chat",
        token=token,
        timeout_seconds=timeout_seconds,
        payload={
            "user_id": user_id,
            "session_id": session_id,
            "prompt": prompt,
        },
    )
    latency_ms = (time.perf_counter() - started_at) * 1000.0
    return {
        "status_code": status_code,
        "payload": payload,
        "error": error,
        "latency_ms": latency_ms,
        "user_id": user_id,
    }


def _poll_status_once(
    *,
    base_url: str,
    token: str,
    timeout_seconds: float,
    job_id: str,
) -> tuple[str, str]:
    status_code, payload, error = _http_json(
        method="GET",
        url=f"{base_url}/api/v1/chat/{job_id}",
        token=token,
        timeout_seconds=timeout_seconds,
    )
    if status_code != 200:
        if error:
            return "poll_error", error
        return "poll_error", f"http_status:{status_code}"
    job_status = str(payload.get("status", "")).strip().lower()
    if job_status in {"queued", "processing"}:
        return job_status, ""
    if job_status in {"completed", "failed"}:
        return job_status, str(payload.get("error", ""))
    return "unknown", ""


def _load_run_settings(config: dict) -> dict:
    return {
        "base_url": str(_cfg(config, "target.base_url", "http://127.0.0.1:18000")).rstrip("/"),
        "users": int(_cfg(config, "workload.users", 100)),
        "requests_per_user": int(_cfg(config, "workload.requests_per_user", 1)),
        "concurrency": int(_cfg(config, "workload.concurrency", 25)),
        "prompt": str(_cfg(config, "workload.prompt", "Tell me about AI universities in Germany.")),
        "enqueue_timeout": float(_cfg(config, "timeouts.enqueue_seconds", 20)),
        "poll_timeout": float(_cfg(config, "timeouts.poll_seconds", 240)),
        "poll_interval": float(_cfg(config, "timeouts.poll_interval_seconds", 1)),
        "status_timeout": float(_cfg(config, "timeouts.status_request_seconds", 10)),
        "max_enqueue_error_rate": float(_cfg(config, "assertions.max_enqueue_error_rate", 0.02)),
        "max_failed_jobs_rate": float(_cfg(config, "assertions.max_failed_jobs_rate", 0.05)),
    }


def _build_work_items(users: int, requests_per_user: int) -> list[tuple[str, str]]:
    work_items: list[tuple[str, str]] = []
    for user_index in range(users):
        user_id = f"load-user-{user_index + 1:04d}"
        session_id = user_id
        for _ in range(requests_per_user):
            work_items.append((user_id, session_id))
    return work_items


def _run_enqueue_phase(
    *,
    base_url: str,
    token: str,
    timeout_seconds: float,
    prompt: str,
    concurrency: int,
    work_items: list[tuple[str, str]],
) -> tuple[list[dict], Counter[str], Counter[str], list[float], float]:
    enqueue_latencies: list[float] = []
    enqueue_status_counts: Counter[str] = Counter()
    enqueue_error_counts: Counter[str] = Counter()
    enqueue_results: list[dict] = []
    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _enqueue_one,
                base_url=base_url,
                token=token,
                timeout_seconds=timeout_seconds,
                user_id=user_id,
                session_id=session_id,
                prompt=prompt,
            )
            for user_id, session_id in work_items
        ]
        for future in futures:
            item = future.result()
            enqueue_results.append(item)
            enqueue_latencies.append(item["latency_ms"])
            status_code = item["status_code"]
            if status_code is None:
                enqueue_status_counts["connection_error"] += 1
            else:
                enqueue_status_counts[str(status_code)] += 1
            if item["error"]:
                enqueue_error_counts[item["error"]] += 1
    enqueue_duration = max(1e-9, time.perf_counter() - started_at)
    return (
        enqueue_results,
        enqueue_status_counts,
        enqueue_error_counts,
        enqueue_latencies,
        enqueue_duration,
    )


def _accepted_job_ids(enqueue_results: list[dict]) -> list[str]:
    accepted = [
        item
        for item in enqueue_results
        if item.get("status_code") == 202 and isinstance(item.get("payload"), dict)
    ]
    job_ids = [str(item["payload"].get("job_id", "")).strip() for item in accepted]
    return [job_id for job_id in job_ids if job_id]


def _poll_job_statuses(
    *,
    base_url: str,
    token: str,
    timeout_seconds: float,
    poll_timeout: float,
    poll_interval: float,
    job_ids: list[str],
) -> tuple[dict[str, str], Counter[str], Counter[str], float]:
    status_by_job: dict[str, str] = dict.fromkeys(job_ids, "queued")
    failed_job_errors: Counter[str] = Counter()
    poll_errors: Counter[str] = Counter()
    poll_started = time.perf_counter()
    deadline = poll_started + poll_timeout
    while time.perf_counter() < deadline:
        pending = [
            job_id for job_id, status in status_by_job.items() if status in {"queued", "processing"}
        ]
        if not pending:
            break
        for job_id in pending:
            status, error = _poll_status_once(
                base_url=base_url,
                token=token,
                timeout_seconds=timeout_seconds,
                job_id=job_id,
            )
            status_by_job[job_id] = status
            if status == "failed" and error:
                failed_job_errors[error] += 1
            if status == "poll_error":
                poll_errors[error or "poll_error"] += 1
        time.sleep(max(0.0, poll_interval))
    poll_duration = max(1e-9, time.perf_counter() - poll_started)
    return status_by_job, failed_job_errors, poll_errors, poll_duration


def _print_summary(summary: dict) -> None:
    print("=== Async Queue Load Summary ===")
    print(f"config:                    {summary['config_path']}")
    print(f"target_base_url:           {summary['base_url']}")
    print(f"users:                     {summary['users']}")
    print(f"requests_per_user:         {summary['requests_per_user']}")
    print(f"total_enqueue_requests:    {summary['total_requests']}")
    print(f"concurrency:               {summary['concurrency']}")
    print(f"enqueue_duration_sec:      {summary['enqueue_duration']:.2f}")
    print(
        "enqueue_rps:               "
        f"{summary['total_requests'] / summary['enqueue_duration']:.2f}"
    )
    print(f"enqueue_success:           {summary['enqueue_success']}")
    print(f"enqueue_error_count:       {summary['enqueue_error_count']}")
    print(f"enqueue_error_rate:        {summary['enqueue_error_rate']:.4f}")
    print(f"enqueue_latency_p50_ms:    {percentile(summary['enqueue_latencies'], 50):.2f}")
    print(f"enqueue_latency_p95_ms:    {percentile(summary['enqueue_latencies'], 95):.2f}")
    print(f"enqueue_latency_p99_ms:    {percentile(summary['enqueue_latencies'], 99):.2f}")
    print(f"enqueue_status_counts:     {dict(summary['enqueue_status_counts'])}")
    if summary["enqueue_error_counts"]:
        print(f"enqueue_errors:            {dict(summary['enqueue_error_counts'])}")
    print(f"poll_duration_sec:         {summary['poll_duration']:.2f}")
    print(f"jobs_total:                {summary['total_jobs']}")
    print(f"jobs_completed:            {summary['completed_jobs']}")
    print(f"jobs_failed:               {summary['failed_jobs']}")
    print(f"jobs_unresolved:           {summary['unresolved_jobs']}")
    print(f"jobs_failed_rate:          {summary['failed_jobs_rate']:.4f}")
    print(f"job_status_counts:         {dict(summary['final_status_counts'])}")
    if summary["failed_job_errors"]:
        print(f"failed_job_errors:         {dict(summary['failed_job_errors'])}")
    if summary["poll_errors"]:
        print(f"poll_errors:               {dict(summary['poll_errors'])}")


def _exit_code_for_assertions(
    *,
    enqueue_error_rate: float,
    max_enqueue_error_rate: float,
    failed_jobs_rate: float,
    max_failed_jobs_rate: float,
    unresolved_jobs: int,
) -> int:
    if enqueue_error_rate > max_enqueue_error_rate:
        return 1
    if failed_jobs_rate > max_failed_jobs_rate:
        return 1
    if unresolved_jobs > 0:
        return 1
    return 0


def main() -> int:
    args = _parse_args()
    config = _read_config(args.config)
    settings = _load_run_settings(config)
    base_url = settings["base_url"]
    users = settings["users"]
    requests_per_user = settings["requests_per_user"]
    concurrency = settings["concurrency"]
    prompt = settings["prompt"]
    enqueue_timeout = settings["enqueue_timeout"]
    poll_timeout = settings["poll_timeout"]
    poll_interval = settings["poll_interval"]
    status_timeout = settings["status_timeout"]
    max_enqueue_error_rate = settings["max_enqueue_error_rate"]
    max_failed_jobs_rate = settings["max_failed_jobs_rate"]

    if users <= 0 or requests_per_user <= 0 or concurrency <= 0:
        print("users, requests_per_user, and concurrency must be > 0", file=sys.stderr)
        return 2

    work_items = _build_work_items(users, requests_per_user)
    (
        enqueue_results,
        enqueue_status_counts,
        enqueue_error_counts,
        enqueue_latencies,
        enqueue_duration,
    ) = _run_enqueue_phase(
        base_url=base_url,
        token=args.token,
        timeout_seconds=enqueue_timeout,
        prompt=prompt,
        concurrency=concurrency,
        work_items=work_items,
    )
    job_ids = _accepted_job_ids(enqueue_results)

    total_requests = len(work_items)
    enqueue_success = len(job_ids)
    enqueue_error_count = total_requests - enqueue_success
    enqueue_error_rate = enqueue_error_count / max(1, total_requests)

    status_by_job, failed_job_errors, poll_errors, poll_duration = _poll_job_statuses(
        base_url=base_url,
        token=args.token,
        timeout_seconds=status_timeout,
        poll_timeout=poll_timeout,
        poll_interval=poll_interval,
        job_ids=job_ids,
    )
    final_status_counts: Counter[str] = Counter(status_by_job.values())
    completed_jobs = final_status_counts.get("completed", 0)
    failed_jobs = final_status_counts.get("failed", 0)
    unresolved_jobs = final_status_counts.get("queued", 0) + final_status_counts.get(
        "processing", 0
    )
    total_jobs = len(job_ids)
    failed_jobs_rate = failed_jobs / max(1, total_jobs)

    summary_payload = {
        "config_path": args.config,
        "base_url": base_url,
        "users": users,
        "requests_per_user": requests_per_user,
        "total_requests": total_requests,
        "concurrency": concurrency,
        "enqueue_duration": enqueue_duration,
        "enqueue_success": enqueue_success,
        "enqueue_error_count": enqueue_error_count,
        "enqueue_error_rate": enqueue_error_rate,
        "enqueue_latencies": enqueue_latencies,
        "enqueue_status_counts": enqueue_status_counts,
        "enqueue_error_counts": enqueue_error_counts,
        "poll_duration": poll_duration,
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "unresolved_jobs": unresolved_jobs,
        "failed_jobs_rate": failed_jobs_rate,
        "final_status_counts": final_status_counts,
        "failed_job_errors": failed_job_errors,
        "poll_errors": poll_errors,
    }
    _print_summary(summary_payload)
    return _exit_code_for_assertions(
        enqueue_error_rate=enqueue_error_rate,
        max_enqueue_error_rate=max_enqueue_error_rate,
        failed_jobs_rate=failed_jobs_rate,
        max_failed_jobs_rate=max_failed_jobs_rate,
        unresolved_jobs=unresolved_jobs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
