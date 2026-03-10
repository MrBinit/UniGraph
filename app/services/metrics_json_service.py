import json
import threading
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.core.paths import resolve_project_path
from app.services.metrics_dynamodb_service import persist_chat_metrics_dynamodb

settings = get_settings()

_THREAD_LOCK = threading.Lock()

try:  # pragma: no cover
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


def _metrics_dir() -> Path:
    path = resolve_project_path(settings.app.metrics_json_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _requests_jsonl_path() -> Path:
    return _metrics_dir() / "chat_metrics_requests.jsonl"


def _aggregate_json_path() -> Path:
    return _metrics_dir() / "chat_metrics_aggregate.json"


def _aggregate_lock_path() -> Path:
    return _metrics_dir() / ".chat_metrics.lock"


@contextmanager
def _aggregate_lock():
    if fcntl is None:
        with _THREAD_LOCK:
            yield
        return

    lock_path = _aggregate_lock_path()
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _series_template() -> dict:
    return {
        "count": 0,
        "total": 0.0,
        "average": 0.0,
        "max": 0.0,
        "p95": 0.0,
        "p99": 0.0,
    }


def _default_aggregate() -> dict:
    return {
        "version": 1,
        "updated_at": _now_iso(),
        "total_requests": 0,
        "outcomes": {},
        "latency_ms": {
            "overall": _series_template(),
            "llm_response": _series_template(),
            "short_term_memory": _series_template(),
            "long_term_memory": _series_template(),
            "memory_update": _series_template(),
            "cache_read": _series_template(),
            "cache_write": _series_template(),
            "evaluation_trace": _series_template(),
        },
        "token_usage": {
            "requests_with_usage": 0,
            "prompt_tokens_total": 0,
            "completion_tokens_total": 0,
            "total_tokens_total": 0,
            "prompt_tokens_average": 0.0,
            "completion_tokens_average": 0.0,
            "total_tokens_average": 0.0,
        },
        "latest_request": {},
    }


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _update_series(series: dict, value) -> None:
    numeric = _to_float(value)
    if numeric is None:
        return
    series["count"] = int(series.get("count", 0)) + 1
    series["total"] = float(series.get("total", 0.0)) + numeric
    series["average"] = round(series["total"] / max(1, series["count"]), 3)
    series["max"] = round(max(float(series.get("max", 0.0)), numeric), 3)


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil((p / 100) * len(ordered)))
    return float(ordered[rank - 1])


def _refresh_latency_percentiles(aggregate: dict) -> None:
    series_to_timing_key = {
        "overall": "overall_response_ms",
        "llm_response": "llm_response_ms",
        "short_term_memory": "short_term_memory_ms",
        "long_term_memory": "long_term_memory_ms",
        "memory_update": "memory_update_ms",
        "cache_read": "cache_read_ms",
        "cache_write": "cache_write_ms",
        "evaluation_trace": "evaluation_trace_ms",
    }

    requests_path = _requests_jsonl_path()
    samples: dict[str, list[float]] = {series: [] for series in series_to_timing_key}
    if requests_path.exists():
        with open(requests_path, "r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    request = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                timings = request.get("timings_ms", {})
                if not isinstance(timings, dict):
                    continue
                for series_name, timing_key in series_to_timing_key.items():
                    value = _to_float(timings.get(timing_key))
                    if value is not None:
                        samples[series_name].append(value)

    latency = aggregate.setdefault("latency_ms", {})
    for series_name, series_samples in samples.items():
        series = latency.setdefault(series_name, _series_template())
        series["p95"] = round(_percentile(series_samples, 95), 3)
        series["p99"] = round(_percentile(series_samples, 99), 3)


def _load_aggregate(path: Path) -> dict:
    if not path.exists():
        return _default_aggregate()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _default_aggregate()
    return raw if isinstance(raw, dict) else _default_aggregate()


def _save_aggregate(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _normalize_request_record(record: dict) -> dict:
    payload = dict(record) if isinstance(record, dict) else {}
    payload["timestamp"] = payload.get("timestamp") or _now_iso()
    payload["session_id"] = payload.get("session_id") or payload.get("user_id", "")
    return payload


def _append_request_record(record: dict) -> None:
    with open(_requests_jsonl_path(), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _update_aggregate_payload(aggregate: dict, record: dict) -> dict:
    aggregate["updated_at"] = _now_iso()
    aggregate["total_requests"] = int(aggregate.get("total_requests", 0)) + 1

    outcomes = aggregate.setdefault("outcomes", {})
    outcome = str(record.get("outcome", "unknown"))
    outcomes[outcome] = int(outcomes.get(outcome, 0)) + 1

    timings = record.get("timings_ms", {})
    if not isinstance(timings, dict):
        timings = {}
    latency = aggregate.setdefault("latency_ms", {})
    _update_series(
        latency.setdefault("overall", _series_template()), timings.get("overall_response_ms")
    )
    _update_series(
        latency.setdefault("llm_response", _series_template()), timings.get("llm_response_ms")
    )
    _update_series(
        latency.setdefault("short_term_memory", _series_template()),
        timings.get("short_term_memory_ms"),
    )
    _update_series(
        latency.setdefault("long_term_memory", _series_template()),
        timings.get("long_term_memory_ms"),
    )
    _update_series(
        latency.setdefault("memory_update", _series_template()), timings.get("memory_update_ms")
    )
    _update_series(
        latency.setdefault("cache_read", _series_template()), timings.get("cache_read_ms")
    )
    _update_series(
        latency.setdefault("cache_write", _series_template()), timings.get("cache_write_ms")
    )
    _update_series(
        latency.setdefault("evaluation_trace", _series_template()),
        timings.get("evaluation_trace_ms"),
    )

    usage = record.get("llm_usage", {})
    if not isinstance(usage, dict):
        usage = {}
    prompt_tokens = _to_int(usage.get("prompt_tokens"))
    completion_tokens = _to_int(usage.get("completion_tokens"))
    total_tokens = _to_int(usage.get("total_tokens"))
    usage_summary = aggregate.setdefault("token_usage", {})
    if prompt_tokens is not None and completion_tokens is not None and total_tokens is not None:
        usage_summary["requests_with_usage"] = int(usage_summary.get("requests_with_usage", 0)) + 1
        usage_summary["prompt_tokens_total"] = (
            int(usage_summary.get("prompt_tokens_total", 0)) + prompt_tokens
        )
        usage_summary["completion_tokens_total"] = (
            int(usage_summary.get("completion_tokens_total", 0)) + completion_tokens
        )
        usage_summary["total_tokens_total"] = (
            int(usage_summary.get("total_tokens_total", 0)) + total_tokens
        )
        count = max(1, int(usage_summary["requests_with_usage"]))
        usage_summary["prompt_tokens_average"] = round(
            usage_summary["prompt_tokens_total"] / count, 3
        )
        usage_summary["completion_tokens_average"] = round(
            usage_summary["completion_tokens_total"] / count,
            3,
        )
        usage_summary["total_tokens_average"] = round(
            usage_summary["total_tokens_total"] / count, 3
        )

    aggregate["latest_request"] = {
        "request_id": record.get("request_id", ""),
        "timestamp": record.get("timestamp", ""),
        "user_id": record.get("user_id", ""),
        "outcome": outcome,
        "overall_response_ms": timings.get("overall_response_ms"),
    }
    return aggregate


def append_chat_metrics_json(record: dict) -> None:
    """Persist per-request chat metrics to JSON and optionally DynamoDB."""
    normalized = _normalize_request_record(record)
    aggregate = None

    if settings.app.metrics_json_enabled:
        _append_request_record(normalized)

        aggregate_path = _aggregate_json_path()
        with _aggregate_lock():
            aggregate = _load_aggregate(aggregate_path)
            aggregate = _update_aggregate_payload(aggregate, normalized)
            _refresh_latency_percentiles(aggregate)
            _save_aggregate(aggregate_path, aggregate)

    persist_chat_metrics_dynamodb(normalized, aggregate)


def load_chat_metrics_aggregate_json() -> dict | None:
    """Return the latest aggregate JSON snapshot for out-of-band DynamoDB sync."""
    aggregate_path = _aggregate_json_path()
    if not aggregate_path.exists():
        return None
    with _aggregate_lock():
        payload = _load_aggregate(aggregate_path)
    if not isinstance(payload, dict):
        return None
    return payload
