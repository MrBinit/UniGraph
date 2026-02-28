from app.core.config import get_settings
from app.infra.redis_client import app_redis_client, app_scoped_key
from app.services.summary_queue_service import get_summary_dlq_state, get_summary_queue_state

settings = get_settings()


def _safe_ping(client) -> bool:
    try:
        return bool(client.ping())
    except Exception:
        return False


def _read_hash(key: str) -> dict:
    try:
        data = app_redis_client.hgetall(key)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_ops_status() -> dict:
    memory_available = _safe_ping(app_redis_client)
    queue_state = get_summary_queue_state()
    dlq_state = get_summary_dlq_state()

    compaction_key = app_scoped_key("metrics", "memory", "compaction", "global")
    compaction_raw = _read_hash(compaction_key)

    latency_key = app_scoped_key("metrics", "llm", "latency")
    latency_raw = _read_hash(latency_key)
    latency_count = _to_int(latency_raw.get("count", 0))
    latency_total_ms = _to_int(latency_raw.get("total_ms", 0))

    status = "ok"
    if not memory_available:
        status = "down"
    elif dlq_state["depth"] >= settings.memory.summary_queue_dlq_alert_threshold:
        status = "degraded"

    latest_dlq = dlq_state.get("latest") or {}
    average_ms = round(latency_total_ms / latency_count, 2) if latency_count else 0.0

    return {
        "status": status,
        "memory": {
            "redis_available": memory_available,
            "ttl_seconds": settings.memory.redis_ttl_seconds,
            "encryption_enabled": True,
        },
        "queue": {
            "stream_depth": queue_state["stream_depth"],
            "pending_jobs": queue_state["pending_jobs"],
            "dlq_depth": dlq_state["depth"],
            "consumer_group": settings.memory.summary_queue_group,
            "last_dlq_error": latest_dlq.get("error", ""),
        },
        "compaction": {
            "events": _to_int(compaction_raw.get("events", 0)),
            "removed_messages": _to_int(compaction_raw.get("removed_messages", 0)),
            "removed_tokens": _to_int(compaction_raw.get("removed_tokens", 0)),
        },
        "latency": {
            "count": latency_count,
            "average_ms": average_ms,
            "max_ms": _to_int(latency_raw.get("max_ms", 0)),
            "last_ms": _to_int(latency_raw.get("last_ms", 0)),
            "last_outcome": str(latency_raw.get("last_outcome", "")),
        },
    }
