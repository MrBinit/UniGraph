import json
import logging
from datetime import datetime, timezone

from app.core.config import get_settings
from app.infra.redis_client import app_scoped_key, redis_client

settings = get_settings()
logger = logging.getLogger(__name__)


def _token_count_text(text: str, token_counter) -> int:
    """Estimate token count for a summary string using the shared token counter."""
    if not text:
        return 0
    return token_counter([{"role": "assistant", "content": text}])


def _summary_quality_indicator(summary_tokens: int, removed_tokens: int) -> str:
    """Classify summary quality based on how compressed the result is."""
    if summary_tokens == 0:
        return "empty"
    if removed_tokens <= 0:
        return "n/a"

    ratio = summary_tokens / removed_tokens
    if ratio < 0.05:
        return "too_short"
    if ratio > settings.memory.summary_quality_max_ratio:
        return "too_verbose"
    return "good"


def record_compaction_metrics(
    *,
    user_id: str,
    trigger: str,
    removed_messages: int,
    removed_tokens: int,
    before_tokens: int,
    after_tokens: int,
    summary_text: str,
    token_counter,
):
    """Record compaction metrics in logs and Redis for later operational analysis."""
    summary_tokens = _token_count_text(summary_text, token_counter)
    quality = _summary_quality_indicator(summary_tokens, removed_tokens)

    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "trigger": trigger,
        "removed_messages": removed_messages,
        "removed_tokens": removed_tokens,
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "summary_tokens": summary_tokens,
        "summary_quality": quality,
    }

    logger.info("MemoryCompactionMetrics | %s", json.dumps(payload, sort_keys=True))

    try:
        global_key = app_scoped_key("metrics", "memory", "compaction", "global")
        user_key = app_scoped_key("metrics", "memory", "compaction", "user", user_id)

        redis_client.hincrby(global_key, "events", 1)
        redis_client.hincrby(global_key, "removed_messages", removed_messages)
        redis_client.hincrby(global_key, "removed_tokens", removed_tokens)

        redis_client.hincrby(user_key, "events", 1)
        redis_client.hincrby(user_key, "removed_messages", removed_messages)
        redis_client.hincrby(user_key, "removed_tokens", removed_tokens)
        redis_client.hset(
            user_key,
            mapping={
                "last_trigger": trigger,
                "last_before_tokens": before_tokens,
                "last_after_tokens": after_tokens,
                "last_summary_quality": quality,
                "last_summary_tokens": summary_tokens,
                "last_updated_at": payload["ts"],
            },
        )
        redis_client.expire(user_key, settings.memory.redis_ttl_seconds)
    except Exception as exc:
        logger.warning("Compaction metrics persistence failed; continuing. %s", exc)
