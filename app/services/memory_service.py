import json
import logging
from copy import deepcopy

from redis.exceptions import RedisError, WatchError

from app.core.config import get_settings, get_prompts
from app.core.memory_crypto import decrypt_memory_payload, encrypt_memory_payload
from app.core.token_utils import count_tokens
from app.infra.azure_openai_client import client
from app.infra.redis_client import app_scoped_key, redis_client
from app.services.memory_compaction_service import (
    safe_token_count,
    select_summary_cutoff,
    truncate_context_without_summary,
)
from app.services.memory_metrics_service import record_compaction_metrics
from app.services.summary_queue_service import enqueue_summary_job
from app.services.token_budget_service import resolve_user_budget

settings = get_settings()
prompts = get_prompts()
logger = logging.getLogger(__name__)


def _redis_key(user_id: str) -> str:
    return app_scoped_key("memory", "chat", user_id)


def _legacy_redis_key(user_id: str) -> str:
    return f"chat:{user_id}"


def _empty_memory() -> dict:
    return {
        "summary": "",
        "messages": [],
        "version": 0,
        "next_seq": 1,
        "last_summarized_seq": 0,
        "summary_pending": False,
        "last_summary_job_id": "",
    }


def _coerce_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_memory(raw_memory: dict | None) -> dict:
    normalized = _empty_memory()
    if not isinstance(raw_memory, dict):
        return normalized

    summary = raw_memory.get("summary", "")
    normalized["summary"] = summary if isinstance(summary, str) else ""
    normalized["version"] = max(0, _coerce_int(raw_memory.get("version", 0), 0))
    normalized["last_summarized_seq"] = max(
        0,
        _coerce_int(raw_memory.get("last_summarized_seq", 0), 0),
    )
    normalized["summary_pending"] = bool(raw_memory.get("summary_pending", False))
    last_job_id = raw_memory.get("last_summary_job_id", "")
    normalized["last_summary_job_id"] = last_job_id if isinstance(last_job_id, str) else ""

    next_seq_hint = max(1, _coerce_int(raw_memory.get("next_seq", 1), 1))
    seq_counter = 1
    cleaned_messages = []

    for msg in raw_memory.get("messages", []) if isinstance(raw_memory.get("messages"), list) else []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        seq = _coerce_int(msg.get("seq"), 0)
        if seq <= 0:
            seq = seq_counter
        seq_counter = max(seq_counter, seq + 1)
        cleaned_messages.append({"seq": seq, "role": role, "content": content})

    normalized["messages"] = cleaned_messages
    normalized["next_seq"] = max(next_seq_hint, seq_counter)
    return normalized


def _get_user_budget(user_id: str) -> tuple[int, int, int]:
    return resolve_user_budget(user_id)


def _strip_seq(messages: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def _serialize_memory_payload(memory: dict) -> str:
    return encrypt_memory_payload(_normalize_memory(memory))


def _deserialize_memory_payload(raw: str) -> tuple[dict, bool]:
    parsed = decrypt_memory_payload(raw)
    if not isinstance(parsed, dict):
        return _empty_memory(), False
    return _normalize_memory(parsed), True


async def load_memory(user_id: str) -> dict:
    try:
        raw = redis_client.get(_redis_key(user_id))
    except RedisError as exc:
        logger.warning("Redis memory read failed; using empty memory. %s", exc)
        return _empty_memory()

    if not raw:
        try:
            raw = redis_client.get(_legacy_redis_key(user_id))
        except RedisError as exc:
            logger.warning("Redis legacy memory read failed; using empty memory. %s", exc)
            return _empty_memory()

    if not raw:
        return _empty_memory()

    parsed, ok = _deserialize_memory_payload(raw)
    if not ok:
        logger.warning("Corrupted encrypted memory payload for user_id=%s; resetting memory.", user_id)
    return parsed


def save_memory(user_id: str, memory: dict):
    normalized = _normalize_memory(memory)
    try:
        redis_client.setex(
            _redis_key(user_id),
            settings.memory.redis_ttl_seconds,
            _serialize_memory_payload(normalized),
        )
    except RedisError as exc:
        logger.warning("Redis memory write failed; skipping persistence. %s", exc)


def save_memory_if_version(user_id: str, expected_version: int, memory: dict) -> tuple[bool, dict]:
    key = _redis_key(user_id)
    normalized_target = _normalize_memory(memory)

    for _ in range(3):
        try:
            with redis_client.pipeline() as pipe:
                pipe.watch(key)
                current_raw = pipe.get(key)
                if current_raw:
                    current_memory, ok = _deserialize_memory_payload(current_raw)
                    if not ok:
                        current_memory = _empty_memory()
                else:
                    current_memory = _empty_memory()

                if current_memory["version"] != expected_version:
                    pipe.unwatch()
                    return False, current_memory

                pipe.multi()
                pipe.setex(
                    key,
                    settings.memory.redis_ttl_seconds,
                    _serialize_memory_payload(normalized_target),
                )
                pipe.execute()
                return True, normalized_target
        except WatchError:
            continue
        except RedisError as exc:
            logger.warning("Versioned memory save failed for user=%s. %s", user_id, exc)
            return False, _empty_memory()

    return False, _empty_memory()


async def summarize_messages(messages: list) -> str:
    if not messages:
        return ""

    response = await client.chat.completions.create(
        model=settings.azure_openai.primary_deployment,
        messages=[
            {
                "role": "system",
                "content": prompts["memory"]["summary_system_prompt"],
            },
            {
                "role": "user",
                "content": json.dumps(messages),
            },
        ],
        timeout=settings.azure_openai.timeout,
    )

    content = response.choices[0].message.content
    return content if isinstance(content, str) else ""


async def build_context(user_id: str, new_user_message: str) -> list:
    memory = await load_memory(user_id)
    soft_limit, hard_limit, min_recent = _get_user_budget(user_id)

    messages = deepcopy(memory["messages"])
    summary = memory["summary"]

    token_count_pre_compaction = safe_token_count(
        count_tokens,
        messages + [{"role": "user", "content": new_user_message}],
    )

    logger.info(
        "MemoryCheck | user=%s | token_count=%s | message_count=%s | summary_exists=%s | soft_limit=%s | hard_limit=%s",
        user_id,
        token_count_pre_compaction,
        len(messages),
        bool(summary),
        soft_limit,
        hard_limit,
    )

    if (
        token_count_pre_compaction > settings.memory.summary_trigger
        and messages
        and not memory["summary_pending"]
    ):
        summary_candidates, cutoff_seq = select_summary_cutoff(
            messages,
            settings.memory.summary_ratio,
        )
        if cutoff_seq is not None:
            removed_tokens = safe_token_count(count_tokens, summary_candidates)
            job_id = enqueue_summary_job(
                user_id=user_id,
                cutoff_seq=cutoff_seq,
                trigger="summary_trigger",
                enqueue_version=memory["version"],
                approx_removed_tokens=removed_tokens,
            )
            if job_id:
                memory["summary_pending"] = True
                memory["last_summary_job_id"] = job_id
                memory["version"] += 1
                save_memory(user_id, memory)

    truncation_result = truncate_context_without_summary(
        summary=summary,
        messages=messages,
        new_user_message=new_user_message,
        soft_limit=soft_limit,
        hard_limit=hard_limit,
        min_recent=min_recent,
        token_counter=count_tokens,
    )

    for event in truncation_result["events"]:
        record_compaction_metrics(
            user_id=user_id,
            trigger=event["trigger"],
            removed_messages=event["removed_messages"],
            removed_tokens=event["removed_tokens"],
            before_tokens=event["before_tokens"],
            after_tokens=event["after_tokens"],
            summary_text=event["summary_text"],
            token_counter=count_tokens,
        )

    if truncation_result["memory_changed"]:
        memory["summary"] = truncation_result["summary"]
        memory["messages"] = truncation_result["messages"]
        memory["version"] += 1
        memory["next_seq"] = max(
            memory["next_seq"],
            (memory["messages"][-1]["seq"] + 1) if memory["messages"] else 1,
        )
        save_memory(user_id, memory)

    logger.info(
        "ContextBuilt | user=%s | final_tokens=%s | summary_present=%s",
        user_id,
        truncation_result["final_tokens"],
        bool(truncation_result["summary"]),
    )

    return truncation_result["final_context"]


async def update_memory(user_id: str, user_message: str, assistant_reply: str):
    memory = await load_memory(user_id)

    user_seq = memory["next_seq"]
    assistant_seq = user_seq + 1
    memory["messages"].append({"seq": user_seq, "role": "user", "content": user_message})
    memory["messages"].append({"seq": assistant_seq, "role": "assistant", "content": assistant_reply})
    memory["next_seq"] = assistant_seq + 1
    memory["version"] += 1

    save_memory(user_id, memory)
