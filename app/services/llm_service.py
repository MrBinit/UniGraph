import asyncio
import logging
import time
from redis.exceptions import RedisError

from app.infra.redis_client import app_scoped_key, redis_client
from app.infra.azure_openai_client import client
from app.core.config import get_prompts, get_settings
from app.services.guardrails_service import (
    apply_context_guardrails,
    guard_model_output,
    guard_user_input,
    refusal_response,
)
from app.services.memory_service import build_context, update_memory

settings = get_settings()
prompts = get_prompts()
logger = logging.getLogger(__name__)

SEMAPHORE = asyncio.Semaphore(settings.azure_openai.max_concurrency)


def _latency_metrics_key() -> str:
    """Return the Redis key used to store aggregate LLM latency metrics."""
    return app_scoped_key("metrics", "llm", "latency")


def _record_latency_metrics(started_at: float, outcome: str):
    """Persist request latency metrics for observability and ops reporting."""
    latency_ms = max(0, int((time.perf_counter() - started_at) * 1000))
    key = _latency_metrics_key()
    try:
        redis_client.hincrby(key, "count", 1)
        redis_client.hincrby(key, "total_ms", latency_ms)
        current_max = redis_client.hget(key, "max_ms")
        if current_max is None or latency_ms > int(current_max):
            redis_client.hset(key, "max_ms", latency_ms)
        redis_client.hset(
            key,
            mapping={
                "last_ms": latency_ms,
                "last_outcome": outcome,
            },
        )
    except Exception:
        logger.warning("Latency metrics persistence failed; continuing.")


async def _call_primary(messages: list):
    """Send the request to the primary Azure OpenAI deployment."""
    return await client.chat.completions.create(
        model=settings.azure_openai.primary_deployment,
        messages=messages,
        timeout=settings.azure_openai.timeout,
    )


async def _call_fallback(messages: list):
    """Send the request to the fallback Azure OpenAI deployment."""
    return await client.chat.completions.create(
        model=settings.azure_openai.fallback_deployment,
        messages=messages,
        timeout=settings.azure_openai.timeout,
    )


async def generate_response(user_id: str, user_prompt: str) -> str:
    """Run the full chat pipeline: guardrails, memory, model call, cache, and persistence."""
    started_at = time.perf_counter()
    input_guard = guard_user_input(user_id, user_prompt)
    if input_guard["blocked"]:
        logger.info(
            "GuardrailDecision | stage=input | user=%s | blocked=true | reason=%s",
            user_id,
            input_guard["reason"],
        )
        _record_latency_metrics(started_at, "blocked_input")
        return refusal_response()

    safe_user_prompt = input_guard["sanitized_text"]
    cache_key = app_scoped_key("cache", "chat", user_id, safe_user_prompt)

    # Cache lookup
    try:
        cached = redis_client.get(cache_key)
        if cached:
            _record_latency_metrics(started_at, "cache_hit")
            return cached
    except RedisError as exc:
        logger.warning("Redis cache read failed. %s", exc)

    # Build hybrid memory context
    messages = await build_context(user_id, safe_user_prompt)
    chat_system_prompt = prompts.get("chat", {}).get("system_prompt", "")
    if isinstance(chat_system_prompt, str) and chat_system_prompt.strip():
        messages = [{"role": "system", "content": chat_system_prompt.strip()}] + messages

    context_guard = apply_context_guardrails(messages)
    if context_guard["blocked"]:
        logger.info(
            "GuardrailDecision | stage=context | user=%s | blocked=true | reason=%s",
            user_id,
            context_guard["reason"],
        )
        _record_latency_metrics(started_at, "blocked_context")
        return refusal_response()
    messages = context_guard["messages"]

    async with SEMAPHORE:
        try:
            response = await _call_primary(messages)
        except Exception as primary_exc:
            logger.warning("Primary model failed; attempting fallback. %s", primary_exc)
            try:
                response = await _call_fallback(messages)
            except Exception:
                logger.exception("Fallback model also failed.")
                raise

    raw_result = response.choices[0].message.content
    guarded_output = guard_model_output(raw_result)
    result = guarded_output["text"]
    if guarded_output["blocked"]:
        logger.info(
            "GuardrailDecision | stage=output | user=%s | blocked=true | reason=%s",
            user_id,
            guarded_output["reason"],
        )

    # LLM usage observability
    usage = getattr(response, "usage", None)
    if usage:
        logger.info(
            "LLMUsage | user=%s | prompt_tokens=%s | completion_tokens=%s | total_tokens=%s",
            user_id,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
        )

    # Update memory
    try:
        await update_memory(user_id, safe_user_prompt, result)
    except Exception as exc:
        logger.warning("Memory update failed. %s", exc)

    # Cache result
    try:
        redis_client.setex(
            cache_key,
            settings.memory.redis_ttl_seconds,
            result,
        )
    except RedisError as exc:
        logger.warning("Redis cache write failed. %s", exc)

    _record_latency_metrics(started_at, "success")
    return result
