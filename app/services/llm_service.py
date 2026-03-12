import asyncio
import hashlib
import inspect
import logging
import time
from typing import AsyncIterator
from uuid import uuid4
from redis.exceptions import RedisError
from app.core.config import get_prompts, get_settings
from app.infra.bedrock_chat_client import client
from app.infra.io_limiters import dependency_limiter
from app.infra.redis_client import app_scoped_key, async_redis_client, redis_client
from app.services.evaluation_service import store_chat_trace
from app.services.guardrails_service import (
    apply_context_guardrails,
    guard_model_output,
    guard_user_input,
    refusal_response,
)
from app.services.memory_service import build_context, update_memory
from app.services.sqs_event_queue_service import enqueue_metrics_record_event
from app.services.retrieval_service import aretrieve_document_chunks

settings = get_settings()
prompts = get_prompts()
logger = logging.getLogger(__name__)

_BACKGROUND_TASKS: set[asyncio.Task] = set()
_RETRIEVAL_QUERY_MAX_CHARS = 900
_RETRIEVAL_CONTEXT_MAX_CHARS = 1500
_RETRIEVAL_CHUNK_MAX_CHARS = 360
_RETRIEVAL_MAX_PROMPT_RESULTS = 2
_RETRIEVAL_EVIDENCE_MAX_ITEMS = 3
_RETRIEVAL_EVIDENCE_CONTENT_MAX_CHARS = 700
_STREAM_GUARD_HOLDBACK_CHARS = 120


def _chat_cache_key(user_id: str, prompt: str, session_id: str | None = None) -> str:
    """Build a fixed-length chat cache key without embedding raw prompt text."""
    normalized_session = str(session_id or user_id).strip() or str(user_id).strip()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return app_scoped_key("cache", "chat", user_id, normalized_session, f"sha256:{prompt_hash}")


def _resolve_session_id(user_id: str, session_id: str | None) -> str:
    """Resolve an effective session identifier with a safe fallback."""
    candidate = str(session_id or "").strip()
    return candidate or str(user_id).strip()


def _conversation_user_id(user_id: str, session_id: str) -> str:
    """Return memory key space id; isolate when a distinct session id is provided."""
    normalized_user = str(user_id).strip()
    normalized_session = str(session_id).strip()
    if not normalized_session or normalized_session == normalized_user:
        return normalized_user
    return f"{normalized_user}::session::{normalized_session}"


async def _redis_call(method, *args, **kwargs):
    """Execute a Redis operation using the async client and limiter."""
    async with dependency_limiter("redis"):
        result = method(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


def _elapsed_ms(started_at: float) -> int:
    """Return elapsed milliseconds from a monotonic start time."""
    return max(0, int((time.perf_counter() - started_at) * 1000))


def _safe_int(value) -> int | None:
    """Convert a numeric-like value to int when possible."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_llm_usage(response) -> dict:
    """Normalize response usage information into a JSON-safe dictionary."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    prompt_tokens = _safe_int(getattr(usage, "prompt_tokens", None))
    completion_tokens = _safe_int(getattr(usage, "completion_tokens", None))
    total_tokens = _safe_int(getattr(usage, "total_tokens", None))
    if prompt_tokens is None or completion_tokens is None or total_tokens is None:
        return {}

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


async def _record_json_metrics(record: dict) -> None:
    """Enqueue per-request metrics for background persistence."""
    if not settings.queue.metrics_aggregation_queue_enabled:
        return
    if not settings.queue.metrics_aggregation_queue_url.strip():
        logger.warning("Metrics queue URL missing; dropping metrics event.")
        return

    async def _enqueue() -> None:
        try:
            await asyncio.to_thread(enqueue_metrics_record_event, record)
        except Exception:
            logger.warning("Metrics event enqueue failed; continuing.")

    _track_background_task(
        asyncio.create_task(_enqueue()),
        label="Metrics event enqueue",
    )


def _build_json_metrics_record(
    *,
    request_id: str,
    started_at: float,
    user_id: str,
    session_id: str | None,
    user_prompt: str,
    safe_user_prompt: str,
    answer: str,
    outcome: str,
    build_context_ms: int | None,
    retrieval_ms: int | None,
    model_ms: int | None,
    memory_update_ms: int | None,
    cache_read_ms: int | None,
    cache_write_ms: int | None,
    evaluation_trace_ms: int | None,
    retrieval_strategy: str,
    retrieved_count: int,
    retrieval_evidence: list[dict] | None,
    quality: dict,
    llm_usage: dict,
    input_guard_reason: str,
    context_guard_reason: str,
    output_guard_reason: str,
    used_fallback_model: bool,
    error_message: str = "",
) -> dict:
    """Build the per-request metrics payload persisted to JSON."""
    return {
        "request_id": request_id,
        "user_id": user_id,
        "session_id": str(session_id or user_id),
        "question": user_prompt,
        "question_sanitized": safe_user_prompt,
        "answer": answer,
        "outcome": outcome,
        "timings_ms": {
            "overall_response_ms": _elapsed_ms(started_at),
            "llm_response_ms": model_ms,
            "short_term_memory_ms": build_context_ms,
            "long_term_memory_ms": retrieval_ms,
            "memory_update_ms": memory_update_ms,
            "cache_read_ms": cache_read_ms,
            "cache_write_ms": cache_write_ms,
            "evaluation_trace_ms": evaluation_trace_ms,
        },
        "retrieval": {
            "strategy": retrieval_strategy,
            "result_count": retrieved_count,
            "evidence": retrieval_evidence or [],
        },
        "quality": quality,
        "hallucination_proxy": quality.get("hallucination_proxy"),
        "llm_usage": llm_usage,
        "guardrails": {
            "input_reason": input_guard_reason,
            "context_reason": context_guard_reason,
            "output_reason": output_guard_reason,
        },
        "model": {
            "provider": "amazon_bedrock",
            "used_fallback": used_fallback_model,
            "primary_model_id": settings.bedrock.primary_model_id,
            "fallback_model_id": settings.bedrock.fallback_model_id,
        },
        "error": error_message,
    }


def _latency_metrics_key() -> str:
    """Return the Redis key used to store aggregate LLM latency metrics."""
    return app_scoped_key("metrics", "llm", "latency")


async def _record_latency_metrics(started_at: float, outcome: str):
    """Persist request latency metrics for observability and ops reporting."""
    latency_ms = _elapsed_ms(started_at)
    key = _latency_metrics_key()
    try:
        await _redis_call(async_redis_client.hincrby, key, "count", 1)
        await _redis_call(async_redis_client.hincrby, key, "total_ms", latency_ms)
        current_max = await _redis_call(async_redis_client.hget, key, "max_ms")
        if current_max is None or latency_ms > int(current_max):
            await _redis_call(async_redis_client.hset, key, "max_ms", latency_ms)
        await _redis_call(
            async_redis_client.hset,
            key,
            mapping={
                "last_ms": latency_ms,
                "last_outcome": outcome,
            },
        )
    except Exception:
        logger.warning("Latency metrics persistence failed; continuing.")


async def _record_pipeline_stage_metrics(
    *,
    build_context_ms: int,
    retrieval_ms: int,
    model_ms: int,
    retrieval_strategy: str,
    retrieved_count: int,
):
    """Persist stage-level latency metrics for successful full chat pipeline runs."""
    key = _latency_metrics_key()
    try:
        await _redis_call(async_redis_client.hincrby, key, "pipeline_count", 1)
        await _redis_call(
            async_redis_client.hincrby, key, "build_context_total_ms", build_context_ms
        )
        await _redis_call(async_redis_client.hincrby, key, "retrieval_total_ms", retrieval_ms)
        await _redis_call(async_redis_client.hincrby, key, "model_total_ms", model_ms)
        await _redis_call(
            async_redis_client.hset,
            key,
            mapping={
                "last_build_context_ms": build_context_ms,
                "last_retrieval_ms": retrieval_ms,
                "last_model_ms": model_ms,
                "last_retrieval_strategy": retrieval_strategy,
                "last_retrieved_count": retrieved_count,
            },
        )
    except Exception:
        logger.warning("Pipeline stage metrics persistence failed; continuing.")


def _build_retrieval_query(messages: list[dict]) -> str:
    """Build a retrieval query from the latest short-term conversation context."""
    text_parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role != "user" or not isinstance(content, str) or not content.strip():
            continue
        text_parts.append(content.strip())
    if not text_parts:
        return ""
    return "\n\n".join(text_parts[-2:])[:_RETRIEVAL_QUERY_MAX_CHARS].strip()


def _format_retrieval_context(retrieval_result: dict) -> dict | None:
    """Convert retrieved long-term chunks into a single system-context message."""
    results = retrieval_result.get("results", []) if isinstance(retrieval_result, dict) else []
    if not isinstance(results, list) or not results:
        return None

    lines = [
        "Retrieved long-term knowledge. Use this only when relevant to the user's request.",
    ]
    seen_chunks: set[str] = set()
    used_results = 0
    for result in results:
        if not isinstance(result, dict):
            continue
        content = result.get("content")
        metadata = result.get("metadata") or {}
        if not isinstance(content, str) or not content.strip():
            continue
        dedupe_key = " ".join(content.lower().split())[:180]
        if dedupe_key in seen_chunks:
            continue
        seen_chunks.add(dedupe_key)
        used_results += 1

        label_parts = []
        if isinstance(metadata, dict):
            university = metadata.get("university")
            section_heading = metadata.get("section_heading")
            if isinstance(university, str) and university.strip():
                label_parts.append(university.strip())
            if isinstance(section_heading, str) and section_heading.strip():
                label_parts.append(section_heading.strip())
        label = " | ".join(label_parts) if label_parts else f"Result {used_results}"
        compact_content = " ".join(content.split())[:_RETRIEVAL_CHUNK_MAX_CHARS]
        lines.append(f"{used_results}. {label}: {compact_content}")
        if used_results >= _RETRIEVAL_MAX_PROMPT_RESULTS:
            break

    if len(lines) == 1:
        return None
    joined = "\n".join(lines)
    return {"role": "system", "content": joined[:_RETRIEVAL_CONTEXT_MAX_CHARS]}


def _build_retrieval_evidence(results: list[dict]) -> list[dict]:
    """Build compact retrieval evidence for grounded hallucination evaluation."""
    if not isinstance(results, list):
        return []

    evidence: list[dict] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        content = str(result.get("content", "")).strip()
        if not content:
            continue
        metadata = result.get("metadata")
        evidence.append(
            {
                "chunk_id": str(result.get("chunk_id", "")),
                "source_path": str(result.get("source_path", "")),
                "distance": result.get("distance"),
                "metadata": metadata if isinstance(metadata, dict) else {},
                "content": " ".join(content.split())[:_RETRIEVAL_EVIDENCE_CONTENT_MAX_CHARS],
            }
        )
        if len(evidence) >= _RETRIEVAL_EVIDENCE_MAX_ITEMS:
            break
    return evidence


def _track_background_task(task: asyncio.Task, *, label: str) -> None:
    """Track and log fire-and-forget tasks so they are not silently lost."""
    _BACKGROUND_TASKS.add(task)

    def _on_done(completed: asyncio.Task) -> None:
        _BACKGROUND_TASKS.discard(completed)
        try:
            completed.result()
        except Exception as exc:
            logger.warning("%s failed in background. %s", label, exc)

    task.add_done_callback(_on_done)


async def _persist_evaluation_trace(
    *,
    user_id: str,
    prompt: str,
    answer: str,
    retrieved_results: list[dict],
    retrieval_strategy: str,
    build_context_ms: int,
    retrieval_ms: int,
    model_ms: int,
) -> None:
    """Persist evaluation traces outside the request critical path."""
    await asyncio.to_thread(
        store_chat_trace,
        user_id=user_id,
        prompt=prompt,
        answer=answer,
        retrieved_results=retrieved_results,
        retrieval_strategy=retrieval_strategy,
        timings_ms={
            "build_context": build_context_ms,
            "retrieval": retrieval_ms,
            "model": model_ms,
        },
        redis=redis_client,
    )


async def _call_primary(messages: list):
    """Send the request to the primary Bedrock model."""
    return await client.chat.completions.create(
        model=settings.bedrock.primary_model_id,
        messages=messages,
        timeout=settings.bedrock.timeout,
    )


async def _call_fallback(messages: list):
    """Send the request to the fallback Bedrock model."""
    return await client.chat.completions.create(
        model=settings.bedrock.fallback_model_id,
        messages=messages,
        timeout=settings.bedrock.timeout,
    )


async def _stream_primary(messages: list) -> AsyncIterator[str]:
    """Stream token deltas from the primary deployment."""
    async for delta in client.chat.completions.stream(
        model=settings.bedrock.primary_model_id,
        messages=messages,
        timeout=settings.bedrock.timeout,
    ):
        yield delta


async def _stream_fallback(messages: list) -> AsyncIterator[str]:
    """Stream token deltas from the fallback deployment."""
    async for delta in client.chat.completions.stream(
        model=settings.bedrock.fallback_model_id,
        messages=messages,
        timeout=settings.bedrock.timeout,
    ):
        yield delta


async def generate_response(user_id: str, user_prompt: str, session_id: str | None = None) -> str:
    """Run the full chat pipeline: guardrails, memory, model call, cache, and persistence."""
    started_at = time.perf_counter()
    request_id = uuid4().hex
    effective_session_id = _resolve_session_id(user_id, session_id)
    conversation_user_id = _conversation_user_id(user_id, effective_session_id)
    safe_user_prompt = user_prompt
    build_context_ms: int | None = None
    retrieval_ms: int | None = None
    model_ms: int | None = None
    memory_update_ms: int | None = None
    cache_read_ms: int | None = None
    cache_write_ms: int | None = None
    evaluation_trace_ms: int | None = None
    retrieval_strategy = "none"
    retrieved_count = 0
    retrieved_results: list[dict] = []
    retrieval_evidence: list[dict] = []
    llm_usage: dict = {}
    quality: dict = {}
    input_guard_reason = ""
    context_guard_reason = ""
    output_guard_reason = ""
    used_fallback_model = False

    input_guard = guard_user_input(user_id, user_prompt)
    if input_guard["blocked"]:
        input_guard_reason = str(input_guard.get("reason", "blocked_input"))
        refusal = refusal_response()
        logger.info(
            "GuardrailDecision | stage=input | user=%s | blocked=true | reason=%s",
            user_id,
            input_guard_reason,
        )
        await _record_json_metrics(
            _build_json_metrics_record(
                request_id=request_id,
                started_at=started_at,
                user_id=user_id,
                session_id=effective_session_id,
                user_prompt=user_prompt,
                safe_user_prompt=safe_user_prompt,
                answer=refusal,
                outcome="blocked_input",
                build_context_ms=build_context_ms,
                retrieval_ms=retrieval_ms,
                model_ms=model_ms,
                memory_update_ms=memory_update_ms,
                cache_read_ms=cache_read_ms,
                cache_write_ms=cache_write_ms,
                evaluation_trace_ms=evaluation_trace_ms,
                retrieval_strategy=retrieval_strategy,
                retrieved_count=retrieved_count,
                retrieval_evidence=retrieval_evidence,
                quality=quality,
                llm_usage=llm_usage,
                input_guard_reason=input_guard_reason,
                context_guard_reason=context_guard_reason,
                output_guard_reason=output_guard_reason,
                used_fallback_model=used_fallback_model,
            )
        )
        await _record_latency_metrics(started_at, "blocked_input")
        return refusal

    safe_user_prompt = str(input_guard.get("sanitized_text", user_prompt))
    cache_key = _chat_cache_key(user_id, safe_user_prompt, effective_session_id)

    # Cache lookup
    cached = None
    cache_read_started_at = time.perf_counter()
    try:
        cached = await _redis_call(async_redis_client.get, cache_key)
    except RedisError as exc:
        logger.warning("Redis cache read failed. %s", exc)
    finally:
        cache_read_ms = _elapsed_ms(cache_read_started_at)

    if cached:
        await _record_json_metrics(
            _build_json_metrics_record(
                request_id=request_id,
                started_at=started_at,
                user_id=user_id,
                session_id=effective_session_id,
                user_prompt=user_prompt,
                safe_user_prompt=safe_user_prompt,
                answer=str(cached),
                outcome="cache_hit",
                build_context_ms=build_context_ms,
                retrieval_ms=retrieval_ms,
                model_ms=model_ms,
                memory_update_ms=memory_update_ms,
                cache_read_ms=cache_read_ms,
                cache_write_ms=cache_write_ms,
                evaluation_trace_ms=evaluation_trace_ms,
                retrieval_strategy=retrieval_strategy,
                retrieved_count=retrieved_count,
                retrieval_evidence=retrieval_evidence,
                quality=quality,
                llm_usage=llm_usage,
                input_guard_reason=input_guard_reason,
                context_guard_reason=context_guard_reason,
                output_guard_reason=output_guard_reason,
                used_fallback_model=used_fallback_model,
            )
        )
        await _record_latency_metrics(started_at, "cache_hit")
        return str(cached)

    # Build hybrid memory context
    build_context_started_at = time.perf_counter()
    messages = await build_context(conversation_user_id, safe_user_prompt)
    build_context_ms = _elapsed_ms(build_context_started_at)

    retrieval_query = _build_retrieval_query(messages)
    if retrieval_query:
        retrieval_started_at = time.perf_counter()
        try:
            retrieval_result = await aretrieve_document_chunks(
                retrieval_query,
                top_k=settings.postgres.default_top_k,
            )
            retrieval_strategy = str(retrieval_result.get("retrieval_strategy", "unknown"))
            results = retrieval_result.get("results", [])
            if isinstance(results, list):
                retrieved_results = results
                retrieved_count = len(results)
                retrieval_evidence = _build_retrieval_evidence(results)
            retrieval_message = _format_retrieval_context(retrieval_result)
            if retrieval_message:
                messages = [retrieval_message] + messages
        except Exception as exc:
            retrieval_strategy = "error"
            logger.warning(
                "Long-term retrieval failed; continuing without retrieved context. %s",
                exc,
            )
        finally:
            retrieval_ms = _elapsed_ms(retrieval_started_at)
    else:
        retrieval_ms = 0

    chat_system_prompt = prompts.get("chat", {}).get("system_prompt", "")
    if isinstance(chat_system_prompt, str) and chat_system_prompt.strip():
        messages = [{"role": "system", "content": chat_system_prompt.strip()}] + messages

    context_guard = apply_context_guardrails(messages)
    if context_guard["blocked"]:
        context_guard_reason = str(context_guard.get("reason", "blocked_context"))
        refusal = refusal_response()
        logger.info(
            "GuardrailDecision | stage=context | user=%s | blocked=true | reason=%s",
            user_id,
            context_guard_reason,
        )
        await _record_json_metrics(
            _build_json_metrics_record(
                request_id=request_id,
                started_at=started_at,
                user_id=user_id,
                session_id=effective_session_id,
                user_prompt=user_prompt,
                safe_user_prompt=safe_user_prompt,
                answer=refusal,
                outcome="blocked_context",
                build_context_ms=build_context_ms,
                retrieval_ms=retrieval_ms,
                model_ms=model_ms,
                memory_update_ms=memory_update_ms,
                cache_read_ms=cache_read_ms,
                cache_write_ms=cache_write_ms,
                evaluation_trace_ms=evaluation_trace_ms,
                retrieval_strategy=retrieval_strategy,
                retrieved_count=retrieved_count,
                retrieval_evidence=retrieval_evidence,
                quality=quality,
                llm_usage=llm_usage,
                input_guard_reason=input_guard_reason,
                context_guard_reason=context_guard_reason,
                output_guard_reason=output_guard_reason,
                used_fallback_model=used_fallback_model,
            )
        )
        await _record_latency_metrics(started_at, "blocked_context")
        return refusal
    messages = context_guard["messages"]

    model_started_at = time.perf_counter()
    try:
        response = await _call_primary(messages)
    except Exception as primary_exc:
        used_fallback_model = True
        logger.warning("Primary model failed; attempting fallback. %s", primary_exc)
        try:
            response = await _call_fallback(messages)
        except Exception as fallback_exc:
            logger.exception("Fallback model also failed.")
            model_ms = _elapsed_ms(model_started_at)
            await _record_json_metrics(
                _build_json_metrics_record(
                    request_id=request_id,
                    started_at=started_at,
                    user_id=user_id,
                    session_id=effective_session_id,
                    user_prompt=user_prompt,
                    safe_user_prompt=safe_user_prompt,
                    answer="",
                    outcome="model_error",
                    build_context_ms=build_context_ms,
                    retrieval_ms=retrieval_ms,
                    model_ms=model_ms,
                    memory_update_ms=memory_update_ms,
                    cache_read_ms=cache_read_ms,
                    cache_write_ms=cache_write_ms,
                    evaluation_trace_ms=evaluation_trace_ms,
                    retrieval_strategy=retrieval_strategy,
                    retrieved_count=retrieved_count,
                    retrieval_evidence=retrieval_evidence,
                    quality=quality,
                    llm_usage=llm_usage,
                    input_guard_reason=input_guard_reason,
                    context_guard_reason=context_guard_reason,
                    output_guard_reason=output_guard_reason,
                    used_fallback_model=used_fallback_model,
                    error_message=str(fallback_exc),
                )
            )
            await _record_latency_metrics(started_at, "model_error")
            raise
    model_ms = _elapsed_ms(model_started_at)

    raw_result = response.choices[0].message.content
    guarded_output = guard_model_output(raw_result)
    result = guarded_output["text"]
    if guarded_output["blocked"]:
        output_guard_reason = str(guarded_output.get("reason", "blocked_output"))
        logger.info(
            "GuardrailDecision | stage=output | user=%s | blocked=true | reason=%s",
            user_id,
            output_guard_reason,
        )

    # LLM usage observability
    llm_usage = _extract_llm_usage(response)
    if llm_usage:
        logger.info(
            "LLMUsage | user=%s | prompt_tokens=%s | completion_tokens=%s | total_tokens=%s",
            user_id,
            llm_usage["prompt_tokens"],
            llm_usage["completion_tokens"],
            llm_usage["total_tokens"],
        )

    # Update memory
    memory_update_started_at = time.perf_counter()
    try:
        await update_memory(conversation_user_id, safe_user_prompt, result)
    except Exception as exc:
        logger.warning("Memory update failed. %s", exc)
    finally:
        memory_update_ms = _elapsed_ms(memory_update_started_at)

    # Cache result
    cache_write_started_at = time.perf_counter()
    try:
        await _redis_call(
            async_redis_client.setex,
            cache_key,
            settings.memory.redis_ttl_seconds,
            result,
        )
    except RedisError as exc:
        logger.warning("Redis cache write failed. %s", exc)
    finally:
        cache_write_ms = _elapsed_ms(cache_write_started_at)

    evaluation_trace_started_at = time.perf_counter()
    _track_background_task(
        asyncio.create_task(
            _persist_evaluation_trace(
                user_id=user_id,
                prompt=user_prompt,
                answer=result,
                retrieved_results=retrieved_results,
                retrieval_strategy=retrieval_strategy,
                build_context_ms=build_context_ms or 0,
                retrieval_ms=retrieval_ms or 0,
                model_ms=model_ms or 0,
            )
        ),
        label="Evaluation trace persistence",
    )
    evaluation_trace_ms = _elapsed_ms(evaluation_trace_started_at)

    # Online path no longer computes hallucination/relevance metrics.
    # These are evaluated asynchronously by a separate evaluator job.
    quality = {}

    await _record_pipeline_stage_metrics(
        build_context_ms=build_context_ms or 0,
        retrieval_ms=retrieval_ms or 0,
        model_ms=model_ms or 0,
        retrieval_strategy=retrieval_strategy,
        retrieved_count=retrieved_count,
    )
    logger.info(
        (
            "ChatPipelineLatency | user=%s | build_context_ms=%s | retrieval_ms=%s "
            "| model_ms=%s | retrieval_strategy=%s | retrieved_count=%s"
        ),
        user_id,
        build_context_ms,
        retrieval_ms,
        model_ms,
        retrieval_strategy,
        retrieved_count,
    )
    await _record_json_metrics(
        _build_json_metrics_record(
            request_id=request_id,
            started_at=started_at,
            user_id=user_id,
            session_id=effective_session_id,
            user_prompt=user_prompt,
            safe_user_prompt=safe_user_prompt,
            answer=result,
            outcome="success",
            build_context_ms=build_context_ms,
            retrieval_ms=retrieval_ms,
            model_ms=model_ms,
            memory_update_ms=memory_update_ms,
            cache_read_ms=cache_read_ms,
            cache_write_ms=cache_write_ms,
            evaluation_trace_ms=evaluation_trace_ms,
            retrieval_strategy=retrieval_strategy,
            retrieved_count=retrieved_count,
            retrieval_evidence=retrieval_evidence,
            quality=quality,
            llm_usage=llm_usage,
            input_guard_reason=input_guard_reason,
            context_guard_reason=context_guard_reason,
            output_guard_reason=output_guard_reason,
            used_fallback_model=used_fallback_model,
        )
    )
    await _record_latency_metrics(started_at, "success")
    return result


async def generate_response_stream(
    user_id: str,
    user_prompt: str,
    session_id: str | None = None,
    *,
    chunk_size: int = 120,
    chunk_delay_ms: int = 12,
) -> AsyncIterator[str]:
    """Stream true model output from Bedrock and yield progressively assembled text."""

    async def _yield_guarded_stream(
        delta_stream: AsyncIterator[str],
        *,
        stream_state: dict[str, object],
    ) -> AsyncIterator[str]:
        assembled = ""
        emitted = ""
        size = max(1, int(chunk_size))
        delay_seconds = max(0.0, float(chunk_delay_ms) / 1000.0)
        holdback_chars = max(0, _STREAM_GUARD_HOLDBACK_CHARS)

        async for delta in delta_stream:
            text = str(delta or "")
            if not text:
                continue
            for start in range(0, len(text), size):
                piece = text[start : start + size]
                if not piece:
                    continue
                assembled += piece
                guarded = guard_model_output(assembled)
                guarded_text = str(guarded.get("text", ""))
                blocked = bool(guarded.get("blocked"))
                reason = str(guarded.get("reason", "blocked_output")) if blocked else ""

                stream_state["blocked"] = blocked
                stream_state["reason"] = reason
                stream_state["final_text"] = guarded_text

                if blocked:
                    if guarded_text and guarded_text != emitted:
                        emitted = guarded_text
                        yield emitted
                    return

                stable_len = len(guarded_text) - holdback_chars
                if stable_len <= 0:
                    continue

                stable_text = guarded_text[:stable_len]
                if stable_text and stable_text != emitted:
                    emitted = stable_text
                    yield emitted
                    if delay_seconds > 0:
                        await asyncio.sleep(delay_seconds)

        guarded = guard_model_output(assembled)
        guarded_text = str(guarded.get("text", ""))
        blocked = bool(guarded.get("blocked"))
        reason = str(guarded.get("reason", "blocked_output")) if blocked else ""
        stream_state["blocked"] = blocked
        stream_state["reason"] = reason
        stream_state["final_text"] = guarded_text
        if guarded_text and guarded_text != emitted:
            yield guarded_text

    started_at = time.perf_counter()
    request_id = uuid4().hex
    effective_session_id = _resolve_session_id(user_id, session_id)
    conversation_user_id = _conversation_user_id(user_id, effective_session_id)
    safe_user_prompt = user_prompt
    build_context_ms: int | None = None
    retrieval_ms: int | None = None
    model_ms: int | None = None
    memory_update_ms: int | None = None
    cache_read_ms: int | None = None
    cache_write_ms: int | None = None
    evaluation_trace_ms: int | None = None
    retrieval_strategy = "none"
    retrieved_count = 0
    retrieved_results: list[dict] = []
    retrieval_evidence: list[dict] = []
    llm_usage: dict = {}
    quality: dict = {}
    input_guard_reason = ""
    context_guard_reason = ""
    output_guard_reason = ""
    used_fallback_model = False

    input_guard = guard_user_input(user_id, user_prompt)
    if input_guard["blocked"]:
        input_guard_reason = str(input_guard.get("reason", "blocked_input"))
        refusal = refusal_response()
        logger.info(
            "GuardrailDecision | stage=input | user=%s | blocked=true | reason=%s",
            user_id,
            input_guard_reason,
        )
        await _record_json_metrics(
            _build_json_metrics_record(
                request_id=request_id,
                started_at=started_at,
                user_id=user_id,
                session_id=effective_session_id,
                user_prompt=user_prompt,
                safe_user_prompt=safe_user_prompt,
                answer=refusal,
                outcome="blocked_input",
                build_context_ms=build_context_ms,
                retrieval_ms=retrieval_ms,
                model_ms=model_ms,
                memory_update_ms=memory_update_ms,
                cache_read_ms=cache_read_ms,
                cache_write_ms=cache_write_ms,
                evaluation_trace_ms=evaluation_trace_ms,
                retrieval_strategy=retrieval_strategy,
                retrieved_count=retrieved_count,
                retrieval_evidence=retrieval_evidence,
                quality=quality,
                llm_usage=llm_usage,
                input_guard_reason=input_guard_reason,
                context_guard_reason=context_guard_reason,
                output_guard_reason=output_guard_reason,
                used_fallback_model=used_fallback_model,
            )
        )
        await _record_latency_metrics(started_at, "blocked_input")
        yield refusal
        return

    safe_user_prompt = str(input_guard.get("sanitized_text", user_prompt))
    cache_key = _chat_cache_key(user_id, safe_user_prompt, effective_session_id)

    cached = None
    cache_read_started_at = time.perf_counter()
    try:
        cached = await _redis_call(async_redis_client.get, cache_key)
    except RedisError as exc:
        logger.warning("Redis cache read failed. %s", exc)
    finally:
        cache_read_ms = _elapsed_ms(cache_read_started_at)

    if cached:
        cached_text = str(cached)
        await _record_json_metrics(
            _build_json_metrics_record(
                request_id=request_id,
                started_at=started_at,
                user_id=user_id,
                session_id=effective_session_id,
                user_prompt=user_prompt,
                safe_user_prompt=safe_user_prompt,
                answer=cached_text,
                outcome="cache_hit",
                build_context_ms=build_context_ms,
                retrieval_ms=retrieval_ms,
                model_ms=model_ms,
                memory_update_ms=memory_update_ms,
                cache_read_ms=cache_read_ms,
                cache_write_ms=cache_write_ms,
                evaluation_trace_ms=evaluation_trace_ms,
                retrieval_strategy=retrieval_strategy,
                retrieved_count=retrieved_count,
                retrieval_evidence=retrieval_evidence,
                quality=quality,
                llm_usage=llm_usage,
                input_guard_reason=input_guard_reason,
                context_guard_reason=context_guard_reason,
                output_guard_reason=output_guard_reason,
                used_fallback_model=used_fallback_model,
            )
        )
        await _record_latency_metrics(started_at, "cache_hit")
        yield cached_text
        return

    build_context_started_at = time.perf_counter()
    messages = await build_context(conversation_user_id, safe_user_prompt)
    build_context_ms = _elapsed_ms(build_context_started_at)

    retrieval_query = _build_retrieval_query(messages)
    if retrieval_query:
        retrieval_started_at = time.perf_counter()
        try:
            retrieval_result = await aretrieve_document_chunks(
                retrieval_query,
                top_k=settings.postgres.default_top_k,
            )
            retrieval_strategy = str(retrieval_result.get("retrieval_strategy", "unknown"))
            results = retrieval_result.get("results", [])
            if isinstance(results, list):
                retrieved_results = results
                retrieved_count = len(results)
                retrieval_evidence = _build_retrieval_evidence(results)
            retrieval_message = _format_retrieval_context(retrieval_result)
            if retrieval_message:
                messages = [retrieval_message] + messages
        except Exception as exc:
            retrieval_strategy = "error"
            logger.warning(
                "Long-term retrieval failed; continuing without retrieved context. %s",
                exc,
            )
        finally:
            retrieval_ms = _elapsed_ms(retrieval_started_at)
    else:
        retrieval_ms = 0

    chat_system_prompt = prompts.get("chat", {}).get("system_prompt", "")
    if isinstance(chat_system_prompt, str) and chat_system_prompt.strip():
        messages = [{"role": "system", "content": chat_system_prompt.strip()}] + messages

    context_guard = apply_context_guardrails(messages)
    if context_guard["blocked"]:
        context_guard_reason = str(context_guard.get("reason", "blocked_context"))
        refusal = refusal_response()
        logger.info(
            "GuardrailDecision | stage=context | user=%s | blocked=true | reason=%s",
            user_id,
            context_guard_reason,
        )
        await _record_json_metrics(
            _build_json_metrics_record(
                request_id=request_id,
                started_at=started_at,
                user_id=user_id,
                session_id=effective_session_id,
                user_prompt=user_prompt,
                safe_user_prompt=safe_user_prompt,
                answer=refusal,
                outcome="blocked_context",
                build_context_ms=build_context_ms,
                retrieval_ms=retrieval_ms,
                model_ms=model_ms,
                memory_update_ms=memory_update_ms,
                cache_read_ms=cache_read_ms,
                cache_write_ms=cache_write_ms,
                evaluation_trace_ms=evaluation_trace_ms,
                retrieval_strategy=retrieval_strategy,
                retrieved_count=retrieved_count,
                retrieval_evidence=retrieval_evidence,
                quality=quality,
                llm_usage=llm_usage,
                input_guard_reason=input_guard_reason,
                context_guard_reason=context_guard_reason,
                output_guard_reason=output_guard_reason,
                used_fallback_model=used_fallback_model,
            )
        )
        await _record_latency_metrics(started_at, "blocked_context")
        yield refusal
        return
    messages = context_guard["messages"]

    streamed_text = ""
    stream_guard_state: dict[str, object] = {
        "blocked": False,
        "reason": "",
        "final_text": "",
    }
    model_started_at = time.perf_counter()
    try:
        async for partial in _yield_guarded_stream(
            _stream_primary(messages),
            stream_state=stream_guard_state,
        ):
            streamed_text = partial
            yield partial
    except Exception as primary_exc:
        used_fallback_model = True
        logger.warning("Primary model stream failed; attempting fallback. %s", primary_exc)
        if streamed_text:
            model_ms = _elapsed_ms(model_started_at)
            await _record_json_metrics(
                _build_json_metrics_record(
                    request_id=request_id,
                    started_at=started_at,
                    user_id=user_id,
                    session_id=effective_session_id,
                    user_prompt=user_prompt,
                    safe_user_prompt=safe_user_prompt,
                    answer=streamed_text,
                    outcome="model_error",
                    build_context_ms=build_context_ms,
                    retrieval_ms=retrieval_ms,
                    model_ms=model_ms,
                    memory_update_ms=memory_update_ms,
                    cache_read_ms=cache_read_ms,
                    cache_write_ms=cache_write_ms,
                    evaluation_trace_ms=evaluation_trace_ms,
                    retrieval_strategy=retrieval_strategy,
                    retrieved_count=retrieved_count,
                    retrieval_evidence=retrieval_evidence,
                    quality=quality,
                    llm_usage=llm_usage,
                    input_guard_reason=input_guard_reason,
                    context_guard_reason=context_guard_reason,
                    output_guard_reason=output_guard_reason,
                    used_fallback_model=used_fallback_model,
                    error_message=str(primary_exc),
                )
            )
            await _record_latency_metrics(started_at, "model_error")
            raise
        try:
            stream_guard_state = {
                "blocked": False,
                "reason": "",
                "final_text": "",
            }
            async for partial in _yield_guarded_stream(
                _stream_fallback(messages),
                stream_state=stream_guard_state,
            ):
                streamed_text = partial
                yield partial
        except Exception as fallback_exc:
            logger.exception("Fallback model stream also failed.")
            model_ms = _elapsed_ms(model_started_at)
            await _record_json_metrics(
                _build_json_metrics_record(
                    request_id=request_id,
                    started_at=started_at,
                    user_id=user_id,
                    session_id=effective_session_id,
                    user_prompt=user_prompt,
                    safe_user_prompt=safe_user_prompt,
                    answer=streamed_text,
                    outcome="model_error",
                    build_context_ms=build_context_ms,
                    retrieval_ms=retrieval_ms,
                    model_ms=model_ms,
                    memory_update_ms=memory_update_ms,
                    cache_read_ms=cache_read_ms,
                    cache_write_ms=cache_write_ms,
                    evaluation_trace_ms=evaluation_trace_ms,
                    retrieval_strategy=retrieval_strategy,
                    retrieved_count=retrieved_count,
                    retrieval_evidence=retrieval_evidence,
                    quality=quality,
                    llm_usage=llm_usage,
                    input_guard_reason=input_guard_reason,
                    context_guard_reason=context_guard_reason,
                    output_guard_reason=output_guard_reason,
                    used_fallback_model=used_fallback_model,
                    error_message=str(fallback_exc),
                )
            )
            await _record_latency_metrics(started_at, "model_error")
            raise
    model_ms = _elapsed_ms(model_started_at)

    result = str(stream_guard_state.get("final_text", streamed_text) or streamed_text)
    if bool(stream_guard_state.get("blocked")):
        output_guard_reason = str(stream_guard_state.get("reason", "blocked_output"))
        logger.info(
            "GuardrailDecision | stage=output | user=%s | blocked=true | reason=%s",
            user_id,
            output_guard_reason,
        )

    memory_update_started_at = time.perf_counter()
    try:
        await update_memory(conversation_user_id, safe_user_prompt, result)
    except Exception as exc:
        logger.warning("Memory update failed. %s", exc)
    finally:
        memory_update_ms = _elapsed_ms(memory_update_started_at)

    cache_write_started_at = time.perf_counter()
    try:
        await _redis_call(
            async_redis_client.setex,
            cache_key,
            settings.memory.redis_ttl_seconds,
            result,
        )
    except RedisError as exc:
        logger.warning("Redis cache write failed. %s", exc)
    finally:
        cache_write_ms = _elapsed_ms(cache_write_started_at)

    evaluation_trace_started_at = time.perf_counter()
    _track_background_task(
        asyncio.create_task(
            _persist_evaluation_trace(
                user_id=user_id,
                prompt=user_prompt,
                answer=result,
                retrieved_results=retrieved_results,
                retrieval_strategy=retrieval_strategy,
                build_context_ms=build_context_ms or 0,
                retrieval_ms=retrieval_ms or 0,
                model_ms=model_ms or 0,
            )
        ),
        label="Evaluation trace persistence",
    )
    evaluation_trace_ms = _elapsed_ms(evaluation_trace_started_at)

    quality = {}
    await _record_pipeline_stage_metrics(
        build_context_ms=build_context_ms or 0,
        retrieval_ms=retrieval_ms or 0,
        model_ms=model_ms or 0,
        retrieval_strategy=retrieval_strategy,
        retrieved_count=retrieved_count,
    )
    await _record_json_metrics(
        _build_json_metrics_record(
            request_id=request_id,
            started_at=started_at,
            user_id=user_id,
            session_id=effective_session_id,
            user_prompt=user_prompt,
            safe_user_prompt=safe_user_prompt,
            answer=result,
            outcome="success",
            build_context_ms=build_context_ms,
            retrieval_ms=retrieval_ms,
            model_ms=model_ms,
            memory_update_ms=memory_update_ms,
            cache_read_ms=cache_read_ms,
            cache_write_ms=cache_write_ms,
            evaluation_trace_ms=evaluation_trace_ms,
            retrieval_strategy=retrieval_strategy,
            retrieved_count=retrieved_count,
            retrieval_evidence=retrieval_evidence,
            quality=quality,
            llm_usage=llm_usage,
            input_guard_reason=input_guard_reason,
            context_guard_reason=context_guard_reason,
            output_guard_reason=output_guard_reason,
            used_fallback_model=used_fallback_model,
        )
    )
    await _record_latency_metrics(started_at, "success")
