import asyncio
import logging
import time
from typing import AsyncIterator
from uuid import uuid4

from redis.exceptions import RedisError

from app.core.config import get_prompts, get_settings
from app.infra.azure_openai_client import client
from app.infra.redis_client import app_scoped_key, redis_client
from app.services.evaluation_service import store_chat_trace
from app.services.guardrails_service import (
    apply_context_guardrails,
    guard_model_output,
    guard_user_input,
    refusal_response,
)
from app.services.memory_service import build_context, update_memory
from app.services.metrics_json_service import append_chat_metrics_json
from app.services.quality_metrics_service import generation_metrics
from app.services.retrieval_service import aretrieve_document_chunks

settings = get_settings()
prompts = get_prompts()
logger = logging.getLogger(__name__)

SEMAPHORE = asyncio.Semaphore(settings.azure_openai.max_concurrency)
_BACKGROUND_TASKS: set[asyncio.Task] = set()
_RETRIEVAL_QUERY_MAX_CHARS = 900
_RETRIEVAL_CONTEXT_MAX_CHARS = 1500
_RETRIEVAL_CHUNK_MAX_CHARS = 360
_RETRIEVAL_MAX_PROMPT_RESULTS = 2


async def _redis_call(method, *args, **kwargs):
    """Run a blocking Redis operation without blocking the event loop."""
    return await asyncio.to_thread(method, *args, **kwargs)


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
    """Persist per-request chat metrics to JSON files without blocking the event loop."""
    try:
        await asyncio.to_thread(append_chat_metrics_json, record)
    except Exception:
        logger.warning("JSON metrics persistence failed; continuing.")


def _build_json_metrics_record(
    *,
    request_id: str,
    started_at: float,
    user_id: str,
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
        "session_id": user_id,
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
            "provider": "azure_openai",
            "used_fallback": used_fallback_model,
            "primary_deployment": settings.azure_openai.primary_deployment,
            "fallback_deployment": settings.azure_openai.fallback_deployment,
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
        await _redis_call(redis_client.hincrby, key, "count", 1)
        await _redis_call(redis_client.hincrby, key, "total_ms", latency_ms)
        current_max = await _redis_call(redis_client.hget, key, "max_ms")
        if current_max is None or latency_ms > int(current_max):
            await _redis_call(redis_client.hset, key, "max_ms", latency_ms)
        await _redis_call(
            redis_client.hset,
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
        await _redis_call(redis_client.hincrby, key, "pipeline_count", 1)
        await _redis_call(redis_client.hincrby, key, "build_context_total_ms", build_context_ms)
        await _redis_call(redis_client.hincrby, key, "retrieval_total_ms", retrieval_ms)
        await _redis_call(redis_client.hincrby, key, "model_total_ms", model_ms)
        await _redis_call(
            redis_client.hset,
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

    lines = ["Retrieved context (use only if relevant):"]
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
    request_id = uuid4().hex
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
    cache_key = app_scoped_key("cache", "chat", user_id, safe_user_prompt)

    # Cache lookup
    cached = None
    cache_read_started_at = time.perf_counter()
    try:
        cached = await _redis_call(redis_client.get, cache_key)
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
    messages = await build_context(user_id, safe_user_prompt)
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
    async with SEMAPHORE:
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
        await update_memory(user_id, safe_user_prompt, result)
    except Exception as exc:
        logger.warning("Memory update failed. %s", exc)
    finally:
        memory_update_ms = _elapsed_ms(memory_update_started_at)

    # Cache result
    cache_write_started_at = time.perf_counter()
    try:
        await _redis_call(redis_client.setex, cache_key, settings.memory.redis_ttl_seconds, result)
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

    try:
        quality = generation_metrics(
            query=user_prompt,
            answer=result,
            retrieved_results=retrieved_results,
        )
    except Exception as exc:
        quality = {}
        logger.warning("Quality metrics computation failed; skipping metrics. %s", exc)

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
    *,
    chunk_size: int = 120,
    chunk_delay_ms: int = 12,
) -> AsyncIterator[str]:
    """Yield a progressively growing answer string for Gradio streaming UX.

    This streams the already-validated final answer text in chunks so guardrail,
    memory, cache, and metrics behavior remains identical to `generate_response`.
    """
    result = await generate_response(user_id, user_prompt)
    if not isinstance(result, str):
        result = str(result)

    size = max(1, int(chunk_size))
    delay_seconds = max(0.0, float(chunk_delay_ms) / 1000.0)
    assembled = ""
    for start in range(0, len(result), size):
        assembled += result[start : start + size]
        yield assembled
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
