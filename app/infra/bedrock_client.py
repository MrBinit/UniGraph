import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator

import boto3

from app.core.config import get_settings
from app.infra.circuit import get_embedding_breaker, get_llm_breaker

settings = get_settings()
_bedrock_runtime_client = None
_BEDROCK_EXECUTOR = ThreadPoolExecutor(
    max_workers=settings.io.bedrock_executor_workers,
    thread_name_prefix="bedrock-io",
)


def get_bedrock_runtime_client():
    """Return a cached Bedrock runtime client for the configured AWS region."""
    global _bedrock_runtime_client
    if _bedrock_runtime_client is None:
        _bedrock_runtime_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=settings.embedding.region_name,
        )
    return _bedrock_runtime_client


async def _run_in_bedrock_executor(fn, *, timeout: int | None = None):
    """Run one blocking Bedrock SDK operation on the dedicated Bedrock thread pool."""
    loop = asyncio.get_running_loop()
    result = loop.run_in_executor(_BEDROCK_EXECUTOR, fn)
    if timeout and int(timeout) > 0:
        return await asyncio.wait_for(result, timeout=float(timeout))
    return await result


async def aconverse(payload: dict[str, Any], *, timeout: int | None = None) -> dict[str, Any]:
    """Invoke Bedrock Converse asynchronously using the dedicated Bedrock executor."""

    def _invoke():
        client = get_bedrock_runtime_client()
        model_id = str(payload.get("modelId", "")).strip() or "default"
        return get_llm_breaker(model_id).call(client.converse, **payload)

    return await _run_in_bedrock_executor(_invoke, timeout=timeout)


async def ainvoke_model(
    payload: dict[str, Any],
    *,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Invoke Bedrock invoke_model asynchronously using the dedicated Bedrock executor."""

    def _invoke():
        client = get_bedrock_runtime_client()
        return get_embedding_breaker().call(client.invoke_model, **payload)

    return await _run_in_bedrock_executor(_invoke, timeout=timeout)


async def ainvoke_model_json(
    payload: dict[str, Any],
    *,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Invoke Bedrock and decode the model body into a JSON object in one async call."""

    def _invoke_and_decode():
        client = get_bedrock_runtime_client()
        response = get_embedding_breaker().call(client.invoke_model, **payload)
        raw_body = response.get("body").read()
        if isinstance(raw_body, bytes):
            raw_body = raw_body.decode("utf-8")
        payload_obj = json.loads(raw_body)
        if not isinstance(payload_obj, dict):
            raise ValueError("Bedrock response body must decode to a JSON object.")
        return payload_obj

    return await _run_in_bedrock_executor(_invoke_and_decode, timeout=timeout)


def _parse_converse_stream_event(event: dict[str, Any]) -> tuple[str, Exception | None]:
    """Extract text deltas or service-side stream errors from one ConverseStream event."""
    if not isinstance(event, dict):
        return "", None

    content_delta = event.get("contentBlockDelta")
    if isinstance(content_delta, dict):
        delta_payload = content_delta.get("delta")
        if isinstance(delta_payload, dict):
            delta_text = delta_payload.get("text")
            if isinstance(delta_text, str) and delta_text:
                return delta_text, None

    error_keys = (
        "internalServerException",
        "modelStreamErrorException",
        "validationException",
        "throttlingException",
        "serviceUnavailableException",
    )
    for key in error_keys:
        if key not in event:
            continue
        raw = event.get(key)
        if isinstance(raw, dict):
            message = str(raw.get("message") or raw)
        else:
            message = str(raw)
        return "", RuntimeError(f"Bedrock stream error ({key}): {message}")
    return "", None


async def aconverse_stream_text(
    payload: dict[str, Any],
    *,
    timeout: int | None = None,  # noqa: ARG001 - reserved for future stream timeout control
) -> AsyncIterator[str]:
    """Yield incremental text deltas from Bedrock ConverseStream."""
    del timeout
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()

    def _produce():
        try:
            client = get_bedrock_runtime_client()
            model_id = str(payload.get("modelId", "")).strip() or "default"
            response = get_llm_breaker(model_id).call(client.converse_stream, **payload)
            stream = response.get("stream", [])
            for event in stream:
                text, stream_error = _parse_converse_stream_event(event)
                if stream_error is not None:
                    loop.call_soon_threadsafe(queue.put_nowait, stream_error)
                    return
                if text:
                    loop.call_soon_threadsafe(queue.put_nowait, text)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    producer_task = loop.run_in_executor(_BEDROCK_EXECUTOR, _produce)
    try:
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield str(item)
    finally:
        await producer_task
