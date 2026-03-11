import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import boto3

from app.core.config import get_settings

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
        return client.converse(**payload)

    return await _run_in_bedrock_executor(_invoke, timeout=timeout)


async def ainvoke_model(
    payload: dict[str, Any],
    *,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Invoke Bedrock invoke_model asynchronously using the dedicated Bedrock executor."""

    def _invoke():
        client = get_bedrock_runtime_client()
        return client.invoke_model(**payload)

    return await _run_in_bedrock_executor(_invoke, timeout=timeout)


async def ainvoke_model_json(
    payload: dict[str, Any],
    *,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Invoke Bedrock and decode the model body into a JSON object in one async call."""

    def _invoke_and_decode():
        client = get_bedrock_runtime_client()
        response = client.invoke_model(**payload)
        raw_body = response.get("body").read()
        if isinstance(raw_body, bytes):
            raw_body = raw_body.decode("utf-8")
        payload_obj = json.loads(raw_body)
        if not isinstance(payload_obj, dict):
            raise ValueError("Bedrock response body must decode to a JSON object.")
        return payload_obj

    return await _run_in_bedrock_executor(_invoke_and_decode, timeout=timeout)
