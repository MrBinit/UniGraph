import asyncio
import json
from io import BytesIO

import pytest

from app.infra import bedrock_client
from app.infra.circuit import CircuitBreakerError


class _FakeBreaker:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls = []

    def call(self, fn, *args, **kwargs):
        self.calls.append((fn, args, kwargs))
        if self.error is not None:
            raise self.error
        return fn(*args, **kwargs)


class _FakeBedrockRuntimeClient:
    def converse(self, **payload):
        return {
            "payload": payload,
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }

    def invoke_model(self, **_payload):
        return {"body": BytesIO(json.dumps({"embedding": [0.1, 0.2]}).encode("utf-8"))}

    def converse_stream(self, **_payload):
        return {
            "stream": [
                {"contentBlockDelta": {"delta": {"text": "hel"}}},
                {"contentBlockDelta": {"delta": {"text": "lo"}}},
            ]
        }


async def _run_inline(fn, *, timeout=None):  # noqa: ARG001
    return fn()


def test_aconverse_uses_model_scoped_circuit_breaker(monkeypatch):
    fake_breaker = _FakeBreaker()
    fake_client = _FakeBedrockRuntimeClient()
    monkeypatch.setattr(bedrock_client, "_run_in_bedrock_executor", _run_inline)
    monkeypatch.setattr(bedrock_client, "get_bedrock_runtime_client", lambda: fake_client)
    monkeypatch.setattr(bedrock_client, "get_llm_breaker", lambda _model_id: fake_breaker)

    payload = {
        "modelId": "test-model",
        "messages": [{"role": "user", "content": [{"text": "hi"}]}],
    }
    response = asyncio.run(bedrock_client.aconverse(payload, timeout=3))

    assert response["output"]["message"]["content"][0]["text"] == "ok"
    assert len(fake_breaker.calls) == 1
    _fn, _args, kwargs = fake_breaker.calls[0]
    assert kwargs["modelId"] == "test-model"


def test_ainvoke_model_json_uses_embedding_circuit_breaker(monkeypatch):
    fake_breaker = _FakeBreaker()
    fake_client = _FakeBedrockRuntimeClient()
    monkeypatch.setattr(bedrock_client, "_run_in_bedrock_executor", _run_inline)
    monkeypatch.setattr(bedrock_client, "get_bedrock_runtime_client", lambda: fake_client)
    monkeypatch.setattr(bedrock_client, "get_embedding_breaker", lambda: fake_breaker)

    payload = {
        "modelId": "embed-model",
        "body": json.dumps({"inputText": "hello"}),
        "contentType": "application/json",
        "accept": "application/json",
    }
    response = asyncio.run(bedrock_client.ainvoke_model_json(payload, timeout=4))

    assert response == {"embedding": [0.1, 0.2]}
    assert len(fake_breaker.calls) == 1
    _fn, _args, kwargs = fake_breaker.calls[0]
    assert kwargs["modelId"] == "embed-model"


def test_ainvoke_model_json_propagates_open_circuit(monkeypatch):
    fake_breaker = _FakeBreaker(error=CircuitBreakerError("open"))
    fake_client = _FakeBedrockRuntimeClient()
    monkeypatch.setattr(bedrock_client, "_run_in_bedrock_executor", _run_inline)
    monkeypatch.setattr(bedrock_client, "get_bedrock_runtime_client", lambda: fake_client)
    monkeypatch.setattr(bedrock_client, "get_embedding_breaker", lambda: fake_breaker)

    payload = {
        "modelId": "embed-model",
        "body": json.dumps({"inputText": "hello"}),
        "contentType": "application/json",
        "accept": "application/json",
    }
    with pytest.raises(CircuitBreakerError):
        asyncio.run(bedrock_client.ainvoke_model_json(payload, timeout=4))


def test_aconverse_stream_text_yields_deltas(monkeypatch):
    fake_breaker = _FakeBreaker()
    fake_client = _FakeBedrockRuntimeClient()
    monkeypatch.setattr(bedrock_client, "get_bedrock_runtime_client", lambda: fake_client)
    monkeypatch.setattr(bedrock_client, "get_llm_breaker", lambda _model_id: fake_breaker)

    async def _collect():
        chunks = []
        payload = {
            "modelId": "test-model",
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
        }
        async for delta in bedrock_client.aconverse_stream_text(payload):
            chunks.append(delta)
        return chunks

    chunks = asyncio.run(_collect())
    assert chunks == ["hel", "lo"]
    assert len(fake_breaker.calls) == 1
