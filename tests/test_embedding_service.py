import asyncio
import json
from io import BytesIO
from pathlib import Path

from app.services import embedding_service


class _FakeBedrockClient:
    def __init__(self, vector):
        self._vector = vector

    def invoke_model(self, **kwargs):
        payload = json.loads(kwargs["body"])
        assert "inputText" in payload
        return {
            "body": BytesIO(json.dumps({"embedding": self._vector}).encode("utf-8")),
        }


class _FakeBreaker:
    def __init__(self):
        self.calls = []

    def call(self, fn, *args, **kwargs):
        self.calls.append((fn, args, kwargs))
        return fn(*args, **kwargs)


class _FakeRedis:
    def __init__(self, initial=None):
        self.store = dict(initial or {})
        self.setex_calls = []

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.setex_calls.append((key, ttl, value))
        self.store[key] = value


def test_embed_text_returns_vector(monkeypatch):
    monkeypatch.setattr(embedding_service, "redis_client", _FakeRedis())
    monkeypatch.setattr(
        embedding_service,
        "get_bedrock_runtime_client",
        lambda: _FakeBedrockClient([0.1, 0.2, 0.3]),
    )

    vector = embedding_service.embed_text("Amazon Bedrock provides managed foundation models.")

    assert vector == [0.1, 0.2, 0.3]


def test_embed_text_uses_cached_embedding(monkeypatch):
    cache_key = embedding_service._embedding_cache_key("cached text")
    fake_redis = _FakeRedis({cache_key: json.dumps([7.0, 8.0, 9.0])})
    monkeypatch.setattr(embedding_service, "redis_client", fake_redis)

    class _FailingClient:
        def invoke_model(self, **kwargs):
            raise AssertionError("Bedrock should not be called on cache hit")

    monkeypatch.setattr(
        embedding_service,
        "get_bedrock_runtime_client",
        lambda: _FailingClient(),
    )

    vector = embedding_service.embed_text("cached text")

    assert vector == [7.0, 8.0, 9.0]
    assert fake_redis.setex_calls == []


def test_embed_text_caches_embedding_on_miss(monkeypatch):
    fake_redis = _FakeRedis()
    monkeypatch.setattr(embedding_service, "redis_client", fake_redis)
    monkeypatch.setattr(
        embedding_service,
        "get_bedrock_runtime_client",
        lambda: _FakeBedrockClient([4.0, 5.0, 6.0]),
    )

    vector = embedding_service.embed_text("cache miss")

    assert vector == [4.0, 5.0, 6.0]
    assert len(fake_redis.setex_calls) == 1
    cache_key, ttl, payload = fake_redis.setex_calls[0]
    assert cache_key == embedding_service._embedding_cache_key("cache miss")
    assert ttl == embedding_service.settings.embedding.cache_ttl_seconds
    assert json.loads(payload) == [4.0, 5.0, 6.0]


def test_embed_text_uses_embedding_circuit_breaker(monkeypatch):
    fake_redis = _FakeRedis()
    fake_breaker = _FakeBreaker()
    monkeypatch.setattr(embedding_service, "redis_client", fake_redis)
    monkeypatch.setattr(embedding_service, "get_embedding_breaker", lambda: fake_breaker)
    monkeypatch.setattr(
        embedding_service,
        "get_bedrock_runtime_client",
        lambda: _FakeBedrockClient([1.5, 2.5, 3.5]),
    )

    vector = embedding_service.embed_text("breaker path")

    assert vector == [1.5, 2.5, 3.5]
    assert len(fake_breaker.calls) == 1
    _fn, _args, kwargs = fake_breaker.calls[0]
    assert kwargs["modelId"] == embedding_service.settings.embedding.model_id
    assert json.loads(kwargs["body"]) == {"inputText": "breaker path"}


def test_embed_chunk_manifest_writes_embedding_output(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(embedding_service, "redis_client", _FakeRedis())
    chunk_manifest = tmp_path / "sample.chunks.json"
    chunk_manifest.write_text(
        json.dumps(
            {
                "source_file": "sample.md",
                "source_path": "/tmp/sample.md",
                "document_metadata": {"document_id": "sample"},
                "chunk_count": 1,
                "chunk_size_chars": 900,
                "chunk_overlap_chars": 120,
                "chunks": [
                    {
                        "chunk_id": "sample:0000",
                        "chunk_index": 0,
                        "source_file": "sample.md",
                        "source_path": "/tmp/sample.md",
                        "char_count": 42,
                        "metadata": {"document_id": "sample"},
                        "content": "Sample chunk for embedding.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        embedding_service,
        "get_bedrock_runtime_client",
        lambda: _FakeBedrockClient([1.0, 2.0, 3.0, 4.0]),
    )

    output_path = embedding_service.embed_chunk_manifest(chunk_manifest, tmp_path / "embeddings")
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.name == "sample.embeddings.json"
    assert payload["embedding_model"] == embedding_service.settings.embedding.model_id
    assert payload["embedding_dimensions"] == 4
    assert payload["chunks"][0]["embedding"] == [1.0, 2.0, 3.0, 4.0]


def test_aembed_text_uses_async_wrapper(monkeypatch):
    fake_redis = _FakeRedis()
    monkeypatch.setattr(embedding_service, "async_redis_client", fake_redis)

    async def _fake_ainvoke_model_json(_payload, timeout=None):
        assert timeout == embedding_service.settings.bedrock.timeout
        return {"embedding": [9.0, 8.0, 7.0]}

    monkeypatch.setattr(embedding_service, "ainvoke_model_json", _fake_ainvoke_model_json)

    vector = asyncio.run(embedding_service.aembed_text("async test"))

    assert vector == [9.0, 8.0, 7.0]
    assert len(fake_redis.setex_calls) == 1


def test_aembed_chunk_manifest_writes_embedding_output(tmp_path: Path, monkeypatch):
    chunk_manifest = tmp_path / "sample.chunks.json"
    chunk_manifest.write_text(
        json.dumps(
            {
                "source_file": "sample.md",
                "source_path": "/tmp/sample.md",
                "document_metadata": {"document_id": "sample"},
                "chunk_count": 2,
                "chunk_size_chars": 900,
                "chunk_overlap_chars": 120,
                "chunks": [
                    {
                        "chunk_id": "sample:0000",
                        "chunk_index": 0,
                        "source_file": "sample.md",
                        "source_path": "/tmp/sample.md",
                        "char_count": 21,
                        "metadata": {"document_id": "sample"},
                        "content": "Chunk one.",
                    },
                    {
                        "chunk_id": "sample:0001",
                        "chunk_index": 1,
                        "source_file": "sample.md",
                        "source_path": "/tmp/sample.md",
                        "char_count": 21,
                        "metadata": {"document_id": "sample"},
                        "content": "Chunk two.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    async def _fake_aembed_text(text: str) -> list[float]:
        return [float(len(text)), 1.0]

    monkeypatch.setattr(embedding_service, "aembed_text", _fake_aembed_text)

    output_path = asyncio.run(
        embedding_service.aembed_chunk_manifest(chunk_manifest, tmp_path / "embeddings")
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.name == "sample.embeddings.json"
    assert payload["embedding_dimensions"] == 2
    assert len(payload["chunks"]) == 2
    assert payload["chunks"][0]["embedding"] == [10.0, 1.0]
