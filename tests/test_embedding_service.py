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


def test_embed_text_returns_vector(monkeypatch):
    monkeypatch.setattr(
        embedding_service,
        "get_bedrock_runtime_client",
        lambda: _FakeBedrockClient([0.1, 0.2, 0.3]),
    )

    vector = embedding_service.embed_text("Amazon Bedrock provides managed foundation models.")

    assert vector == [0.1, 0.2, 0.3]


def test_embed_chunk_manifest_writes_embedding_output(tmp_path: Path, monkeypatch):
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
    monkeypatch.setattr(embedding_service, "embed_text", lambda text: [9.0, 8.0, 7.0])

    vector = asyncio.run(embedding_service.aembed_text("async test"))

    assert vector == [9.0, 8.0, 7.0]
