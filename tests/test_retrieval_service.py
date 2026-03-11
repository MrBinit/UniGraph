import asyncio

from app.services import retrieval_service


def test_retrieve_document_chunks_returns_results_with_timing(monkeypatch):
    monkeypatch.setattr(retrieval_service, "embed_text", lambda query: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        retrieval_service,
        "resolve_document_chunk_search_strategy",
        lambda filters: "filtered_exact",
    )
    monkeypatch.setattr(
        retrieval_service,
        "search_document_chunks",
        lambda **kwargs: [
            {
                "chunk_id": "university_1:0001",
                "document_id": "university_1",
                "chunk_index": 1,
                "source_file": "university_1.md",
                "source_path": "/tmp/university_1.md",
                "content": "Master of Science in Artificial Intelligence Systems",
                "char_count": 52,
                "metadata": {"country": "Germany"},
                "distance": 0.05,
            }
        ],
    )

    result = retrieval_service.retrieve_document_chunks(
        "Find master's programs in Germany for AI systems.",
        top_k=4,
        metadata_filters={"country": "Germany"},
    )

    assert result["top_k"] == 4
    assert result["retrieval_strategy"] == "filtered_exact"
    assert result["metadata_filters"] == {"country": "Germany"}
    assert result["results"][0]["chunk_id"] == "university_1:0001"
    assert set(result["timings_ms"].keys()) == {"embedding", "database", "total"}
    assert result["timings_ms"]["total"] >= result["timings_ms"]["database"]


def test_aretrieve_document_chunks_uses_async_wrapper(monkeypatch):
    async def _fake_aembed_text(_query):
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(retrieval_service, "aembed_text", _fake_aembed_text)
    monkeypatch.setattr(
        retrieval_service,
        "resolve_document_chunk_search_strategy",
        lambda _filters: "ann",
    )

    async def _fake_search_document_chunks_async(**kwargs):
        assert kwargs["embedding"] == [0.1, 0.2, 0.3]
        return []

    monkeypatch.setattr(
        retrieval_service,
        "search_document_chunks_async",
        _fake_search_document_chunks_async,
    )

    result = asyncio.run(
        retrieval_service.aretrieve_document_chunks(
            "Find AI labs in Germany",
            top_k=3,
            metadata_filters={"country": "Germany"},
        )
    )

    assert result["query"] == "Find AI labs in Germany"
    assert result["top_k"] == 3
    assert result["metadata_filters"] == {"country": "Germany"}
    assert result["retrieval_strategy"] == "ann"
