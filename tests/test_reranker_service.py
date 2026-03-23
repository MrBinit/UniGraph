import pytest

from app.services import reranker_service as service


@pytest.mark.asyncio
async def test_arerank_retrieval_results_returns_passthrough_when_disabled(monkeypatch):
    monkeypatch.setattr(service.settings.bedrock, "reranker_enabled", False)
    candidates = [{"content": "A"}, {"content": "B"}]
    result = await service.arerank_retrieval_results("query", candidates)
    assert result["applied"] is False
    assert result["results"] == candidates


@pytest.mark.asyncio
async def test_arerank_retrieval_results_sorts_by_ranked_indices(monkeypatch):
    monkeypatch.setattr(service.settings.bedrock, "reranker_enabled", True)
    monkeypatch.setattr(service.settings.bedrock, "reranker_model_id", "cohere.rerank-v3-5:0")
    monkeypatch.setattr(service.settings.bedrock, "reranker_top_n", 2)
    monkeypatch.setattr(service.settings.bedrock, "reranker_min_documents", 2)
    monkeypatch.setattr(service.settings.bedrock, "reranker_max_documents", 10)
    monkeypatch.setattr(service.settings.bedrock, "reranker_max_query_chars", 700)
    monkeypatch.setattr(service.settings.bedrock, "reranker_max_document_chars", 1400)
    monkeypatch.setattr(
        service.boto3.session,
        "Session",
        lambda: type("FakeSession", (), {"get_credentials": lambda self: object()})(),
    )

    async def _fake_invoke_model_json(payload):
        assert payload["modelId"] == "cohere.rerank-v3-5:0"
        return {
            "results": [
                {"index": 2, "relevance_score": 0.91},
                {"index": 0, "relevance_score": 0.73},
            ]
        }

    candidates = [
        {"chunk_id": "v1", "content": "Vector one", "metadata": {"university": "U1"}},
        {"chunk_id": "v2", "content": "Vector two", "metadata": {"university": "U2"}},
        {"chunk_id": "w1", "content": "Web top", "metadata": {"url": "https://x.de/a"}},
    ]

    monkeypatch.setattr(service, "ainvoke_model_json", _fake_invoke_model_json)
    result = await service.arerank_retrieval_results("best result", candidates)
    assert result["applied"] is True
    assert [item["chunk_id"] for item in result["results"]] == ["w1", "v1"]
    assert "rerank_score" in result["results"][0]
