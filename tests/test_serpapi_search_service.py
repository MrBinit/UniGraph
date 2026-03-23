import pytest

from app.services import serpapi_search_service as service


def test_search_google_builds_expected_params(monkeypatch):
    monkeypatch.setattr(service.settings.serpapi, "enabled", True)
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")

    captured = {}

    def _fake_request_json(url: str, params: dict, timeout_seconds: float):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout_seconds
        return {"ok": True}

    monkeypatch.setattr(service, "_request_json", _fake_request_json)

    result = service.search_google("oxford ai masters", gl="us", hl="en", num=5)
    assert result["ok"] is True
    assert captured["url"] == service.settings.serpapi.google_search_url
    assert captured["params"]["engine"] == "google"
    assert captured["params"]["q"] == "oxford ai masters"
    assert captured["params"]["gl"] == "us"
    assert captured["params"]["hl"] == "en"
    assert captured["params"]["num"] == 5
    assert captured["params"]["api_key"] == "test-key"


@pytest.mark.asyncio
async def test_asearch_google_batch_uses_queue_workers(monkeypatch):
    monkeypatch.setattr(service.settings.serpapi, "enabled", True)
    monkeypatch.setattr(service.settings.serpapi, "queue_workers", 2)
    monkeypatch.setattr(service.settings.serpapi, "queue_max_size", 10)

    async def _fake_asearch_google(query: str, **kwargs):
        if query == "bad":
            raise RuntimeError("boom")
        return {"query_echo": query}

    monkeypatch.setattr(service, "asearch_google", _fake_asearch_google)
    results = await service.asearch_google_batch(["q1", "bad", "q2"])

    assert [item["query"] for item in results] == ["q1", "bad", "q2"]
    assert results[0]["result"]["query_echo"] == "q1"
    assert "boom" in results[1]["error"]
    assert results[2]["result"]["query_echo"] == "q2"
