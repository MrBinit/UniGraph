import pytest

from app.services import web_retrieval_service as service


@pytest.mark.asyncio
async def test_aretrieve_web_chunks_merges_ai_overview_and_organic(monkeypatch):
    monkeypatch.setattr(service.settings.serpapi, "max_context_results", 3)
    monkeypatch.setattr(service.settings.serpapi, "max_page_chars", 1200)
    monkeypatch.setattr(service.settings.serpapi, "default_num", 10)
    monkeypatch.setattr(service.settings.serpapi, "multi_query_enabled", True)
    monkeypatch.setattr(service.settings.serpapi, "max_query_variants", 3)
    monkeypatch.setattr(service.settings.serpapi, "allowed_domain_suffixes", [])

    async def _fake_batch(queries: list[str], **kwargs):
        assert "oxford ai admission" in queries
        assert len(queries) >= 2
        return [
            {
                "query": queries[0],
                "result": {
                    "ai_overview": {
                        "title": "Summary",
                        "text": "Oxford admission requires strong profile.",
                    },
                    "organic_results": [
                        {
                            "title": "Oxford MSc AI",
                            "link": "https://example.edu/oxford-ai",
                            "snippet": "Entry requirements and deadlines.",
                        }
                    ],
                },
                "error": "",
            },
            {
                "query": queries[1],
                "result": {
                    "organic_results": [
                        {
                            "title": "Oxford MSc AI",
                            "link": "https://example.edu/oxford-ai",
                            "snippet": "Duplicate row from another variant.",
                        }
                    ],
                },
                "error": "",
            },
        ]

    async def _fake_fetch_pages(rows: list[dict]):
        assert rows[0]["url"] == "https://example.edu/oxford-ai"
        return {"https://example.edu/oxford-ai": "Detailed page content from source site."}

    async def _should_not_call_single(*_args, **_kwargs):
        raise AssertionError("single-query search should not run when multi-query is enabled")

    monkeypatch.setattr(service, "asearch_google_batch", _fake_batch)
    monkeypatch.setattr(service, "asearch_google", _should_not_call_single)
    monkeypatch.setattr(service, "_afetch_organic_pages", _fake_fetch_pages)

    result = await service.aretrieve_web_chunks("oxford ai admission", top_k=2)
    assert result["retrieval_strategy"] == "web_search"
    assert len(result["query_variants"]) >= 2
    assert len(result["results"]) >= 2
    assert "Oxford admission requires strong profile" in result["results"][0]["content"]
    assert any("Detailed page content" in item["content"] for item in result["results"])
    organic_items = [
        item
        for item in result["results"]
        if item.get("metadata", {}).get("source_type") == "google_organic"
    ]
    assert organic_items
    assert "title" in organic_items[0]["metadata"]
    assert "published_date" in organic_items[0]["metadata"]


@pytest.mark.asyncio
async def test_aretrieve_web_chunks_filters_to_allowed_domain_suffixes(monkeypatch):
    monkeypatch.setattr(service.settings.serpapi, "max_context_results", 4)
    monkeypatch.setattr(service.settings.serpapi, "max_page_chars", 1200)
    monkeypatch.setattr(service.settings.serpapi, "default_num", 10)
    monkeypatch.setattr(service.settings.serpapi, "multi_query_enabled", True)
    monkeypatch.setattr(service.settings.serpapi, "max_query_variants", 3)
    monkeypatch.setattr(service.settings.serpapi, "allowed_domain_suffixes", [".de", ".eu"])

    async def _fake_batch(queries: list[str], **kwargs):
        assert "eu ai universities" in queries
        return [
            {
                "query": queries[0],
                "result": {
                    "ai_overview": {
                        "title": "Summary",
                        "text": "Should be excluded when domain allowlist is active.",
                    },
                    "organic_results": [
                        {
                            "title": "LMU Munich AI",
                            "link": "https://www.lmu.de/programs/ai",
                            "snippet": "German source.",
                        },
                        {
                            "title": "US Blog",
                            "link": "https://example.com/ai",
                            "snippet": "Should be filtered.",
                        },
                    ],
                },
                "error": "",
            },
            {
                "query": queries[1],
                "result": {
                    "organic_results": [
                        {
                            "title": "EU Research",
                            "link": "https://research.example.eu/ai",
                            "snippet": "EU source.",
                        }
                    ],
                },
                "error": "",
            },
        ]

    async def _fake_fetch_pages(rows: list[dict]):
        assert len(rows) == 2
        urls = {row["url"] for row in rows}
        assert "https://www.lmu.de/programs/ai" in urls
        assert "https://research.example.eu/ai" in urls
        return {
            "https://www.lmu.de/programs/ai": "DE content",
            "https://research.example.eu/ai": "EU content",
        }

    monkeypatch.setattr(service, "asearch_google_batch", _fake_batch)
    monkeypatch.setattr(service, "_afetch_organic_pages", _fake_fetch_pages)

    result = await service.aretrieve_web_chunks("eu ai universities", top_k=3)
    assert result["retrieval_strategy"] == "web_search"
    assert len(result["results"]) == 2
    assert all(
        item["metadata"]["url"].endswith((".de/programs/ai", ".eu/ai"))
        for item in result["results"]
    )


@pytest.mark.asyncio
async def test_aretrieve_web_chunks_dedupes_same_url_from_multiple_variants(monkeypatch):
    monkeypatch.setattr(service.settings.serpapi, "max_context_results", 4)
    monkeypatch.setattr(service.settings.serpapi, "max_page_chars", 1200)
    monkeypatch.setattr(service.settings.serpapi, "default_num", 10)
    monkeypatch.setattr(service.settings.serpapi, "multi_query_enabled", True)
    monkeypatch.setattr(service.settings.serpapi, "max_query_variants", 3)
    monkeypatch.setattr(service.settings.serpapi, "allowed_domain_suffixes", [])

    async def _fake_batch(queries: list[str], **kwargs):
        return [
            {
                "query": queries[0],
                "result": {
                    "organic_results": [
                        {
                            "title": "RWTH Program",
                            "link": "https://www.rwth-aachen.de/ai",
                            "snippet": "Program overview.",
                        }
                    ],
                },
                "error": "",
            },
            {
                "query": queries[1],
                "result": {
                    "organic_results": [
                        {
                            "title": "RWTH Program Duplicate",
                            "link": "https://www.rwth-aachen.de/ai",
                            "snippet": "Same URL from another variant.",
                        },
                        {
                            "title": "TUM Program",
                            "link": "https://www.tum.de/ai",
                            "snippet": "Second unique URL.",
                        },
                    ],
                },
                "error": "",
            },
        ]

    async def _fake_fetch_pages(rows: list[dict]):
        urls = [row["url"] for row in rows]
        assert urls.count("https://www.rwth-aachen.de/ai") == 1
        return {
            "https://www.rwth-aachen.de/ai": "RWTH details",
            "https://www.tum.de/ai": "TUM details",
        }

    monkeypatch.setattr(service, "asearch_google_batch", _fake_batch)
    monkeypatch.setattr(service, "_afetch_organic_pages", _fake_fetch_pages)

    result = await service.aretrieve_web_chunks("germany ai program", top_k=3)
    urls = [
        item.get("metadata", {}).get("url", "")
        for item in result["results"]
        if item.get("metadata")
    ]
    assert urls.count("https://www.rwth-aachen.de/ai") == 1


@pytest.mark.asyncio
async def test_aretrieve_web_chunks_preserves_published_date(monkeypatch):
    monkeypatch.setattr(service.settings.serpapi, "max_context_results", 3)
    monkeypatch.setattr(service.settings.serpapi, "max_page_chars", 1200)
    monkeypatch.setattr(service.settings.serpapi, "default_num", 10)
    monkeypatch.setattr(service.settings.serpapi, "multi_query_enabled", False)
    monkeypatch.setattr(service.settings.serpapi, "allowed_domain_suffixes", [])

    async def _fake_single(_query: str, **_kwargs):
        return {
            "organic_results": [
                {
                    "title": "University News",
                    "link": "https://www.example.edu/news/ai",
                    "snippet": "Scholarship updates.",
                    "date": "2026-03-20",
                }
            ]
        }

    async def _fake_fetch_pages(rows: list[dict]):
        assert len(rows) == 1
        return {
            "https://www.example.edu/news/ai": {
                "content": "Scholarship updates and eligibility details.",
                "published_date": "2026-03-19",
            }
        }

    monkeypatch.setattr(service, "asearch_google", _fake_single)
    monkeypatch.setattr(service, "_afetch_organic_pages", _fake_fetch_pages)

    result = await service.aretrieve_web_chunks("example scholarship update", top_k=2)
    organic = [
        item
        for item in result["results"]
        if item.get("metadata", {}).get("source_type") == "google_organic"
    ]
    assert organic
    assert organic[0]["metadata"]["published_date"] == "2026-03-20"
    assert organic[0]["metadata"]["title"] == "University News"
