import pytest

from app.infra.io_limiters import DependencyBackpressureError
from app.services import web_retrieval_service as service


@pytest.fixture(autouse=True)
def _disable_llm_planner_by_default(monkeypatch):
    monkeypatch.setattr(service.settings.web_search, "query_planner_use_llm", False)
    monkeypatch.setattr(service.settings.web_search, "retrieval_loop_use_llm", False)
    monkeypatch.setattr(service.settings.web_search, "retrieval_min_unique_domains", 1)
    monkeypatch.setattr(service.settings.web_search, "deep_min_unique_domains", 1)
    monkeypatch.setattr(service.settings.web_search, "official_source_filter_enabled", False)


@pytest.mark.asyncio
async def test_aretrieve_web_chunks_merges_ai_overview_and_organic(monkeypatch):
    monkeypatch.setattr(service.settings.web_search, "max_context_results", 3)
    monkeypatch.setattr(service.settings.web_search, "max_page_chars", 1200)
    monkeypatch.setattr(service.settings.web_search, "default_num", 10)
    monkeypatch.setattr(service.settings.web_search, "multi_query_enabled", True)
    monkeypatch.setattr(service.settings.web_search, "max_query_variants", 3)
    monkeypatch.setattr(service.settings.web_search, "allowed_domain_suffixes", [])

    async def _fake_batch(queries: list[str], **kwargs):
        assert any("oxford ai admission" in item for item in queries)
        assert len(queries) >= 1
        responses = [
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
                            "link": "https://uni-example.de/oxford-ai",
                            "snippet": "Entry requirements and deadlines.",
                        }
                    ],
                },
                "error": "",
            }
        ]
        if len(queries) > 1:
            responses.append(
                {
                    "query": queries[1],
                    "result": {
                        "organic_results": [
                            {
                                "title": "Oxford MSc AI",
                                "link": "https://uni-example.de/oxford-ai",
                                "snippet": "Duplicate row from another variant.",
                            }
                        ],
                    },
                    "error": "",
                }
            )
        return responses

    async def _fake_fetch_pages(rows: list[dict]):
        assert rows[0]["url"] == "https://uni-example.de/oxford-ai"
        return {"https://uni-example.de/oxford-ai": "Detailed page content from source site."}

    async def _should_not_call_single(*_args, **_kwargs):
        raise AssertionError("single-query search should not run when multi-query is enabled")

    monkeypatch.setattr(service, "asearch_google_batch", _fake_batch)
    monkeypatch.setattr(service, "asearch_google", _should_not_call_single)
    monkeypatch.setattr(service, "_afetch_organic_pages", _fake_fetch_pages)

    result = await service.aretrieve_web_chunks("oxford ai admission", top_k=2)
    assert result["retrieval_strategy"] == "web_search"
    assert len(result["query_variants"]) >= 2
    assert len(result["results"]) >= 1
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
    monkeypatch.setattr(service.settings.web_search, "max_context_results", 4)
    monkeypatch.setattr(service.settings.web_search, "max_page_chars", 1200)
    monkeypatch.setattr(service.settings.web_search, "default_num", 10)
    monkeypatch.setattr(service.settings.web_search, "multi_query_enabled", True)
    monkeypatch.setattr(service.settings.web_search, "max_query_variants", 3)
    monkeypatch.setattr(service.settings.web_search, "allowed_domain_suffixes", [".de", ".eu"])

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
                            "title": "EU Research",
                            "link": "https://research.example.eu/ai",
                            "snippet": "Should be filtered because it is not an official university page.",
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
                            "link": "https://www2.daad.de/programmes/ai",
                            "snippet": "DAAD source.",
                        }
                    ],
                },
                "error": "",
            },
        ]

    async def _fake_fetch_pages(rows: list[dict]):
        assert len(rows) == 3
        urls = {row["url"] for row in rows}
        assert "https://www.lmu.de/programs/ai" in urls
        assert "https://www2.daad.de/programmes/ai" in urls
        assert "https://research.example.eu/ai" in urls
        return {
            "https://www.lmu.de/programs/ai": "DE content",
            "https://www2.daad.de/programmes/ai": "DAAD content",
            "https://research.example.eu/ai": "EU content",
        }

    monkeypatch.setattr(service, "asearch_google_batch", _fake_batch)
    monkeypatch.setattr(service, "_afetch_organic_pages", _fake_fetch_pages)

    result = await service.aretrieve_web_chunks("eu ai universities", top_k=3)
    assert result["retrieval_strategy"] == "web_search"
    assert len(result["results"]) == 3
    hosts = {
        service._domain_group_key(
            service._normalized_host(str(item.get("metadata", {}).get("url", "")))
        )
        for item in result["results"]
    }
    assert hosts == {"lmu.de", "daad.de", "example.eu"}


@pytest.mark.asyncio
async def test_aretrieve_web_chunks_dedupes_same_url_from_multiple_variants(monkeypatch):
    monkeypatch.setattr(service.settings.web_search, "max_context_results", 4)
    monkeypatch.setattr(service.settings.web_search, "max_page_chars", 1200)
    monkeypatch.setattr(service.settings.web_search, "default_num", 10)
    monkeypatch.setattr(service.settings.web_search, "multi_query_enabled", True)
    monkeypatch.setattr(service.settings.web_search, "max_query_variants", 3)
    monkeypatch.setattr(service.settings.web_search, "allowed_domain_suffixes", [])

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
    monkeypatch.setattr(service.settings.web_search, "max_context_results", 3)
    monkeypatch.setattr(service.settings.web_search, "max_page_chars", 1200)
    monkeypatch.setattr(service.settings.web_search, "default_num", 10)
    monkeypatch.setattr(service.settings.web_search, "multi_query_enabled", False)
    monkeypatch.setattr(service.settings.web_search, "allowed_domain_suffixes", [])

    async def _fake_single(_query: str, **_kwargs):
        return {
            "organic_results": [
                {
                    "title": "University News",
                    "link": "https://uni-example.de/news/ai",
                    "snippet": "Scholarship updates.",
                    "date": "2026-03-20",
                }
            ]
        }

    async def _fake_fetch_pages(rows: list[dict]):
        assert len(rows) == 1
        return {
            "https://uni-example.de/news/ai": {
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


@pytest.mark.asyncio
async def test_aretrieve_web_chunks_requeries_for_missing_subquestions(monkeypatch):
    monkeypatch.setattr(service.settings.web_search, "max_context_results", 3)
    monkeypatch.setattr(service.settings.web_search, "query_planner_enabled", True)
    monkeypatch.setattr(service.settings.web_search, "query_planner_use_llm", False)
    monkeypatch.setattr(service.settings.web_search, "retrieval_loop_enabled", True)
    monkeypatch.setattr(service.settings.web_search, "retrieval_loop_max_steps", 2)
    monkeypatch.setattr(service.settings.web_search, "retrieval_loop_max_gap_queries", 1)
    monkeypatch.setattr(service.settings.web_search, "retrieval_gap_min_token_coverage", 0.6)
    monkeypatch.setattr(service.settings.web_search, "allowed_domain_suffixes", [])

    async def _fake_plan(_query: str, _allowed_suffixes: list[str]):
        return {
            "queries": ["germany ai admissions"],
            "subquestions": ["tuition fees", "language requirement"],
            "planner": "heuristic",
            "llm_used": False,
        }

    calls = {"count": 0}

    async def _fake_payloads(queries: list[str], *, top_k: int):
        calls["count"] += 1
        if any("language requirement" in query.lower() for query in queries):
            return [
                {
                    "organic_results": [
                        {
                            "title": "Language Requirement",
                            "link": "https://uni-example.de/language",
                            "snippet": "English language requirement IELTS 6.5",
                        }
                    ]
                }
            ]
        return [
            {
                "organic_results": [
                    {
                        "title": "Tuition Details",
                        "link": "https://uni-example.de/tuition",
                        "snippet": "Tuition fees are EUR 2000.",
                    }
                ]
            }
        ]

    async def _fake_fetch_pages(rows: list[dict]):
        urls = {row["url"] for row in rows}
        payload = {}
        if "https://uni-example.de/tuition" in urls:
            payload["https://uni-example.de/tuition"] = {
                "content": "Tuition fees are EUR 2000 per semester.",
                "published_date": "2026-02-01",
            }
        if "https://uni-example.de/language" in urls:
            payload["https://uni-example.de/language"] = {
                "content": "Language requirement: IELTS 6.5 overall.",
                "published_date": "2026-02-02",
            }
        return payload

    monkeypatch.setattr(service, "_resolve_query_plan", _fake_plan)
    monkeypatch.setattr(service, "_asearch_payloads", _fake_payloads)
    monkeypatch.setattr(service, "_afetch_organic_pages", _fake_fetch_pages)

    result = await service.aretrieve_web_chunks("germany ai admissions", top_k=3)

    assert calls["count"] == 2
    assert result["retrieval_loop"]["iterations"] == 2
    facts_text = " ".join(fact["fact"].lower() for fact in result["facts"])
    assert "tuition fees" in facts_text
    assert "language requirement" in facts_text


@pytest.mark.asyncio
async def test_aretrieve_web_chunks_requeries_for_domain_diversity(monkeypatch):
    monkeypatch.setattr(service.settings.web_search, "max_context_results", 4)
    monkeypatch.setattr(service.settings.web_search, "query_planner_enabled", True)
    monkeypatch.setattr(service.settings.web_search, "query_planner_use_llm", False)
    monkeypatch.setattr(service.settings.web_search, "retrieval_loop_enabled", True)
    monkeypatch.setattr(service.settings.web_search, "retrieval_loop_max_steps", 2)
    monkeypatch.setattr(service.settings.web_search, "retrieval_loop_max_gap_queries", 2)
    monkeypatch.setattr(service.settings.web_search, "retrieval_min_unique_domains", 2)
    monkeypatch.setattr(service.settings.web_search, "allowed_domain_suffixes", [])

    async def _fake_plan(_query: str, _allowed_suffixes: list[str]):
        return {
            "queries": ["germany ai admissions"],
            "subquestions": [],
            "planner": "heuristic",
            "llm_used": False,
        }

    calls: list[list[str]] = []

    async def _fake_payloads(queries: list[str], *, top_k: int):
        calls.append(list(queries))
        if any(query.strip().lower() != "germany ai admissions" for query in queries):
            return [
                {
                    "organic_results": [
                        {
                            "title": "Second Source",
                            "link": "https://uni-second.de/admissions",
                            "snippet": "Independent confirmation.",
                        }
                    ]
                }
            ]
        return [
            {
                "organic_results": [
                    {
                        "title": "Primary Source",
                        "link": "https://uni-first.de/admissions",
                        "snippet": "Official admissions details.",
                    }
                ]
            }
        ]

    async def _fake_fetch_pages(rows: list[dict]):
        payload = {}
        for row in rows:
            url = row["url"]
            payload[url] = {
                "content": f"Admissions details from {url}.",
                "published_date": "2026-03-01",
            }
        return payload

    monkeypatch.setattr(service, "_resolve_query_plan", _fake_plan)
    monkeypatch.setattr(service, "_asearch_payloads", _fake_payloads)
    monkeypatch.setattr(service, "_afetch_organic_pages", _fake_fetch_pages)

    result = await service.aretrieve_web_chunks("germany ai admissions", top_k=3)

    assert len(calls) == 2
    assert result["retrieval_loop"]["iterations"] == 2
    assert result["verification"]["unique_domain_count"] >= 2
    assert result["verification"]["verified"] is True


@pytest.mark.asyncio
async def test_aretrieve_web_chunks_adds_trust_score_metadata(monkeypatch):
    monkeypatch.setattr(service.settings.web_search, "max_context_results", 2)
    monkeypatch.setattr(service.settings.web_search, "query_planner_enabled", False)
    monkeypatch.setattr(service.settings.web_search, "multi_query_enabled", False)
    monkeypatch.setattr(service.settings.web_search, "allowed_domain_suffixes", [])

    async def _fake_single(_query: str, **_kwargs):
        return {
            "organic_results": [
                {
                    "title": "Official Program Page",
                    "link": "https://uni-example.de/programs/ai",
                    "snippet": "Admission requirements and curriculum.",
                }
            ]
        }

    async def _fake_fetch_pages(_rows: list[dict]):
        return {
            "https://uni-example.de/programs/ai": {
                "content": "Admission requirements include transcripts and language proof.",
                "published_date": "2026-03-01",
            }
        }

    monkeypatch.setattr(service, "asearch_google", _fake_single)
    monkeypatch.setattr(service, "_afetch_organic_pages", _fake_fetch_pages)

    result = await service.aretrieve_web_chunks("example ai admission", top_k=2)
    assert result["results"]
    metadata = result["results"][0]["metadata"]
    assert "trust_score" in metadata
    assert metadata["trust_score"] >= 0.0
    assert "trust_components" in metadata


@pytest.mark.asyncio
async def test_aretrieve_web_chunks_uses_llm_gap_queries_when_enabled(monkeypatch):
    monkeypatch.setattr(service.settings.web_search, "query_planner_enabled", True)
    monkeypatch.setattr(service.settings.web_search, "retrieval_loop_enabled", True)
    monkeypatch.setattr(service.settings.web_search, "retrieval_loop_use_llm", True)
    monkeypatch.setattr(service.settings.web_search, "retrieval_loop_max_steps", 2)
    monkeypatch.setattr(service.settings.web_search, "retrieval_loop_max_gap_queries", 1)
    monkeypatch.setattr(service.settings.web_search, "allowed_domain_suffixes", [])

    async def _fake_plan(_query: str, _allowed_suffixes: list[str]):
        return {
            "queries": ["germany ai admissions"],
            "subquestions": ["tuition fees", "language requirement"],
            "planner": "llm",
            "llm_used": True,
        }

    async def _fake_gap_plan(_query: str, **_kwargs):
        return {
            "missing_subquestions": ["language requirement"],
            "queries": ["germany ai admissions official language requirement"],
        }

    call_queries: list[list[str]] = []

    async def _fake_payloads(queries: list[str], *, top_k: int):
        call_queries.append(list(queries))
        if any("official language requirement" in query.lower() for query in queries):
            return [
                {
                    "organic_results": [
                        {
                            "title": "Language Requirement",
                            "link": "https://uni-example.de/lang",
                            "snippet": "English language requirement IELTS 6.5",
                        }
                    ]
                }
            ]
        return [
            {
                "organic_results": [
                    {
                        "title": "Tuition",
                        "link": "https://uni-example.de/tuition",
                        "snippet": "Tuition fees are EUR 2000.",
                    }
                ]
            }
        ]

    async def _fake_fetch_pages(rows: list[dict]):
        urls = {row["url"] for row in rows}
        payload = {}
        if "https://uni-example.de/tuition" in urls:
            payload["https://uni-example.de/tuition"] = {
                "content": "Tuition fees are EUR 2000.",
                "published_date": "2026-02-01",
            }
        if "https://uni-example.de/lang" in urls:
            payload["https://uni-example.de/lang"] = {
                "content": "Language requirement IELTS 6.5.",
                "published_date": "2026-02-03",
            }
        return payload

    monkeypatch.setattr(service, "_resolve_query_plan", _fake_plan)
    monkeypatch.setattr(service, "_aidentify_gap_plan_with_llm", _fake_gap_plan)
    monkeypatch.setattr(service, "_asearch_payloads", _fake_payloads)
    monkeypatch.setattr(service, "_afetch_organic_pages", _fake_fetch_pages)

    result = await service.aretrieve_web_chunks("germany ai admissions", top_k=3)

    assert len(call_queries) == 2
    assert any("official language requirement" in query.lower() for query in call_queries[1])
    assert result["retrieval_loop"]["llm_used"] is True


@pytest.mark.asyncio
async def test_aretrieve_web_chunks_runs_llm_query_planner_before_search(monkeypatch):
    monkeypatch.setattr(service.settings.web_search, "query_planner_enabled", True)
    monkeypatch.setattr(service.settings.web_search, "query_planner_use_llm", True)
    monkeypatch.setattr(service.settings.web_search, "retrieval_loop_enabled", False)
    monkeypatch.setattr(service.settings.web_search, "allowed_domain_suffixes", [])

    call_order: list[str] = []

    async def _fake_llm_plan(query: str, allowed_suffixes: list[str]):
        assert query == "germany ai admissions"
        assert allowed_suffixes == []
        call_order.append("planner")
        return {
            "queries": ["germany ai admissions official site"],
            "subquestions": ["tuition fees"],
            "planner": "llm",
            "llm_used": True,
        }

    async def _fake_payloads(queries: list[str], *, top_k: int):
        assert top_k == 2
        # Search must start only after planner has produced query variants.
        assert call_order == ["planner"]
        assert "germany ai admissions official site" in queries
        assert any("tuition fees" in item for item in queries)
        call_order.append("search")
        return [
            {
                "organic_results": [
                    {
                        "title": "Admissions",
                        "link": "https://uni-example.de/admissions",
                        "snippet": "Tuition fees and requirements.",
                    }
                ]
            }
        ]

    async def _fake_fetch_pages(rows: list[dict]):
        assert rows and rows[0]["url"] == "https://uni-example.de/admissions"
        return {
            "https://uni-example.de/admissions": {
                "content": "Tuition fees are EUR 2000 and language requirement is IELTS 6.5.",
                "published_date": "2026-03-20",
            }
        }

    monkeypatch.setattr(service, "_aplan_queries_with_llm", _fake_llm_plan)
    monkeypatch.setattr(service, "_asearch_payloads", _fake_payloads)
    monkeypatch.setattr(service, "_afetch_organic_pages", _fake_fetch_pages)

    result = await service.aretrieve_web_chunks("germany ai admissions", top_k=2)

    assert call_order == ["planner", "search"]
    assert result["query_plan"]["planner"] == "llm"
    assert result["query_plan"]["llm_used"] is True
    assert "germany ai admissions official site" in result["query_variants"]
    assert any("tuition fees" in item for item in result["query_variants"])


@pytest.mark.asyncio
async def test_aretrieve_web_chunks_fast_mode_skips_deep_loop(monkeypatch):
    monkeypatch.setattr(service.settings.web_search, "retrieval_loop_enabled", True)

    async def _should_not_call_resolve_plan(*_args, **_kwargs):
        raise AssertionError("fast mode should not call deep planner resolver")

    call_count = {"search": 0}

    async def _fake_payloads(queries: list[str], *, top_k: int):
        assert 1 <= len(queries) <= 2
        call_count["search"] += 1
        return [
            {
                "organic_results": [
                    {
                        "title": "Admissions",
                        "link": "https://uni-example.de/admissions",
                        "snippet": "Admission summary.",
                    }
                ]
            }
        ]

    async def _fake_fetch_pages(rows: list[dict], **_kwargs):
        return {
            "https://uni-example.de/admissions": {
                "content": "Admissions details from official page.",
                "published_date": "2026-03-21",
            }
        }

    monkeypatch.setattr(service, "_resolve_query_plan", _should_not_call_resolve_plan)
    monkeypatch.setattr(service, "_asearch_payloads", _fake_payloads)
    monkeypatch.setattr(service, "_afetch_organic_pages", _fake_fetch_pages)

    result = await service.aretrieve_web_chunks(
        "germany ai admissions",
        top_k=2,
        search_mode="fast",
    )

    assert call_count["search"] == 1
    assert result["search_mode"] == "fast"
    assert result["retrieval_loop"]["enabled"] is False
    assert result["retrieval_loop"]["iterations"] == 1


@pytest.mark.asyncio
async def test_aretrieve_web_chunks_standard_mode_skips_deep_loop(monkeypatch):
    monkeypatch.setattr(service.settings.web_search, "retrieval_loop_enabled", True)

    async def _should_not_call_resolve_plan(*_args, **_kwargs):
        raise AssertionError("standard mode should not call deep planner resolver")

    call_count = {"search": 0}

    async def _fake_payloads(queries: list[str], *, top_k: int):
        assert 1 <= len(queries) <= 2
        _ = queries, top_k
        call_count["search"] += 1
        return [
            {
                "organic_results": [
                    {
                        "title": "Admissions",
                        "link": "https://uni-example.de/admissions",
                        "snippet": "Admission summary.",
                    }
                ]
            }
        ]

    async def _fake_fetch_pages(_rows: list[dict], **_kwargs):
        return {
            "https://uni-example.de/admissions": {
                "content": "Admissions details from official page.",
                "published_date": "2026-03-21",
            }
        }

    monkeypatch.setattr(service, "_resolve_query_plan", _should_not_call_resolve_plan)
    monkeypatch.setattr(service, "_asearch_payloads", _fake_payloads)
    monkeypatch.setattr(service, "_afetch_organic_pages", _fake_fetch_pages)

    result = await service.aretrieve_web_chunks(
        "germany ai admissions",
        top_k=2,
        search_mode="standard",
    )

    assert call_count["search"] == 1
    assert result["search_mode"] == "standard"
    assert result["retrieval_loop"]["enabled"] is False
    assert result["retrieval_loop"]["iterations"] == 1


@pytest.mark.asyncio
async def test_asearch_payloads_uses_mode_specific_search_depth(monkeypatch):
    monkeypatch.setattr(service.settings.web_search, "default_num", 3)
    captured_depths: list[str] = []

    async def _fake_batch(queries: list[str], **kwargs):
        _ = queries
        captured_depths.append(str(kwargs.get("search_depth", "")))
        return [
            {
                "query": "q1",
                "result": {"organic_results": []},
                "error": "",
            }
        ]

    monkeypatch.setattr(service, "asearch_google_batch", _fake_batch)

    token = service._RETRIEVAL_MODE_CTX.set("deep")
    try:
        await service._asearch_payloads(["q1", "q2"], top_k=2)
    finally:
        service._RETRIEVAL_MODE_CTX.reset(token)

    token = service._RETRIEVAL_MODE_CTX.set("standard")
    try:
        await service._asearch_payloads(["q1", "q2"], top_k=2)
    finally:
        service._RETRIEVAL_MODE_CTX.reset(token)

    assert captured_depths == ["advanced", "basic"]


@pytest.mark.asyncio
async def test_asearch_payloads_uses_mode_specific_result_count(monkeypatch):
    monkeypatch.setattr(service.settings.web_search, "default_num", 3)
    monkeypatch.setattr(service.settings.web_search, "deep_default_num", 6)
    captured_nums: list[int] = []

    async def _fake_batch(queries: list[str], **kwargs):
        _ = queries
        captured_nums.append(int(kwargs.get("num", 0)))
        return [
            {
                "query": "q1",
                "result": {"organic_results": []},
                "error": "",
            }
        ]

    monkeypatch.setattr(service, "asearch_google_batch", _fake_batch)

    token = service._RETRIEVAL_MODE_CTX.set("deep")
    try:
        await service._asearch_payloads(["q1", "q2"], top_k=2)
    finally:
        service._RETRIEVAL_MODE_CTX.reset(token)

    token = service._RETRIEVAL_MODE_CTX.set("standard")
    try:
        await service._asearch_payloads(["q1", "q2"], top_k=2)
    finally:
        service._RETRIEVAL_MODE_CTX.reset(token)

    assert captured_nums == [6, 3]


@pytest.mark.asyncio
async def test_query_planner_uses_cache_before_llm_call(monkeypatch):
    monkeypatch.setattr(service.settings.web_search, "query_planner_use_llm", True)
    monkeypatch.setattr(service.settings.web_search, "query_planner_cache_enabled", True)

    async def _fake_cache_read(_cache_key: str):
        return {
            "queries": ["germany ai admissions official site"],
            "subquestions": ["tuition fees"],
        }

    async def _should_not_call_create(**_kwargs):
        raise AssertionError("planner should use cache and skip model call")

    from app.infra import bedrock_chat_client

    monkeypatch.setattr(service, "_read_cache_json", _fake_cache_read)
    monkeypatch.setattr(
        bedrock_chat_client.client.chat.completions,
        "create",
        _should_not_call_create,
    )

    plan = await service._aplan_queries_with_llm("germany ai admissions", [])

    assert plan is not None
    assert plan["planner"] == "llm_cache"
    assert plan["llm_used"] is True


@pytest.mark.asyncio
async def test_query_planner_backpressure_falls_back(monkeypatch):
    monkeypatch.setattr(service.settings.web_search, "query_planner_use_llm", True)
    monkeypatch.setattr(service.settings.web_search, "query_planner_cache_enabled", False)

    async def _raise_backpressure(**_kwargs):
        raise DependencyBackpressureError("llm_planner", 0.75)

    from app.infra import bedrock_chat_client

    monkeypatch.setattr(
        bedrock_chat_client.client.chat.completions,
        "create",
        _raise_backpressure,
    )

    plan = await service._aplan_queries_with_llm("germany ai admissions", [])

    assert plan is None


def test_build_query_variants_includes_suffix_scoped_variant(monkeypatch):
    monkeypatch.setattr(service.settings.web_search, "multi_query_enabled", True)
    monkeypatch.setattr(service.settings.web_search, "max_query_variants", 6)

    variants = service._build_query_variants(
        "Compare TUM vs LMU data science admissions",
        [".de", ".eu"],
    )

    assert any("site:.de" in item and "site:.eu" in item for item in variants)


def test_build_query_variants_adds_entity_focused_queries(monkeypatch):
    monkeypatch.setattr(service.settings.web_search, "multi_query_enabled", True)
    monkeypatch.setattr(service.settings.web_search, "max_query_variants", 8)

    variants = service._build_query_variants(
        "Compare TUM vs LMU for English-taught data science master's programs",
        [".de", ".eu"],
    )

    assert any("tum data science master's program" in item.lower() for item in variants)
    assert any("lmu data science master's program" in item.lower() for item in variants)


def test_build_query_variants_fast_mode_is_lightweight(monkeypatch):
    monkeypatch.setattr(service.settings.web_search, "multi_query_enabled", True)
    monkeypatch.setattr(service.settings.web_search, "max_query_variants", 5)

    token = service._RETRIEVAL_MODE_CTX.set("fast")
    try:
        variants = service._build_query_variants(
            "University of Hamburg MSc Data Science and Artificial Intelligence",
            [".de", ".eu"],
        )
    finally:
        service._RETRIEVAL_MODE_CTX.reset(token)

    assert 1 <= len(variants) <= 2


def test_max_query_variants_for_mode_uses_deep_override(monkeypatch):
    monkeypatch.setattr(service.settings.web_search, "max_query_variants", 3)
    monkeypatch.setattr(service.settings.web_search, "deep_max_query_variants", 5)

    token = service._RETRIEVAL_MODE_CTX.set("deep")
    try:
        deep_variants = service._max_query_variants_for_mode()
    finally:
        service._RETRIEVAL_MODE_CTX.reset(token)

    token = service._RETRIEVAL_MODE_CTX.set("standard")
    try:
        standard_variants = service._max_query_variants_for_mode()
    finally:
        service._RETRIEVAL_MODE_CTX.reset(token)

    assert deep_variants == 5
    assert standard_variants == 2


def test_url_matches_allowed_suffix_filters_to_de_and_eu():
    assert service._url_matches_allowed_suffix("https://www.uni-tuebingen.de/en/", [".de", ".eu"])
    assert service._url_matches_allowed_suffix("https://research.example.eu/ai", [".de", ".eu"])
    assert not service._url_matches_allowed_suffix("https://example.com/ai", [".de", ".eu"])


def test_normalized_allowed_domain_suffixes_reads_settings(monkeypatch):
    monkeypatch.setattr(service.settings.web_search, "allowed_domain_suffixes", ["de", ".eu", "DE"])
    assert service._normalized_allowed_domain_suffixes() == [".de", ".eu"]


def test_retrieval_min_unique_domains_uses_deep_override(monkeypatch):
    monkeypatch.setattr(service.settings.web_search, "retrieval_min_unique_domains", 1)
    monkeypatch.setattr(service.settings.web_search, "deep_min_unique_domains", 3)

    deep_token = service._RETRIEVAL_MODE_CTX.set("deep")
    try:
        deep_value = service._retrieval_min_unique_domains()
    finally:
        service._RETRIEVAL_MODE_CTX.reset(deep_token)

    standard_token = service._RETRIEVAL_MODE_CTX.set("standard")
    try:
        standard_value = service._retrieval_min_unique_domains()
    finally:
        service._RETRIEVAL_MODE_CTX.reset(standard_token)

    assert deep_value == 3
    assert standard_value == 1


def test_filter_rows_by_allowed_domains_keeps_official_and_daad_only(monkeypatch):
    monkeypatch.setattr(service.settings.web_search, "official_source_filter_enabled", True)
    monkeypatch.setattr(service.settings.web_search, "official_source_allowlist", ["daad.de"])

    rows = [
        {
            "title": "M.Sc. Program",
            "url": "https://www.uni-hamburg.de/en/studium/master/programs/data-science.html",
            "snippet": "University of Hamburg master's program details.",
        },
        {
            "title": "DAAD Program Entry",
            "url": "https://www2.daad.de/deutschland/studienangebote/international-programmes/en/detail/5634/",
            "snippet": "DAAD international program details.",
        },
        {
            "title": "Forum discussion",
            "url": "https://research-forum.eu/ai",
            "snippet": "Community notes about admissions.",
        },
    ]
    filtered = service._filter_rows_by_allowed_domains(rows, [".de", ".eu"])
    urls = {str(item["url"]) for item in filtered}
    assert "https://www.uni-hamburg.de/en/studium/master/programs/data-science.html" in urls
    assert (
        "https://www2.daad.de/deutschland/studienangebote/international-programmes/en/detail/5634/"
        in urls
    )
    assert "https://research-forum.eu/ai" not in urls


def test_fetch_page_data_sync_extracts_pdf(monkeypatch):
    class _FakePdfPage:
        def extract_text(self):
            return "Admission requirements\nLanguage: IELTS 6.5"

    class _FakePdfReader:
        def __init__(self, _buffer):
            self.pages = [_FakePdfPage()]

    class _FakeResponse:
        def __init__(self):
            self.headers = {"Content-Type": "application/pdf"}

        def read(self, _max_bytes):
            return b"%PDF-sample"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(service, "PdfReader", _FakePdfReader)
    monkeypatch.setattr(service.urllib.request, "urlopen", lambda *_args, **_kwargs: _FakeResponse())

    page = service._fetch_page_data_sync(
        "https://uni-example.de/program.pdf",
        timeout_seconds=5.0,
        max_chars=500,
    )
    assert "Admission requirements" in page["content"]
    assert page["published_date"] == ""


def test_domain_group_key_collapses_official_subdomains():
    assert service._domain_group_key("cit.tum.de") == "tum.de"
    assert service._domain_group_key("www.tum.de") == "tum.de"


def test_domain_authority_prefers_de_or_eu_over_com():
    de_score = service._domain_authority_score("https://www.lmu.de/programs/ai", [".de", ".eu"])
    eu_score = service._domain_authority_score("https://research.example.eu/ai", [".de", ".eu"])
    com_score = service._domain_authority_score("https://example.com/ai", [".de", ".eu"])

    assert de_score > com_score
    assert eu_score > com_score


def test_required_fields_from_query_detects_explicit_fields():
    fields = service._required_fields_from_query(
        "Tell me course requirements, language requirements for international students, and application deadline."
    )
    ids = [str(item.get("id", "")).strip() for item in fields]
    assert "admission_requirements" in ids
    assert "language_requirements" in ids
    assert "application_deadline" in ids


def test_required_field_coverage_for_language_requires_score():
    required_fields = service._required_fields_from_query(
        "language requirements for international students"
    )
    language_only = [
        {
            "content": "Applicants must provide proof of English proficiency.",
            "metadata": {"url": "https://uni-example.de/admission"},
        }
    ]
    with_scores = [
        {
            "content": "English requirement: IELTS 6.5 or TOEFL iBT 90.",
            "metadata": {"url": "https://uni-example.de/admission"},
        }
    ]

    weak = service._required_field_coverage(required_fields, language_only)
    strong = service._required_field_coverage(required_fields, with_scores)

    assert "language_requirements" in weak["missing_ids"]
    assert weak["coverage"] < 1.0
    assert "language_requirements" not in strong["missing_ids"]
    assert strong["coverage"] == 1.0


@pytest.mark.asyncio
async def test_aretrieve_web_chunks_deep_loop_requeries_until_required_field_is_complete(monkeypatch):
    monkeypatch.setattr(service.settings.web_search, "multi_query_enabled", False)
    monkeypatch.setattr(service.settings.web_search, "query_planner_enabled", True)
    monkeypatch.setattr(service.settings.web_search, "retrieval_loop_enabled", True)
    monkeypatch.setattr(service.settings.web_search, "retrieval_loop_max_steps", 2)
    monkeypatch.setattr(service.settings.web_search, "retrieval_loop_max_gap_queries", 1)
    monkeypatch.setattr(service.settings.web_search, "allowed_domain_suffixes", [])

    async def _fake_plan(_query: str, _allowed_suffixes: list[str]):
        return {
            "queries": ["uni sample msc ai language requirements"],
            "subquestions": [],
            "planner": "heuristic",
            "llm_used": False,
        }

    calls: list[list[str]] = []

    async def _fake_payloads(queries: list[str], *, top_k: int):
        _ = top_k
        calls.append(list(queries))
        query_text = " ".join(queries).lower()
        if "minimum score" in query_text or "ielts" in query_text or "toefl" in query_text:
            return [
                {
                    "organic_results": [
                        {
                            "title": "Language Requirements",
                            "link": "https://uni-example.de/language",
                            "snippet": "IELTS 6.5 or TOEFL iBT 90.",
                        }
                    ]
                }
            ]
        return [
            {
                "organic_results": [
                    {
                        "title": "Admission Overview",
                        "link": "https://uni-example.de/admission",
                        "snippet": "Proof of English proficiency is required.",
                    }
                ]
            }
        ]

    async def _fake_fetch_pages(rows: list[dict], **_kwargs):
        payload: dict[str, dict] = {}
        for row in rows:
            url = str(row.get("url", "")).strip()
            if "language" in url:
                payload[url] = {
                    "content": "Language requirement: IELTS 6.5 overall or TOEFL iBT 90.",
                    "published_date": "2026-03-10",
                }
            else:
                payload[url] = {
                    "content": "Applicants must provide proof of English proficiency.",
                    "published_date": "2026-03-10",
                }
        return payload

    monkeypatch.setattr(service, "_resolve_query_plan", _fake_plan)
    monkeypatch.setattr(service, "_asearch_payloads", _fake_payloads)
    monkeypatch.setattr(service, "_afetch_organic_pages", _fake_fetch_pages)

    result = await service.aretrieve_web_chunks(
        "tell me language requirements for international students",
        top_k=3,
        search_mode="deep",
    )

    assert len(calls) == 2
    assert result["verification"]["required_field_coverage"] == 1.0
    assert result["verification"]["required_fields_missing"] == []
    assert result["verification"]["verified"] is True
