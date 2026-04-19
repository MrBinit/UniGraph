import asyncio
import hashlib
import pytest
from app.services import llm_service

_REAL_ENFORCE_CITATION_GROUNDING = llm_service._enforce_citation_grounding


@pytest.fixture(autouse=True)
def _stub_json_metrics(monkeypatch):
    async def noop(_record):
        return None

    monkeypatch.setattr(llm_service, "_record_json_metrics", noop)


@pytest.fixture(autouse=True)
def _stub_reranker(monkeypatch):
    async def passthrough(_query, candidates):
        return {"results": candidates, "applied": False, "timings_ms": {"total": 0}}

    monkeypatch.setattr(llm_service, "arerank_retrieval_results", passthrough)


@pytest.fixture(autouse=True)
def _stub_evidence_urls(monkeypatch):
    monkeypatch.setattr(
        llm_service,
        "_evidence_urls",
        lambda results: (
            ["https://example.edu/evidence"] if isinstance(results, list) and results else []
        ),
    )


@pytest.fixture(autouse=True)
def _stub_citation_enforcement(monkeypatch):
    monkeypatch.setattr(llm_service, "_enforce_citation_grounding", lambda result, _state: result)


@pytest.fixture(autouse=True)
def _stub_citation_grounding_policy(monkeypatch):
    monkeypatch.setattr(llm_service, "_is_citation_grounding_required", lambda: False)


@pytest.fixture(autouse=True)
def _disable_always_web_hybrid_by_default(monkeypatch):
    monkeypatch.setattr(llm_service.settings.web_search, "always_web_retrieval_enabled", False)
    monkeypatch.setattr(llm_service.settings.web_search, "retrieval_min_unique_domains", 1)
    monkeypatch.setattr(llm_service.settings.web_search, "deep_min_unique_domains", 1)


@pytest.fixture(autouse=True)
def _enable_response_cache_by_default_for_tests(monkeypatch):
    monkeypatch.setattr(llm_service.settings.web_search, "response_cache_enabled", True)


@pytest.fixture(autouse=True)
def _disable_agentic_planner_verifier_by_default(monkeypatch):
    monkeypatch.setattr(llm_service, "_agentic_planner_enabled", lambda: False)
    monkeypatch.setattr(llm_service, "_agentic_verifier_enabled", lambda: False)


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value

    def delete(self, key):
        existed = key in self.store
        self.store.pop(key, None)
        return 1 if existed else 0


def _attach_fake_redis(monkeypatch, fake_redis: FakeRedis) -> None:
    monkeypatch.setattr(llm_service, "redis_client", fake_redis)
    monkeypatch.setattr(llm_service, "async_redis_client", fake_redis)


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [_Choice(content)]


@pytest.mark.asyncio
async def test_generate_response_returns_cache_hit(monkeypatch):
    fake_redis = FakeRedis()
    cache_key = llm_service._chat_cache_key("user-1", "find ai professor")
    fake_redis.store[cache_key] = "from-cache"
    _attach_fake_redis(monkeypatch, fake_redis)

    async def should_not_run(*_args, **_kwargs):
        raise AssertionError("build_context should not run on cache hit")

    monkeypatch.setattr(llm_service, "build_context", should_not_run)

    result = await llm_service.generate_response("user-1", "find ai professor")
    assert result == "from-cache"


@pytest.mark.asyncio
async def test_generate_response_bypasses_cache_when_response_cache_disabled(monkeypatch):
    monkeypatch.setattr(llm_service.settings.web_search, "response_cache_enabled", False)
    fake_redis = FakeRedis()
    cache_key = llm_service._chat_cache_key("user-1", "find ai professor")
    fake_redis.store[cache_key] = "from-cache"
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def fake_primary(_messages):
        return FakeResponse("fresh-response")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {"results": []}

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)

    result = await llm_service.generate_response("user-1", "find ai professor")
    assert result == "fresh-response"
    assert fake_redis.store[cache_key] == "from-cache"


@pytest.mark.asyncio
async def test_generate_response_rejects_low_quality_cache_entry(monkeypatch):
    fake_redis = FakeRedis()
    cache_key = llm_service._chat_cache_key("user-1", "find ai professor")
    fake_redis.store[cache_key] = "Sorry, no relevant information is found."
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def fake_primary(_messages):
        return FakeResponse("fresh-answer")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {"results": []}

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)

    result = await llm_service.generate_response("user-1", "find ai professor")
    assert result == "fresh-answer"
    assert fake_redis.store.get(cache_key) == "fresh-answer"


@pytest.mark.asyncio
async def test_generate_response_cache_is_session_scoped(monkeypatch):
    fake_redis = FakeRedis()
    session_a_key = llm_service._chat_cache_key("user-1", "find ai professor", "session-a")
    fake_redis.store[session_a_key] = "from-cache-session-a"
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def fake_primary(_messages):
        return FakeResponse("fresh-session-b")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {"results": []}

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)

    result = await llm_service.generate_response(
        "user-1",
        "find ai professor",
        session_id="session-b",
    )
    assert result == "fresh-session-b"


@pytest.mark.asyncio
async def test_generate_response_uses_primary_and_updates_memory(monkeypatch):
    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)
    captured_metrics = []

    async def fake_build_context(user_id, user_prompt):
        assert user_id == "user-1"
        assert user_prompt == "find ai professor"
        return [{"role": "user", "content": user_prompt}]

    async def fake_primary(_messages):
        return FakeResponse("primary-response")

    async def fake_fallback(_messages):
        raise AssertionError("fallback should not run when primary succeeds")

    memory_updates = []
    retrieval_queries = []

    async def fake_update_memory(user_id, user_message, assistant_reply):
        memory_updates.append((user_id, user_message, assistant_reply))

    async def fake_retrieve_document_chunks(query, **kwargs):
        retrieval_queries.append((query, kwargs))
        return {
            "results": [
                {
                    "content": (
                        "Distributed Security Systems Lab focuses on scalable "
                        "and secure AI infrastructure."
                    ),
                    "metadata": {
                        "university": "Falkenberg University of Cybernetics (FUC)",
                        "section_heading": "Distributed Security Systems Lab (DSSL)",
                    },
                }
            ]
        }

    async def fake_record_json_metrics(record):
        captured_metrics.append(record)

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "_call_fallback", fake_fallback)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)
    monkeypatch.setattr(llm_service, "_record_json_metrics", fake_record_json_metrics)

    result = await llm_service.generate_response("user-1", "find ai professor")
    assert result == "primary-response"
    assert memory_updates == [("user-1", "find ai professor", "primary-response")]
    assert retrieval_queries[0][0] == "find ai professor"
    cache_key = llm_service._chat_cache_key("user-1", "find ai professor")
    assert fake_redis.store[cache_key] == "primary-response"
    assert captured_metrics[-1]["question"] == "find ai professor"
    assert captured_metrics[-1]["answer"] == "primary-response"
    assert captured_metrics[-1]["outcome"] == "success"
    assert captured_metrics[-1]["timings_ms"]["llm_response_ms"] is not None
    assert captured_metrics[-1]["timings_ms"]["short_term_memory_ms"] is not None
    assert captured_metrics[-1]["timings_ms"]["long_term_memory_ms"] is not None
    assert "groundedness" in captured_metrics[-1]["quality"]
    assert "citation_accuracy" in captured_metrics[-1]["quality"]


@pytest.mark.asyncio
async def test_generate_response_does_not_cache_abstain_answer(monkeypatch):
    monkeypatch.setattr(llm_service, "_is_citation_grounding_required", lambda: True)
    monkeypatch.setattr(
        llm_service, "_enforce_citation_grounding", _REAL_ENFORCE_CITATION_GROUNDING
    )

    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    model_calls = {"count": 0}

    async def fake_primary(_messages):
        model_calls["count"] += 1
        return FakeResponse("This answer has no citations.")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {
            "results": [
                {
                    "content": "Admission details for AI program.",
                    "metadata": {"university": "Sample University"},
                }
            ]
        }

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)

    result_first = await llm_service.generate_response("user-1", "find university course")
    assert result_first == llm_service._NO_RELEVANT_INFORMATION_DETAIL
    calls_after_first = model_calls["count"]

    result_second = await llm_service.generate_response("user-1", "find university course")
    assert result_second == llm_service._NO_RELEVANT_INFORMATION_DETAIL

    assert model_calls["count"] > calls_after_first
    cache_key = llm_service._chat_cache_key("user-1", "find university course")
    assert cache_key not in fake_redis.store


@pytest.mark.asyncio
async def test_generate_response_agentic_retries_when_citations_missing(monkeypatch):
    monkeypatch.setattr(llm_service, "_is_citation_grounding_required", lambda: True)
    monkeypatch.setattr(
        llm_service, "_enforce_citation_grounding", _REAL_ENFORCE_CITATION_GROUNDING
    )

    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    model_calls = {"count": 0}

    async def fake_primary(_messages):
        model_calls["count"] += 1
        if model_calls["count"] == 1:
            return FakeResponse("Oxford admission details are available.")
        return FakeResponse(
            "Oxford admission details are available. Source: https://example.edu/evidence"
        )

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {
            "results": [
                {
                    "content": "Oxford AI admission details from official content.",
                    "metadata": {"university": "Oxford"},
                }
            ]
        }

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)

    result = await llm_service.generate_response("user-1", "oxford ai admission")
    assert "https://example.edu/evidence" in result
    assert model_calls["count"] == 2


@pytest.mark.asyncio
async def test_generate_response_agentic_retries_for_source_diversity(monkeypatch):
    monkeypatch.setattr(llm_service, "_is_citation_grounding_required", lambda: True)
    monkeypatch.setattr(
        llm_service, "_enforce_citation_grounding", _REAL_ENFORCE_CITATION_GROUNDING
    )
    monkeypatch.setattr(llm_service.settings.web_search, "retrieval_min_unique_domains", 2)
    monkeypatch.setattr(
        llm_service,
        "_evidence_urls",
        lambda _results: [
            "https://first.example.edu/evidence",
            "https://second.example.org/evidence",
        ],
    )

    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    model_calls = {"count": 0}

    async def fake_primary(_messages):
        model_calls["count"] += 1
        if model_calls["count"] == 1:
            return FakeResponse(
                "Admission details summary. Source: https://first.example.edu/evidence"
            )
        return FakeResponse(
            "Admission details summary. "
            "Sources: https://first.example.edu/evidence and https://second.example.org/evidence"
        )

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {
            "results": [
                {
                    "content": "Evidence from source one.",
                    "metadata": {"university": "Example University"},
                },
                {
                    "content": "Evidence from source two.",
                    "metadata": {"university": "Example University"},
                },
            ]
        }

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)

    result = await llm_service.generate_response("user-1", "oxford ai admission")
    assert "https://second.example.org/evidence" in result
    assert model_calls["count"] == 2


@pytest.mark.asyncio
async def test_generate_response_agentic_planner_and_verifier_orchestration(monkeypatch):
    monkeypatch.setattr(llm_service, "_agentic_planner_enabled", lambda: True)
    monkeypatch.setattr(llm_service, "_agentic_verifier_enabled", lambda: True)
    monkeypatch.setattr(llm_service, "_is_citation_grounding_required", lambda: False)

    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    responses = [
        FakeResponse(
            '{"intent":"south westphalia msc business informatics",'
            '"subquestions":["admissions","fees"],'
            '"search_queries":["south westphalia msc business informatics admissions official"],'
            '"success_criteria":["answer admission requirements","include fees if available"]}'
        ),
        FakeResponse("Draft answer without fees."),
        FakeResponse(
            '{"pass":false,"coverage_score":0.4,'
            '"issues":["missing_fees"],'
            '"missing_points":["tuition and semester contribution"],'
            '"revision_guidance":"Add fee details with evidence-backed citation."}'
        ),
        FakeResponse("Final answer with fees. Source: https://example.edu/evidence"),
        FakeResponse(
            '{"pass":true,"coverage_score":0.92,'
            '"issues":[],"missing_points":[],"revision_guidance":""}'
        ),
    ]
    model_calls = {"count": 0}

    async def fake_primary(_messages):
        index = model_calls["count"]
        model_calls["count"] += 1
        return responses[index]

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {
            "results": [
                {
                    "content": "Admissions and fee details from official page.",
                    "metadata": {"university": "South Westphalia University"},
                }
            ]
        }

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)

    result = await llm_service.generate_response(
        "user-1", "South Westphalia MSc Business Informatics"
    )
    assert result == "Final answer with fees. Source: https://example.edu/evidence"
    assert model_calls["count"] == 5


@pytest.mark.asyncio
async def test_generate_response_returns_best_partial_when_verification_never_passes(monkeypatch):
    monkeypatch.setattr(llm_service, "_agentic_planner_enabled", lambda: True)
    monkeypatch.setattr(llm_service, "_agentic_verifier_enabled", lambda: True)
    monkeypatch.setattr(llm_service, "_is_citation_grounding_required", lambda: False)

    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    responses = [
        FakeResponse(
            '{"intent":"compare programs","subquestions":["deadline","fees"],'
            '"search_queries":["compare programs official"],'
            '"success_criteria":["compare both programs","cite sources"]}'
        ),
        FakeResponse(
            "Partial comparison with one verified detail. " "Source: https://example.edu/evidence"
        ),
        FakeResponse(
            '{"pass":false,"coverage_score":0.55,'
            '"issues":["missing second program details"],'
            '"missing_points":["second program fees"],'
            '"revision_guidance":"Add missing details if supported by evidence."}'
        ),
        FakeResponse("Sources: https://example.edu/evidence"),
        FakeResponse(
            '{"pass":false,"coverage_score":0.25,'
            '"issues":["no comparison table","missing key details"],'
            '"missing_points":["deadlines","fees","requirements"],'
            '"revision_guidance":"Needs substantially more detail."}'
        ),
        FakeResponse("Insufficient details. Sources: https://example.edu/evidence"),
        FakeResponse(
            '{"pass":false,"coverage_score":0.22,'
            '"issues":["still missing key details"],'
            '"missing_points":["deadlines","fees","requirements"],'
            '"revision_guidance":"Insufficient evidence to complete."}'
        ),
    ]
    model_calls = {"count": 0}

    async def fake_primary(_messages):
        idx = model_calls["count"]
        model_calls["count"] += 1
        return responses[idx]

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {
            "results": [
                {
                    "content": "Program detail evidence.",
                    "metadata": {"university": "Example University"},
                }
            ]
        }

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)

    result = await llm_service.generate_response("user-1", "compare programs", mode="deep")

    assert result.startswith("Partial comparison with one verified detail.")
    assert result != llm_service._NO_RELEVANT_INFORMATION_DETAIL
    assert model_calls["count"] == 7


def test_targeted_required_field_rescue_queries_include_site_and_field_focus():
    state = {
        "query_intent": "fact_lookup",
        "required_answer_fields": [
            "application_deadline",
            "application_portal",
            "language_test_score_thresholds",
        ],
        "required_fields_missing": ["application_deadline"],
        "web_required_fields_missing": ["application_portal"],
        "evidence_urls": ["https://www.uni-mannheim.de/studium/bewerbung/"],
    }
    queries = llm_service._targeted_required_field_rescue_queries(
        base_query="University of Mannheim MSc Business Informatics admission requirements",
        state=state,
        issues=["missing_required_answer_fields", "web_missing:application_portal"],
    )
    assert queries
    assert any("site:uni-mannheim.de" in query.lower() for query in queries)
    assert any("deadline" in query.lower() for query in queries)
    assert any("portal" in query.lower() for query in queries)


def test_targeted_required_field_rescue_queries_infer_site_hint_from_query_without_evidence():
    state = {
        "query_intent": "fact_lookup",
        "required_answer_fields": ["application_deadline", "application_portal"],
        "required_fields_missing": ["application_deadline"],
        "web_required_fields_missing": ["application_portal"],
        "evidence_urls": [],
        "safe_user_prompt": (
            "Tell me about University of Mannheim MSc Business Informatics including "
            "deadline and where to apply"
        ),
    }
    queries = llm_service._targeted_required_field_rescue_queries(
        base_query="University of Mannheim MSc Business Informatics admission requirements",
        state=state,
        issues=["web_missing:application_portal"],
    )
    assert queries
    assert any("site:uni-mannheim.de" in query.lower() for query in queries)


@pytest.mark.asyncio
async def test_generate_agentic_answer_runs_required_field_rescue_extra_round(monkeypatch):
    monkeypatch.setattr(llm_service, "_web_retrieval_ready", lambda: (True, "ready"))
    monkeypatch.setattr(llm_service, "_agentic_required_field_rescue_max_rounds", lambda: 1)

    async def fake_finalize_candidate_with_llm(**_kwargs):
        return "", {}, 0

    monkeypatch.setattr(llm_service, "_finalize_candidate_with_llm", fake_finalize_candidate_with_llm)

    worker_calls = {"count": 0}
    rescue_calls = {"count": 0}

    async def fake_call_model_with_fallback(_messages, _state, role="worker", attempt=1):
        _ = attempt
        assert role == "worker"
        worker_calls["count"] += 1
        if worker_calls["count"] == 1:
            return FakeResponse("I could not verify this from current evidence.")
        return FakeResponse("Application deadline: 31 May 2026.")

    async def fake_required_field_rescue(*, issues, state, base_query, search_mode):
        _ = state
        assert "missing_required_answer_fields" in issues
        assert "deadline" in base_query.lower()
        assert search_mode == "deep"
        rescue_calls["count"] += 1
        return ([{"role": "system", "content": "Rescue retrieval context"}], True)

    monkeypatch.setattr(llm_service, "_call_model_with_fallback", fake_call_model_with_fallback)
    monkeypatch.setattr(llm_service, "_attempt_required_field_web_rescue", fake_required_field_rescue)

    state = llm_service._new_metrics_state()
    state.update(
        {
            "safe_user_prompt": "Tell me the application deadline for this program",
            "query_intent": "deadline",
            "required_answer_fields": ["application_deadline"],
        }
    )
    policy = {
        "max_attempts": 1,
        "planner_enabled": False,
        "verifier_enabled": False,
        "web_search_mode": "deep",
        "mode": "deep",
    }

    result, _usage = await llm_service._generate_agentic_answer(
        user_id="user-1",
        messages=[{"role": "user", "content": "deadline?"}],
        policy=policy,
        state=state,
    )
    assert result == "Application deadline: 31 May 2026."
    assert worker_calls["count"] == 2
    assert rescue_calls["count"] == 1
    assert state["agent_required_field_rescue_rounds"] == 1


@pytest.mark.asyncio
async def test_generate_response_skips_retrieval_when_disabled(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_DISABLED", "true")
    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def fake_primary(_messages):
        return FakeResponse("primary-response")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def should_not_retrieve(*_args, **_kwargs):
        raise AssertionError("retrieval should be skipped when RETRIEVAL_DISABLED=true")

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", should_not_retrieve)

    result = await llm_service.generate_response("user-1", "find ai professor")

    assert result == "primary-response"


@pytest.mark.asyncio
async def test_generate_response_uses_web_fallback_when_vector_empty(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.web_search, "enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "max_context_results", 2)

    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def fake_primary(messages):
        assert any(
            isinstance(message, dict)
            and "Live web fallback context" in str(message.get("content", ""))
            for message in messages
        )
        return FakeResponse("primary-response")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {"retrieval_strategy": "ann", "results": []}

    async def fake_web_fallback(*_args, **_kwargs):
        return {
            "results": [
                {
                    "content": "Oxford AI entry requirements from official site.",
                    "metadata": {
                        "university": "Oxford MSc AI",
                        "url": "https://example.edu/oxford",
                    },
                }
            ]
        }

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)
    monkeypatch.setattr(llm_service, "aretrieve_web_chunks", fake_web_fallback)

    result = await llm_service.generate_response("user-1", "latest oxford ai admission")
    assert result == "primary-response"


@pytest.mark.asyncio
async def test_generate_response_skips_web_fallback_when_vector_confident(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.web_search, "enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_similarity_threshold", 0.35)

    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def fake_primary(messages):
        assert any(
            isinstance(message, dict)
            and "Retrieved long-term knowledge" in str(message.get("content", ""))
            for message in messages
        )
        assert not any(
            isinstance(message, dict)
            and "Live web fallback context" in str(message.get("content", ""))
            for message in messages
        )
        return FakeResponse("primary-response")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {
            "retrieval_strategy": "ann",
            "results": [
                {
                    "content": "Reliable on-platform result.",
                    "distance": 0.1,
                    "metadata": {"university": "Oxford"},
                }
            ],
        }

    async def should_not_call_web(*_args, **_kwargs):
        raise AssertionError("web fallback should not run for confident vector matches")

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)
    monkeypatch.setattr(llm_service, "aretrieve_web_chunks", should_not_call_web)

    result = await llm_service.generate_response("user-1", "oxford ai admission")
    assert result == "primary-response"


@pytest.mark.asyncio
async def test_generate_response_always_web_hybrid_runs_web_when_vector_confident(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.web_search, "enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "always_web_retrieval_enabled", True)

    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def fake_primary(messages):
        assert any(
            isinstance(message, dict)
            and "Live web fallback context" in str(message.get("content", ""))
            for message in messages
        )
        return FakeResponse("primary-response")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {
            "retrieval_strategy": "ann",
            "results": [
                {
                    "content": "Reliable vector result.",
                    "distance": 0.1,
                    "metadata": {"university": "Oxford"},
                }
            ],
        }

    web_calls = {"count": 0}

    async def fake_web_fallback(*_args, **_kwargs):
        web_calls["count"] += 1
        return {
            "results": [
                {
                    "content": "Oxford official admissions web evidence.",
                    "metadata": {"university": "Oxford Web", "url": "https://example.edu/oxford"},
                }
            ]
        }

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)
    monkeypatch.setattr(llm_service, "aretrieve_web_chunks", fake_web_fallback)

    result = await llm_service.generate_response("user-1", "oxford ai admission")
    assert result == "primary-response"
    assert web_calls["count"] >= 1


@pytest.mark.asyncio
async def test_augment_messages_with_retrieval_fanout_prefetches_web(monkeypatch):
    monkeypatch.setattr(llm_service, "_web_retrieval_ready", lambda: (True, "ready"))
    monkeypatch.setattr(llm_service, "_should_run_web_retrieval", lambda: True)

    web_started = asyncio.Event()
    release_web = asyncio.Event()

    async def fake_web_with_mode(_retrieval_query, *, top_k, search_mode):
        assert top_k == llm_service.settings.postgres.default_top_k
        assert search_mode == "deep"
        web_started.set()
        await release_web.wait()
        return {
            "results": [
                {
                    "content": "Web evidence.",
                    "metadata": {"url": "https://web.example.edu/a"},
                }
            ],
            "query_plan": {"planner": "llm", "llm_used": True, "subquestions": []},
            "query_variants": ["query"],
            "facts": [],
            "retrieval_loop": {"enabled": True},
        }

    async def fake_vector(_retrieval_query, _state):
        # If fan-out works, web prefetch has already started before vector resolves.
        await asyncio.wait_for(web_started.wait(), timeout=0.3)
        release_web.set()
        return (
            [
                {
                    "content": "Vector evidence.",
                    "metadata": {"url": "https://vector.example.edu/a"},
                }
            ],
            0.1,
        )

    async def fake_rerank(_query, merged_results, _state):
        return merged_results

    def fake_apply_grounded_retrieval_context(*, messages, merged_results, used_web_results, state):
        _ = state
        assert used_web_results is True
        assert len(merged_results) >= 2
        return messages, None

    monkeypatch.setattr(llm_service, "_aretrieve_web_chunks_with_mode", fake_web_with_mode)
    monkeypatch.setattr(llm_service, "_retrieve_vector_candidates", fake_vector)
    monkeypatch.setattr(llm_service, "_rerank_if_configured", fake_rerank)
    monkeypatch.setattr(
        llm_service, "_apply_grounded_retrieval_context", fake_apply_grounded_retrieval_context
    )

    base_messages = [{"role": "user", "content": "hello"}]
    state = {"safe_user_prompt": "hello"}
    messages, detail = await llm_service._augment_messages_with_retrieval(
        messages=base_messages,
        retrieval_query="hello",
        search_mode="deep",
        state=state,
    )

    assert detail is None
    assert messages == base_messages


@pytest.mark.asyncio
async def test_prepare_messages_for_model_fanout_prefetches_vector(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_DISABLED", raising=False)
    monkeypatch.setattr(llm_service, "_retrieval_fanout_enabled", lambda: True)

    vector_started = asyncio.Event()
    release_vector = asyncio.Event()
    captured: dict = {}

    async def fake_vector_retrieval(query, **kwargs):
        _ = kwargs
        assert query == "hello"
        vector_started.set()
        await release_vector.wait()
        return {"retrieval_strategy": "ann", "results": []}

    async def fake_build_context(_conversation_user_id, safe_user_prompt):
        await asyncio.wait_for(vector_started.wait(), timeout=0.3)
        release_vector.set()
        return [{"role": "user", "content": safe_user_prompt}]

    async def fake_augment_messages_with_retrieval(
        *,
        messages,
        retrieval_query,
        search_mode,
        state,
        vector_prefetch_result=None,
    ):
        _ = (search_mode, state)
        captured["retrieval_query"] = retrieval_query
        captured["vector_prefetch_result"] = vector_prefetch_result
        return messages, None

    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_vector_retrieval)
    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(
        llm_service, "_augment_messages_with_retrieval", fake_augment_messages_with_retrieval
    )
    monkeypatch.setattr(
        llm_service,
        "apply_context_guardrails",
        lambda messages: {"blocked": False, "messages": messages, "reason": ""},
    )

    state: dict = {}
    messages, detail = await llm_service._prepare_messages_for_model(
        user_id="user-1",
        conversation_user_id="session-1",
        safe_user_prompt="hello",
        execution_mode="deep",
        policy={"web_search_mode": "deep"},
        state=state,
    )

    assert detail is None
    assert messages
    assert captured["retrieval_query"] == "hello"
    assert isinstance(captured["vector_prefetch_result"], dict)


@pytest.mark.asyncio
async def test_prepare_messages_for_model_fanout_cancels_vector_prefetch_on_query_mismatch(
    monkeypatch,
):
    monkeypatch.delenv("RETRIEVAL_DISABLED", raising=False)
    monkeypatch.setattr(llm_service, "_retrieval_fanout_enabled", lambda: True)

    vector_started = asyncio.Event()
    vector_cancelled = asyncio.Event()
    captured: dict = {}

    async def fake_vector_retrieval(query, **kwargs):
        _ = kwargs
        assert query == "What about RWTH Aachen?"
        vector_started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            vector_cancelled.set()
            raise

    async def fake_build_context(_conversation_user_id, _safe_user_prompt):
        await asyncio.wait_for(vector_started.wait(), timeout=0.3)
        return [
            {"role": "user", "content": "What is TU Munich MSc AI deadline?"},
            {"role": "assistant", "content": "Here is TU Munich info."},
            {"role": "user", "content": "What about RWTH Aachen?"},
        ]

    async def fake_augment_messages_with_retrieval(
        *,
        messages,
        retrieval_query,
        search_mode,
        state,
        vector_prefetch_result=None,
    ):
        _ = (messages, search_mode, state)
        captured["retrieval_query"] = retrieval_query
        captured["vector_prefetch_result"] = vector_prefetch_result
        return [{"role": "user", "content": "ok"}], None

    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_vector_retrieval)
    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(
        llm_service, "_augment_messages_with_retrieval", fake_augment_messages_with_retrieval
    )
    monkeypatch.setattr(
        llm_service,
        "apply_context_guardrails",
        lambda messages: {"blocked": False, "messages": messages, "reason": ""},
    )

    state: dict = {}
    messages, detail = await llm_service._prepare_messages_for_model(
        user_id="user-1",
        conversation_user_id="session-1",
        safe_user_prompt="What about RWTH Aachen?",
        execution_mode="deep",
        policy={"web_search_mode": "deep"},
        state=state,
    )

    assert detail is None
    assert any(
        isinstance(item, dict) and item.get("role") == "user" and item.get("content") == "ok"
        for item in messages
    )
    assert "What is TU Munich MSc AI deadline?" in captured["retrieval_query"]
    assert captured["vector_prefetch_result"] is None
    assert vector_cancelled.is_set()


@pytest.mark.asyncio
async def test_generate_response_uses_web_fallback_when_vector_low_confidence(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.web_search, "enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_similarity_threshold", 0.35)

    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def fake_primary(messages):
        assert any(
            isinstance(message, dict)
            and "Live web fallback context" in str(message.get("content", ""))
            for message in messages
        )
        return FakeResponse("primary-response")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {
            "retrieval_strategy": "ann",
            "results": [
                {
                    "content": "Potentially irrelevant vector hit.",
                    "distance": 0.9,
                    "metadata": {"university": "Unknown"},
                }
            ],
        }

    async def fake_web_fallback(*_args, **_kwargs):
        return {
            "results": [
                {
                    "content": "Official Oxford admissions page details.",
                    "metadata": {
                        "university": "Oxford MSc AI",
                        "url": "https://example.edu/oxford",
                    },
                }
            ]
        }

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)
    monkeypatch.setattr(llm_service, "aretrieve_web_chunks", fake_web_fallback)

    result = await llm_service.generate_response("user-1", "oxford ai admission")
    assert result == "primary-response"


@pytest.mark.asyncio
async def test_generate_response_reranks_combined_vector_and_web_results(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.web_search, "enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_similarity_threshold", 0.35)

    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def fake_primary(messages):
        joined = "\n".join(
            str(message.get("content", "")) for message in messages if isinstance(message, dict)
        )
        assert "Best grounded web evidence." in joined
        assert "Lower priority vector text." not in joined
        return FakeResponse("primary-response")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {
            "retrieval_strategy": "ann",
            "results": [
                {
                    "content": "Lower priority vector text.",
                    "distance": 0.85,
                    "metadata": {"university": "Vector Source"},
                }
            ],
        }

    async def fake_web_fallback(*_args, **_kwargs):
        return {
            "results": [
                {
                    "content": "Best grounded web evidence.",
                    "metadata": {"university": "Web Source 1", "url": "https://x.de/a"},
                },
                {
                    "content": "Second web evidence.",
                    "metadata": {"university": "Web Source 2", "url": "https://y.eu/b"},
                },
            ]
        }

    async def fake_rerank(_query, candidates):
        assert len(candidates) == 3
        return {
            "results": [candidates[1], candidates[2]],
            "applied": True,
            "timings_ms": {"total": 7},
        }

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)
    monkeypatch.setattr(llm_service, "aretrieve_web_chunks", fake_web_fallback)
    monkeypatch.setattr(llm_service, "arerank_retrieval_results", fake_rerank)

    result = await llm_service.generate_response("user-1", "best ai program evidence")
    assert result == "primary-response"


@pytest.mark.asyncio
async def test_generate_response_rerank_restores_domain_diversity(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.web_search, "enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "retrieval_min_unique_domains", 2)
    monkeypatch.setattr(llm_service.settings.web_search, "max_context_results", 4)

    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def fake_primary(messages):
        joined = "\n".join(
            str(message.get("content", "")) for message in messages if isinstance(message, dict)
        )
        assert "Evidence from domain X." in joined
        assert "Evidence from domain Y." in joined
        return FakeResponse("Comparison: X has systems focus, while Y has research focus.")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {"retrieval_strategy": "ann", "results": []}

    async def fake_web_fallback(*_args, **_kwargs):
        return {
            "results": [
                {
                    "content": "Evidence from domain X.",
                    "metadata": {"university": "Web X", "url": "https://x.de/a"},
                },
                {
                    "content": "Evidence from domain Y.",
                    "metadata": {"university": "Web Y", "url": "https://y.eu/b"},
                },
            ]
        }

    async def fake_rerank(_query, candidates):
        assert len(candidates) == 2
        # Simulate reranker collapse to one domain; pipeline should restore domain diversity.
        return {
            "results": [candidates[0]],
            "applied": True,
            "timings_ms": {"total": 6},
        }

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)
    monkeypatch.setattr(llm_service, "aretrieve_web_chunks", fake_web_fallback)
    monkeypatch.setattr(llm_service, "arerank_retrieval_results", fake_rerank)

    result = await llm_service.generate_response("user-1", "compare x and y")
    assert "Comparison:" in result


@pytest.mark.asyncio
async def test_generate_response_rerank_restores_comparison_entity_coverage(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.web_search, "enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "retrieval_min_unique_domains", 1)
    monkeypatch.setattr(llm_service.settings.web_search, "max_context_results", 4)

    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def fake_primary(messages):
        joined = "\n".join(
            str(message.get("content", "")) for message in messages if isinstance(message, dict)
        )
        assert "TUM data evidence." in joined
        assert "LMU data evidence." in joined
        return FakeResponse("Comparison: TUM and LMU both have relevant programs.")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {"retrieval_strategy": "ann", "results": []}

    async def fake_web_fallback(*_args, **_kwargs):
        return {
            "results": [
                {
                    "content": "TUM data evidence.",
                    "metadata": {"url": "https://www.tum.de/programs/data"},
                },
                {
                    "content": "LMU data evidence.",
                    "metadata": {"url": "https://www.lmu.de/programs/data"},
                },
            ]
        }

    async def fake_rerank(_query, candidates):
        assert len(candidates) == 2
        # Simulate reranker collapsing to only one side of the comparison.
        return {"results": [candidates[1]], "applied": True, "timings_ms": {"total": 5}}

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)
    monkeypatch.setattr(llm_service, "aretrieve_web_chunks", fake_web_fallback)
    monkeypatch.setattr(llm_service, "arerank_retrieval_results", fake_rerank)

    result = await llm_service.generate_response("user-1", "Compare TUM vs LMU data science programs")
    assert "Comparison:" in result


@pytest.mark.asyncio
async def test_generate_response_returns_sorry_when_web_fallback_has_no_results(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.web_search, "enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_enabled", True)

    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def should_not_call_model(_messages):
        raise AssertionError("model should not run when retrieval+web fallback found nothing")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {"retrieval_strategy": "ann", "results": []}

    async def fake_web_fallback(*_args, **_kwargs):
        return {"results": []}

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", should_not_call_model)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)
    monkeypatch.setattr(llm_service, "aretrieve_web_chunks", fake_web_fallback)

    result = await llm_service.generate_response("user-1", "latest oxford ai admission")
    assert result == "Sorry, no relevant information is found."


@pytest.mark.asyncio
async def test_generate_response_returns_timeout_detail_when_web_retrieval_times_out(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.web_search, "enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_enabled", True)

    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def should_not_call_model(_messages):
        raise AssertionError("model should not run when retrieval times out with no evidence")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {"retrieval_strategy": "ann", "results": []}

    async def fake_web_fallback(*args, **kwargs):
        _ = args, kwargs
        return {"results": [], "_timed_out": True, "_query": "timeout-query", "_search_mode": "deep"}

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", should_not_call_model)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)
    monkeypatch.setattr(llm_service, "aretrieve_web_chunks", fake_web_fallback)

    result = await llm_service.generate_response("user-1", "latest oxford ai admission")
    assert result == llm_service._WEB_RETRIEVAL_TIMEOUT_DETAIL


@pytest.mark.asyncio
async def test_generate_response_uses_vector_when_web_fallback_empty(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.web_search, "enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_similarity_threshold", 0.35)

    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def fake_primary(messages):
        assert any(
            isinstance(message, dict)
            and "Retrieved long-term knowledge" in str(message.get("content", ""))
            for message in messages
        )
        return FakeResponse("vector-response")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {
            "retrieval_strategy": "ann",
            "results": [
                {
                    "content": "Vector-backed result remains available.",
                    "distance": 0.9,
                    "metadata": {"university": "Saarland University"},
                }
            ],
        }

    async def fake_web_fallback(*_args, **_kwargs):
        return {"results": []}

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)
    monkeypatch.setattr(llm_service, "aretrieve_web_chunks", fake_web_fallback)

    result = await llm_service.generate_response("user-1", "saarland ai")
    assert result == "vector-response"


@pytest.mark.asyncio
async def test_generate_response_tries_web_when_vector_has_no_urls(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.web_search, "enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_similarity_threshold", 0.05)
    monkeypatch.setattr(llm_service, "_evidence_urls", lambda _results: [])

    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def fake_primary(_messages):
        raise AssertionError("model should not run when evidence has no URLs and web is empty")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {
            "retrieval_strategy": "ann",
            "results": [
                {
                    "content": "Vector result without URL metadata.",
                    "distance": 0.1,
                    "metadata": {"university": "Saarland University"},
                }
            ],
        }

    web_calls = {"count": 0}

    async def fake_web_fallback(*_args, **_kwargs):
        web_calls["count"] += 1
        return {"results": []}

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)
    monkeypatch.setattr(llm_service, "aretrieve_web_chunks", fake_web_fallback)

    result = await llm_service.generate_response("user-1", "saarland ai")
    assert result == "Sorry, no relevant information is found."
    assert web_calls["count"] >= 1


@pytest.mark.asyncio
async def test_generate_response_returns_sorry_when_strict_citation_has_no_evidence(monkeypatch):
    monkeypatch.setattr(llm_service, "_is_citation_grounding_required", lambda: True)
    monkeypatch.setattr(llm_service.settings.web_search, "enabled", False)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_enabled", False)

    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {"retrieval_strategy": "ann", "results": []}

    async def should_not_call_model(_messages):
        raise AssertionError("model should not run without evidence in strict citation mode")

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)
    monkeypatch.setattr(llm_service, "_call_primary", should_not_call_model)

    result = await llm_service.generate_response("user-1", "oxford ai admission")
    assert result == "Sorry, no relevant information is found."


@pytest.mark.asyncio
async def test_generate_response_uses_fallback_when_primary_fails(monkeypatch):
    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def fake_primary(_messages):
        raise RuntimeError("primary down")

    async def fake_fallback(_messages):
        return FakeResponse("fallback-response")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {"results": []}

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "_call_fallback", fake_fallback)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)

    result = await llm_service.generate_response("user-1", "find university course")
    assert result == "fallback-response"


@pytest.mark.asyncio
async def test_generate_response_raises_when_both_models_fail(monkeypatch):
    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def fake_primary(_messages):
        raise RuntimeError("primary down")

    async def fake_fallback(_messages):
        raise RuntimeError("fallback down")

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {"results": []}

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "_call_fallback", fake_fallback)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)

    with pytest.raises(RuntimeError, match="fallback down"):
        await llm_service.generate_response("user-1", "find university course")


@pytest.mark.asyncio
async def test_generate_response_blocks_on_input_guardrail(monkeypatch):
    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    def fake_input_guard(_user_id, _prompt):
        return {"blocked": True, "sanitized_text": "", "reason": "blocked_input_pattern"}

    async def should_not_run(*_args, **_kwargs):
        raise AssertionError("build_context should not run when input is blocked")

    monkeypatch.setattr(llm_service, "guard_user_input", fake_input_guard)
    monkeypatch.setattr(llm_service, "build_context", should_not_run)

    result = await llm_service.generate_response("user-1", "bad prompt")
    assert result == llm_service.refusal_response()


@pytest.mark.asyncio
async def test_generate_response_injects_chat_system_prompt(monkeypatch):
    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    def fake_apply_context_guardrails(messages):
        assert messages[0]["role"] == "system"
        assert "UniGraph" in messages[0]["content"]
        assert messages[1]["role"] == "system"
        assert "Citation policy" in messages[1]["content"]
        assert messages[2]["role"] == "system"
        assert "Retrieved long-term knowledge" in messages[2]["content"]
        return {"blocked": False, "messages": messages, "reason": ""}

    async def fake_primary(_messages):
        return FakeResponse("primary-response")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {
            "results": [
                {
                    "content": (
                        "Master of Science in Artificial Intelligence Systems "
                        "is offered in Germany."
                    ),
                    "metadata": {
                        "university": "Rheinberg Technical University (RTU)",
                        "section_heading": "Master of Science in Artificial Intelligence Systems",
                    },
                }
            ]
        }

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "apply_context_guardrails", fake_apply_context_guardrails)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)

    result = await llm_service.generate_response("user-1", "find university course")
    assert result == "primary-response"


@pytest.mark.asyncio
async def test_generate_response_stream_primary_success(monkeypatch):
    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def fake_stream_primary(_messages):
        yield "he"
        yield "llo"

    async def fake_stream_fallback(_messages):
        raise AssertionError("fallback stream should not run when primary stream succeeds")
        if False:  # pragma: no cover
            yield ""

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {"results": []}

    async def fake_persist_evaluation_trace(*_args, **_kwargs):
        return None

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_stream_primary", fake_stream_primary)
    monkeypatch.setattr(llm_service, "_stream_fallback", fake_stream_fallback)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)
    monkeypatch.setattr(llm_service, "_persist_evaluation_trace", fake_persist_evaluation_trace)
    monkeypatch.setattr(
        llm_service,
        "guard_user_input",
        lambda _user_id, prompt: {"blocked": False, "sanitized_text": prompt, "reason": ""},
    )
    monkeypatch.setattr(
        llm_service,
        "apply_context_guardrails",
        lambda messages: {"blocked": False, "messages": messages, "reason": ""},
    )
    monkeypatch.setattr(
        llm_service,
        "guard_model_output",
        lambda text: {"blocked": False, "text": text, "reason": ""},
    )

    outputs = []
    async for partial in llm_service.generate_response_stream("user-1", "hello"):
        outputs.append(partial)

    assert outputs
    assert outputs[-1] == "hello"
    cache_key = llm_service._chat_cache_key("user-1", "hello")
    assert fake_redis.store[cache_key] == "hello"


@pytest.mark.asyncio
async def test_generate_response_stream_uses_fallback_on_primary_failure(monkeypatch):
    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def fake_stream_primary(_messages):
        raise RuntimeError("primary down")
        if False:  # pragma: no cover
            yield ""

    async def fake_stream_fallback(_messages):
        yield "fallback"

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {"results": []}

    async def fake_persist_evaluation_trace(*_args, **_kwargs):
        return None

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_stream_primary", fake_stream_primary)
    monkeypatch.setattr(llm_service, "_stream_fallback", fake_stream_fallback)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)
    monkeypatch.setattr(llm_service, "_persist_evaluation_trace", fake_persist_evaluation_trace)
    monkeypatch.setattr(
        llm_service,
        "guard_user_input",
        lambda _user_id, prompt: {"blocked": False, "sanitized_text": prompt, "reason": ""},
    )
    monkeypatch.setattr(
        llm_service,
        "apply_context_guardrails",
        lambda messages: {"blocked": False, "messages": messages, "reason": ""},
    )
    monkeypatch.setattr(
        llm_service,
        "guard_model_output",
        lambda text: {"blocked": False, "text": text, "reason": ""},
    )

    outputs = []
    async for partial in llm_service.generate_response_stream("user-1", "hello"):
        outputs.append(partial)

    assert outputs
    assert outputs[-1] == "fallback"


@pytest.mark.asyncio
async def test_generate_response_stream_blocks_unsafe_output_before_emit(monkeypatch):
    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def fake_stream_primary(_messages):
        yield "safe intro"
        yield " unsafe payload"

    async def fake_stream_fallback(_messages):
        raise AssertionError("fallback should not run")
        if False:  # pragma: no cover
            yield ""

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {"results": []}

    async def fake_persist_evaluation_trace(*_args, **_kwargs):
        return None

    def fake_guard_model_output(text: str):
        if "unsafe payload" in text:
            return {
                "blocked": True,
                "text": "guardrail refusal",
                "reason": "blocked_output_pattern",
            }
        return {"blocked": False, "text": text, "reason": ""}

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_stream_primary", fake_stream_primary)
    monkeypatch.setattr(llm_service, "_stream_fallback", fake_stream_fallback)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)
    monkeypatch.setattr(llm_service, "_persist_evaluation_trace", fake_persist_evaluation_trace)
    monkeypatch.setattr(llm_service, "guard_model_output", fake_guard_model_output)
    monkeypatch.setattr(llm_service, "_STREAM_GUARD_HOLDBACK_CHARS", 0)
    monkeypatch.setattr(
        llm_service,
        "guard_user_input",
        lambda _user_id, prompt: {"blocked": False, "sanitized_text": prompt, "reason": ""},
    )
    monkeypatch.setattr(
        llm_service,
        "apply_context_guardrails",
        lambda messages: {"blocked": False, "messages": messages, "reason": ""},
    )

    outputs = []
    async for partial in llm_service.generate_response_stream("user-1", "hello"):
        outputs.append(partial)

    assert outputs[-1] == "guardrail refusal"
    assert all("unsafe payload" not in partial for partial in outputs)


@pytest.mark.asyncio
async def test_call_primary_uses_mock_mode_without_bedrock_client(monkeypatch):
    monkeypatch.setenv("LLM_MOCK_MODE", "true")
    monkeypatch.setenv("LLM_MOCK_TEXT", "mocked completion")

    async def should_not_call(**_kwargs):
        raise AssertionError("Bedrock client should not be called in mock mode")

    monkeypatch.setattr(llm_service.client.chat.completions, "create", should_not_call)

    response = await llm_service._call_primary([{"role": "user", "content": "hello"}])

    assert response.choices[0].message.content == "mocked completion"
    assert response.usage.total_tokens >= response.usage.prompt_tokens


def test_model_ids_for_role_uses_role_specific_settings(monkeypatch):
    monkeypatch.setattr(llm_service.settings.bedrock, "primary_model_id", "primary-model")
    monkeypatch.setattr(llm_service.settings.bedrock, "fallback_model_id", "fallback-model")
    monkeypatch.setattr(llm_service.settings.bedrock, "planner_model_id", "planner-model")
    monkeypatch.setattr(llm_service.settings.bedrock, "planner_fallback_model_id", "planner-fallback")
    monkeypatch.setattr(llm_service.settings.bedrock, "worker_model_id", "worker-model")
    monkeypatch.setattr(llm_service.settings.bedrock, "worker_fallback_model_id", "worker-fallback")
    monkeypatch.setattr(
        llm_service.settings.bedrock, "worker_escalation_model_id", "worker-escalated"
    )
    monkeypatch.setattr(llm_service.settings.bedrock, "verifier_model_id", "verifier-model")
    monkeypatch.setattr(llm_service.settings.bedrock, "verifier_fallback_model_id", "")
    monkeypatch.setattr(llm_service.settings.bedrock, "finalizer_model_id", "finalizer-model")
    monkeypatch.setattr(llm_service.settings.bedrock, "finalizer_fallback_model_id", "")

    assert llm_service._model_ids_for_role("planner") == ("planner-model", "planner-fallback")
    assert llm_service._model_ids_for_role("worker", attempt=1) == (
        "worker-model",
        "worker-fallback",
    )
    assert llm_service._model_ids_for_role("worker", attempt=2) == (
        "worker-escalated",
        "worker-fallback",
    )
    assert llm_service._model_ids_for_role("verifier") == ("verifier-model", "fallback-model")
    assert llm_service._model_ids_for_role("finalizer") == ("finalizer-model", "fallback-model")


@pytest.mark.asyncio
async def test_call_model_with_fallback_uses_role_model_selection(monkeypatch):
    monkeypatch.setattr(llm_service.settings.bedrock, "primary_model_id", "primary-model")
    monkeypatch.setattr(llm_service.settings.bedrock, "fallback_model_id", "fallback-model")
    monkeypatch.setattr(llm_service.settings.bedrock, "planner_model_id", "planner-model")
    monkeypatch.setattr(llm_service.settings.bedrock, "planner_fallback_model_id", "planner-fallback")
    monkeypatch.setattr(llm_service.settings.bedrock, "worker_model_id", "worker-model")
    monkeypatch.setattr(llm_service.settings.bedrock, "worker_fallback_model_id", "worker-fallback")
    monkeypatch.setattr(
        llm_service.settings.bedrock, "worker_escalation_model_id", "worker-escalated"
    )

    calls: list[str] = []

    async def fake_call_model_by_id(_messages, *, model_id: str):
        calls.append(model_id)
        return FakeResponse("ok")

    monkeypatch.setattr(llm_service, "_call_model_by_id", fake_call_model_by_id)

    state = {"used_fallback_model": False}
    await llm_service._call_model_with_fallback(
        [{"role": "user", "content": "hello"}],
        state,
        role="planner",
    )
    await llm_service._call_model_with_fallback(
        [{"role": "user", "content": "hello"}],
        state,
        role="worker",
        attempt=2,
    )

    assert calls == ["planner-model", "worker-escalated"]
    assert state["role_model_ids"]["planner"]["primary"] == "planner-model"
    assert state["role_model_ids"]["worker"]["primary"] == "worker-escalated"


@pytest.mark.asyncio
async def test_stream_primary_uses_mock_mode_without_bedrock_client(monkeypatch):
    monkeypatch.setenv("LLM_MOCK_MODE", "true")
    monkeypatch.setenv("LLM_MOCK_TEXT", "mock stream text")
    monkeypatch.setenv("LLM_MOCK_STREAM_CHUNK_CHARS", "4")

    async def should_not_stream(**_kwargs):
        raise AssertionError("Bedrock stream should not be called in mock mode")
        if False:  # pragma: no cover
            yield ""

    monkeypatch.setattr(llm_service.client.chat.completions, "stream", should_not_stream)

    chunks = []
    async for chunk in llm_service._stream_primary([{"role": "user", "content": "hello"}]):
        chunks.append(chunk)

    assert "".join(chunks) == "mock stream text"
    assert len(chunks) >= 2


def test_chat_cache_key_uses_hash_without_raw_prompt_text():
    prompt = "my secret prompt text"
    cache_key = llm_service._chat_cache_key("user-1", prompt)

    assert prompt not in cache_key
    assert "sha256:" in cache_key
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() in cache_key


def test_chat_cache_key_changes_across_sessions():
    prompt = "same prompt"
    key_a = llm_service._chat_cache_key("user-1", prompt, "session-a")
    key_b = llm_service._chat_cache_key("user-1", prompt, "session-b")

    assert key_a != key_b


def test_chat_cache_key_changes_across_modes():
    prompt = "same prompt"
    key_fast = llm_service._chat_cache_key("user-1", prompt, "session-a", "fast")
    key_auto = llm_service._chat_cache_key("user-1", prompt, "session-a", "auto")

    assert key_fast != key_auto
    assert "mode:fast" in key_fast
    assert "mode:auto" in key_auto


def test_resolve_initial_execution_mode_routes_auto_queries():
    assert llm_service._resolve_initial_execution_mode("auto", "hello") == "fast"
    assert (
        llm_service._resolve_initial_execution_mode(
            "auto",
            "tell me fau erlangen-nuernberg msc artificial intelligence course requirements and language requirements for international students",
        )
        == "deep"
    )
    assert (
        llm_service._resolve_initial_execution_mode(
            "auto",
            "compare top AI universities in germany and include latest deadlines",
        )
        == "deep"
    )


def test_normalized_request_mode_maps_standard_to_fast():
    assert llm_service._normalized_request_mode("standard") == "fast"
    assert llm_service._resolve_initial_execution_mode("standard", "hello") == "fast"


def test_apply_answer_policy_skips_template_for_fast_mode(monkeypatch):
    monkeypatch.setattr(llm_service, "_is_citation_grounding_required", lambda: True)
    state = {
        "execution_mode": "fast",
        "citation_required": True,
        "evidence_urls": ["https://example.edu/source"],
        "output_guard_reason": "",
    }
    answer = "University details are available (https://example.edu/source)."
    result = llm_service._apply_answer_policy(answer, state)
    assert result == answer


def test_apply_answer_policy_skips_template_for_deep_mode_when_not_partial(monkeypatch):
    monkeypatch.setattr(llm_service, "_is_citation_grounding_required", lambda: True)
    state = {
        "execution_mode": "deep",
        "citation_required": True,
        "evidence_urls": ["https://example.edu/source"],
        "output_guard_reason": "",
    }
    answer = "University details are available (https://example.edu/source)."
    result = llm_service._apply_answer_policy(answer, state)
    assert result == answer


def test_apply_answer_policy_keeps_answer_unchanged_for_partial_fallback(monkeypatch):
    monkeypatch.setattr(llm_service, "_is_citation_grounding_required", lambda: True)
    state = {
        "execution_mode": "deep",
        "citation_required": True,
        "evidence_urls": ["https://example.edu/source"],
        "output_guard_reason": "agent_verification_partial",
    }
    answer = "University details are available (https://example.edu/source)."

    result = llm_service._apply_answer_policy(answer, state)
    assert result == answer


def test_append_uncertainty_and_missing_info_skip_for_fast_mode(monkeypatch):
    monkeypatch.setattr(llm_service, "_is_citation_grounding_required", lambda: True)
    state = {
        "execution_mode": "fast",
        "citation_required": True,
        "evidence_urls": ["https://example.edu/source"],
        "agent_last_issues": ["missing: exact application deadline"],
        "trust_uncertainty_reasons": ["Independent source corroboration is limited."],
        "output_guard_reason": "agent_verification_partial",
    }
    answer = "Program details are available (https://example.edu/source)."

    with_uncertainty = llm_service._append_uncertainty_section(answer, state)
    with_missing = llm_service._append_missing_info_section(with_uncertainty, state)

    assert with_missing == answer


def test_build_retrieval_query_uses_only_latest_user_turn_for_new_topic():
    messages = [
        {"role": "user", "content": "Compare TU Munich vs RWTH Aachen MSc AI"},
        {"role": "assistant", "content": "Working on it."},
        {"role": "user", "content": "tell me about open admission in german universities"},
    ]

    query = llm_service._build_retrieval_query(messages)

    assert query == "tell me about open admission in german universities"


def test_build_retrieval_query_keeps_previous_user_turn_for_followup():
    messages = [
        {"role": "user", "content": "What is TU Munich MSc AI deadline?"},
        {"role": "assistant", "content": "Here is TU Munich info."},
        {"role": "user", "content": "What about RWTH Aachen?"},
    ]

    query = llm_service._build_retrieval_query(messages)

    assert "What is TU Munich MSc AI deadline?" in query
    assert "What about RWTH Aachen?" in query


def test_build_retrieval_query_keeps_previous_user_turn_for_context_dependent_deadline_prompt():
    messages = [
        {"role": "user", "content": "tell me about university of darmstadt Msc AI"},
        {"role": "assistant", "content": "Here is TU Darmstadt info."},
        {
            "role": "user",
            "content": "tell the deadline of the university and the application requirements.",
        },
    ]

    query = llm_service._build_retrieval_query(messages)

    assert "tell me about university of darmstadt Msc AI" in query
    assert "tell the deadline of the university and the application requirements." in query


@pytest.mark.asyncio
async def test_prepare_request_auto_escalates_to_deep_on_context_refusal(monkeypatch):
    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_read_cached_response(_cache_key):
        return None, 0

    call_modes: list[str] = []

    async def fake_prepare_messages_for_model(
        *,
        user_id,
        conversation_user_id,
        safe_user_prompt,
        execution_mode,
        policy,
        state,
    ):
        _ = (user_id, conversation_user_id, policy)
        call_modes.append(execution_mode)
        if execution_mode == "fast":
            state["context_guard_reason"] = "no_relevant_information"
            return None, llm_service._NO_RELEVANT_INFORMATION_DETAIL
        return [{"role": "user", "content": safe_user_prompt}], None

    monkeypatch.setattr(llm_service, "_read_cached_response", fake_read_cached_response)
    monkeypatch.setattr(llm_service, "_prepare_messages_for_model", fake_prepare_messages_for_model)

    context, messages, refusal = await llm_service._prepare_request(
        "user-1",
        "hello",
        None,
        mode="auto",
    )

    assert refusal is None
    assert messages == [{"role": "user", "content": "hello"}]
    assert call_modes == ["fast", "deep"]
    assert context["execution_mode"] == "deep"
    assert context["state"]["auto_escalated"] is True


def test_build_json_metrics_record_uses_metrics_state_and_legacy_kwargs():
    record = llm_service._build_json_metrics_record(
        request_id="req-1",
        started_at=0.0,
        user_id="user-1",
        session_id="session-1",
        user_prompt="prompt",
        safe_user_prompt="prompt",
        answer="answer",
        outcome="success",
        metrics_state={
            "model_ms": 11,
            "build_context_ms": 12,
            "retrieval_ms": 13,
            "retrieved_count": 2,
            "retrieval_strategy": "hybrid",
            "quality": {"hallucination_proxy": 0.1},
        },
        llm_usage={"total_tokens": 5},
        used_fallback_model=True,
    )

    assert record["timings_ms"]["llm_response_ms"] == 11
    assert record["timings_ms"]["short_term_memory_ms"] == 12
    assert record["timings_ms"]["long_term_memory_ms"] == 13
    assert record["retrieval"]["strategy"] == "hybrid"
    assert record["retrieval"]["result_count"] == 2
    assert record["llm_usage"] == {"total_tokens": 5}
    assert record["model"]["used_fallback"] is True
    assert record["hallucination_proxy"] == 0.1


def test_retrieval_helpers_handle_invalid_and_valid_inputs():
    assert llm_service._retrieval_result_label(None, 3) == "Result 3"
    assert (
        llm_service._retrieval_result_label({"university": "Uni", "section_heading": "Programs"}, 1)
        == "Uni | Programs"
    )
    assert llm_service._retrieval_content_and_metadata(None) == ("", {})
    assert llm_service._retrieval_content_and_metadata({"content": "   "}) == ("", {})
    assert llm_service._retrieval_content_and_metadata(
        {"content": "Chunk text", "metadata": {"country": "DE"}}
    ) == ("Chunk text", {"country": "DE"})


def test_format_retrieval_context_dedupes_and_limits_results():
    payload = {
        "results": [
            {
                "content": "Alpha program details",
                "metadata": {"university": "A", "section_heading": "Programs"},
            },
            {
                "content": "Alpha   program   details",
                "metadata": {"university": "A2", "section_heading": "Programs 2"},
            },
            {
                "content": "Beta research details",
                "metadata": {"university": "B", "section_heading": "Research"},
            },
        ]
    }
    message = llm_service._format_retrieval_context(payload)

    assert message is not None
    assert message["role"] == "system"
    content = message["content"]
    assert "1. A | Programs: Alpha program details" in content
    assert "2. B | Research: Beta research details" in content
    assert "A2 | Programs 2" not in content


def test_enforce_citation_grounding_accepts_allowed_host():
    state = {
        "citation_required": True,
        "evidence_urls": ["https://www.ox.ac.uk/admissions"],
        "output_guard_reason": "",
    }
    answer = "Check https://www.ox.ac.uk/admissions for official requirements."
    assert _REAL_ENFORCE_CITATION_GROUNDING(answer, state) == answer


def test_enforce_citation_grounding_returns_abstain_without_citation():
    state = {
        "citation_required": True,
        "evidence_urls": ["https://www.ox.ac.uk/admissions"],
        "output_guard_reason": "",
    }
    result = _REAL_ENFORCE_CITATION_GROUNDING("Requirements are competitive.", state)
    assert result == "Sorry, no relevant information is found."
    assert state["output_guard_reason"] == "missing_citations"


def test_enforce_citation_grounding_requires_evidence_when_policy_enabled(monkeypatch):
    monkeypatch.setattr(llm_service, "_is_citation_grounding_required", lambda: True)
    state = {
        "citation_required": False,
        "evidence_urls": [],
        "output_guard_reason": "",
    }
    result = _REAL_ENFORCE_CITATION_GROUNDING("Some answer text.", state)
    assert result == "Sorry, no relevant information is found."
    assert state["output_guard_reason"] == "weak_evidence_missing"


def test_enforce_citation_grounding_allows_uncited_comparison_fallback(monkeypatch):
    monkeypatch.setattr(llm_service, "_is_citation_grounding_required", lambda: True)
    state = {
        "citation_required": False,
        "allow_uncited_comparison_fallback": True,
        "query_intent": "comparison",
        "output_guard_reason": "",
    }
    answer = "Comparison: TUM vs LMU."
    assert _REAL_ENFORCE_CITATION_GROUNDING(answer, state) == answer


def test_extract_guarded_result_deadline_missing_date_is_partial_not_abstain(monkeypatch):
    monkeypatch.setattr(
        llm_service,
        "guard_model_output",
        lambda raw: {"text": str(raw), "blocked": False, "reason": ""},
    )
    monkeypatch.setattr(llm_service, "_enforce_citation_grounding", lambda text, _state: text)
    state = {"deadline_query": True}
    result = llm_service._extract_guarded_result(
        user_id="user-1",
        raw_result="Program language is English and TOEFL is accepted.",
        state=state,
    )
    assert result != llm_service._NO_RELEVANT_INFORMATION_DETAIL
    assert "Application deadline: Not verified from sources." in result
    assert state["output_guard_reason"] == "deadline_missing_date"


def test_required_answer_fields_and_missing_detection_for_comparison():
    prompt = (
        "Compare TUM vs LMU for English-taught data science master's programs, "
        "including admission requirements and application deadlines."
    )
    required = llm_service._required_answer_fields(prompt, intent="comparison")
    assert "comparison_between_requested_entities" in required
    assert "application_deadline" in required
    assert "eligibility_requirements" in required

    state = {
        "required_answer_fields": required,
        "comparison_entities": ["TUM", "LMU"],
    }
    missing = llm_service._missing_required_answer_fields(
        "TUM has strong data programs with good labs.",
        state,
    )
    assert "comparison_between_requested_entities" in missing
    assert "application_deadline" in missing


def test_required_answer_fields_for_admissions_requirements_query():
    prompt = (
        "Tell me MSc AI course requirements, language requirements for international students, "
        "minimum IELTS/TOEFL scores, and whether I can get admission."
    )
    required = llm_service._required_answer_fields(prompt, intent="fact_lookup")
    assert "eligibility_requirements" in required
    assert "gpa_or_grade_threshold" in required
    assert "ects_or_prerequisite_credit_breakdown" in required
    assert "language_requirements" in required
    assert "language_test_score_thresholds" in required
    assert "admission_decision_signal" in required


def test_required_answer_fields_language_requirement_only_does_not_force_gpa_or_ects():
    prompt = "What is the language requirement for international students in MSc AI?"
    required = llm_service._required_answer_fields(prompt, intent="fact_lookup")
    assert "language_requirements" in required
    assert "language_test_score_thresholds" not in required
    assert "eligibility_requirements" not in required
    assert "gpa_or_grade_threshold" not in required
    assert "ects_or_prerequisite_credit_breakdown" not in required


def test_required_answer_fields_include_application_portal_when_requested():
    prompt = (
        "Tell me MSc AI course requirements, language requirements, admission deadline, "
        "and application portal for international students."
    )
    required = llm_service._required_answer_fields(prompt, intent="fact_lookup")
    assert "application_deadline" in required
    assert "application_portal" in required


def test_required_answer_fields_requirements_language_do_not_force_decision_signal():
    prompt = (
        "Tell me MSc AI course requirements and language requirements for international students."
    )
    required = llm_service._required_answer_fields(prompt, intent="fact_lookup")
    assert "eligibility_requirements" in required
    assert "language_requirements" in required
    assert "admission_decision_signal" not in required


def test_required_answer_fields_detects_where_to_apply_as_portal():
    prompt = "Tell me from where can I apply for this MSc course."
    required = llm_service._required_answer_fields(prompt, intent="fact_lookup")
    assert "application_portal" in required


def test_required_answer_fields_detects_taught_subjects_as_curriculum():
    prompt = "What are the things taught in this MSc course?"
    required = llm_service._required_answer_fields(prompt, intent="fact_lookup")
    assert "curriculum_focus" in required


def test_admissions_answer_schema_message_enabled_for_requirements_query():
    state = {
        "required_answer_fields": [
            "eligibility_requirements",
            "gpa_or_grade_threshold",
            "language_test_score_thresholds",
        ]
    }
    message = llm_service._admissions_answer_schema_message(state)
    assert message is not None
    assert "Admissions answer schema" in message["content"]
    assert "Eligibility Requirements" in message["content"]
    assert "Language Requirements" in message["content"]
    assert "Do not include an Admission Decision section unless explicitly requested." in message["content"]


def test_missing_comparison_entities_detects_uncovered_entity():
    results = [
        {
            "content": "Data Engineering and Analytics at TUM.",
            "metadata": {"url": "https://www.tum.de/en/studies/degree-programs/detail/program"},
        }
    ]
    missing = llm_service._missing_comparison_entities(results, ["TUM", "LMU"])
    assert missing == ["LMU"]


def test_build_structured_comparison_from_evidence_contains_required_fields():
    state = {
        "comparison_entities": ["TUM", "LMU"],
        "retrieved_results": [
            {
                "content": "TUM Data Engineering and Analytics is an English-taught master's program.",
                "metadata": {"url": "https://www.tum.de/en/studies/degree-programs/detail/program"},
            },
            {
                "content": "LMU statistics program provides data science focus areas.",
                "metadata": {"url": "https://www.lmu.de/en/studies/programs/data-science"},
            },
        ],
        "evidence_urls": [
            "https://www.tum.de/en/studies/degree-programs/detail/program",
            "https://www.lmu.de/en/studies/programs/data-science",
        ],
    }
    answer = llm_service._build_structured_comparison_from_evidence(state)
    assert "Comparison: TUM vs LMU" in answer
    assert "TUM:" in answer and "LMU:" in answer
    assert "Eligibility requirements:" in answer
    assert "Application deadline:" in answer
    assert "https://www.tum.de/en/studies/degree-programs/detail/program" in answer
    assert "https://www.lmu.de/en/studies/programs/data-science" in answer


def test_apply_grounded_retrieval_context_relaxes_for_comparison_without_urls(monkeypatch):
    monkeypatch.setattr(llm_service, "_evidence_urls", lambda _results: [])
    state = {
        "query_intent": "comparison",
        "comparison_entities": ["TUM", "LMU"],
    }
    messages = [{"role": "user", "content": "compare tum vs lmu"}]
    merged_results = [
        {
            "content": "TUM data evidence.",
            "metadata": {},
        },
        {
            "content": "LMU data evidence.",
            "metadata": {},
        },
    ]
    updated_messages, detail = llm_service._apply_grounded_retrieval_context(
        messages=messages,
        merged_results=merged_results,
        used_web_results=False,
        state=state,
    )
    assert detail is None
    assert state["allow_uncited_comparison_fallback"] is True
    assert state["citation_required"] is False
    assert isinstance(updated_messages, list)


def test_web_expansion_queries_for_comparison_include_missing_entity_target():
    state = {
        "query_intent": "comparison",
        "comparison_entities": ["TUM", "LMU"],
    }
    current_results = [
        {
            "content": "TUM Data Engineering and Analytics details.",
            "metadata": {"url": "https://www.tum.de/en/studies/degree-programs/detail/program"},
        }
    ]
    queries = llm_service._web_expansion_queries(
        base_query=(
            "Compare TUM vs LMU for English-taught data science master's programs, "
            "including admission requirements and application deadlines."
        ),
        state=state,
        low_similarity=True,
        insufficient_domains=True,
        current_results=current_results,
    )
    assert any("site:lmu.de" in query.lower() for query in queries)


def test_agentic_result_issues_marks_query_not_addressed_for_low_required_coverage():
    state = {
        "citation_required": False,
        "evidence_urls": [],
        "required_answer_fields": [
            "comparison_between_requested_entities",
            "application_deadline",
            "required_documents",
        ],
        "comparison_entities": ["TUM", "LMU"],
        "retrieval_single_domain_low_quality": False,
        "deadline_query": False,
    }
    issues = llm_service._agentic_result_issues(
        "I cannot provide a comprehensive comparison with the current evidence.",
        state,
    )
    assert "missing_required_answer_fields" in issues
    assert "query_not_addressed" in issues
    assert state["required_field_coverage"] == 0.0


def test_answer_matches_required_field_for_application_portal():
    assert llm_service._answer_matches_required_field(
        "application_portal",
        "Apply online via the application portal: https://campus.uni-example.de",
    )
    assert not llm_service._answer_matches_required_field(
        "application_portal",
        "Use the application portal.",
    )


def test_agentic_result_issues_marks_missing_web_required_fields_for_admissions():
    state = {
        "citation_required": False,
        "evidence_urls": [],
        "required_answer_fields": [
            "eligibility_requirements",
            "language_requirements",
            "language_test_score_thresholds",
        ],
        "web_required_fields_missing": ["language_score_thresholds", "application_deadline"],
        "retrieval_single_domain_low_quality": False,
        "deadline_query": False,
    }
    issues = llm_service._agentic_result_issues(
        "Program is English taught. Language scores are not verified from sources.",
        state,
    )
    assert "web_required_fields_missing" in issues
    assert any(item.startswith("web_missing:") for item in issues)


def test_agentic_result_issues_marks_confidence_below_target_in_deep_mode():
    state = {
        "citation_required": False,
        "evidence_urls": [],
        "required_answer_fields": [
            "eligibility_requirements",
            "language_requirements",
            "application_deadline",
        ],
        "web_required_fields_missing": [],
        "retrieval_single_domain_low_quality": False,
        "deadline_query": False,
        "execution_mode": "deep",
        "trust_confidence": 0.58,
    }
    issues = llm_service._agentic_result_issues(
        "Eligibility requirement is a relevant bachelor's degree.",
        state,
    )
    assert "confidence_below_target" in issues


def test_agentic_result_issues_marks_speculative_factual_claim_for_admissions():
    state = {
        "citation_required": False,
        "evidence_urls": [],
        "required_answer_fields": [
            "eligibility_requirements",
            "language_requirements",
            "language_test_score_thresholds",
        ],
        "web_required_fields_missing": [],
        "retrieval_single_domain_low_quality": False,
        "deadline_query": False,
    }
    issues = llm_service._agentic_result_issues(
        "Language of instruction is likely English, but exact IELTS score is not confirmed.",
        state,
    )
    assert "speculative_factual_claim" in issues


def test_is_hard_verification_failure_for_admissions_missing_web_fields():
    state = {
        "retrieval_source_count": 2,
        "evidence_urls": ["https://uni-example.de/program"],
        "required_answer_fields": ["eligibility_requirements", "language_requirements"],
        "web_required_fields_missing": ["language_score_thresholds"],
    }
    issues = ["web_required_fields_missing", "web_missing:language_score_thresholds"]
    assert (
        llm_service._is_hard_verification_failure(
            issues,
            "Sample answer with citation (https://uni-example.de/program).",
            state,
        )
        is True
    )


def test_allow_partial_admissions_answer_requires_rescue_exhaustion_for_missing_web_fields(
    monkeypatch,
):
    monkeypatch.setattr(llm_service.settings.web_search, "agentic_required_field_rescue_max_rounds", 2)
    state = {
        "retrieval_source_count": 2,
        "evidence_urls": ["https://uni-example.de/program"],
        "required_answer_fields": [
            "eligibility_requirements",
            "language_requirements",
            "application_deadline",
        ],
        "required_field_coverage": 0.8,
        "web_required_field_coverage": 0.75,
        "web_required_fields_missing": ["application_deadline"],
        "execution_mode": "deep",
        "web_fallback_attempted": True,
        "agent_required_field_rescue_rounds": 0,
        "web_timeout_count": 0,
    }
    assert (
        llm_service._allow_partial_admissions_answer(
            issues=["web_required_fields_missing"],
            result="Program uses English (https://uni-example.de/program).",
            state=state,
        )
        is False
    )


def test_allow_partial_admissions_answer_allows_after_rescue_budget_exhausted(monkeypatch):
    monkeypatch.setattr(llm_service.settings.web_search, "agentic_required_field_rescue_max_rounds", 2)
    state = {
        "retrieval_source_count": 2,
        "evidence_urls": ["https://uni-example.de/program"],
        "required_answer_fields": [
            "eligibility_requirements",
            "language_requirements",
            "application_deadline",
        ],
        "required_field_coverage": 0.8,
        "web_required_field_coverage": 0.75,
        "web_required_fields_missing": ["application_deadline"],
        "execution_mode": "deep",
        "web_fallback_attempted": True,
        "agent_required_field_rescue_rounds": 2,
        "web_timeout_count": 0,
    }
    assert (
        llm_service._allow_partial_admissions_answer(
            issues=["web_required_fields_missing"],
            result="Program uses English (https://uni-example.de/program).",
            state=state,
        )
        is True
    )


def test_is_hard_verification_failure_for_speculative_factual_claim():
    state = {
        "retrieval_source_count": 2,
        "evidence_urls": ["https://uni-example.de/program"],
        "required_answer_fields": ["language_requirements", "language_test_score_thresholds"],
        "web_required_fields_missing": [],
    }
    assert (
        llm_service._is_hard_verification_failure(
            [],
            "Language is likely English based on available pages (https://uni-example.de/program).",
            state,
        )
        is True
    )


def test_cache_skip_reason_handles_abstain_like_variants():
    reason = llm_service._cache_skip_reason(
        "Could not verify the requested details from available evidence. Not enough information.",
        {},
    )
    assert reason == "abstain_like_variant"


def test_cache_skip_reason_rejects_low_confidence_deep_answer():
    reason = llm_service._cache_skip_reason(
        "Answer text with citations.",
        {
            "execution_mode": "deep",
            "trust_confidence": 0.51,
            "web_fallback_attempted": True,
            "web_retrieval_verified": True,
            "retrieval_source_count": 3,
            "web_required_field_coverage": 1.0,
        },
    )
    assert reason == "deep_confidence_low"


@pytest.mark.asyncio
async def test_augment_messages_with_retrieval_fanout_prefetches_web_for_deep_mode(monkeypatch):
    monkeypatch.setattr(llm_service, "_web_retrieval_ready", lambda: (True, "ready"))
    monkeypatch.setattr(llm_service, "_should_run_web_retrieval", lambda: False)
    monkeypatch.setattr(llm_service, "_retrieval_fanout_enabled", lambda: True)

    web_started = asyncio.Event()
    release_web = asyncio.Event()

    async def fake_web_with_timeout(_retrieval_query, *, top_k, search_mode):
        assert top_k == llm_service.settings.postgres.default_top_k
        assert search_mode == "deep"
        web_started.set()
        await release_web.wait()
        return {
            "results": [
                {
                    "content": "Web evidence.",
                    "metadata": {"url": "https://web.example.edu/a"},
                }
            ],
            "query_plan": {"planner": "llm", "llm_used": True, "subquestions": []},
            "query_variants": ["query"],
            "facts": [],
            "retrieval_loop": {"enabled": True},
        }

    async def fake_vector(_retrieval_query, _state):
        await asyncio.wait_for(web_started.wait(), timeout=0.3)
        release_web.set()
        return (
            [
                {
                    "content": "Vector evidence.",
                    "metadata": {"url": "https://vector.example.edu/a"},
                }
            ],
            0.1,
        )

    async def fake_rerank(_query, merged_results, _state):
        return merged_results

    def fake_apply_grounded_retrieval_context(*, messages, merged_results, used_web_results, state):
        _ = state
        assert used_web_results is True
        assert len(merged_results) >= 2
        return messages, None

    monkeypatch.setattr(llm_service, "_run_one_web_query_with_timeout", fake_web_with_timeout)
    monkeypatch.setattr(llm_service, "_retrieve_vector_candidates", fake_vector)
    monkeypatch.setattr(llm_service, "_rerank_if_configured", fake_rerank)
    monkeypatch.setattr(
        llm_service, "_apply_grounded_retrieval_context", fake_apply_grounded_retrieval_context
    )

    base_messages = [{"role": "user", "content": "hello"}]
    state = {"safe_user_prompt": "hello"}
    messages, detail = await llm_service._augment_messages_with_retrieval(
        messages=base_messages,
        retrieval_query="hello",
        search_mode="deep",
        state=state,
    )

    assert detail is None
    assert messages == base_messages


def test_derive_evidence_trust_signals_penalizes_incomplete_web_required_fields():
    results = [
        {
            "content": "Official requirements include ECTS and language scores.",
            "metadata": {
                "url": "https://uni-a.de/program",
                "published_date": "2026-03-10",
                "trust_components": {"authority": 0.9, "agreement": 0.85, "recency": 0.9},
            },
        },
        {
            "content": "Application timeline and documents are listed.",
            "metadata": {
                "url": "https://uni-b.de/admission",
                "published_date": "2026-03-12",
                "trust_components": {"authority": 0.88, "agreement": 0.83, "recency": 0.9},
            },
        },
    ]

    state_complete = {
        "safe_user_prompt": "admission requirements and language scores",
        "retrieval_source_count": 2,
        "web_required_field_coverage": 1.0,
        "web_required_fields_missing": [],
    }
    llm_service._derive_evidence_trust_signals(results, state_complete)

    state_incomplete = {
        "safe_user_prompt": "admission requirements and language scores",
        "retrieval_source_count": 2,
        "web_required_field_coverage": 0.25,
        "web_required_fields_missing": ["language_requirements"],
    }
    llm_service._derive_evidence_trust_signals(results, state_incomplete)

    complete_conf = float(state_complete.get("trust_confidence") or 0.0)
    incomplete_conf = float(state_incomplete.get("trust_confidence") or 0.0)

    assert incomplete_conf < complete_conf
    assert (
        "Some requested fields are not fully verified from web evidence."
        in state_incomplete["trust_uncertainty_reasons"]
    )


def test_apply_answer_policy_keeps_plain_text_without_urls_or_scaffold():
    state = llm_service._new_metrics_state()
    text = "Primary response without URL citations."
    assert llm_service._apply_answer_policy(text, state) == text


def test_apply_answer_policy_rebuilds_clean_sources_section():
    state = llm_service._new_metrics_state()
    state["execution_mode"] = "deep"
    state["evidence_urls"] = [
        "https://uni-example.de/program",
        "https://daad.de/program",
    ]
    raw = (
        "Evidence and caveats:\n"
        "- Program is English-taught (https://uni-example.de/program)\n"
        "Claim-by-Claim Citations\n"
        "Sources:\n"
        "- https://uni-example.de/program\n"
        "- https://uni-example.de/program\n"
    )
    cleaned = llm_service._apply_answer_policy(raw, state)
    assert "Evidence and caveats" not in cleaned
    assert "Claim-by-Claim Citations" not in cleaned
    assert "Sources" in cleaned
    assert cleaned.count("https://uni-example.de/program") == 2
