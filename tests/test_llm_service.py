import asyncio
import hashlib
import logging
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


def test_truncate_query_safely_keeps_whole_words():
    text = "about university application language requirements and deadline details"
    clipped = llm_service._truncate_query_safely(text, max_chars=34)
    assert len(clipped) <= 34
    assert clipped in {
        "about university application",
        "about university application language",
    }


def test_config_prompts_require_natural_synthesis_not_raw_dumping():
    chat_prompts = llm_service.prompts["chat"]
    prompt_text = "\n".join(
        [
            chat_prompts["mode_prompts"]["fast_system_prompt"],
            chat_prompts["mode_prompts"]["deep_system_prompt"],
            chat_prompts["system_prompt"],
        ]
    )

    assert "Summarize evidence into clean" in prompt_text
    assert "never dump raw retrieved chunks" in prompt_text.lower() or "never dump raw source snippets" in prompt_text.lower()
    assert 'Never write "Not verified from official sources"' in prompt_text
    assert "The retrieved official evidence does not state a specific IELTS band score." in prompt_text


def test_runtime_answer_style_prompt_blocks_raw_internal_wording():
    message = llm_service._answer_style_instruction_message(
        "deep",
        {"required_answer_fields": ["application_deadline", "ielts_score"]},
    )
    content = message["content"]

    assert "Summarize retrieved evidence into natural wording" in content
    assert "raw chunks" in content
    assert "answered_fields" in content
    assert "Never write 'Not verified from official sources'" in content


@pytest.mark.asyncio
async def test_finalize_success_logs_and_sanitizes_final_answer_path(monkeypatch, caplog):
    async def noop_async(*_args, **_kwargs):
        return None

    monkeypatch.setattr(llm_service, "_update_memory_with_timing", noop_async)
    monkeypatch.setattr(llm_service, "_write_cache_with_timing", noop_async)
    monkeypatch.setattr(llm_service, "_record_success_metrics", noop_async)
    monkeypatch.setattr(llm_service, "_schedule_evaluation_trace", lambda *_args, **_kwargs: None)

    context = {
        "state": {},
        "conversation_user_id": "user-1",
        "safe_user_prompt": "When is the deadline?",
        "cache_key": "cache-key",
        "user_id": "user-1",
        "user_prompt": "When is the deadline?",
        "request_id": "req-1",
        "started_at": 0.0,
        "effective_session_id": "session-1",
    }
    raw = "Application deadline: ### Application Periods Winter semester: 01 February - 31 May"

    with caplog.at_level(logging.INFO, logger="app.services.llm_service"):
        answer = await llm_service._finalize_success(context, raw)

    assert "###" not in answer
    assert context["state"]["final_answer_source"] == "llm_synthesis"
    assert context["state"]["final_prompt_used"] is True
    assert context["state"]["raw_span_rendered"] is True
    assert "final_answer_before_sanitizer" in context["state"]
    assert "final_answer_after_sanitizer" in context["state"]
    assert "FinalAnswerPath" in caplog.text


@pytest.mark.asyncio
async def test_run_one_web_query_marks_hard_provider_errors(monkeypatch):
    async def _raise_usage_limit(*_args, **_kwargs):
        raise RuntimeError("This request exceeds your plan's set usage limit.")

    monkeypatch.setattr(llm_service, "_aretrieve_web_chunks_with_mode", _raise_usage_limit)

    result = await llm_service._run_one_web_query_with_timeout(
        "mannheim msc business informatics",
        top_k=3,
        search_mode="deep",
    )
    assert isinstance(result, dict)
    assert result.get("_failed") is True
    assert result.get("_hard_error") is True


def test_timeout_rescue_queries_dedup_after_truncation():
    state = {
        "required_answer_fields": ["application_deadline", "language_requirements"],
        "query_intent": "fact_lookup",
    }
    base_query = (
        "about University of Mannheim MSc Business Informatics language of instruction "
        "IELTS/German requirement GPA and ECTS requirements application deadline for international "
        "students where to apply (portal link) application official requirements"
    )
    queries = llm_service._timeout_rescue_queries(base_query=base_query, state=state)
    assert queries
    assert len(queries) == len({item.lower() for item in queries})


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
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.web_search, "enabled", True)
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
    web_queries = []

    async def fake_update_memory(user_id, user_message, assistant_reply):
        memory_updates.append((user_id, user_message, assistant_reply))

    async def fake_web_fallback(query, **kwargs):
        web_queries.append((query, kwargs))
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
                        "url": "https://example.edu/labs/dssl",
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
    monkeypatch.setattr(llm_service, "aretrieve_web_chunks", fake_web_fallback)
    monkeypatch.setattr(llm_service, "_record_json_metrics", fake_record_json_metrics)

    result = await llm_service.generate_response("user-1", "find ai professor")
    assert result == "primary-response"
    assert memory_updates == [("user-1", "find ai professor", "primary-response")]
    assert web_queries
    assert "find ai professor" in web_queries[0][0].lower()
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
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.web_search, "enabled", True)
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

    async def fake_web_fallback(*_args, **_kwargs):
        return {
            "results": [
                {
                    "content": "Admission details for AI program.",
                    "metadata": {
                        "university": "Sample University",
                        "url": "https://example.edu/admission",
                    },
                }
            ]
        }

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_web_chunks", fake_web_fallback)

    result_first = await llm_service.generate_response("user-1", "find university course")
    assert result_first == "This answer has no citations."
    calls_after_first = model_calls["count"]

    result_second = await llm_service.generate_response("user-1", "find university course")
    assert result_second == "This answer has no citations."

    assert model_calls["count"] > calls_after_first
    cache_key = llm_service._chat_cache_key("user-1", "find university course")
    assert cache_key not in fake_redis.store


@pytest.mark.asyncio
async def test_generate_response_agentic_retries_when_citations_missing(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.web_search, "enabled", True)
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

    async def fake_web_fallback(*_args, **_kwargs):
        return {
            "results": [
                {
                    "content": "Oxford AI admission details from official content.",
                    "metadata": {"university": "Oxford", "url": "https://example.edu/oxford"},
                }
            ]
        }

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_web_chunks", fake_web_fallback)

    result = await llm_service.generate_response("user-1", "oxford ai admission")
    assert "https://example.edu/evidence" in result
    assert model_calls["count"] == 2


@pytest.mark.asyncio
async def test_generate_response_agentic_retries_for_source_diversity(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.web_search, "enabled", True)
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

    async def fake_web_fallback(*_args, **_kwargs):
        return {
            "results": [
                {
                    "content": "Evidence from source one.",
                    "metadata": {"university": "Example University", "url": "https://first.example.edu/evidence"},
                },
                {
                    "content": "Evidence from source two.",
                    "metadata": {"university": "Example University", "url": "https://second.example.org/evidence"},
                },
            ]
        }

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_web_chunks", fake_web_fallback)

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
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.web_search, "enabled", True)
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

    async def fake_web_fallback(*_args, **_kwargs):
        return {
            "results": [
                {
                    "content": "Program detail evidence.",
                    "metadata": {"university": "Example University", "url": "https://example.edu/evidence"},
                }
            ]
        }

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_web_chunks", fake_web_fallback)

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


def test_german_selective_filter_preserves_verified_ledger_sources_and_drops_wrong_scope():
    state = llm_service._new_metrics_state()
    state.update(
        {
            "safe_user_prompt": "Tell me about University of Mannheim MSc Business Informatics IELTS ECTS deadline portal",
            "coverage_ledger": [
                {
                    "id": "language_test_score_thresholds",
                    "label": "Language test score thresholds",
                    "status": "found",
                    "value": "TOEFL iBT 72; IELTS 6.0",
                    "source_url": "https://www.uni-mannheim.de/en/academics/before-your-studies/applying/the-a-to-z-of-applying/masters-programs-foreign-language-requirements/",
                    "confidence": 0.95,
                },
                {
                    "id": "application_deadline",
                    "label": "Application deadline",
                    "status": "found",
                    "value": "1 April - 15 May; 15 October - 15 November",
                    "source_url": "https://www.uni-mannheim.de/en/academics/dates/application-deadlines/",
                    "confidence": 0.95,
                },
            ],
        }
    )
    results = [
        {
            "chunk_id": "wrong-bsc",
            "source_path": "https://www.uni-mannheim.de/en/academics/before-your-studies/programs/bsc-business-informatics/",
            "content": "Bachelor's Program in Business Informatics. German C1.",
            "metadata": {"url": "https://www.uni-mannheim.de/en/academics/before-your-studies/programs/bsc-business-informatics/", "title": "Bachelor's Program in Business Informatics"},
        },
        {
            "chunk_id": "overview",
            "source_path": "https://www.uni-mannheim.de/en/academics/before-your-studies/programs/masters-program-in-business-informatics/",
            "content": "Master's Program in Business Informatics. Language of instruction: English.",
            "metadata": {"url": "https://www.uni-mannheim.de/en/academics/before-your-studies/programs/masters-program-in-business-informatics/", "title": "Master's Program in Business Informatics"},
        },
        {
            "chunk_id": "lang",
            "source_path": "https://www.uni-mannheim.de/en/academics/before-your-studies/applying/the-a-to-z-of-applying/masters-programs-foreign-language-requirements/",
            "content": "Master's Program in Business Informatics. TOEFL iBT 72. IELTS 6.0.",
            "metadata": {"url": "https://www.uni-mannheim.de/en/academics/before-your-studies/applying/the-a-to-z-of-applying/masters-programs-foreign-language-requirements/", "title": "Master's Programs Foreign Language Requirements"},
        },
        {
            "chunk_id": "deadline",
            "source_path": "https://www.uni-mannheim.de/en/academics/dates/application-deadlines/",
            "content": "Business Informatics taught in English. 1 April - 15 May. 15 October - 15 November.",
            "metadata": {"url": "https://www.uni-mannheim.de/en/academics/dates/application-deadlines/", "title": "Application deadlines"},
        },
    ]

    selected = llm_service._selective_retrieval_results(results, state)
    urls = {llm_service._result_source_url(item) for item in selected}

    assert "https://www.uni-mannheim.de/en/academics/before-your-studies/programs/bsc-business-informatics/" not in urls
    assert "https://www.uni-mannheim.de/en/academics/before-your-studies/applying/the-a-to-z-of-applying/masters-programs-foreign-language-requirements/" in urls
    assert "https://www.uni-mannheim.de/en/academics/dates/application-deadlines/" in urls


def test_german_answer_issues_detect_not_verified_against_found_ledger_and_wrong_deadline():
    state = llm_service._new_metrics_state()
    state.update(
        {
            "safe_user_prompt": "Tell me about University of Mannheim MSc Business Informatics IELTS deadline",
            "citation_required": False,
            "required_answer_fields": ["language_test_score_thresholds", "application_deadline"],
            "coverage_ledger": [
                {
                    "id": "language_test_score_thresholds",
                    "label": "Language test score thresholds",
                    "status": "found",
                    "value": "TOEFL iBT 72; IELTS 6.0",
                    "source_url": "https://www.uni-mannheim.de/en/academics/before-your-studies/applying/the-a-to-z-of-applying/masters-programs-foreign-language-requirements/",
                },
                {
                    "id": "application_deadline",
                    "label": "Application deadline",
                    "status": "found",
                    "value": "1 April - 15 May; 15 October - 15 November",
                    "source_url": "https://www.uni-mannheim.de/en/academics/dates/application-deadlines/",
                },
            ],
        }
    )
    answer = (
        "IELTS/TOEFL exact minimum score thresholds: Not verified from official sources.\n"
        "Application Deadline\n"
        "- Fall semester: 15 August\n"
        "- Spring semester: 15 January"
    )

    issues = llm_service._agentic_result_issues(answer, state)

    assert "verified_ledger_field_marked_unverified" in issues
    assert "deadline_conflicts_with_verified_ledger" in issues


def test_has_date_like_value_accepts_month_range_deadlines():
    text = "Application deadline: 1 April - 15 May; Spring intake: 15 October - 15 November."
    assert llm_service._has_date_like_value(text) is True


def test_structured_recovery_answer_usable_with_partial_verified_admissions_answer():
    state = llm_service._new_metrics_state()
    state.update(
        {
            "safe_user_prompt": "Tell me about University of Mannheim MSc Business Informatics IELTS ECTS deadline portal",
            "citation_required": True,
            "required_answer_fields": [
                "language_test_score_thresholds",
                "ects_prerequisites",
                "application_deadline",
                "application_portal",
            ],
            "evidence_urls": [
                "https://www.uni-mannheim.de/en/academics/dates/application-deadlines/",
                "https://portal2.uni-mannheim.de/portal2/pages/cs/sys/portal/hisinoneStartPage.faces?page=Bewerbung",
            ],
            "coverage_ledger": [
                {
                    "id": "application_deadline",
                    "label": "Application deadline",
                    "status": "found",
                    "value": "1 April - 15 May; 15 October - 15 November",
                    "source_url": "https://www.uni-mannheim.de/en/academics/dates/application-deadlines/",
                },
                {
                    "id": "application_portal",
                    "label": "Application portal",
                    "status": "found",
                    "value": "https://portal2.uni-mannheim.de/portal2/pages/cs/sys/portal/hisinoneStartPage.faces?page=Bewerbung",
                    "source_url": "https://portal2.uni-mannheim.de/portal2/pages/cs/sys/portal/hisinoneStartPage.faces?page=Bewerbung",
                },
                {
                    "id": "language_test_score_thresholds",
                    "label": "Language test score thresholds",
                    "status": "missing",
                    "value": "Not verified from official sources.",
                    "source_url": "",
                },
            ],
        }
    )
    answer = llm_service._build_structured_field_evidence_answer(state)
    assert "1 April - 15 May" in answer
    assert "portal2.uni-mannheim.de" in answer
    assert llm_service._structured_recovery_answer_usable(answer, state) is True


def test_force_structured_recovery_when_evidence_exists_accepts_partial_verified_answer():
    state = llm_service._new_metrics_state()
    state.update(
        {
            "safe_user_prompt": "Tell me about University of Mannheim MSc Business Informatics language deadline portal ECTS",
            "citation_required": True,
            "web_required_field_coverage": 0.625,
            "required_answer_fields": [
                "language_of_instruction",
                "ects_prerequisites",
                "application_deadline",
                "application_portal",
                "gpa_threshold",
            ],
            "evidence_urls": [
                "https://www.uni-mannheim.de/en/academics/before-your-studies/programs/masters-program-in-business-informatics/",
                "https://www.uni-mannheim.de/en/academics/dates/application-deadlines/",
                "https://portal2.uni-mannheim.de/portal2/pages/cs/sys/portal/hisinoneStartPage.faces?page=Bewerbung",
            ],
            "coverage_ledger": [
                {
                    "id": "language_of_instruction",
                    "label": "Language of instruction",
                    "status": "found",
                    "value": "English",
                    "source_url": "https://www.uni-mannheim.de/en/academics/before-your-studies/programs/masters-program-in-business-informatics/",
                },
                {
                    "id": "ects_prerequisites",
                    "label": "ECTS / prerequisite credits",
                    "status": "found",
                    "value": "30 ECTS informatics; 30 ECTS business/business informatics; 18 ECTS mathematics/statistics; 8 ECTS programming",
                    "source_url": "https://www.uni-mannheim.de/en/academics/before-your-studies/programs/masters-program-in-business-informatics/",
                },
                {
                    "id": "application_deadline",
                    "label": "Application deadline",
                    "status": "found",
                    "value": "1 April - 15 May; 15 October - 15 November",
                    "source_url": "https://www.uni-mannheim.de/en/academics/dates/application-deadlines/",
                },
                {
                    "id": "application_portal",
                    "label": "Application portal",
                    "status": "found",
                    "value": "https://portal2.uni-mannheim.de/portal2/pages/cs/sys/portal/hisinoneStartPage.faces?page=Bewerbung",
                    "source_url": "https://portal2.uni-mannheim.de/portal2/pages/cs/sys/portal/hisinoneStartPage.faces?page=Bewerbung",
                },
                {
                    "id": "gpa_threshold",
                    "label": "Minimum GPA / grade threshold",
                    "status": "missing",
                    "value": "Not verified from official sources.",
                    "source_url": "",
                },
            ],
        }
    )
    answer = llm_service._build_structured_field_evidence_answer(state)
    assert "Language of Instruction" in answer
    assert "Application Deadline" in answer
    assert llm_service._force_structured_recovery_when_evidence_exists(answer, state) is True


def test_force_structured_recovery_allows_speculative_wording_when_evidence_is_strong():
    state = llm_service._new_metrics_state()
    state.update(
        {
            "safe_user_prompt": "Tell me about University of Mannheim MSc Business Informatics deadline and portal",
            "citation_required": True,
            "web_required_field_coverage": 0.625,
            "required_answer_fields": ["application_deadline", "application_portal"],
            "evidence_urls": [
                "https://www.uni-mannheim.de/media/Einrichtungen/zula/Auswahlsatzungen_master/satzung_ma_wifo_en.pdf",
                "https://portal2.uni-mannheim.de/portal2/pages/cs/sys/portal/hisinoneStartPage.faces?page=Bewerbung",
                "https://www.uni-mannheim.de/en/academics/before-your-studies/programs/masters-program-in-business-informatics/",
            ],
            "coverage_ledger": [
                {
                    "id": "application_deadline",
                    "status": "found",
                    "value": "Proof of English proficiency may be submitted by 15 August / 15 January.",
                    "source_url": "https://www.uni-mannheim.de/media/Einrichtungen/zula/Auswahlsatzungen_master/satzung_ma_wifo_en.pdf",
                },
                {
                    "id": "application_portal",
                    "status": "found",
                    "value": "https://portal2.uni-mannheim.de/portal2/pages/cs/sys/portal/hisinoneStartPage.faces?page=Bewerbung",
                    "source_url": "https://portal2.uni-mannheim.de/portal2/pages/cs/sys/portal/hisinoneStartPage.faces?page=Bewerbung",
                },
            ],
        }
    )
    answer = llm_service._build_structured_field_evidence_answer(state)
    assert llm_service._has_speculative_factual_language(answer, state) is True
    assert llm_service._force_structured_recovery_when_evidence_exists(answer, state) is True


def test_can_return_best_effort_admissions_answer_requires_evidence():
    state = llm_service._new_metrics_state()
    state.update(
        {
            "safe_user_prompt": "Tell me about University of Mannheim MSc Business Informatics IELTS GPA ECTS deadline and portal",
            "required_answer_fields": ["language_test_score_thresholds", "gpa_threshold", "application_deadline"],
            "evidence_urls": ["https://www.uni-mannheim.de/en/academics/before-your-studies/programs/masters-program-in-business-informatics/"],
        }
    )
    assert (
        llm_service._can_return_best_effort_admissions_answer(
            "Language of instruction: English", state
        )
        is True
    )
    state["evidence_urls"] = []
    assert (
        llm_service._can_return_best_effort_admissions_answer(
            "Language of instruction: English", state
        )
        is False
    )


def test_structured_recovery_sources_use_only_allowed_evidence_urls_for_german_queries():
    state = llm_service._new_metrics_state()
    state.update(
        {
            "safe_user_prompt": "Tell me about University of Mannheim MSc Business Informatics IELTS GPA ECTS deadline portal",
            "required_answer_fields": ["application_deadline", "application_portal", "gpa_threshold"],
            "evidence_urls": [
                "https://www.uni-mannheim.de/en/academics/before-your-studies/programs/masters-program-in-business-informatics/",
                "https://portal2.uni-mannheim.de/portal2/pages/cs/sys/portal/hisinoneStartPage.faces?page=Bewerbung",
            ],
            "coverage_ledger": [
                {
                    "id": "application_portal",
                    "status": "found",
                    "value": "https://portal2.uni-mannheim.de/portal2/pages/cs/sys/portal/hisinoneStartPage.faces?page=Bewerbung",
                    "source_url": "https://www.uni-mannheim.de/en/news/apply-for-academic-year-2025-2026/",
                },
                {
                    "id": "application_deadline",
                    "status": "missing",
                    "value": "Not verified from official sources.",
                    "source_url": "",
                },
            ],
        }
    )
    answer = llm_service._build_structured_field_evidence_answer(state)
    assert "https://www.uni-mannheim.de/en/news/apply-for-academic-year-2025-2026/" not in answer
    assert "https://portal2.uni-mannheim.de/portal2/pages/cs/sys/portal/hisinoneStartPage.faces?page=Bewerbung" in answer


def test_structured_answer_suppresses_unverified_german_requirement_for_english_msc():
    state = llm_service._new_metrics_state()
    state.update(
        {
            "safe_user_prompt": "Tell me about University of Mannheim MSc Business Informatics IELTS German GPA ECTS deadline portal and if it is safe for 3.2 GPA",
            "evidence_urls": [
                "https://www.uni-mannheim.de/en/academics/before-your-studies/programs/masters-program-in-business-informatics/",
                "https://www.uni-mannheim.de/en/academics/before-your-studies/applying/the-a-to-z-of-applying/masters-programs-foreign-language-requirements/",
            ],
            "coverage_ledger": [
                {
                    "id": "language_of_instruction",
                    "label": "Language of instruction",
                    "status": "found",
                    "value": "English",
                    "source_url": "https://www.uni-mannheim.de/en/academics/before-your-studies/programs/masters-program-in-business-informatics/",
                },
                {
                    "id": "language_requirements",
                    "label": "Language requirements",
                    "status": "found",
                    "value": "Solid knowledge of English",
                    "source_url": "https://www.uni-mannheim.de/en/academics/before-your-studies/applying/the-a-to-z-of-applying/masters-programs-foreign-language-requirements/",
                },
                {
                    "id": "language_test_score_thresholds",
                    "label": "Language test score thresholds",
                    "status": "found",
                    "value": "TOEFL iBT 72; IELTS 6.0",
                    "source_url": "https://www.uni-mannheim.de/en/academics/before-your-studies/applying/the-a-to-z-of-applying/masters-programs-foreign-language-requirements/",
                },
                {
                    "id": "german_language_requirement",
                    "label": "German language requirement",
                    "status": "missing",
                    "value": "Not verified from official sources.",
                    "source_url": "",
                },
                {
                    "id": "selection_criteria",
                    "label": "Selection criteria",
                    "status": "found",
                    "value": "The final grade, professional activities, and semester abroad are selection criteria.",
                    "source_url": "https://www.uni-mannheim.de/en/academics/before-your-studies/programs/masters-program-in-business-informatics/",
                },
            ],
        }
    )

    answer = llm_service._build_structured_field_evidence_answer(state)

    assert "German language requirement: No separate German requirement was verified" in answer
    assert "German language requirement: Not verified from official sources" not in answer
    assert "TOEFL iBT 72; IELTS 6.0" in answer
    assert "No fixed minimum GPA/grade threshold was found" in answer
    assert "final grade or grade average is used as a selection criterion" in answer
    assert "Cannot be classified as safe" in answer


def test_structured_answer_cleans_noisy_deadline_ects_and_german_rows():
    state = llm_service._new_metrics_state()
    state.update(
        {
            "safe_user_prompt": "Tell me about University of Mannheim MSc Business Informatics IELTS German ECTS deadline",
            "evidence_urls": [
                "https://www.uni-mannheim.de/en/academics/dates/application-deadlines/",
                "https://www.wim.uni-mannheim.de/en/academics/organizing-your-studies/msc-business-informatics/",
            ],
            "coverage_ledger": [
                {
                    "id": "application_deadline",
                    "label": "Application deadline",
                    "status": "found",
                    "value": (
                        "1 April \u2013 15 May; 15 October \u2013 15 November; "
                        "1 April \u2013 15 May; 15 October \u2013 15 November"
                    ),
                    "source_url": "https://www.uni-mannheim.de/en/academics/dates/application-deadlines/",
                },
                {
                    "id": "ects_or_subject_credit_requirements",
                    "label": "ECTS / prerequisite credits",
                    "status": "found",
                    "value": "2 ECTS; 4 ECTS; 12 ECTS",
                    "source_url": "https://www.wim.uni-mannheim.de/en/academics/organizing-your-studies/msc-business-informatics/",
                },
                {
                    "id": "german_language_requirement",
                    "label": "German language requirement",
                    "status": "found",
                    "value": (
                        "For some of the master's programs, applicants have to prove a minimum "
                        "of German language skills. German citizens do not have to provide proof."
                    ),
                    "source_url": "https://www.wim.uni-mannheim.de/en/academics/organizing-your-studies/msc-business-informatics/",
                },
                {
                    "id": "language_test_score_thresholds",
                    "label": "Language score thresholds",
                    "status": "found",
                    "value": "DSH passed with at least grade 2",
                    "source_url": "https://www.sowi.uni-mannheim.de/media/Einrichtungen/zula/Dokumente_Zula/masterbroschuere_uni_mannheim_en.pdf",
                },
                {
                    "id": "language_requirements",
                    "label": "Language requirements",
                    "status": "found",
                    "value": "TOEFL with a score of at least 80 from 120",
                    "source_url": "https://www.uni-mannheim.de/en/academics/going-abroad/studying-abroad/proof-of-language-proficiency/",
                },
                {
                    "id": "language_of_instruction",
                    "label": "Language of instruction",
                    "status": "found",
                    "value": "German",
                    "source_url": "https://www.uni-mannheim.de/en/academics/before-your-studies/programs/bsc-business-informatics/",
                },
            ],
        }
    )

    answer = llm_service._build_structured_field_evidence_answer(state)

    assert answer.count("1 April - 15 May") == 1
    assert answer.count("15 October - 15 November") == 1
    assert "ECTS / prerequisite credit breakdown: The retrieved official evidence does not state this requested detail." in answer
    assert "2 ECTS; 4 ECTS; 12 ECTS" not in answer
    assert "For some of the master's programs" not in answer
    assert "DSH passed with at least grade 2" not in answer
    assert "TOEFL with a score of at least 80" not in answer
    assert "Language of instruction: German" not in answer
    assert "IELTS/TOEFL thresholds: The retrieved official evidence does not state a specific IELTS band score." in answer


@pytest.mark.asyncio
async def test_generate_agentic_answer_uses_structured_admissions_ledger_when_model_is_noisy(monkeypatch):
    async def fake_finalize_candidate_with_llm(**_kwargs):
        return "", {}, 0

    monkeypatch.setattr(llm_service, "_finalize_candidate_with_llm", fake_finalize_candidate_with_llm)

    async def fake_call_model_with_fallback(_messages, _state, role="worker", attempt=1):
        _ = role, attempt
        return FakeResponse(
            "Sources Checked\n"
            "Application deadline: 1 April - 15 May; 15 October - 15 November; "
            "1 April - 15 May; 15 October - 15 November\n"
            "ECTS / prerequisite credit breakdown: 2 ECTS; 4 ECTS; 12 ECTS\n"
            "Show citations"
        )

    monkeypatch.setattr(llm_service, "_call_model_with_fallback", fake_call_model_with_fallback)

    state = llm_service._new_metrics_state()
    state.update(
        {
            "safe_user_prompt": "Tell me about University of Mannheim MSc Business Informatics ECTS deadline portal",
            "citation_required": True,
            "required_answer_fields": [
                "ects_or_prerequisite_credit_breakdown",
                "application_deadline",
                "application_portal",
            ],
            "evidence_urls": [
                "https://www.uni-mannheim.de/en/academics/before-your-studies/programs/masters-program-in-business-informatics/",
                "https://www.uni-mannheim.de/en/academics/dates/application-deadlines/",
                "https://portal2.uni-mannheim.de/portal2/pages/cs/sys/portal/hisinoneStartPage.faces?page=Bewerbung",
            ],
            "coverage_ledger": [
                {
                    "id": "ects_or_prerequisite_credit_breakdown",
                    "label": "ECTS / prerequisite credits",
                    "status": "found",
                    "value": "30 ECTS informatics; 30 ECTS business/business informatics; 18 ECTS mathematics/statistics; 8 ECTS programming",
                    "source_url": "https://www.uni-mannheim.de/en/academics/before-your-studies/programs/masters-program-in-business-informatics/",
                },
                {
                    "id": "application_deadline",
                    "label": "Application deadline",
                    "status": "found",
                    "value": "1 April - 15 May; 15 October - 15 November",
                    "source_url": "https://www.uni-mannheim.de/en/academics/dates/application-deadlines/",
                },
                {
                    "id": "application_portal",
                    "label": "Application portal",
                    "status": "found",
                    "value": "https://portal2.uni-mannheim.de/portal2/pages/cs/sys/portal/hisinoneStartPage.faces?page=Bewerbung",
                    "source_url": "https://portal2.uni-mannheim.de/portal2/pages/cs/sys/portal/hisinoneStartPage.faces?page=Bewerbung",
                },
            ],
        }
    )

    result, _usage = await llm_service._generate_agentic_answer(
        user_id="user-1",
        messages=[{"role": "user", "content": "requirements?"}],
        policy={
            "max_attempts": 1,
            "planner_enabled": False,
            "verifier_enabled": False,
            "web_search_mode": "fast",
            "mode": "fast",
        },
        state=state,
    )

    assert "30 ECTS informatics" in result
    assert "2 ECTS; 4 ECTS; 12 ECTS" not in result
    assert "Show citations" not in result
    assert "portal2.uni-mannheim.de" in result


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
            and "Live web fallback context" in str(message.get("content", ""))
            for message in messages
        )
        return FakeResponse("primary-response")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    web_calls = {"count": 0}
    async def fake_web_fallback(*_args, **_kwargs):
        web_calls["count"] += 1
        return {
            "results": [
                {
                    "content": "Reliable official web result.",
                    "metadata": {"university": "Oxford", "url": "https://example.edu/oxford"},
                }
            ]
        }

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_web_chunks", fake_web_fallback)

    result = await llm_service.generate_response("user-1", "oxford ai admission")
    assert result == "primary-response"
    assert web_calls["count"] >= 1


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

    async def fake_retrieve_web_candidates_if_needed(
        _retrieval_query,
        *,
        vector_results,
        vector_has_urls,
        top_similarity,
        search_mode,
        state,
        web_prefetch_task,
    ):
        _ = (top_similarity, state, web_prefetch_task)
        assert vector_results == []
        assert vector_has_urls is False
        assert search_mode == "deep"
        return (
            [
                {
                    "content": "Web evidence.",
                    "metadata": {"url": "https://web.example.edu/a"},
                }
            ],
            True,
        )

    async def fake_rerank(_query, merged_results, _state):
        return merged_results

    def fake_apply_grounded_retrieval_context(*, messages, merged_results, used_web_results, state):
        _ = state
        assert used_web_results is True
        assert len(merged_results) == 1
        return messages, None

    monkeypatch.setattr(
        llm_service,
        "_retrieve_web_candidates_if_needed",
        fake_retrieve_web_candidates_if_needed,
    )
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
    captured: dict = {}
    monkeypatch.setattr(llm_service, "_retrieval_fanout_enabled", lambda: True)

    async def fake_build_context(_conversation_user_id, safe_user_prompt):
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
    assert captured["vector_prefetch_result"] is None


@pytest.mark.asyncio
async def test_prepare_messages_for_model_fanout_cancels_vector_prefetch_on_query_mismatch(
    monkeypatch,
):
    monkeypatch.delenv("RETRIEVAL_DISABLED", raising=False)
    monkeypatch.setattr(llm_service, "_retrieval_fanout_enabled", lambda: True)
    captured: dict = {}

    async def fake_build_context(_conversation_user_id, _safe_user_prompt):
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


def test_web_retrieval_timeout_seconds_boosts_for_high_precision_admissions_query(monkeypatch):
    monkeypatch.setattr(llm_service.settings.web_search, "deep_timeout_seconds", 110.0)
    timeout = llm_service._web_retrieval_timeout_seconds(
        "deep",
        query=(
            "Tell me University of Mannheim MSc Business Informatics language requirement, "
            "GPA, ECTS, deadline for international students and portal."
        ),
    )
    assert timeout >= 150.0


@pytest.mark.asyncio
async def test_retrieve_web_candidates_if_needed_recovers_with_timeout_rescue(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.web_search, "enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "always_web_retrieval_enabled", True)

    calls: list[tuple[str, str]] = []

    async def fake_run_one_web_query_with_timeout(query: str, *, top_k: int, search_mode: str):
        _ = top_k
        calls.append((query, search_mode))
        if len(calls) == 1:
            return {
                "results": [],
                "_timed_out": True,
                "_query": query,
                "_search_mode": search_mode,
            }
        return {
            "results": [
                {
                    "content": "Application deadline: 15 July 2026.",
                    "metadata": {"url": "https://uni-example.de/admission"},
                }
            ],
            "verification": {
                "required_field_coverage": 1.0,
                "required_fields_missing": [],
                "verified": True,
            },
        }

    monkeypatch.setattr(llm_service, "_run_one_web_query_with_timeout", fake_run_one_web_query_with_timeout)
    monkeypatch.setattr(
        llm_service,
        "_timeout_rescue_queries",
        lambda **_kwargs: [
            "mannheim msc business informatics official deadline",
            "mannheim msc business informatics official application portal",
        ],
    )

    state = {
        "safe_user_prompt": (
            "University of Mannheim MSc Business Informatics deadline and portal for international students"
        ),
        "required_answer_fields": ["international_deadline", "application_portal"],
    }
    results, attempted = await llm_service._retrieve_web_candidates_if_needed(
        "University of Mannheim MSc Business Informatics deadline and portal for international students",
        vector_results=[],
        vector_has_urls=False,
        top_similarity=None,
        search_mode="deep",
        state=state,
        web_prefetch_task=None,
    )

    assert attempted is True
    assert results
    assert state["web_timeout_rescued"] is True
    assert state["web_timeout_count"] == 0
    assert any(mode == "fast" for _, mode in calls[1:])


@pytest.mark.asyncio
async def test_retrieve_web_candidates_skips_rescue_after_unigraph_official_field(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.web_search, "enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "always_web_retrieval_enabled", True)

    async def fake_run_one_web_query_with_timeout(query: str, *, top_k: int, search_mode: str):
        _ = top_k, search_mode
        return {
            "query": query,
            "results": [
                {
                    "content": "Application deadline: 31 May.",
                    "metadata": {"url": "https://www.tum.de/en/studies/degree-programs/detail/informatics-master-of-science-msc"},
                }
            ],
            "unigraph_answered_required_field": True,
            "coverage_ledger": [
                {
                    "field": "application_deadline",
                    "status": "found",
                    "value": "31 May",
                    "source_type": "official",
                    "source_url": "https://www.tum.de/en/studies/degree-programs/detail/informatics-master-of-science-msc",
                }
            ],
        }

    async def should_not_augment(*_args, **_kwargs):
        raise AssertionError("german researcher rescue should not run after UniGraph answers")

    monkeypatch.setattr(llm_service, "_run_one_web_query_with_timeout", fake_run_one_web_query_with_timeout)
    monkeypatch.setattr(llm_service, "_augment_with_german_researcher", should_not_augment)

    state = {
        "safe_user_prompt": "When is the winter semester application deadline for MSc Informatics at TU Munich?",
        "required_answer_fields": ["application_deadline"],
    }
    results, attempted = await llm_service._retrieve_web_candidates_if_needed(
        state["safe_user_prompt"],
        vector_results=[],
        vector_has_urls=False,
        top_similarity=None,
        search_mode="deep",
        state=state,
        web_prefetch_task=None,
    )

    assert attempted is True
    assert results
    assert state["unigraph_answered_required_field"] is True
    assert state["rescue_retrieval_skipped_reason"] == "unigraph_answered_required_field"


@pytest.mark.asyncio
async def test_generate_response_uses_vector_when_web_fallback_empty(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.web_search, "enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_similarity_threshold", 0.35)
    monkeypatch.setattr(llm_service, "_is_citation_grounding_required", lambda: True)

    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def fake_primary(_messages):
        raise AssertionError("model should not run when web retrieval has no evidence")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_web_fallback(*_args, **_kwargs):
        return {"results": []}

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_web_chunks", fake_web_fallback)

    result = await llm_service.generate_response("user-1", "saarland ai")
    assert result == llm_service._NO_RELEVANT_INFORMATION_DETAIL


@pytest.mark.asyncio
async def test_generate_response_tries_web_when_vector_has_no_urls(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.web_search, "enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_enabled", True)
    monkeypatch.setattr(llm_service.settings.web_search, "fallback_similarity_threshold", 0.05)
    monkeypatch.setattr(llm_service, "_evidence_urls", lambda _results: [])
    monkeypatch.setattr(llm_service, "_is_citation_grounding_required", lambda: True)

    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    async def fake_primary(_messages):
        raise AssertionError("model should not run when evidence has no URLs and web is empty")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    web_calls = {"count": 0}

    async def fake_web_fallback(*_args, **_kwargs):
        web_calls["count"] += 1
        return {"results": []}

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
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
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.web_search, "enabled", True)
    fake_redis = FakeRedis()
    _attach_fake_redis(monkeypatch, fake_redis)

    async def fake_build_context(_user_id, user_prompt):
        return [{"role": "user", "content": user_prompt}]

    def fake_apply_context_guardrails(messages):
        system_contents = [
            str(item.get("content", ""))
            for item in messages
            if isinstance(item, dict) and item.get("role") == "system"
        ]
        assert any("UniGraph" in content for content in system_contents)
        assert any("Citation policy" in content for content in system_contents)
        assert any("Live web fallback context" in content for content in system_contents)
        return {"blocked": False, "messages": messages, "reason": ""}

    async def fake_primary(_messages):
        return FakeResponse("primary-response")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_web_fallback(*_args, **_kwargs):
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
                        "url": "https://example.edu/program",
                    },
                }
            ]
        }

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "apply_context_guardrails", fake_apply_context_guardrails)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_web_chunks", fake_web_fallback)

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
        == "fast"
    )
    assert (
        llm_service._resolve_initial_execution_mode(
            "auto",
            "compare top AI universities in germany and include latest deadlines",
        )
        == "fast"
    )
    assert llm_service._resolve_initial_execution_mode("deep", "hello") == "deep"


def test_normalized_request_mode_preserves_standard_fast_first_policy():
    assert llm_service._normalized_request_mode("standard") == "standard"
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


@pytest.mark.asyncio
async def test_prepare_request_standard_escalates_to_deep_on_context_refusal(monkeypatch):
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
        mode="standard",
    )

    assert refusal is None
    assert messages == [{"role": "user", "content": "hello"}]
    assert call_modes == ["fast", "deep"]
    assert context["requested_mode"] == "standard"
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
    assert "retrieved official evidence does not state a separate deadline" in result
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


def test_required_answer_fields_detects_researcher_scopes():
    fields = llm_service._required_answer_fields(
        (
            "Find professors, research labs, department contacts, scholarship options, "
            "and publication links for this program."
        ),
        intent="exploration",
    )
    assert "professors_or_supervisors" in fields
    assert "labs_or_research_groups" in fields
    assert "department_or_faculty" in fields
    assert "contact_information" in fields
    assert "funding_or_scholarship" in fields
    assert "publication_or_profile_links" in fields


def test_answer_matches_required_field_for_contact_information():
    assert llm_service._answer_matches_required_field(
        "contact_information",
        "Contact: admissions@uni-example.de, Phone: +49 621 1234567",
    )
    assert llm_service._answer_matches_required_field(
        "contact_information",
        "Contact information: Not verified from sources.",
    )
    assert not llm_service._answer_matches_required_field(
        "contact_information",
        "Contact information is available on the website.",
    )


def test_decompose_retrieval_queries_adds_research_variants():
    state = {
        "query_intent": "exploration",
        "required_answer_fields": [
            "professors_or_supervisors",
            "labs_or_research_groups",
            "publication_or_profile_links",
        ],
    }
    queries = llm_service._decompose_retrieval_queries(
        base_query="University of Mannheim MSc Business Informatics",
        state=state,
    )
    lowered = [query.lower() for query in queries]
    assert any("faculty professors supervisors" in query for query in lowered)
    assert any("professor publication list" in query for query in lowered)


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


def test_agentic_result_issues_marks_web_research_objectives_missing():
    state = {
        "citation_required": False,
        "evidence_urls": [],
        "required_answer_fields": [
            "professors_or_supervisors",
            "labs_or_research_groups",
        ],
        "web_research_objectives_missing": ["labs_and_research"],
        "retrieval_single_domain_low_quality": False,
        "deadline_query": False,
    }
    issues = llm_service._agentic_result_issues(
        "Professor list is available from official sources.",
        state,
    )
    assert "web_research_objectives_missing" in issues
    assert "web_missing:labs_and_research" in issues


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


def test_agentic_result_issues_marks_weak_critical_evidence_claim_for_admissions():
    state = {
        "citation_required": False,
        "evidence_urls": ["https://uni-example.de/selection"],
        "required_answer_fields": ["gpa_or_grade_threshold", "ects_or_prerequisite_credit_breakdown"],
        "web_required_fields_missing": [],
        "retrieval_single_domain_low_quality": False,
        "deadline_query": False,
    }
    issues = llm_service._agentic_result_issues(
        "A related document lists a GPA threshold of 2.0, but direct applicability should be confirmed.",
        state,
    )
    assert "weak_critical_evidence_claim" in issues


def test_structured_field_evidence_answer_is_preferred_for_admissions_ledger():
    state = {
        "required_answer_fields": ["application_portal", "application_deadline"],
        "safe_user_prompt": "Tell me about University of Mannheim MSc Business Informatics deadline and portal",
        "evidence_urls": ["https://portal2.uni-mannheim.de/"],
        "coverage_ledger": [
            {
                "id": "application_portal",
                "label": "Application portal",
                "status": "found",
                "value": "https://portal2.uni-mannheim.de/",
                "source_url": "https://portal2.uni-mannheim.de/",
                "confidence": 0.9,
            },
            {
                "id": "application_deadline",
                "label": "Application deadline",
                "status": "missing",
                "value": "Not verified from official sources.",
            },
        ],
    }
    assert llm_service._should_prefer_structured_field_evidence_answer(state) is True
    answer = llm_service._build_structured_field_evidence_answer(state)
    assert "https://portal2.uni-mannheim.de/" in answer
    assert "Application deadline: The retrieved official evidence does not state a separate deadline for this case." in answer


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
    captured: dict = {}
    async def fake_retrieve_web_candidates_if_needed(
        _retrieval_query,
        *,
        vector_results,
        vector_has_urls,
        top_similarity,
        search_mode,
        state,
        web_prefetch_task,
    ):
        _ = (vector_results, vector_has_urls, top_similarity, state, web_prefetch_task)
        captured["search_mode"] = search_mode
        return (
            [
                {
                    "content": "Web evidence.",
                    "metadata": {"url": "https://web.example.edu/a"},
                }
            ],
            True,
        )

    async def fake_rerank(_query, merged_results, _state):
        return merged_results

    def fake_apply_grounded_retrieval_context(*, messages, merged_results, used_web_results, state):
        _ = state
        assert used_web_results is True
        assert len(merged_results) == 1
        return messages, None

    monkeypatch.setattr(
        llm_service,
        "_retrieve_web_candidates_if_needed",
        fake_retrieve_web_candidates_if_needed,
    )
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
    assert captured["search_mode"] == "deep"


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


@pytest.mark.asyncio
async def test_augment_with_german_researcher_uses_cached_result(monkeypatch):
    calls = {"count": 0}

    async def _fake_research(_query: str) -> dict:
        calls["count"] += 1
        return {
            "applicable": True,
            "results": [
                {
                    "chunk_id": "german:1",
                    "source_path": "https://www.uni-mannheim.de/program",
                    "distance": 0.1,
                    "content": "Language of instruction: English.",
                    "metadata": {"url": "https://www.uni-mannheim.de/program"},
                }
            ],
            "coverage_ledger": [
                {
                    "id": "language_of_instruction",
                    "label": "Language of instruction",
                    "status": "found",
                    "value": "English",
                    "source_url": "https://www.uni-mannheim.de/program",
                    "confidence": 0.9,
                }
            ],
            "verification": {
                "required_field_coverage": 1.0,
                "required_fields_missing": [],
                "unresolved_fields": [],
                "source_policy": "german_official_first",
            },
            "coverage_summary": {"required_field_coverage": 1.0, "unresolved_fields": []},
            "query_variants": ["mannheim msc business informatics official program page"],
            "source_routes_attempted": {},
            "retrieval_budget_usage": {"query_budget": 10, "queries_executed": 4},
            "timings_ms": {"total": 20},
        }

    monkeypatch.setattr(llm_service, "research_german_university", _fake_research)
    monkeypatch.setattr(llm_service, "is_likely_german_university_query", lambda _q: True)
    monkeypatch.setattr(llm_service.settings.web_search, "german_university_mode_enabled", True)

    state = llm_service._new_metrics_state()
    base_result = {"results": [], "retrieval_budget_usage": {"queries_executed": 2}}

    first = await llm_service._augment_with_german_researcher(
        "University of Mannheim MSc Business Informatics",
        base_result,
        state,
    )
    second = await llm_service._augment_with_german_researcher(
        "University of Mannheim MSc Business Informatics",
        base_result,
        state,
    )

    assert calls["count"] == 1
    assert state["german_researcher_cache_hits"] == 1
    assert first["german_research"]["applied"] is True
    assert second["german_research"]["applied"] is True


@pytest.mark.asyncio
async def test_required_field_rescue_uses_fast_mode_when_base_mode_is_deep(monkeypatch):
    captured_modes: list[str] = []

    async def _fake_web_query(query: str, *, top_k: int, search_mode: str):
        captured_modes.append(str(search_mode))
        return {"query": query, "verification": {}, "results": []}

    monkeypatch.setattr(llm_service, "_run_one_web_query_with_timeout", _fake_web_query)
    monkeypatch.setattr(llm_service.settings.web_search, "deep_required_field_rescue_max_queries", 2)

    state = llm_service._new_metrics_state()
    state["required_answer_fields"] = ["application_deadline", "application_portal"]
    state["safe_user_prompt"] = "University of Mannheim MSc Business Informatics deadline and portal"

    _messages, rescued = await llm_service._attempt_required_field_web_rescue(
        issues=["web_missing:application_deadline"],
        state=state,
        base_query=state["safe_user_prompt"],
        search_mode="deep",
    )

    assert rescued is False
    assert captured_modes
    assert all(mode == "fast" for mode in captured_modes)


@pytest.mark.asyncio
async def test_required_field_rescue_skips_when_unigraph_already_answered(monkeypatch):
    async def should_not_query(*_args, **_kwargs):
        raise AssertionError("required field rescue should not launch duplicate web queries")

    monkeypatch.setattr(llm_service, "_run_one_web_query_with_timeout", should_not_query)

    state = llm_service._new_metrics_state()
    state["safe_user_prompt"] = "When is the winter semester application deadline for MSc Informatics at TU Munich?"
    state["required_answer_fields"] = ["application_deadline"]
    state["coverage_ledger"] = [
        {
            "field": "application_deadline",
            "status": "found",
            "value": "31 May",
            "source_type": "official",
        }
    ]

    messages, rescued = await llm_service._attempt_required_field_web_rescue(
        issues=["web_missing:application_deadline"],
        state=state,
        base_query=state["safe_user_prompt"],
        search_mode="deep",
    )

    assert messages == []
    assert rescued is False
    assert state["rescue_retrieval_skipped_reason"] == "unigraph_answered_required_field"
