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


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value


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
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.serpapi, "enabled", True)
    monkeypatch.setattr(llm_service.settings.serpapi, "fallback_enabled", True)
    monkeypatch.setattr(llm_service.settings.serpapi, "max_context_results", 2)

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
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.serpapi, "enabled", True)
    monkeypatch.setattr(llm_service.settings.serpapi, "fallback_enabled", True)
    monkeypatch.setattr(llm_service.settings.serpapi, "fallback_similarity_threshold", 0.35)

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
async def test_generate_response_uses_web_fallback_when_vector_low_confidence(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.serpapi, "enabled", True)
    monkeypatch.setattr(llm_service.settings.serpapi, "fallback_enabled", True)
    monkeypatch.setattr(llm_service.settings.serpapi, "fallback_similarity_threshold", 0.35)

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
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.serpapi, "enabled", True)
    monkeypatch.setattr(llm_service.settings.serpapi, "fallback_enabled", True)
    monkeypatch.setattr(llm_service.settings.serpapi, "fallback_similarity_threshold", 0.35)

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
async def test_generate_response_returns_sorry_when_web_fallback_has_no_results(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setattr(llm_service.settings.serpapi, "enabled", True)
    monkeypatch.setattr(llm_service.settings.serpapi, "fallback_enabled", True)

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
async def test_generate_response_returns_sorry_when_strict_citation_has_no_evidence(monkeypatch):
    monkeypatch.setattr(llm_service, "_is_citation_grounding_required", lambda: True)
    monkeypatch.setattr(llm_service.settings.serpapi, "enabled", False)
    monkeypatch.setattr(llm_service.settings.serpapi, "fallback_enabled", False)

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
