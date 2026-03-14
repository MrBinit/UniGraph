import hashlib
import pytest
from app.services import llm_service


@pytest.fixture(autouse=True)
def _stub_json_metrics(monkeypatch):
    async def noop(_record):
        return None

    monkeypatch.setattr(llm_service, "_record_json_metrics", noop)


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
                    "content": "Distributed Security Systems Lab focuses on scalable and secure AI infrastructure.",
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
    assert captured_metrics[-1]["quality"] == {}


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
        assert "Retrieved long-term knowledge" in messages[1]["content"]
        return {"blocked": False, "messages": messages, "reason": ""}

    async def fake_primary(_messages):
        return FakeResponse("primary-response")

    async def fake_update_memory(*_args, **_kwargs):
        return None

    async def fake_retrieve_document_chunks(*_args, **_kwargs):
        return {
            "results": [
                {
                    "content": "Master of Science in Artificial Intelligence Systems is offered in Germany.",
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
