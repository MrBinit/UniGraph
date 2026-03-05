import pytest
from app.services import llm_service
from app.infra.redis_client import app_scoped_key


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value


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
    cache_key = app_scoped_key("cache", "chat", "user-1", "find ai professor")
    fake_redis.store[cache_key] = "from-cache"
    monkeypatch.setattr(llm_service, "redis_client", fake_redis)

    async def should_not_run(*_args, **_kwargs):
        raise AssertionError("build_context should not run on cache hit")

    monkeypatch.setattr(llm_service, "build_context", should_not_run)

    result = await llm_service.generate_response("user-1", "find ai professor")
    assert result == "from-cache"


@pytest.mark.asyncio
async def test_generate_response_uses_primary_and_updates_memory(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(llm_service, "redis_client", fake_redis)

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

    monkeypatch.setattr(llm_service, "build_context", fake_build_context)
    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(llm_service, "_call_fallback", fake_fallback)
    monkeypatch.setattr(llm_service, "update_memory", fake_update_memory)
    monkeypatch.setattr(llm_service, "aretrieve_document_chunks", fake_retrieve_document_chunks)

    result = await llm_service.generate_response("user-1", "find ai professor")
    assert result == "primary-response"
    assert memory_updates == [("user-1", "find ai professor", "primary-response")]
    assert retrieval_queries[0][0] == "find ai professor"
    cache_key = app_scoped_key("cache", "chat", "user-1", "find ai professor")
    assert fake_redis.store[cache_key] == "primary-response"


@pytest.mark.asyncio
async def test_generate_response_uses_fallback_when_primary_fails(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(llm_service, "redis_client", fake_redis)

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
    monkeypatch.setattr(llm_service, "redis_client", fake_redis)

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
    monkeypatch.setattr(llm_service, "redis_client", fake_redis)

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
    monkeypatch.setattr(llm_service, "redis_client", fake_redis)

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
