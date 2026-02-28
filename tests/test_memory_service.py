import json
from copy import deepcopy

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.infra.redis_client import app_scoped_key
from app.schemas.settings_schema import UserTokenBudgetConfig
from app.services import memory_service


class FakePipeline:
    def __init__(self, redis):
        self.redis = redis
        self.key = None
        self.pending_setex = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def watch(self, key):
        self.key = key

    def get(self, key):
        return self.redis.get(key)

    def unwatch(self):
        return None

    def multi(self):
        return None

    def setex(self, key, ttl, value):
        self.pending_setex = (key, ttl, value)

    def execute(self):
        if self.pending_setex:
            key, ttl, value = self.pending_setex
            self.redis.setex(key, ttl, value)


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value

    def pipeline(self):
        return FakePipeline(self)


def _memory_key(user_id: str) -> str:
    return app_scoped_key("memory", "chat", user_id)


def _seed_memory(fake_redis: FakeRedis, user_id: str, memory: dict):
    fake_redis.store[_memory_key(user_id)] = json.dumps(memory)


@pytest.mark.asyncio
async def test_load_memory_returns_default_on_redis_error(monkeypatch):
    class FailingRedis:
        def get(self, _key):
            raise RedisConnectionError("redis down")

    monkeypatch.setattr(memory_service, "redis_client", FailingRedis())

    memory = await memory_service.load_memory("user-1")
    assert memory["summary"] == ""
    assert memory["messages"] == []
    assert memory["summary_pending"] is False


@pytest.mark.asyncio
async def test_build_context_without_prior_memory(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(memory_service, "redis_client", fake_redis)

    monkeypatch.setattr(
        memory_service,
        "truncate_context_without_summary",
        lambda **_kwargs: {
            "summary": "",
            "messages": [],
            "final_context": [{"role": "user", "content": "hello"}],
            "final_tokens": 1,
            "memory_changed": False,
            "events": [],
        },
    )

    context = await memory_service.build_context("user-1", "hello")
    assert context == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_build_context_enqueues_summary_job(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(memory_service, "redis_client", fake_redis)

    existing = {
        "summary": "",
        "messages": [
            {"seq": 1, "role": "user", "content": "m1"},
            {"seq": 2, "role": "assistant", "content": "m2"},
            {"seq": 3, "role": "user", "content": "m3"},
            {"seq": 4, "role": "assistant", "content": "m4"},
        ],
        "version": 3,
        "next_seq": 5,
        "last_summarized_seq": 0,
        "summary_pending": False,
        "last_summary_job_id": "",
    }
    _seed_memory(fake_redis, "user-1", existing)

    monkeypatch.setattr(
        memory_service,
        "safe_token_count",
        lambda _counter, _messages: memory_service.settings.memory.summary_trigger + 10,
    )
    monkeypatch.setattr(memory_service, "enqueue_summary_job", lambda **_kwargs: "job-123")
    monkeypatch.setattr(
        memory_service,
        "truncate_context_without_summary",
        lambda **kwargs: {
            "summary": kwargs["summary"],
            "messages": kwargs["messages"],
            "final_context": kwargs["messages"] + [{"role": "user", "content": kwargs["new_user_message"]}],
            "final_tokens": 1,
            "memory_changed": False,
            "events": [],
        },
    )

    await memory_service.build_context("user-1", "new prompt")

    saved, ok = memory_service._deserialize_memory_payload(fake_redis.store[_memory_key("user-1")])
    assert ok is True
    assert saved["summary_pending"] is True
    assert saved["last_summary_job_id"] == "job-123"
    assert saved["version"] == 4


@pytest.mark.asyncio
async def test_build_context_does_not_enqueue_when_pending(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(memory_service, "redis_client", fake_redis)

    existing = {
        "summary": "",
        "messages": [{"seq": 1, "role": "user", "content": "m1"}],
        "version": 1,
        "next_seq": 2,
        "last_summarized_seq": 0,
        "summary_pending": True,
        "last_summary_job_id": "job-x",
    }
    _seed_memory(fake_redis, "user-1", existing)

    monkeypatch.setattr(
        memory_service,
        "safe_token_count",
        lambda _counter, _messages: memory_service.settings.memory.summary_trigger + 10,
    )
    monkeypatch.setattr(
        memory_service,
        "enqueue_summary_job",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not enqueue")),
    )
    monkeypatch.setattr(
        memory_service,
        "truncate_context_without_summary",
        lambda **kwargs: {
            "summary": kwargs["summary"],
            "messages": kwargs["messages"],
            "final_context": [{"role": "user", "content": kwargs["new_user_message"]}],
            "final_tokens": 1,
            "memory_changed": False,
            "events": [],
        },
    )

    await memory_service.build_context("user-1", "new prompt")


@pytest.mark.asyncio
async def test_update_memory_appends_messages_with_seq(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(memory_service, "redis_client", fake_redis)

    existing = {
        "summary": "",
        "messages": [],
        "version": 2,
        "next_seq": 10,
        "last_summarized_seq": 0,
        "summary_pending": False,
        "last_summary_job_id": "",
    }
    _seed_memory(fake_redis, "user-1", existing)

    await memory_service.update_memory("user-1", "hello", "hi there")
    saved_raw = fake_redis.store[_memory_key("user-1")]
    assert isinstance(saved_raw, str)
    assert saved_raw.startswith("enc:v1:")
    saved, ok = memory_service._deserialize_memory_payload(saved_raw)
    assert ok is True

    assert saved["messages"] == [
        {"seq": 10, "role": "user", "content": "hello"},
        {"seq": 11, "role": "assistant", "content": "hi there"},
    ]
    assert saved["next_seq"] == 12
    assert saved["version"] == 3


def test_get_user_budget_uses_override():
    original = deepcopy(memory_service.settings.memory.user_token_budgets)
    try:
        memory_service.settings.memory.user_token_budgets = {
            "user-1": UserTokenBudgetConfig(
                soft_limit=1200,
                hard_limit=1800,
                min_recent_messages_to_keep=2,
            )
        }
        soft, hard, min_recent = memory_service._get_user_budget("user-1")
        assert (soft, hard, min_recent) == (1200, 1800, 2)
    finally:
        memory_service.settings.memory.user_token_budgets = original


def test_save_memory_if_version_rejects_mismatch(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(memory_service, "redis_client", fake_redis)

    existing = {
        "summary": "",
        "messages": [],
        "version": 5,
        "next_seq": 1,
        "last_summarized_seq": 0,
        "summary_pending": False,
        "last_summary_job_id": "",
    }
    _seed_memory(fake_redis, "user-1", existing)

    ok, current = memory_service.save_memory_if_version(
        "user-1",
        expected_version=4,
        memory={**existing, "version": 6},
    )
    assert ok is False
    assert current["version"] == 5
