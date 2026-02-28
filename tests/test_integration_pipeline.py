from fastapi.testclient import TestClient

from app.core.security import create_access_token
from app.main import app
from app.services import (
    llm_service,
    memory_metrics_service,
    memory_service,
    ops_status_service,
    summary_queue_service,
    summary_worker_service,
)


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
        self.hashes = {}
        self.streams = {}
        self.delivered = {}
        self.acked = {}

    def ping(self):
        return True

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None, nx=False):
        if nx and key in self.store:
            return False
        self.store[key] = str(value)
        return True

    def setex(self, key, ttl, value):
        self.store[key] = value

    def delete(self, key):
        removed = 0
        if key in self.store:
            del self.store[key]
            removed += 1
        if key in self.hashes:
            del self.hashes[key]
            removed += 1
        return removed

    def pipeline(self):
        return FakePipeline(self)

    def xgroup_create(self, name, groupname, id="0", mkstream=False):
        self.streams.setdefault(name, [])
        return True

    def xadd(self, name, payload, maxlen=None, approximate=None):
        stream = self.streams.setdefault(name, [])
        stream_id = f"{len(stream) + 1}-0"
        stream.append((stream_id, dict(payload)))
        if maxlen and len(stream) > maxlen:
            overflow = len(stream) - maxlen
            del stream[:overflow]
        return stream_id

    def xreadgroup(self, groupname, consumername, streams, count=None, block=None):
        result = []
        for name in streams:
            all_entries = self.streams.get(name, [])
            delivered = self.delivered.setdefault(name, set())
            fresh = [entry for entry in all_entries if entry[0] not in delivered]
            if count is not None:
                fresh = fresh[:count]
            for stream_id, _fields in fresh:
                delivered.add(stream_id)
            if fresh:
                result.append((name, fresh))
        return result

    def xack(self, name, groupname, stream_id):
        self.acked.setdefault(name, set()).add(stream_id)
        return 1

    def xlen(self, name):
        return len(self.streams.get(name, []))

    def xrevrange(self, name, count=1):
        entries = list(reversed(self.streams.get(name, [])))
        return entries[:count]

    def xpending(self, name, groupname):
        delivered = self.delivered.get(name, set())
        acked = self.acked.get(name, set())
        return {"pending": max(0, len(delivered - acked))}

    def hincrby(self, key, field, amount):
        bucket = self.hashes.setdefault(key, {})
        current = int(bucket.get(field, "0"))
        current += int(amount)
        bucket[field] = str(current)
        return current

    def hset(self, key, field=None, value=None, mapping=None):
        bucket = self.hashes.setdefault(key, {})
        if mapping is not None:
            for item_key, item_value in mapping.items():
                bucket[str(item_key)] = str(item_value)
            return len(mapping)
        if field is not None:
            bucket[str(field)] = str(value)
            return 1
        return 0

    def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def expire(self, key, ttl):
        return True


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Usage:
    prompt_tokens = 10
    completion_tokens = 5
    total_tokens = 15


class FakeResponse:
    def __init__(self, content):
        self.choices = [_Choice(content)]
        self.usage = _Usage()


def _base_memory():
    return {
        "summary": "",
        "messages": [
            {"seq": 1, "role": "user", "content": "u1"},
            {"seq": 2, "role": "assistant", "content": "a1"},
            {"seq": 3, "role": "user", "content": "u2"},
            {"seq": 4, "role": "assistant", "content": "a2"},
        ],
        "version": 1,
        "next_seq": 5,
        "last_summarized_seq": 0,
        "summary_pending": False,
        "last_summary_job_id": "",
    }


def test_api_to_queue_to_worker_to_memory_update(monkeypatch):
    fake_redis = FakeRedis()

    monkeypatch.setattr(llm_service, "redis_client", fake_redis)
    monkeypatch.setattr(memory_service, "redis_client", fake_redis)
    monkeypatch.setattr(memory_metrics_service, "redis_client", fake_redis)
    monkeypatch.setattr(ops_status_service, "app_redis_client", fake_redis)
    monkeypatch.setattr(summary_queue_service, "app_redis_client", fake_redis)
    monkeypatch.setattr(summary_queue_service, "worker_redis_client", fake_redis)

    async def fake_primary(_messages):
        return FakeResponse("assistant-output")

    async def fake_summary(_messages):
        return "condensed summary"

    monkeypatch.setattr(llm_service, "_call_primary", fake_primary)
    monkeypatch.setattr(summary_worker_service, "summarize_messages", fake_summary)
    monkeypatch.setattr(memory_service, "safe_token_count", lambda *_args: 9999)
    monkeypatch.setattr(
        memory_service,
        "truncate_context_without_summary",
        lambda **kwargs: {
            "summary": kwargs["summary"],
            "messages": kwargs["messages"],
            "final_context": kwargs["messages"] + [{"role": "user", "content": kwargs["new_user_message"]}],
            "final_tokens": 10,
            "memory_changed": False,
            "events": [],
        },
    )

    memory_service.save_memory("user-1", _base_memory())

    user_token = create_access_token(user_id="user-1", roles=["user"])
    client = TestClient(app)

    chat_response = client.post(
        "/api/v1/chat",
        json={"user_id": "user-1", "prompt": "find ai research lab"},
        headers={"Authorization": f"Bearer {user_token}"},
    )

    assert chat_response.status_code == 200
    assert chat_response.json() == {"response": "assistant-output"}

    jobs = summary_queue_service.read_summary_jobs("worker-1")
    assert len(jobs) == 1

    stream_id, fields = jobs[0]
    import asyncio

    asyncio.run(summary_worker_service.process_summary_job(stream_id, fields))

    final_memory = asyncio.run(memory_service.load_memory("user-1"))
    assert final_memory["summary"] == "condensed summary"
    assert final_memory["last_summarized_seq"] == 2
    assert final_memory["summary_pending"] is False
    assert final_memory["last_summary_job_id"] == ""

    queue_state = summary_queue_service.get_summary_queue_state()
    assert queue_state["pending_jobs"] == 0

    admin_token = create_access_token(user_id="admin-1", roles=["admin"])
    ops_response = client.get(
        "/api/v1/ops/status",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert ops_response.status_code == 200
    payload = ops_response.json()
    assert payload["status"] == "ok"
    assert payload["memory"]["redis_available"] is True
    assert payload["queue"]["dlq_depth"] == 0
    assert payload["compaction"]["events"] == 1
    assert payload["latency"]["count"] >= 1
    assert payload["latency"]["last_outcome"] == "success"
