from decimal import Decimal
from fastapi.testclient import TestClient

from app.api.v1 import chat as chat_api
from app.api.v1 import ops as ops_api
from app.core.security import create_access_token
from app.main import app


def test_chat_enqueue_endpoint_removed():
    token = create_access_token(user_id="user-1", roles=["user"])
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        json={"user_id": "user-1", "prompt": "hello"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404


def test_chat_async_status_requires_owner(monkeypatch):
    monkeypatch.setattr(
        chat_api,
        "get_chat_job",
        lambda _job_id: {
            "job_id": "job-abc",
            "user_id": "user-b",
            "session_id": "user-b",
            "status": "processing",
            "created_at": "2026-03-10T00:00:00+00:00",
            "started_at": "2026-03-10T00:00:02+00:00",
            "completed_at": "",
            "answer": "",
            "error": "",
        },
    )

    token = create_access_token(user_id="user-a", roles=["user"])
    client = TestClient(app)
    response = client.get(
        "/api/v1/chat/job-abc",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


def test_chat_stream_endpoint_success(monkeypatch):
    def fake_enqueue_chat_job(
        *, user_id: str, prompt: str, session_id: str | None = None, mode: str | None = None
    ) -> dict:
        assert user_id == "user-1"
        assert prompt == "hello stream"
        assert session_id is None
        assert mode == "standard"
        return {
            "job_id": "job-stream-1",
            "status": "queued",
            "submitted_at": "2026-03-10T00:00:00+00:00",
        }

    poll_count = {"value": 0}

    def fake_get_chat_job(job_id: str) -> dict:
        assert job_id == "job-stream-1"
        poll_count["value"] += 1
        if poll_count["value"] == 1:
            return {
                "job_id": job_id,
                "user_id": "user-1",
                "session_id": "user-1",
                "status": "processing",
                "answer": "",
                "error": "",
            }
        return {
            "job_id": job_id,
            "user_id": "user-1",
            "session_id": "user-1",
            "status": "completed",
            "answer": "hello",
            "error": "",
        }

    monkeypatch.setattr(chat_api, "enqueue_chat_job", fake_enqueue_chat_job)
    monkeypatch.setattr(chat_api, "get_chat_job", fake_get_chat_job)

    token = create_access_token(user_id="user-1", roles=["user"])
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat/stream",
        json={"user_id": "user-1", "prompt": "hello stream"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert '"type": "queued"' in body
    assert '"type": "status"' in body
    assert '"status": "processing"' in body
    assert '"status": "completed"' in body
    assert '"type": "chunk"' in body
    assert '"text": "hello"' in body
    assert '{"type":"done"}' in body


def test_chat_stream_endpoint_forwards_session_id(monkeypatch):
    def fake_enqueue_chat_job(
        *, user_id: str, prompt: str, session_id: str | None = None, mode: str | None = None
    ) -> dict:
        assert user_id == "user-1"
        assert prompt == "hello stream"
        assert session_id == "session-xyz"
        assert mode == "standard"
        return {
            "job_id": "job-stream-2",
            "status": "queued",
            "submitted_at": "2026-03-10T00:00:00+00:00",
        }

    def fake_get_chat_job(job_id: str) -> dict:
        return {
            "job_id": job_id,
            "user_id": "user-1",
            "session_id": "session-xyz",
            "status": "completed",
            "answer": "hello",
            "error": "",
        }

    monkeypatch.setattr(chat_api, "enqueue_chat_job", fake_enqueue_chat_job)
    monkeypatch.setattr(chat_api, "get_chat_job", fake_get_chat_job)

    token = create_access_token(user_id="user-1", roles=["user"])
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat/stream",
        json={"user_id": "user-1", "session_id": "session-xyz", "prompt": "hello stream"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200


def test_chat_stream_endpoint_forwards_explicit_mode(monkeypatch):
    def fake_enqueue_chat_job(
        *, user_id: str, prompt: str, session_id: str | None = None, mode: str | None = None
    ) -> dict:
        assert user_id == "user-1"
        assert prompt == "hello stream"
        assert session_id is None
        assert mode == "deep"
        return {
            "job_id": "job-stream-mode",
            "status": "queued",
            "submitted_at": "2026-03-10T00:00:00+00:00",
        }

    def fake_get_chat_job(job_id: str) -> dict:
        return {
            "job_id": job_id,
            "user_id": "user-1",
            "session_id": "user-1",
            "status": "completed",
            "answer": "hello",
            "error": "",
        }

    monkeypatch.setattr(chat_api, "enqueue_chat_job", fake_enqueue_chat_job)
    monkeypatch.setattr(chat_api, "get_chat_job", fake_get_chat_job)

    token = create_access_token(user_id="user-1", roles=["user"])
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat/stream",
        json={"user_id": "user-1", "mode": "deep", "prompt": "hello stream"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200


def test_chat_stream_emits_trace_events(monkeypatch):
    def fake_enqueue_chat_job(
        *, user_id: str, prompt: str, session_id: str | None = None, mode: str | None = None
    ) -> dict:
        _ = user_id, prompt, session_id, mode
        return {
            "job_id": "job-trace-1",
            "status": "queued",
            "submitted_at": "2026-03-10T00:00:00+00:00",
        }

    poll_count = {"value": 0}

    def fake_get_chat_job(job_id: str) -> dict:
        assert job_id == "job-trace-1"
        poll_count["value"] += 1
        if poll_count["value"] == 1:
            return {
                "job_id": job_id,
                "user_id": "user-1",
                "session_id": "user-1",
                "status": "processing",
                "answer": "",
                "error": "",
                "trace_events": [
                    {
                        "type": "search_started",
                        "timestamp": "2026-03-10T00:00:01+00:00",
                        "payload": {"queries": ["x"]},
                    }
                ],
            }
        return {
            "job_id": job_id,
            "user_id": "user-1",
            "session_id": "user-1",
            "status": "completed",
            "answer": "hello",
            "error": "",
            "trace_events": [
                {
                    "type": "search_started",
                    "timestamp": "2026-03-10T00:00:01+00:00",
                    "payload": {"queries": ["x"]},
                },
                {
                    "type": "answer_finalized",
                    "timestamp": "2026-03-10T00:00:02+00:00",
                    "payload": {"source_urls": ["https://example.edu/evidence"]},
                },
            ],
        }

    monkeypatch.setattr(chat_api, "enqueue_chat_job", fake_enqueue_chat_job)
    monkeypatch.setattr(chat_api, "get_chat_job", fake_get_chat_job)

    token = create_access_token(user_id="user-1", roles=["user"])
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat/stream",
        json={"user_id": "user-1", "prompt": "hello stream"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    body = response.text
    assert '"type": "trace"' in body
    assert '"type": "search_started"' in body
    assert '"type": "answer_finalized"' in body


def test_chat_stream_serializes_decimal_trace_payload(monkeypatch):
    def fake_enqueue_chat_job(
        *, user_id: str, prompt: str, session_id: str | None = None, mode: str | None = None
    ) -> dict:
        _ = user_id, prompt, session_id, mode
        return {
            "job_id": "job-trace-decimal",
            "status": "queued",
            "submitted_at": "2026-03-10T00:00:00+00:00",
        }

    def fake_get_chat_job(job_id: str) -> dict:
        assert job_id == "job-trace-decimal"
        return {
            "job_id": job_id,
            "user_id": "user-1",
            "session_id": "user-1",
            "status": "completed",
            "answer": "ok",
            "error": "",
            "debug": {"query_decomposition": {"university": "TUM"}},
            "trace_events": [
                {
                    "type": "search_started",
                    "timestamp": "2026-03-10T00:00:01+00:00",
                    "payload": {"step": Decimal("1"), "queries": ["x"]},
                }
            ],
        }

    monkeypatch.setattr(chat_api, "enqueue_chat_job", fake_enqueue_chat_job)
    monkeypatch.setattr(chat_api, "get_chat_job", fake_get_chat_job)

    token = create_access_token(user_id="user-1", roles=["user"])
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat/stream",
        json={"user_id": "user-1", "prompt": "hello stream"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    body = response.text
    assert '"type": "trace"' in body
    assert '"step": 1' in body
    assert '"type":"done"' in body
    assert '"type": "debug"' not in body
    assert "query_decomposition" not in body
    assert '"type": "error"' not in body


def test_chat_stream_endpoint_hides_internal_details(monkeypatch):
    def fake_enqueue_chat_job(
        *, user_id: str, prompt: str, session_id: str | None = None, mode: str | None = None
    ) -> dict:
        _ = user_id, prompt, session_id, mode
        raise RuntimeError("provider timeout at host internal.example")

    monkeypatch.setattr(chat_api, "enqueue_chat_job", fake_enqueue_chat_job)

    token = create_access_token(user_id="user-1", roles=["user"])
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat/stream",
        json={"user_id": "user-1", "prompt": "hello stream"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert '"type": "error"' in response.text
    assert '"detail": "Async chat service is temporarily unavailable."' in response.text
    assert "internal.example" not in response.text


def test_chat_stream_endpoint_forbidden_for_different_user():
    token = create_access_token(user_id="user-a", roles=["user"])
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat/stream",
        json={"user_id": "user-b", "prompt": "hello"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


def test_chat_status_hides_internal_job_error(monkeypatch):
    internal_ip = ".".join(["10", "0", "0", "1"])
    monkeypatch.setattr(
        chat_api,
        "get_chat_job",
        lambda _job_id: {
            "job_id": "job-abc-1234",
            "user_id": "user-1",
            "session_id": "session-1",
            "status": "failed",
            "created_at": "2026-03-10T00:00:00+00:00",
            "started_at": "2026-03-10T00:00:02+00:00",
            "completed_at": "2026-03-10T00:00:03+00:00",
            "answer": "",
            "error": f"db host is {internal_ip}:5432",
        },
    )

    token = create_access_token(user_id="user-1", roles=["user"])
    client = TestClient(app)
    response = client.get(
        "/api/v1/chat/job-abc-1234",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "failed"
    assert payload["error"] == "Async chat job failed."
    assert internal_ip not in payload["error"]


def test_chat_status_includes_trace_events(monkeypatch):
    monkeypatch.setattr(
        chat_api,
        "get_chat_job",
        lambda _job_id: {
            "job_id": "job-trace-status",
            "user_id": "user-1",
            "session_id": "session-1",
            "status": "completed",
            "created_at": "2026-03-10T00:00:00+00:00",
            "started_at": "2026-03-10T00:00:02+00:00",
            "completed_at": "2026-03-10T00:00:03+00:00",
            "answer": "done",
            "error": "",
            "debug": {"query_decomposition": {"university": "TUM"}},
            "trace_events": [
                {
                    "type": "search_results",
                    "timestamp": "2026-03-10T00:00:01+00:00",
                    "payload": {"urls": ["https://example.edu"]},
                }
            ],
        },
    )

    token = create_access_token(user_id="user-1", roles=["user"])
    client = TestClient(app)
    response = client.get(
        "/api/v1/chat/job-trace-status",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("trace_events"), list)
    assert payload["trace_events"][0]["type"] == "search_results"
    assert "debug" not in payload


def test_route_matching_middleware_formats_404():
    client = TestClient(app)
    response = client.get("/api/v1/does-not-exist")
    assert response.status_code == 404
    body = response.json()
    assert body["detail"] == "No route matched this path."
    assert body["path"] == "/api/v1/does-not-exist"


def test_chat_history_clear_endpoint_success(monkeypatch):
    monkeypatch.setattr(
        chat_api,
        "clear_user_chat_state",
        lambda _user_id, session_id=None: {
            "memory_keys_deleted": 2 if not session_id else 1,
            "legacy_memory_keys_deleted": 1 if not session_id else 0,
            "cache_keys_deleted": 5 if not session_id else 2,
        },
    )
    monkeypatch.setattr(
        chat_api,
        "clear_chat_traces",
        lambda _user_id: {
            "trace_keys_deleted": 7,
            "index_key_deleted": 1,
        },
    )

    token = create_access_token(user_id="user-1", roles=["user"])
    client = TestClient(app)
    response = client.delete(
        "/api/v1/chat/history",
        params={"user_id": "user-1"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == "user-1"
    assert payload["session_id"] is None
    assert payload["memory_keys_deleted"] == 2
    assert payload["legacy_memory_keys_deleted"] == 1
    assert payload["cache_keys_deleted"] == 5
    assert payload["trace_keys_deleted"] == 7
    assert payload["trace_index_deleted"] == 1


def test_chat_history_clear_endpoint_requires_owner():
    token = create_access_token(user_id="user-a", roles=["user"])
    client = TestClient(app)
    response = client.delete(
        "/api/v1/chat/history",
        params={"user_id": "user-b"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


def test_ops_status_requires_admin(monkeypatch):
    monkeypatch.setattr(
        ops_api,
        "get_ops_status",
        lambda: {
            "status": "ok",
            "memory": {"redis_available": True, "ttl_seconds": 3600, "encryption_enabled": True},
            "queue": {
                "stream_depth": 0,
                "pending_jobs": 0,
                "dlq_depth": 0,
                "consumer_group": "memory-summary-workers",
                "last_dlq_error": "",
            },
            "compaction": {"events": 0, "removed_messages": 0, "removed_tokens": 0},
            "latency": {
                "count": 0,
                "pipeline_count": 0,
                "average_ms": 0.0,
                "average_build_context_ms": 0.0,
                "average_retrieval_ms": 0.0,
                "average_model_ms": 0.0,
                "max_ms": 0,
                "last_ms": 0,
                "last_build_context_ms": 0,
                "last_retrieval_ms": 0,
                "last_model_ms": 0,
                "last_retrieval_strategy": "",
                "last_retrieved_count": 0,
                "last_outcome": "",
            },
        },
    )

    token = create_access_token(user_id="user-1", roles=["user"])
    client = TestClient(app)
    response = client.get(
        "/api/v1/ops/status",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 403
