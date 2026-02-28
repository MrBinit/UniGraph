import pytest

from app.services import summary_worker_service


def _base_memory():
    return {
        "summary": "",
        "messages": [
            {"seq": 1, "role": "user", "content": "u1"},
            {"seq": 2, "role": "assistant", "content": "a1"},
            {"seq": 3, "role": "user", "content": "u2"},
        ],
        "version": 4,
        "next_seq": 4,
        "last_summarized_seq": 0,
        "summary_pending": True,
        "last_summary_job_id": "job-1",
    }


@pytest.mark.asyncio
async def test_process_summary_job_success(monkeypatch):
    acked = []
    retried = []
    metrics_calls = []
    marked = []

    async def fake_load_memory(_user_id):
        return _base_memory()

    async def fake_summarize(_messages):
        return "summary"

    def fake_save_if_version(_user_id, _expected_version, memory):
        return True, memory

    monkeypatch.setattr(summary_worker_service, "load_memory", fake_load_memory)
    monkeypatch.setattr(summary_worker_service, "summarize_messages", fake_summarize)
    monkeypatch.setattr(summary_worker_service, "save_memory_if_version", fake_save_if_version)
    monkeypatch.setattr(summary_worker_service, "get_summary_job_idempotency_key", lambda _fields: "idem-1")
    monkeypatch.setattr(summary_worker_service, "is_summary_job_processed", lambda _key: False)
    monkeypatch.setattr(summary_worker_service, "claim_summary_job_processing", lambda _key, _stream_id: True)
    monkeypatch.setattr(
        summary_worker_service,
        "mark_summary_job_processed",
        lambda key, stream_id: marked.append((key, stream_id)),
    )
    monkeypatch.setattr(summary_worker_service, "release_summary_job_processing", lambda _key: None)
    monkeypatch.setattr(summary_worker_service, "ack_summary_job", lambda stream_id: acked.append(stream_id))
    monkeypatch.setattr(
        summary_worker_service,
        "retry_or_dlq_summary_job",
        lambda stream_id, fields, error: retried.append((stream_id, fields, error)),
    )
    monkeypatch.setattr(
        summary_worker_service,
        "record_compaction_metrics",
        lambda **kwargs: metrics_calls.append(kwargs),
    )

    await summary_worker_service.process_summary_job(
        "1-0",
        {"job_id": "job-1", "user_id": "user-1", "cutoff_seq": "2", "trigger": "summary_trigger"},
    )

    assert acked == ["1-0"]
    assert retried == []
    assert marked == [("idem-1", "1-0")]
    assert len(metrics_calls) == 1
    assert metrics_calls[0]["trigger"] == "async_summary_trigger"


@pytest.mark.asyncio
async def test_process_summary_job_stale_acks_without_retry(monkeypatch):
    acked = []
    retried = []
    marked = []

    async def fake_load_memory(_user_id):
        memory = _base_memory()
        memory["last_summarized_seq"] = 10
        return memory

    monkeypatch.setattr(summary_worker_service, "load_memory", fake_load_memory)
    monkeypatch.setattr(summary_worker_service, "get_summary_job_idempotency_key", lambda _fields: "idem-2")
    monkeypatch.setattr(summary_worker_service, "is_summary_job_processed", lambda _key: False)
    monkeypatch.setattr(summary_worker_service, "claim_summary_job_processing", lambda _key, _stream_id: True)
    monkeypatch.setattr(
        summary_worker_service,
        "mark_summary_job_processed",
        lambda key, stream_id: marked.append((key, stream_id)),
    )
    monkeypatch.setattr(summary_worker_service, "release_summary_job_processing", lambda _key: None)
    monkeypatch.setattr(summary_worker_service, "ack_summary_job", lambda stream_id: acked.append(stream_id))
    monkeypatch.setattr(
        summary_worker_service,
        "retry_or_dlq_summary_job",
        lambda stream_id, fields, error: retried.append((stream_id, fields, error)),
    )

    await summary_worker_service.process_summary_job(
        "2-0",
        {"job_id": "job-2", "user_id": "user-1", "cutoff_seq": "2", "trigger": "summary_trigger"},
    )

    assert acked == ["2-0"]
    assert retried == []
    assert marked == [("idem-2", "2-0")]


@pytest.mark.asyncio
async def test_process_summary_job_retries_on_error(monkeypatch):
    acked = []
    retried = []
    released = []

    async def fake_load_memory(_user_id):
        return _base_memory()

    async def fake_summarize(_messages):
        return ""

    monkeypatch.setattr(summary_worker_service, "load_memory", fake_load_memory)
    monkeypatch.setattr(summary_worker_service, "summarize_messages", fake_summarize)
    monkeypatch.setattr(summary_worker_service, "get_summary_job_idempotency_key", lambda _fields: "idem-3")
    monkeypatch.setattr(summary_worker_service, "is_summary_job_processed", lambda _key: False)
    monkeypatch.setattr(summary_worker_service, "claim_summary_job_processing", lambda _key, _stream_id: True)
    monkeypatch.setattr(summary_worker_service, "mark_summary_job_processed", lambda _key, _stream_id: None)
    monkeypatch.setattr(
        summary_worker_service,
        "release_summary_job_processing",
        lambda key: released.append(key),
    )
    monkeypatch.setattr(summary_worker_service, "ack_summary_job", lambda stream_id: acked.append(stream_id))
    monkeypatch.setattr(
        summary_worker_service,
        "retry_or_dlq_summary_job",
        lambda stream_id, fields, error: retried.append((stream_id, fields, error)),
    )

    await summary_worker_service.process_summary_job(
        "3-0",
        {"job_id": "job-3", "user_id": "user-1", "cutoff_seq": "2", "trigger": "summary_trigger"},
    )

    assert acked == []
    assert len(retried) == 1
    assert released == ["idem-3"]


@pytest.mark.asyncio
async def test_process_summary_job_skips_replayed_job(monkeypatch):
    acked = []
    retried = []

    async def should_not_load(_user_id):
        raise AssertionError("load_memory should not run for replayed jobs")

    monkeypatch.setattr(summary_worker_service, "load_memory", should_not_load)
    monkeypatch.setattr(summary_worker_service, "get_summary_job_idempotency_key", lambda _fields: "idem-4")
    monkeypatch.setattr(summary_worker_service, "is_summary_job_processed", lambda _key: True)
    monkeypatch.setattr(
        summary_worker_service,
        "claim_summary_job_processing",
        lambda _key, _stream_id: (_ for _ in ()).throw(AssertionError("should not claim")),
    )
    monkeypatch.setattr(summary_worker_service, "mark_summary_job_processed", lambda _key, _stream_id: None)
    monkeypatch.setattr(summary_worker_service, "release_summary_job_processing", lambda _key: None)
    monkeypatch.setattr(summary_worker_service, "ack_summary_job", lambda stream_id: acked.append(stream_id))
    monkeypatch.setattr(
        summary_worker_service,
        "retry_or_dlq_summary_job",
        lambda stream_id, fields, error: retried.append((stream_id, fields, error)),
    )

    await summary_worker_service.process_summary_job(
        "4-0",
        {"job_id": "job-4", "user_id": "user-1", "cutoff_seq": "2", "trigger": "summary_trigger"},
    )

    assert acked == ["4-0"]
    assert retried == []
