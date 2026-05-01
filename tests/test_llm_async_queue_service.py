from app.services import llm_async_queue_service


def test_mark_job_failed_sanitizes_internal_exception_text(monkeypatch):
    captured = []
    monkeypatch.setattr(
        llm_async_queue_service,
        "_update_job",
        lambda job_id, updates: captured.append((job_id, updates)),
    )
    monkeypatch.setattr(llm_async_queue_service, "_now_iso", lambda: "2026-03-11T00:00:00+00:00")

    internal_ip = ".".join(["10", "0", "0", "1"])
    llm_async_queue_service.mark_job_failed("job-1", f"credential=secret db host={internal_ip}")

    assert captured == [
        (
            "job-1",
            {
                "status": "failed",
                "error": "Async chat job failed.",
                "updated_at": "2026-03-11T00:00:00+00:00",
            },
        )
    ]


def test_mark_job_failed_preserves_safe_queue_enqueue_error(monkeypatch):
    captured = []
    monkeypatch.setattr(
        llm_async_queue_service,
        "_update_job",
        lambda job_id, updates: captured.append((job_id, updates)),
    )
    monkeypatch.setattr(llm_async_queue_service, "_now_iso", lambda: "2026-03-11T00:00:00+00:00")

    llm_async_queue_service.mark_job_failed("job-2", "Queue enqueue failed: endpoint timeout")

    assert captured[0][1]["error"] == "Queue enqueue failed."


def test_mark_job_failed_preserves_invalid_payload_error(monkeypatch):
    captured = []
    monkeypatch.setattr(
        llm_async_queue_service,
        "_update_job",
        lambda job_id, updates: captured.append((job_id, updates)),
    )
    monkeypatch.setattr(llm_async_queue_service, "_now_iso", lambda: "2026-03-11T00:00:00+00:00")

    llm_async_queue_service.mark_job_failed("job-3", "Invalid async job payload.")

    assert captured[0][1]["error"] == "Invalid async job payload."


def test_mark_job_completed_writes_debug_artifact_not_dynamodb_debug(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(
        llm_async_queue_service,
        "_update_job",
        lambda job_id, updates: captured.append((job_id, updates)),
    )
    monkeypatch.setattr(llm_async_queue_service, "_now_iso", lambda: "2026-03-11T00:00:00+00:00")
    monkeypatch.setenv("UNIGRAPH_DEBUG_DIR", str(tmp_path))

    llm_async_queue_service.mark_job_completed(
        "job-debug",
        "answer",
        debug_info={
            "request_id": "unigraph-1",
            "raw_search_results": [
                {"query": "q", "results": [{"snippet": "x" * 5000} for _ in range(10)]}
                for _ in range(20)
            ],
            "chunks_created_detail": [{"chunks": ["y" * 5000 for _ in range(20)]}],
            "selected_evidence_chunks": [{"text": "z" * 5000} for _ in range(20)],
            "rejected_urls_with_reasons": [{"url": f"https://example.edu/{i}"} for i in range(100)],
        },
    )

    updates = captured[0][1]
    assert "debug" not in updates
    artifact_path = updates["debug_artifact_path"]
    assert artifact_path.endswith(".json")
    artifact_text = tmp_path.joinpath(artifact_path.split("/")[-1]).read_text(encoding="utf-8")
    assert '"request_id": "unigraph-1"' in artifact_text
    assert '"raw_search_results"' in artifact_text


def test_append_job_trace_event_sanitizes_payload(monkeypatch):
    captured = {}

    class _FakeTable:
        def update_item(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_async_queue_service, "_dynamodb_table", lambda: _FakeTable())
    monkeypatch.setattr(llm_async_queue_service, "_now_iso", lambda: "2026-03-11T00:00:00+00:00")

    llm_async_queue_service.append_job_trace_event(
        "job-4",
        {
            "type": "search_results",
            "timestamp": "2026-03-11T00:00:01+00:00",
            "payload": {"urls": ["https://example.edu"], "nested": {"value": "ok"}},
        },
    )

    assert captured["Key"] == {"job_id": "job-4"}
    new_event = captured["ExpressionAttributeValues"][":new_event"][0]
    assert new_event["type"] == "search_results"
    assert new_event["payload"]["urls"] == ["https://example.edu"]


def test_enqueue_chat_job_persists_normalized_mode(monkeypatch):
    captured_record = {}
    captured_send = {}

    monkeypatch.setattr(llm_async_queue_service.settings.queue, "llm_async_enabled", True)
    monkeypatch.setattr(
        llm_async_queue_service.settings.queue,
        "llm_queue_url",
        "https://sqs.us-east-1.amazonaws.com/123/queue",
    )
    monkeypatch.setattr(llm_async_queue_service.settings.queue, "llm_result_table", "tbl")
    monkeypatch.setattr(llm_async_queue_service.settings.queue, "llm_result_ttl_days", 0)
    monkeypatch.setattr(llm_async_queue_service, "_now_iso", lambda: "2026-03-30T00:00:00+00:00")
    monkeypatch.setattr(
        llm_async_queue_service,
        "_put_initial_job",
        lambda record: captured_record.update(record),
    )

    class _FakeSqs:
        def send_message(self, **kwargs):
            captured_send.update(kwargs)
            return {"MessageId": "mid-1"}

    monkeypatch.setattr(llm_async_queue_service, "_sqs_client", lambda: _FakeSqs())
    monkeypatch.setattr(llm_async_queue_service, "_update_job", lambda *_args, **_kwargs: None)

    response = llm_async_queue_service.enqueue_chat_job(
        user_id="user-1",
        prompt="hello",
        session_id="session-1",
        mode="DEEP",
    )

    assert response["status"] == "queued"
    assert captured_record["mode"] == "deep"
    assert '"mode": "deep"' in captured_send["MessageBody"]


def test_enqueue_chat_job_preserves_standard_mode(monkeypatch):
    captured_record = {}
    captured_send = {}

    monkeypatch.setattr(llm_async_queue_service.settings.queue, "llm_async_enabled", True)
    monkeypatch.setattr(
        llm_async_queue_service.settings.queue,
        "llm_queue_url",
        "https://sqs.us-east-1.amazonaws.com/123/queue",
    )
    monkeypatch.setattr(llm_async_queue_service.settings.queue, "llm_result_table", "tbl")
    monkeypatch.setattr(llm_async_queue_service.settings.queue, "llm_result_ttl_days", 0)
    monkeypatch.setattr(llm_async_queue_service, "_now_iso", lambda: "2026-03-30T00:00:00+00:00")
    monkeypatch.setattr(
        llm_async_queue_service,
        "_put_initial_job",
        lambda record: captured_record.update(record),
    )

    class _FakeSqs:
        def send_message(self, **kwargs):
            captured_send.update(kwargs)
            return {"MessageId": "mid-1"}

    monkeypatch.setattr(llm_async_queue_service, "_sqs_client", lambda: _FakeSqs())
    monkeypatch.setattr(llm_async_queue_service, "_update_job", lambda *_args, **_kwargs: None)

    response = llm_async_queue_service.enqueue_chat_job(
        user_id="user-1",
        prompt="hello",
        session_id="session-1",
        mode="standard",
    )

    assert response["status"] == "queued"
    assert captured_record["mode"] == "standard"
    assert '"mode": "standard"' in captured_send["MessageBody"]
