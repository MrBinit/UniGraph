from app.services import metrics_dynamodb_service


class _FakeDynamoClient:
    def __init__(self):
        self.calls = []

    def put_item(self, **kwargs):
        self.calls.append(kwargs)


def test_persist_chat_metrics_dynamodb_writes_request_and_aggregate(monkeypatch):
    fake = _FakeDynamoClient()
    monkeypatch.setattr(
        metrics_dynamodb_service.settings.app, "metrics_dynamodb_enabled", True
    )
    monkeypatch.setattr(
        metrics_dynamodb_service.settings.app,
        "metrics_dynamodb_requests_table",
        "chat-metrics-requests",
    )
    monkeypatch.setattr(
        metrics_dynamodb_service.settings.app,
        "metrics_dynamodb_aggregate_table",
        "chat-metrics-aggregate",
    )
    monkeypatch.setattr(metrics_dynamodb_service, "_dynamodb_client", lambda: fake)

    record = {
        "request_id": "req-1",
        "timestamp": "2026-03-10T00:00:00+00:00",
        "user_id": "user-1",
        "session_id": "session-1",
        "outcome": "success",
        "retrieval": {
            "strategy": "hnsw",
            "result_count": 3,
            "evidence": [{"chunk_id": "c-1", "content": "sample"}],
        },
        "llm_usage": {"prompt_tokens": 120, "total_tokens": 280},
        "question": "question",
        "answer": "answer",
        "timings_ms": {
            "overall_response_ms": 123,
            "llm_response_ms": 100,
            "short_term_memory_ms": 8,
            "long_term_memory_ms": 12,
        },
    }
    aggregate = {
        "updated_at": "2026-03-10T00:00:00+00:00",
        "total_requests": 1,
        "latency_ms": {"overall": {"average": 123.0}},
    }

    metrics_dynamodb_service.persist_chat_metrics_dynamodb(record, aggregate)

    assert len(fake.calls) == 2
    request_write = next(
        call for call in fake.calls if call["TableName"] == "chat-metrics-requests"
    )
    aggregate_write = next(
        call for call in fake.calls if call["TableName"] == "chat-metrics-aggregate"
    )
    assert request_write["Item"]["request_id"]["S"] == "req-1"
    assert request_write["Item"]["session_id"]["S"] == "session-1"
    assert request_write["Item"]["retrieval_strategy"]["S"] == "hnsw"
    assert request_write["Item"]["retrieval_result_count"]["N"] == "3"
    assert request_write["Item"]["retrieval_evidence_count"]["N"] == "1"
    assert request_write["Item"]["prompt_tokens"]["N"] == "120"
    assert request_write["Item"]["total_tokens"]["N"] == "280"
    assert aggregate_write["Item"]["id"]["S"] == "global"


def test_persist_chat_metrics_dynamodb_skips_when_disabled(monkeypatch):
    fake = _FakeDynamoClient()
    monkeypatch.setattr(
        metrics_dynamodb_service.settings.app, "metrics_dynamodb_enabled", False
    )
    monkeypatch.setattr(metrics_dynamodb_service, "_dynamodb_client", lambda: fake)

    metrics_dynamodb_service.persist_chat_metrics_dynamodb(
        {"request_id": "req-x"},
        {"total_requests": 1},
    )
    assert not fake.calls


def test_persist_chat_metrics_dynamodb_queues_aggregate_when_enabled(monkeypatch):
    fake = _FakeDynamoClient()
    queued = []
    monkeypatch.setattr(
        metrics_dynamodb_service.settings.app, "metrics_dynamodb_enabled", True
    )
    monkeypatch.setattr(
        metrics_dynamodb_service.settings.app,
        "metrics_dynamodb_requests_table",
        "chat-metrics-requests",
    )
    monkeypatch.setattr(
        metrics_dynamodb_service.settings.app,
        "metrics_dynamodb_aggregate_table",
        "chat-metrics-aggregate",
    )
    monkeypatch.setattr(
        metrics_dynamodb_service.settings.queue,
        "metrics_aggregation_queue_enabled",
        True,
    )
    monkeypatch.setattr(
        metrics_dynamodb_service.settings.queue,
        "metrics_aggregation_queue_url",
        "https://sqs.example/metrics",
    )
    monkeypatch.setattr(
        metrics_dynamodb_service.settings.queue,
        "evaluation_queue_enabled",
        False,
    )
    monkeypatch.setattr(
        metrics_dynamodb_service,
        "enqueue_metrics_aggregation_event",
        lambda request_id: queued.append(request_id),
    )
    monkeypatch.setattr(metrics_dynamodb_service, "_dynamodb_client", lambda: fake)

    metrics_dynamodb_service.persist_chat_metrics_dynamodb(
        {
            "request_id": "req-queue-1",
            "timestamp": "2026-03-10T00:00:00+00:00",
            "user_id": "user-1",
            "session_id": "session-1",
            "outcome": "success",
            "retrieval": {"strategy": "hnsw", "result_count": 1, "evidence": []},
            "llm_usage": {"prompt_tokens": 5, "total_tokens": 9},
            "question": "q",
            "answer": "a",
            "timings_ms": {"overall_response_ms": 10},
        },
        {"updated_at": "2026-03-10T00:00:00+00:00", "total_requests": 1},
    )

    # Request record is written inline; aggregate write is offloaded to queue.
    assert len(fake.calls) == 1
    assert fake.calls[0]["TableName"] == "chat-metrics-requests"
    assert queued == ["req-queue-1"]


def test_persist_chat_metrics_dynamodb_queues_eval_event(monkeypatch):
    fake = _FakeDynamoClient()
    eval_events = []
    monkeypatch.setattr(
        metrics_dynamodb_service.settings.app, "metrics_dynamodb_enabled", True
    )
    monkeypatch.setattr(
        metrics_dynamodb_service.settings.app,
        "metrics_dynamodb_requests_table",
        "chat-metrics-requests",
    )
    monkeypatch.setattr(
        metrics_dynamodb_service.settings.app,
        "metrics_dynamodb_aggregate_table",
        "chat-metrics-aggregate",
    )
    monkeypatch.setattr(
        metrics_dynamodb_service.settings.queue,
        "metrics_aggregation_queue_enabled",
        False,
    )
    monkeypatch.setattr(
        metrics_dynamodb_service.settings.queue, "evaluation_queue_enabled", True
    )
    monkeypatch.setattr(
        metrics_dynamodb_service.settings.queue,
        "evaluation_queue_url",
        "https://sqs.example/eval",
    )
    monkeypatch.setattr(
        metrics_dynamodb_service,
        "enqueue_evaluation_event",
        lambda request_id, session_id="": eval_events.append((request_id, session_id)),
    )
    monkeypatch.setattr(metrics_dynamodb_service, "_dynamodb_client", lambda: fake)

    metrics_dynamodb_service.persist_chat_metrics_dynamodb(
        {
            "request_id": "req-eval-1",
            "timestamp": "2026-03-10T00:00:00+00:00",
            "user_id": "user-2",
            "session_id": "session-2",
            "outcome": "success",
            "retrieval": {"strategy": "hnsw", "result_count": 1, "evidence": []},
            "llm_usage": {"prompt_tokens": 5, "total_tokens": 9},
            "question": "q",
            "answer": "a",
            "timings_ms": {"overall_response_ms": 10},
        },
        {
            "updated_at": "2026-03-10T00:00:00+00:00",
            "total_requests": 1,
            "latency_ms": {"overall": {"average": 10.0}},
        },
    )

    # Request + aggregate are still persisted inline when aggregate queue is disabled.
    assert len(fake.calls) == 2
    assert eval_events == [("req-eval-1", "session-2")]
