import json

from app.services import metrics_json_service


def test_append_chat_metrics_json_writes_request_and_aggregate(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics_json_service.settings.app, "metrics_json_enabled", True)
    monkeypatch.setattr(metrics_json_service.settings.app, "metrics_json_dir", str(tmp_path))

    metrics_json_service.append_chat_metrics_json(
        {
            "request_id": "req-1",
            "timestamp": "2026-03-08T00:00:00+00:00",
            "user_id": "user-1",
            "outcome": "success",
            "question": "What is RTU?",
            "answer": "A university in Munich.",
            "timings_ms": {
                "overall_response_ms": 120,
                "llm_response_ms": 60,
                "short_term_memory_ms": 20,
                "long_term_memory_ms": 25,
                "memory_update_ms": 10,
                "cache_read_ms": 2,
                "cache_write_ms": 3,
                "evaluation_trace_ms": 4,
            },
            "quality": {
                "hallucination_proxy": 0.2,
                "context_coverage": 0.8,
                "query_relevance": 0.9,
            },
            "llm_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
    )

    requests_path = tmp_path / "chat_metrics_requests.jsonl"
    aggregate_path = tmp_path / "chat_metrics_aggregate.json"
    assert requests_path.exists()
    assert aggregate_path.exists()

    request_lines = requests_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(request_lines) == 1
    request_payload = json.loads(request_lines[0])
    assert request_payload["request_id"] == "req-1"
    assert request_payload["outcome"] == "success"

    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["total_requests"] == 1
    assert aggregate["outcomes"]["success"] == 1
    assert aggregate["latency_ms"]["overall"]["count"] == 1
    assert aggregate["latency_ms"]["overall"]["average"] == 120.0
    assert aggregate["latency_ms"]["overall"]["p95"] == 120.0
    assert aggregate["latency_ms"]["overall"]["p99"] == 120.0
    assert aggregate["quality"]["hallucination_proxy"]["average"] == 0.2
    assert aggregate["token_usage"]["requests_with_usage"] == 1
    assert aggregate["token_usage"]["total_tokens_total"] == 30
    assert aggregate["latest_request"]["request_id"] == "req-1"


def test_append_chat_metrics_json_skips_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics_json_service.settings.app, "metrics_json_enabled", False)
    monkeypatch.setattr(metrics_json_service.settings.app, "metrics_json_dir", str(tmp_path))

    metrics_json_service.append_chat_metrics_json(
        {
            "request_id": "req-disabled",
            "outcome": "success",
            "timings_ms": {"overall_response_ms": 15},
        }
    )

    assert not (tmp_path / "chat_metrics_requests.jsonl").exists()
    assert not (tmp_path / "chat_metrics_aggregate.json").exists()
