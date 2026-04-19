from app.services import evaluation_service


class FakeRedis:
    def __init__(self):
        self._kv = {}
        self._lists = {}

    def setex(self, key, _ttl, value):
        self._kv[key] = value

    def get(self, key):
        return self._kv.get(key)

    def lpush(self, key, value):
        self._lists.setdefault(key, [])
        self._lists[key].insert(0, value)

    def ltrim(self, key, start, end):
        values = self._lists.get(key, [])
        self._lists[key] = values[start : end + 1]

    def expire(self, _key, _ttl):
        return True

    def lrange(self, key, start, end):
        values = self._lists.get(key, [])
        return values[start : end + 1]


def test_store_and_list_chat_traces():
    redis = FakeRedis()
    conversation_id = evaluation_service.store_chat_trace(
        user_id="user-1",
        prompt="What is RTU?",
        answer="A public research university in Munich.",
        retrieved_results=[
            {"chunk_id": "university_1:0000", "content": "Rheinberg Technical University (RTU) ..."}
        ],
        retrieval_strategy="filtered_exact",
        timings_ms={"retrieval": 10},
        redis=redis,
    )

    assert conversation_id is not None
    stored_raw = redis._kv[evaluation_service._conversation_key(conversation_id)]
    assert stored_raw.startswith("enc:v2:")
    traces = evaluation_service.list_chat_traces("user-1", limit=10, redis=redis)
    assert len(traces) == 1
    assert traces[0]["conversation_id"] == conversation_id
    assert traces[0]["prompt"] == "What is RTU?"


def test_label_chat_trace_and_report():
    redis = FakeRedis()
    conversation_id = evaluation_service.store_chat_trace(
        user_id="user-2",
        prompt="Where is RTU located?",
        answer="RTU is in Munich, Germany.",
        retrieved_results=[
            {
                "chunk_id": "university_1:0000",
                "content": "Location: Munich, Germany",
            }
        ],
        retrieval_strategy="ann",
        timings_ms={"retrieval": 8},
        redis=redis,
    )

    labeled = evaluation_service.label_chat_trace(
        user_id="user-2",
        conversation_id=conversation_id,
        expected_answer="RTU is located in Munich, Germany.",
        relevant_chunk_ids=["university_1:0000"],
        redis=redis,
    )

    assert labeled is not None
    report = evaluation_service.get_user_evaluation_report("user-2", limit=10, redis=redis)

    assert report["total_conversations"] == 1
    assert report["labeled_conversations"] == 1
    assert report["retrieval_metrics"]["hit_at_k"] == 1.0
    assert report["generation_metrics"]["query_relevance"] > 0.0
    assert report["conversations"][0]["metrics"]["generation"]["hallucination_proxy"] < 1.0


def test_report_includes_web_fallback_quality_and_feedback():
    redis = FakeRedis()
    conversation_id = evaluation_service.store_chat_trace(
        user_id="user-3",
        prompt="latest admissions",
        answer="Use https://www.lmu.de/programs/ai.",
        retrieved_results=[
            {
                "chunk_id": "web:1",
                "source_path": "https://www.lmu.de/programs/ai",
                "metadata": {"url": "https://www.lmu.de/programs/ai"},
                "content": "LMU AI admissions details",
            }
        ],
        retrieval_strategy="web_fallback_reranked",
        timings_ms={"retrieval": 5},
        quality={"citation_accuracy": 1.0},
        evidence_urls=["https://www.lmu.de/programs/ai"],
        redis=redis,
    )

    labeled = evaluation_service.label_chat_trace(
        user_id="user-3",
        conversation_id=conversation_id,
        user_feedback="Helpful answer",
        user_feedback_score=1,
        redis=redis,
    )
    assert labeled is not None

    report = evaluation_service.get_user_evaluation_report("user-3", limit=10, redis=redis)
    web_metrics = report["web_fallback_metrics"]
    assert web_metrics["total_web_fallback_answers"] == 1
    assert web_metrics["avg_source_count"] == 1.0
    assert web_metrics["avg_citation_accuracy"] == 1.0
    assert web_metrics["feedback_count"] == 1
    assert web_metrics["positive_feedback_rate"] == 1.0


def test_list_chat_traces_falls_back_to_dynamodb_when_redis_unavailable(monkeypatch):
    class BrokenRedis:
        def lrange(self, *_args, **_kwargs):
            raise TimeoutError("redis unavailable")

    class FakeDynamoClient:
        def scan(self, **_kwargs):
            return {
                "Items": [
                    {
                        "request_id": {"S": "req-1"},
                        "timestamp": {"S": "2026-04-19T10:30:00+00:00"},
                        "user_id": {"S": "user-ddb"},
                        "query": {"S": "Show prior conversation"},
                        "answer": {"S": "Loaded from DynamoDB fallback."},
                        "retrieval_strategy": {"S": "web_fallback_reranked"},
                        "retrieval_result_count": {"N": "3"},
                        "groundedness": {"N": "0.92"},
                        "citation_accuracy": {"N": "1.0"},
                        "outcome": {"S": "success"},
                    }
                ]
            }

    monkeypatch.setattr(evaluation_service, "redis_client", BrokenRedis())
    monkeypatch.setattr(evaluation_service, "_dynamodb_client", lambda: FakeDynamoClient())
    monkeypatch.setattr(evaluation_service.settings.app, "metrics_dynamodb_enabled", True)
    monkeypatch.setattr(
        evaluation_service.settings.app, "metrics_dynamodb_requests_table", "req-table"
    )

    traces = evaluation_service.list_chat_traces("user-ddb", limit=5)
    assert len(traces) == 1
    assert traces[0]["conversation_id"] == "req-1"
    assert traces[0]["prompt"] == "Show prior conversation"
    assert traces[0]["answer"] == "Loaded from DynamoDB fallback."
    assert traces[0]["retrieval_result_count"] == 3
    assert traces[0]["_trace_source"] == "dynamodb_metrics"
