from app.services import summary_queue_service


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.streams = {}
        self.claimed_entries = []
        self.last_xautoclaim = None

    def set(self, key, value, ex=None, nx=False):
        if nx and key in self.store:
            return False
        self.store[key] = value
        return True

    def xlen(self, name):
        return len(self.streams.get(name, []))

    def xrevrange(self, name, count=1):
        entries = list(reversed(self.streams.get(name, [])))
        return entries[:count]

    def xadd(self, name, payload, maxlen=None, approximate=None):
        stream = self.streams.setdefault(name, [])
        stream_id = f"{len(stream) + 1}-0"
        stream.append((stream_id, dict(payload)))
        return stream_id

    def xautoclaim(
        self,
        name,
        groupname,
        consumername,
        min_idle_time,
        start_id="0-0",
        count=None,
    ):
        self.last_xautoclaim = {
            "name": name,
            "groupname": groupname,
            "consumername": consumername,
            "min_idle_time": min_idle_time,
            "start_id": start_id,
            "count": count,
        }
        return ("0-0", list(self.claimed_entries), [])


def test_monitor_summary_dlq_alerts_once_per_cooldown(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(summary_queue_service, "worker_redis_client", fake_redis)
    monkeypatch.setattr(
        summary_queue_service.settings.memory,
        "summary_queue_dlq_alert_threshold",
        1,
    )
    monkeypatch.setattr(
        summary_queue_service.settings.memory,
        "summary_queue_dlq_alert_cooldown_seconds",
        300,
    )

    fake_redis.streams[summary_queue_service._dlq_stream_key()] = [
        (
            "1-0",
            {
                "job_id": "job-1",
                "user_id": "user-1",
                "failed_at": "2026-02-28T00:00:00+00:00",
                "error": "boom",
                "final_attempt": "5",
            },
        )
    ]

    alerts = []
    monkeypatch.setattr(
        summary_queue_service.logger,
        "error",
        lambda message, payload: alerts.append((message, payload)),
    )

    first = summary_queue_service.monitor_summary_dlq()
    second = summary_queue_service.monitor_summary_dlq()

    assert first["depth"] == 1
    assert first["alerted"] is True
    assert second["alerted"] is False
    assert len(alerts) == 1
    assert "SummaryJobDLQAlert" in alerts[0][0]


def test_retry_or_dlq_summary_job_triggers_dlq_monitor(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(summary_queue_service, "worker_redis_client", fake_redis)
    monkeypatch.setattr(
        summary_queue_service.settings.memory,
        "summary_queue_max_attempts",
        5,
    )

    acked = []
    monitored = []
    monkeypatch.setattr(
        summary_queue_service, "ack_summary_job", lambda stream_id: acked.append(stream_id)
    )
    monkeypatch.setattr(
        summary_queue_service, "monitor_summary_dlq", lambda: monitored.append(True)
    )

    summary_queue_service.retry_or_dlq_summary_job(
        "7-0",
        {
            "job_id": "job-7",
            "user_id": "user-7",
            "attempt": "4",
        },
        "failed badly",
    )

    dlq_entries = fake_redis.streams[summary_queue_service._dlq_stream_key()]
    assert len(dlq_entries) == 1
    assert dlq_entries[0][1]["job_id"] == "job-7"
    assert acked == ["7-0"]
    assert monitored == [True]


def test_claim_stale_summary_jobs_reads_xautoclaim(monkeypatch):
    fake_redis = FakeRedis()
    fake_redis.claimed_entries = [
        ("8-0", {"job_id": "job-8", "user_id": "user-8", "cutoff_seq": "12"}),
    ]
    monkeypatch.setattr(summary_queue_service, "worker_redis_client", fake_redis)
    monkeypatch.setattr(summary_queue_service, "ensure_consumer_group", lambda: None)
    monkeypatch.setattr(
        summary_queue_service.settings.memory,
        "summary_queue_claim_idle_ms",
        12345,
    )
    monkeypatch.setattr(
        summary_queue_service.settings.memory,
        "summary_queue_claim_batch_size",
        42,
    )

    jobs = summary_queue_service.claim_stale_summary_jobs("worker-a")

    assert jobs == fake_redis.claimed_entries
    assert fake_redis.last_xautoclaim is not None
    assert fake_redis.last_xautoclaim["consumername"] == "worker-a"
    assert fake_redis.last_xautoclaim["min_idle_time"] == 12345
    assert fake_redis.last_xautoclaim["count"] == 42


def test_enqueue_summary_job_uses_worker_redis_client(monkeypatch):
    fake_redis = FakeRedis()
    monkeypatch.setattr(summary_queue_service, "worker_redis_client", fake_redis)

    job_id = summary_queue_service.enqueue_summary_job(
        user_id="user-1",
        cutoff_seq=10,
        trigger="token_limit",
        enqueue_version=2,
        approx_removed_tokens=900,
    )

    assert isinstance(job_id, str) and job_id
    entries = fake_redis.streams[summary_queue_service._stream_key()]
    assert len(entries) == 1
    assert entries[0][1]["user_id"] == "user-1"
    assert entries[0][1]["cutoff_seq"] == "10"
