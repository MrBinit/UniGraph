import asyncio
from datetime import datetime, timedelta, timezone

from app.services import offline_evaluation_service


class _FakeDynamoClient:
    def __init__(self):
        self.query_calls = []

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        if kwargs.get("ProjectionExpression") == "request_id":
            return {"Items": [{"request_id": {"S": "req-1"}}]}
        return {"Items": [{"timestamp": {"S": "2026-03-11T00:00:00+00:00"}}]}


class _FakeSchedulerRedis:
    def __init__(self, *, acquire: bool = True, refresh_results: list[bool] | None = None):
        self.acquire = acquire
        self.refresh_results = list(refresh_results or [])
        self.store = {}

    def set(self, key, value, ex=None, nx=False):
        _ = ex
        if nx and key in self.store:
            return False
        if nx and not self.acquire:
            return False
        self.store[key] = value
        return True

    def eval(self, script, numkeys, key, *args):
        _ = numkeys
        if "EXPIRE" in script:
            token = str(args[0]) if args else ""
            if self.store.get(key) != token:
                return 0
            if self.refresh_results:
                refreshed = bool(self.refresh_results.pop(0))
                if not refreshed:
                    return 0
            return 1
        if "DEL" in script:
            token = str(args[0]) if args else ""
            if self.store.get(key) != token:
                return 0
            self.store.pop(key, None)
            return 1
        raise AssertionError("Unexpected scheduler Redis script")


class _FakeTask:
    def __init__(self):
        self._done = False
        self.cancelled = False

    def done(self):
        return self._done

    def cancel(self):
        self.cancelled = True
        self._done = True

    def __await__(self):
        async def _noop():
            return None

        return _noop().__await__()


def _reset_scheduler_state(monkeypatch):
    monkeypatch.setattr(offline_evaluation_service, "_scheduler_task", None)
    monkeypatch.setattr(offline_evaluation_service, "_scheduler_lock_token", None)


def test_latest_timestamp_from_table_uses_status_index_query(monkeypatch):
    fake = _FakeDynamoClient()
    monkeypatch.setattr(offline_evaluation_service, "_dynamodb_client", lambda: fake)

    ts = offline_evaluation_service._latest_timestamp_from_table(
        "requests-table",
        index_name="eval-status-timestamp-index",
        status_attr="eval_status",
        status_value="pending",
    )

    assert ts == datetime.fromisoformat("2026-03-11T00:00:00+00:00")
    assert len(fake.query_calls) == 1
    query = fake.query_calls[0]
    assert query["IndexName"] == "eval-status-timestamp-index"
    assert query["KeyConditionExpression"] == "#status = :status_value"


def test_get_offline_eval_status_uses_pending_as_new_data(monkeypatch):
    now = datetime(2026, 3, 11, 12, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(offline_evaluation_service, "_utc_now", lambda: now)
    monkeypatch.setattr(offline_evaluation_service.settings.evaluation, "enabled", True)
    monkeypatch.setattr(offline_evaluation_service.settings.evaluation, "schedule_enabled", True)
    monkeypatch.setattr(
        offline_evaluation_service.settings.evaluation, "schedule_interval_hours", 1
    )
    monkeypatch.setattr(
        offline_evaluation_service.settings.app,
        "metrics_dynamodb_requests_table",
        "requests-table",
    )
    monkeypatch.setattr(
        offline_evaluation_service.settings.evaluation, "dynamodb_table", "eval-table"
    )

    def _fake_latest(table_name: str, **kwargs):
        if table_name == "eval-table":
            return now - timedelta(hours=2)
        if (
            kwargs.get("status_value")
            == offline_evaluation_service.settings.evaluation.request_pending_value
        ):
            return now
        return now - timedelta(minutes=30)

    monkeypatch.setattr(offline_evaluation_service, "_latest_timestamp_from_table", _fake_latest)
    monkeypatch.setattr(offline_evaluation_service, "_new_requests_pending", lambda: True)

    status = offline_evaluation_service.get_offline_eval_status()

    assert status["has_new_requests"] is True
    assert status["due_by_interval"] is True
    assert status["should_auto_run"] is True


def test_start_offline_eval_scheduler_skips_when_lock_not_acquired(monkeypatch):
    _reset_scheduler_state(monkeypatch)
    fake_redis = _FakeSchedulerRedis(acquire=False)
    monkeypatch.setattr(offline_evaluation_service, "app_redis_client", fake_redis)
    monkeypatch.setattr(offline_evaluation_service.settings.evaluation, "enabled", True)
    monkeypatch.setattr(offline_evaluation_service.settings.evaluation, "schedule_enabled", True)

    offline_evaluation_service.start_offline_eval_scheduler()

    assert offline_evaluation_service._scheduler_task is None
    assert offline_evaluation_service._scheduler_lock_token is None


def test_start_offline_eval_scheduler_acquires_lock_and_starts_task(monkeypatch):
    _reset_scheduler_state(monkeypatch)
    fake_redis = _FakeSchedulerRedis(acquire=True)
    fake_task = _FakeTask()
    monkeypatch.setattr(offline_evaluation_service, "app_redis_client", fake_redis)
    monkeypatch.setattr(offline_evaluation_service.settings.evaluation, "enabled", True)
    monkeypatch.setattr(offline_evaluation_service.settings.evaluation, "schedule_enabled", True)

    def _fake_create_task(coro):
        # Close unscheduled coroutine to avoid runtime warnings in test.
        coro.close()
        return fake_task

    monkeypatch.setattr(offline_evaluation_service.asyncio, "create_task", _fake_create_task)

    offline_evaluation_service.start_offline_eval_scheduler()

    assert offline_evaluation_service._scheduler_task is fake_task
    lock_token = offline_evaluation_service._scheduler_lock_token
    assert isinstance(lock_token, str) and lock_token
    assert fake_redis.store[offline_evaluation_service._SCHEDULER_LOCK_KEY] == lock_token


def test_scheduler_loop_stops_when_leader_lock_is_lost(monkeypatch):
    _reset_scheduler_state(monkeypatch)
    fake_redis = _FakeSchedulerRedis(refresh_results=[True, False])
    lock_token = "lock-token"
    fake_redis.store[offline_evaluation_service._SCHEDULER_LOCK_KEY] = lock_token
    monkeypatch.setattr(offline_evaluation_service, "app_redis_client", fake_redis)
    monkeypatch.setattr(offline_evaluation_service, "_scheduler_lock_token", lock_token)

    runs = []

    async def _fake_run_offline_eval(*, limit: int | None = None, force: bool = False):
        runs.append({"limit": limit, "force": force})
        return {"ran": False}

    async def _fake_sleep(_seconds):
        return None

    monkeypatch.setattr(offline_evaluation_service, "run_offline_eval", _fake_run_offline_eval)
    monkeypatch.setattr(offline_evaluation_service.asyncio, "sleep", _fake_sleep)

    asyncio.run(offline_evaluation_service._scheduler_loop(lock_token))

    assert runs == [
        {
            "limit": offline_evaluation_service.settings.evaluation.batch_size,
            "force": False,
        }
    ]
    assert offline_evaluation_service._scheduler_lock_token is None


def test_stop_offline_eval_scheduler_releases_owned_lock(monkeypatch):
    _reset_scheduler_state(monkeypatch)
    fake_redis = _FakeSchedulerRedis(acquire=True)
    lock_token = "lock-token"
    fake_redis.store[offline_evaluation_service._SCHEDULER_LOCK_KEY] = lock_token
    monkeypatch.setattr(offline_evaluation_service, "app_redis_client", fake_redis)
    fake_task = _FakeTask()
    monkeypatch.setattr(offline_evaluation_service, "_scheduler_task", fake_task)
    monkeypatch.setattr(offline_evaluation_service, "_scheduler_lock_token", lock_token)

    asyncio.run(offline_evaluation_service.stop_offline_eval_scheduler())

    assert fake_task.cancelled is True
    assert offline_evaluation_service._scheduler_task is None
    assert offline_evaluation_service._scheduler_lock_token is None
    assert offline_evaluation_service._SCHEDULER_LOCK_KEY not in fake_redis.store
