import asyncio
from types import SimpleNamespace

import pytest
from fastapi.responses import JSONResponse

from app.middlewares import backpressure
from app.middlewares.backpressure import BackpressureMiddleware


def _request(path: str = "/api/v1/chat/stream"):
    return SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path=path),
    )


@pytest.mark.asyncio
async def test_local_backpressure_rejects_when_slot_is_full():
    middleware = BackpressureMiddleware(
        app=lambda *_args, **_kwargs: None,
        max_in_flight_requests=1,
        use_redis=False,
    )
    first_entered = asyncio.Event()
    release_first = asyncio.Event()

    async def slow_call_next(_request):
        first_entered.set()
        await release_first.wait()
        return "ok"

    first_request_task = asyncio.create_task(middleware.dispatch(_request(), slow_call_next))
    await first_entered.wait()

    rejected = await middleware.dispatch(_request(), slow_call_next)
    assert isinstance(rejected, JSONResponse)
    assert rejected.status_code == 503

    release_first.set()
    first_result = await first_request_task
    assert first_result == "ok"
    assert middleware._local_in_flight == 0


class _FakeRedisBackpressure:
    def __init__(self):
        self.released = []

    def eval(self, *_args, **_kwargs):
        return [1, 0]

    def zrem(self, key, token):
        self.released.append((key, token))


@pytest.mark.asyncio
async def test_distributed_token_released_when_local_rejects(monkeypatch):
    fake_redis = _FakeRedisBackpressure()
    monkeypatch.setattr(backpressure, "app_redis_client", fake_redis)

    middleware = BackpressureMiddleware(
        app=lambda *_args, **_kwargs: None,
        max_in_flight_requests=1,
        use_redis=True,
        redis_key="backpressure:inflight",
        distributed_lease_seconds=45,
    )
    first_entered = asyncio.Event()
    release_first = asyncio.Event()

    async def slow_call_next(_request):
        first_entered.set()
        await release_first.wait()
        return "ok"

    first_request_task = asyncio.create_task(middleware.dispatch(_request(), slow_call_next))
    await first_entered.wait()

    rejected = await middleware.dispatch(_request(), slow_call_next)
    assert isinstance(rejected, JSONResponse)
    assert rejected.status_code == 503

    release_first.set()
    await first_request_task

    assert len(fake_redis.released) == 2
