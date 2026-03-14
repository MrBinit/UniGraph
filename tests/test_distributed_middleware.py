from app.middlewares import backpressure, rate_limit


def _ip(*octets: int) -> str:
    return ".".join(str(octet) for octet in octets)


class _FakeRedisCounter:
    def __init__(self):
        self.counts = {}
        self.expire_calls = []

    def incr(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    def expire(self, key, ttl):
        self.expire_calls.append((key, ttl))


def test_redis_fixed_window_limiter_enforces_limit(monkeypatch):
    fake_redis = _FakeRedisCounter()
    monkeypatch.setattr(rate_limit, "app_redis_client", fake_redis)

    limiter = rate_limit._RedisFixedWindowLimiter(
        limit=2,
        window_seconds=60,
        key_prefix="app:ratelimit",
    )

    client_ip = _ip(1, 2, 3, 4)
    limiter_key = f"user:1|ip:{client_ip}|path:/api/v1/chat/stream"
    allowed_1, retry_1 = limiter.allow(limiter_key)
    allowed_2, retry_2 = limiter.allow(limiter_key)
    allowed_3, retry_3 = limiter.allow(limiter_key)

    assert allowed_1 is True and retry_1 == 0
    assert allowed_2 is True and retry_2 == 0
    assert allowed_3 is False and retry_3 >= 1
    assert fake_redis.expire_calls, "expected limiter to set expiry on first increment"


class _FakeRedisBackpressure:
    def __init__(self, responses):
        self.responses = list(responses)
        self.released = []

    def eval(self, *_args, **_kwargs):
        return self.responses.pop(0)

    def zrem(self, key, token):
        self.released.append((key, token))


def test_redis_backpressure_gate_acquire_and_release(monkeypatch):
    fake_redis = _FakeRedisBackpressure(responses=[[1, 0], [0, 4]])
    monkeypatch.setattr(backpressure, "app_redis_client", fake_redis)

    gate = backpressure._RedisBackpressureGate(
        key="app:backpressure:inflight",
        max_in_flight_requests=2,
        lease_seconds=45,
    )

    allowed_1, retry_1 = gate.acquire("token-1")
    allowed_2, retry_2 = gate.acquire("token-2")
    gate.release("token-1")

    assert allowed_1 is True and retry_1 >= 1
    assert allowed_2 is False and retry_2 == 4
    assert fake_redis.released == [("app:backpressure:inflight", "token-1")]
