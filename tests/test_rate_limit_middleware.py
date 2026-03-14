from types import SimpleNamespace

from app.core.security import create_access_token
from app.middlewares.rate_limit import RateLimitMiddleware


def _ip(*octets: int) -> str:
    return ".".join(str(octet) for octet in octets)


def _request(
    path: str,
    host: str,
    auth_header: str | None = None,
    user_id: str | None = None,
    x_forwarded_for: str | None = None,
):
    headers = {}
    if auth_header:
        headers["authorization"] = auth_header
    if x_forwarded_for:
        headers["x-forwarded-for"] = x_forwarded_for
    state = SimpleNamespace()
    if user_id is not None:
        state.user_id = user_id
    return SimpleNamespace(
        headers=headers,
        client=SimpleNamespace(host=host),
        url=SimpleNamespace(path=path),
        state=state,
    )


def test_rate_limit_key_uses_user_id_and_ip():
    middleware = RateLimitMiddleware(
        app=lambda *_args, **_kwargs: None, limit=10, window_seconds=60
    )

    token_1 = create_access_token(user_id="user-1", roles=["user"])
    token_2 = create_access_token(user_id="user-2", roles=["user"])

    client_ip = _ip(1, 2, 3, 4)
    req_1 = _request("/api/v1/chat/stream", client_ip, auth_header=f"Bearer {token_1}")
    req_2 = _request("/api/v1/chat/stream", client_ip, auth_header=f"Bearer {token_2}")

    key_1 = middleware._rate_limit_key(req_1)
    key_2 = middleware._rate_limit_key(req_2)

    assert "user:user-1" in key_1
    assert f"ip:{client_ip}" in key_1
    assert key_1 != key_2


def test_rate_limit_key_falls_back_to_anonymous_when_token_invalid():
    middleware = RateLimitMiddleware(
        app=lambda *_args, **_kwargs: None, limit=10, window_seconds=60
    )
    client_ip = _ip(5, 6, 7, 8)
    req = _request("/api/v1/chat/stream", client_ip, auth_header="Bearer invalid-token")
    key = middleware._rate_limit_key(req)
    assert "user:anonymous" in key
    assert f"ip:{client_ip}" in key


def test_rate_limit_ignores_x_forwarded_for_when_proxy_not_trusted():
    middleware = RateLimitMiddleware(
        app=lambda *_args, **_kwargs: None, limit=10, window_seconds=60
    )
    client_ip = _ip(10, 0, 0, 5)
    forwarded_ip = _ip(203, 0, 113, 9)
    req = _request(
        "/api/v1/chat/stream",
        client_ip,
        x_forwarded_for=forwarded_ip,
    )
    key = middleware._rate_limit_key(req)
    assert f"ip:{client_ip}" in key
    assert f"ip:{forwarded_ip}" not in key


def test_rate_limit_uses_x_forwarded_for_when_proxy_trusted():
    trusted_proxy_cidr = f"{_ip(10, 0, 0, 0)}/8"
    client_ip = _ip(10, 0, 0, 5)
    forwarded_ip = _ip(203, 0, 113, 9)
    middleware = RateLimitMiddleware(
        app=lambda *_args, **_kwargs: None,
        limit=10,
        window_seconds=60,
        trusted_proxy_cidrs=[trusted_proxy_cidr],
    )
    req = _request(
        "/api/v1/chat/stream",
        client_ip,
        x_forwarded_for=f"{forwarded_ip}, {client_ip}",
    )
    key = middleware._rate_limit_key(req)
    assert f"ip:{forwarded_ip}" in key
