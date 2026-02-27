import os
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from app.core.config import get_settings

settings = get_settings()


def _secret_key() -> str:
    return os.getenv("JWT_SECRET", settings.security.jwt_secret)


def create_access_token(
    *,
    user_id: str,
    roles: list[str] | None = None,
    expires_minutes: int | None = None,
) -> str:
    now = datetime.now(timezone.utc)
    exp_minutes = expires_minutes or settings.security.jwt_exp_minutes
    payload = {
        "sub": user_id,
        "roles": roles or ["user"],
        "iss": settings.security.jwt_issuer,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=exp_minutes)).timestamp()),
    }
    return jwt.encode(payload, _secret_key(), algorithm=settings.security.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    return jwt.decode(
        token,
        _secret_key(),
        algorithms=[settings.security.jwt_algorithm],
        issuer=settings.security.jwt_issuer,
        options={"verify_aud": False},
    )


def is_jwt_error(exc: Exception) -> bool:
    return isinstance(exc, JWTError)
