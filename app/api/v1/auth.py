import json
import hmac
import logging
import os
from functools import lru_cache

from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.core.passwords import verify_password
from app.core.security import create_access_token
from app.schemas.auth_schema import PasswordLoginRequest, PasswordLoginResponse

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

_INVALID_CREDENTIALS_DETAIL = "Invalid username or password."


def _normalize_roles(raw_roles) -> list[str]:
    if not isinstance(raw_roles, list):
        return ["user"]
    normalized = [str(role).strip() for role in raw_roles if str(role).strip()]
    return normalized or ["user"]


def _normalize_user_id(user: dict) -> str:
    user_id = str(user.get("user_id", "")).strip()
    if user_id:
        return user_id
    return str(user.get("username", "")).strip()


def _normalize_user_roles(user: dict) -> list[str]:
    return _normalize_roles(user.get("roles"))


def _normalize_auth_user(user: dict) -> dict | None:
    if not isinstance(user, dict):
        return None
    username = str(user.get("username", "")).strip()
    if not username:
        return None
    normalized = dict(user)
    normalized["username"] = username
    return normalized


@lru_cache(maxsize=1)
def _configured_auth_users() -> list[dict]:
    raw = os.getenv("SECURITY_LOGIN_USERS_JSON", "").strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except Exception as exc:
        logger.warning("SECURITY_LOGIN_USERS_JSON is invalid JSON. %s", exc)
        return []
    if not isinstance(payload, list):
        logger.warning("SECURITY_LOGIN_USERS_JSON must be a JSON array of users.")
        return []
    users: list[dict] = []
    for item in payload:
        normalized = _normalize_auth_user(item)
        if normalized is not None:
            users.append(normalized)
    return users


def _password_matches(user: dict, plaintext_password: str) -> bool:
    password_hash = str(user.get("password_hash", "")).strip()
    if password_hash and verify_password(plaintext_password, password_hash):
        return True
    # Backward compatibility for deployments storing plaintext passwords in SECURITY_LOGIN_USERS_JSON.
    plaintext = str(user.get("password", "")).strip()
    if plaintext:
        return hmac.compare_digest(str(plaintext_password), plaintext)
    return False


def _fetch_auth_user(username: str) -> dict | None:
    target = str(username).strip()
    if not target:
        return None
    target_key = target.lower()
    for user in _configured_auth_users():
        username_value = str(user.get("username", "")).strip().lower()
        if username_value == target_key:
            return user
    return None


@router.post("/auth/login", response_model=PasswordLoginResponse)
async def password_login(request: PasswordLoginRequest):
    """Authenticate a username/password pair and return a JWT bearer token."""
    user = _fetch_auth_user(request.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_INVALID_CREDENTIALS_DETAIL,
        )

    if not bool(user.get("is_active", True)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_INVALID_CREDENTIALS_DETAIL,
        )

    if not _password_matches(user, request.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_INVALID_CREDENTIALS_DETAIL,
        )

    roles = _normalize_user_roles(user)
    user_id = _normalize_user_id(user)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_INVALID_CREDENTIALS_DETAIL,
        )
    token = create_access_token(user_id=user_id, roles=roles)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user_id,
        "roles": roles,
        "expires_in_seconds": int(settings.security.jwt_exp_minutes) * 60,
    }
