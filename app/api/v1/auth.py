import hmac
import json
import logging
import os
from fastapi import APIRouter, HTTPException, status
from app.core.config import get_settings
from app.core.security import create_access_token
from app.schemas.auth_schema import PasswordLoginRequest, PasswordLoginResponse

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

_LOGIN_USERS_ENV = "SECURITY_LOGIN_USERS_JSON"
_DEFAULT_LOGIN_USERS: list[dict] = []
_INVALID_CREDENTIALS_DETAIL = "Invalid username or password."


def _normalize_roles(raw_roles) -> list[str]:
    if not isinstance(raw_roles, list):
        return ["user"]
    normalized = [str(role).strip() for role in raw_roles if str(role).strip()]
    return normalized or ["user"]


def _normalize_login_users(raw_users) -> list[dict]:
    normalized: list[dict] = []
    if not isinstance(raw_users, list):
        return normalized

    for raw_user in raw_users:
        if not isinstance(raw_user, dict):
            continue
        username = str(raw_user.get("username", "")).strip()
        password = str(raw_user.get("password", "")).strip()
        user_id = str(raw_user.get("user_id", username)).strip()
        if not username or not password or not user_id:
            continue
        normalized.append(
            {
                "username": username,
                "password": password,
                "user_id": user_id,
                "roles": _normalize_roles(raw_user.get("roles")),
            }
        )
    return normalized


def _login_users() -> list[dict]:
    raw = os.getenv(_LOGIN_USERS_ENV, "").strip()
    if not raw:
        return list(_DEFAULT_LOGIN_USERS)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON in %s; no login users loaded.", _LOGIN_USERS_ENV)
        return list(_DEFAULT_LOGIN_USERS)

    users = _normalize_login_users(parsed)
    if users:
        return users

    logger.warning("%s did not contain valid users; no login users loaded.", _LOGIN_USERS_ENV)
    return list(_DEFAULT_LOGIN_USERS)


def _find_login_user(username: str) -> dict | None:
    target = str(username).strip()
    for user in _login_users():
        if str(user.get("username", "")).strip() == target:
            return user
    return None


@router.post("/auth/login", response_model=PasswordLoginResponse)
async def password_login(request: PasswordLoginRequest):
    """Authenticate a username/password pair and return a JWT bearer token."""
    user = _find_login_user(request.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_INVALID_CREDENTIALS_DETAIL,
        )

    configured_password = str(user.get("password", "")).strip()
    if not configured_password or not hmac.compare_digest(request.password, configured_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_INVALID_CREDENTIALS_DETAIL,
        )

    roles = _normalize_roles(user.get("roles"))
    user_id = str(user.get("user_id", "")).strip()
    token = create_access_token(user_id=user_id, roles=roles)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user_id,
        "roles": roles,
        "expires_in_seconds": int(settings.security.jwt_exp_minutes) * 60,
    }
