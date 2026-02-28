from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings
from app.core.security import decode_access_token
from app.schemas.auth_schema import Principal

settings = get_settings()
http_bearer = HTTPBearer(auto_error=False)


def _roles_from_claim(claim) -> list[str]:
    if isinstance(claim, list):
        return [str(role) for role in claim if isinstance(role, str)]
    if isinstance(claim, str):
        return [claim]
    return []


async def get_current_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(http_bearer),
) -> Principal:
    if not settings.security.auth_enabled:
        principal = Principal(user_id="anonymous", roles=["admin"])
        request.state.user_id = principal.user_id
        request.state.roles = principal.roles
        return principal

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid bearer token.",
        )

    try:
        payload = decode_access_token(credentials.credentials)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )

    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token subject is invalid.",
        )

    principal = Principal(user_id=user_id, roles=_roles_from_claim(payload.get("roles")))
    request.state.user_id = principal.user_id
    request.state.roles = principal.roles
    return principal


def authorize_user_access(principal: Principal, target_user_id: str):
    if principal.user_id == target_user_id:
        return

    if set(principal.roles).intersection(set(settings.security.admin_roles)):
        return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="You are not allowed to access this user's resources.",
    )


def authorize_admin_access(principal: Principal):
    if set(principal.roles).intersection(set(settings.security.admin_roles)):
        return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Admin access is required for this resource.",
    )
