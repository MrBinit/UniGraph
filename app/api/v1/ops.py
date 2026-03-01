from fastapi import APIRouter, Depends

from app.api.dependencies.security import authorize_admin_access, get_current_principal
from app.schemas.auth_schema import Principal
from app.schemas.ops_schema import OpsStatusResponse
from app.services.ops_status_service import get_ops_status

router = APIRouter()


@router.get("/ops/status", response_model=OpsStatusResponse)
async def ops_status(
    principal: Principal = Depends(get_current_principal),
):
    """Return the current operational status snapshot for admin users."""
    authorize_admin_access(principal)
    return get_ops_status()
