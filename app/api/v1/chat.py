from fastapi import APIRouter, Depends

from app.api.dependencies.security import authorize_user_access, get_current_principal
from app.schemas.auth_schema import Principal
from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.services.llm_service import generate_response

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    principal: Principal = Depends(get_current_principal),
):
    """Handle the chat endpoint for an authenticated user-scoped request."""
    authorize_user_access(principal, request.user_id)
    result = await generate_response(request.user_id, request.prompt)
    return ChatResponse(response=result)
