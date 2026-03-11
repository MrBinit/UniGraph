import asyncio
import json
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from app.api.dependencies.security import authorize_user_access, get_current_principal
from app.schemas.auth_schema import Principal
from app.schemas.chat_schema import (
    AsyncChatEnqueueResponse,
    AsyncChatStatusResponse,
    ChatRequest,
)
from app.services.llm_async_queue_service import enqueue_chat_job, get_chat_job
from app.services.llm_service import generate_response_stream

router = APIRouter()


@router.post(
    "/chat",
    response_model=AsyncChatEnqueueResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def chat(
    request: ChatRequest,
    principal: Principal = Depends(get_current_principal),
):
    """Enqueue an async chat job and return a job id immediately."""
    authorize_user_access(principal, request.user_id)
    try:
        job = await asyncio.to_thread(
            enqueue_chat_job,
            user_id=request.user_id,
            prompt=request.prompt,
            session_id=request.user_id,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue async chat job: {exc}",
        ) from exc
    return AsyncChatEnqueueResponse(**job)


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    principal: Principal = Depends(get_current_principal),
):
    """Stream one chat completion over Server-Sent Events."""
    authorize_user_access(principal, request.user_id)

    async def _event_stream():
        try:
            async for partial in generate_response_stream(request.user_id, request.prompt):
                chunk = {"type": "chunk", "text": str(partial)}
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield 'data: {"type":"done"}\n\n'
        except Exception as exc:
            payload = {"type": "error", "detail": str(exc)}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/chat/{job_id}", response_model=AsyncChatStatusResponse)
async def chat_status(
    job_id: str,
    principal: Principal = Depends(get_current_principal),
):
    """Return status/result for one async chat job."""
    record = await asyncio.to_thread(get_chat_job, job_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Async chat job not found.",
        )

    record_user_id = str(record.get("user_id", "")).strip()
    authorize_user_access(principal, record_user_id)
    return AsyncChatStatusResponse(
        job_id=str(record.get("job_id", job_id)),
        user_id=record_user_id,
        session_id=str(record.get("session_id", record_user_id)),
        status=str(record.get("status", "failed")),
        submitted_at=str(record.get("created_at", "")),
        started_at=str(record.get("started_at", "")),
        completed_at=str(record.get("completed_at", "")),
        response=str(record.get("answer", "")),
        error=str(record.get("error", "")),
    )
