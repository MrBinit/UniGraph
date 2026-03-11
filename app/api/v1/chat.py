import asyncio
import json
import logging
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
logger = logging.getLogger(__name__)

_ENQUEUE_UNAVAILABLE_DETAIL = "Async chat service is temporarily unavailable."
_ENQUEUE_FAILED_DETAIL = "Failed to enqueue async chat job."
_STREAM_FAILED_DETAIL = "Failed to stream chat response."
_FAILED_JOB_DETAIL = "Async chat job failed."


def _public_job_error(status: str, error: str) -> str:
    """Return a client-safe error message for async job status responses."""
    if status.strip().lower() != "failed":
        return ""
    if not error.strip():
        return ""
    return _FAILED_JOB_DETAIL


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
    session_id = request.session_id
    try:
        job = await asyncio.to_thread(
            enqueue_chat_job,
            user_id=request.user_id,
            prompt=request.prompt,
            session_id=session_id,
        )
    except RuntimeError as exc:
        logger.warning(
            "Async chat enqueue unavailable for user_id=%s", request.user_id, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_ENQUEUE_UNAVAILABLE_DETAIL,
        ) from exc
    except Exception as exc:
        logger.exception("Async chat enqueue failed for user_id=%s", request.user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_ENQUEUE_FAILED_DETAIL,
        ) from exc
    return AsyncChatEnqueueResponse(**job)


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    principal: Principal = Depends(get_current_principal),
):
    """Stream one chat completion over Server-Sent Events."""
    authorize_user_access(principal, request.user_id)
    session_id = request.session_id

    async def _event_stream():
        try:
            async for partial in generate_response_stream(
                request.user_id,
                request.prompt,
                session_id=session_id,
            ):
                chunk = {"type": "chunk", "text": str(partial)}
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield 'data: {"type":"done"}\n\n'
        except Exception:
            logger.exception("Chat stream failed for user_id=%s", request.user_id)
            payload = {"type": "error", "detail": _STREAM_FAILED_DETAIL}
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
    record_status = str(record.get("status", "failed"))
    public_error = _public_job_error(record_status, str(record.get("error", "")))
    return AsyncChatStatusResponse(
        job_id=str(record.get("job_id", job_id)),
        user_id=record_user_id,
        session_id=str(record.get("session_id", record_user_id)),
        status=record_status,
        submitted_at=str(record.get("created_at", "")),
        started_at=str(record.get("started_at", "")),
        completed_at=str(record.get("completed_at", "")),
        response=str(record.get("answer", "")),
        error=public_error,
    )
