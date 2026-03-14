import asyncio
import json
import logging
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from app.api.dependencies.security import authorize_user_access, get_current_principal
from app.core.config import get_settings
from app.schemas.auth_schema import Principal
from app.schemas.chat_schema import (
    AsyncChatStatusResponse,
    ChatRequest,
)
from app.services.llm_async_queue_service import enqueue_chat_job, get_chat_job

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

_ENQUEUE_UNAVAILABLE_DETAIL = "Async chat service is temporarily unavailable."
_ENQUEUE_FAILED_DETAIL = "Failed to enqueue async chat job."
_STREAM_FAILED_DETAIL = "Failed to stream chat response."
_FAILED_JOB_DETAIL = "Async chat job failed."
_STREAM_STATUS_POLL_SECONDS = max(0.1, float(settings.queue.llm_poll_sleep_seconds))


def _public_job_error(status: str, error: str) -> str:
    """Return a client-safe error message for async job status responses."""
    if status.strip().lower() != "failed":
        return ""
    if not error.strip():
        return ""
    return _FAILED_JOB_DETAIL


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    principal: Annotated[Principal, Depends(get_current_principal)],
):
    """Stream async chat status and final response over Server-Sent Events."""
    authorize_user_access(principal, request.user_id)
    session_id = request.session_id

    async def _event_stream():
        try:
            try:
                job = await asyncio.to_thread(
                    enqueue_chat_job,
                    user_id=request.user_id,
                    prompt=request.prompt,
                    session_id=session_id,
                )
            except RuntimeError:
                logger.warning(
                    "Async chat enqueue unavailable for stream user_id=%s",
                    request.user_id,
                    exc_info=True,
                )
                payload = {"type": "error", "detail": _ENQUEUE_UNAVAILABLE_DETAIL}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                return
            except Exception:
                logger.exception(
                    "Async chat enqueue failed for stream user_id=%s", request.user_id
                )
                payload = {"type": "error", "detail": _ENQUEUE_FAILED_DETAIL}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                return

            job_id = str(job.get("job_id", "")).strip()
            if not job_id:
                payload = {"type": "error", "detail": _ENQUEUE_FAILED_DETAIL}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                return

            queued = {
                "type": "queued",
                "job_id": job_id,
                "status": str(job.get("status", "queued")),
                "submitted_at": str(job.get("submitted_at", "")),
            }
            yield f"data: {json.dumps(queued, ensure_ascii=False)}\n\n"

            last_status = ""
            while True:
                record = await asyncio.to_thread(get_chat_job, job_id)
                if not record:
                    await asyncio.sleep(_STREAM_STATUS_POLL_SECONDS)
                    continue

                record_status = str(record.get("status", "processing")).strip().lower()
                if not record_status:
                    record_status = "processing"

                if record_status != last_status:
                    status_payload = {
                        "type": "status",
                        "job_id": job_id,
                        "status": record_status,
                    }
                    yield f"data: {json.dumps(status_payload, ensure_ascii=False)}\n\n"
                    last_status = record_status

                if record_status == "completed":
                    chunk = {"type": "chunk", "text": str(record.get("answer", ""))}
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    yield 'data: {"type":"done"}\n\n'
                    return

                if record_status == "failed":
                    public_error = _public_job_error(record_status, str(record.get("error", "")))
                    payload = {"type": "error", "detail": public_error or _FAILED_JOB_DETAIL}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    return

                await asyncio.sleep(_STREAM_STATUS_POLL_SECONDS)
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
    principal: Annotated[Principal, Depends(get_current_principal)],
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
