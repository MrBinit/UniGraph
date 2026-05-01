import asyncio
import json
import logging
from decimal import Decimal
from typing import Annotated, AsyncIterator
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from app.api.dependencies.security import authorize_user_access, get_current_principal
from app.core.config import get_settings
from app.schemas.auth_schema import Principal
from app.schemas.chat_schema import (
    AsyncChatStatusResponse,
    ChatHistoryClearResponse,
    ChatRequest,
)
from app.services.evaluation_service import clear_chat_traces
from app.services.llm_async_queue_service import enqueue_chat_job, get_chat_job
from app.services.memory_service import clear_user_chat_state

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


class _StreamClientError(RuntimeError):
    """Represent one client-safe stream error."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


def _sse_data(payload: dict) -> str:
    """Return one SSE data frame."""
    return f"data: {json.dumps(_json_safe(payload), ensure_ascii=False)}\n\n"


def _json_safe(value):
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _queued_payload(job: dict, job_id: str) -> dict:
    """Build one queue-ack payload for SSE clients."""
    return {
        "type": "queued",
        "job_id": job_id,
        "status": str(job.get("status", "queued")),
        "submitted_at": str(job.get("submitted_at", "")),
    }


def _record_status(record: dict) -> str:
    """Normalize one async job status string."""
    status = str(record.get("status", "processing")).strip().lower()
    return status or "processing"


def _record_trace_events(record: dict) -> list[dict]:
    events = record.get("trace_events", []) if isinstance(record, dict) else []
    if not isinstance(events, list):
        return []
    return [_json_safe(item) for item in events if isinstance(item, dict)]


async def _enqueue_stream_job(request: ChatRequest, session_id: str) -> dict:
    """Enqueue one async chat job and return its queued payload."""
    enqueue_kwargs = {
        "user_id": request.user_id,
        "prompt": request.prompt,
        "session_id": session_id,
        "mode": request.mode,
    }
    if request.debug:
        enqueue_kwargs["debug"] = True
    try:
        try:
            job = await asyncio.to_thread(enqueue_chat_job, **enqueue_kwargs)
        except TypeError as exc:
            if "debug" not in str(exc):
                raise
            legacy_kwargs = dict(enqueue_kwargs)
            legacy_kwargs.pop("debug", None)
            job = await asyncio.to_thread(enqueue_chat_job, **legacy_kwargs)
    except RuntimeError as exc:
        logger.warning(
            "Async chat enqueue unavailable for stream user_id=%s",
            request.user_id,
            exc_info=True,
        )
        raise _StreamClientError(_ENQUEUE_UNAVAILABLE_DETAIL) from exc
    except Exception as exc:
        logger.exception("Async chat enqueue failed for stream user_id=%s", request.user_id)
        raise _StreamClientError(_ENQUEUE_FAILED_DETAIL) from exc

    job_id = str(job.get("job_id", "")).strip()
    if not job_id:
        raise _StreamClientError(_ENQUEUE_FAILED_DETAIL)
    return _queued_payload(job, job_id)


async def _stream_job_events(job_id: str) -> AsyncIterator[str]:
    """Yield status/result SSE frames for one async chat job."""
    last_status = ""
    last_trace_count = 0
    while True:
        record = await asyncio.to_thread(get_chat_job, job_id)
        if not record:
            await asyncio.sleep(_STREAM_STATUS_POLL_SECONDS)
            continue

        record_status = _record_status(record)
        if record_status != last_status:
            yield _sse_data(
                {
                    "type": "status",
                    "job_id": job_id,
                    "status": record_status,
                }
            )
            last_status = record_status

        trace_events = _record_trace_events(record)
        if len(trace_events) > last_trace_count:
            for event in trace_events[last_trace_count:]:
                yield _sse_data(
                    {
                        "type": "trace",
                        "job_id": job_id,
                        "event": event,
                    }
                )
            last_trace_count = len(trace_events)

        if record_status == "completed":
            yield _sse_data({"type": "chunk", "text": str(record.get("answer", ""))})
            yield 'data: {"type":"done"}\n\n'
            return

        if record_status == "failed":
            public_error = _public_job_error(record_status, str(record.get("error", "")))
            yield _sse_data({"type": "error", "detail": public_error or _FAILED_JOB_DETAIL})
            return

        await asyncio.sleep(_STREAM_STATUS_POLL_SECONDS)


async def _chat_event_stream(request: ChatRequest, session_id: str) -> AsyncIterator[str]:
    """Yield full SSE stream lifecycle for one chat request."""
    try:
        queued = await _enqueue_stream_job(request, session_id)
        yield _sse_data(queued)
        async for event in _stream_job_events(str(queued["job_id"])):
            yield event
    except _StreamClientError as exc:
        yield _sse_data({"type": "error", "detail": exc.detail})
    except Exception:
        logger.exception("Chat stream failed for user_id=%s", request.user_id)
        yield _sse_data({"type": "error", "detail": _STREAM_FAILED_DETAIL})


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    principal: Annotated[Principal, Depends(get_current_principal)],
):
    """Stream async chat status and final response over Server-Sent Events."""
    authorize_user_access(principal, request.user_id)
    return StreamingResponse(
        _chat_event_stream(request, request.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/chat/{job_id}",
    response_model=AsyncChatStatusResponse,
    response_model_exclude_none=True,
)
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
        trace_events=_record_trace_events(record),
    )


@router.delete("/chat/history", response_model=ChatHistoryClearResponse)
async def clear_chat_history(
    user_id: Annotated[str, Query(min_length=3, max_length=128)],
    principal: Annotated[Principal, Depends(get_current_principal)],
    session_id: Annotated[
        str | None,
        Query(
            min_length=3,
            max_length=128,
            pattern=r"^[A-Za-z0-9_.:@\-]+$",
            description=(
                "Optional session id. If provided, clear only that session state; "
                "if omitted, clear all conversation state for the user."
            ),
        ),
    ] = None,
):
    """Delete prior conversation state for one user (memory, cache, and eval traces)."""
    authorize_user_access(principal, user_id)
    memory_result = clear_user_chat_state(user_id, session_id=session_id)
    trace_result = clear_chat_traces(user_id)
    return ChatHistoryClearResponse(
        user_id=user_id,
        session_id=session_id,
        memory_keys_deleted=int(memory_result.get("memory_keys_deleted", 0)),
        legacy_memory_keys_deleted=int(memory_result.get("legacy_memory_keys_deleted", 0)),
        cache_keys_deleted=int(memory_result.get("cache_keys_deleted", 0)),
        trace_keys_deleted=int(trace_result.get("trace_keys_deleted", 0)),
        trace_index_deleted=int(trace_result.get("index_key_deleted", 0)),
    )
