"""SQS-backed async chat queue with DynamoDB job status tracking.

This module supports the async chat flow:
1) API enqueues a chat request to SQS,
2) worker consumes and runs model inference,
3) worker persists final job status/result in DynamoDB.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError

from app.core.config import get_settings

settings = get_settings()

_JOB_STATUS_QUEUED = "queued"
_JOB_STATUS_PROCESSING = "processing"
_JOB_STATUS_COMPLETED = "completed"
_JOB_STATUS_FAILED = "failed"
_VALID_STATUSES = {
    _JOB_STATUS_QUEUED,
    _JOB_STATUS_PROCESSING,
    _JOB_STATUS_COMPLETED,
    _JOB_STATUS_FAILED,
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _utc_now().isoformat()


def _ttl_epoch_seconds(ttl_days: int) -> int:
    if ttl_days <= 0:
        return 0
    expires_at = _utc_now() + timedelta(days=ttl_days)
    return int(expires_at.timestamp())


def _region_name() -> str | None:
    region = (
        os.getenv("AWS_REGION", "").strip()
        or os.getenv("AWS_DEFAULT_REGION", "").strip()
        or os.getenv("AWS_SECRETS_MANAGER_REGION", "").strip()
    )
    return region or None


@lru_cache()
def _sqs_client():
    kwargs = {"region_name": _region_name()} if _region_name() else {}
    return boto3.client("sqs", **kwargs)


@lru_cache()
def _dynamodb_table():
    kwargs = {"region_name": _region_name()} if _region_name() else {}
    resource = boto3.resource("dynamodb", **kwargs)
    return resource.Table(settings.queue.llm_result_table.strip())


def _require_async_chat_config() -> None:
    if not settings.queue.llm_async_enabled:
        raise RuntimeError("Async chat queue is disabled.")
    if not settings.queue.llm_queue_url.strip():
        raise RuntimeError("LLM_QUEUE_URL is not configured.")
    if not settings.queue.llm_result_table.strip():
        raise RuntimeError("LLM_RESULT_TABLE is not configured.")


def _safe_message_group_id(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return "anonymous"
    return text[:128]


def _put_initial_job(job: dict) -> None:
    _dynamodb_table().put_item(
        Item=job,
        ConditionExpression="attribute_not_exists(job_id)",
    )


def _update_job(job_id: str, updates: dict) -> None:
    if not updates:
        return
    keys = []
    values = {}
    names = {}
    for idx, (key, value) in enumerate(updates.items(), start=1):
        name_key = f"#k{idx}"
        value_key = f":v{idx}"
        names[name_key] = key
        values[value_key] = value
        keys.append(f"{name_key}={value_key}")
    _dynamodb_table().update_item(
        Key={"job_id": job_id},
        UpdateExpression="SET " + ", ".join(keys),
        ConditionExpression="attribute_exists(job_id)",
        ExpressionAttributeNames=names,
        ExpressionAttributeValues=values,
    )


def enqueue_chat_job(*, user_id: str, prompt: str, session_id: str | None = None) -> dict:
    """Create async chat job state and enqueue it into SQS."""
    _require_async_chat_config()

    normalized_user = str(user_id).strip()
    normalized_prompt = str(prompt).strip()
    normalized_session = str(session_id or normalized_user).strip() or normalized_user
    job_id = uuid4().hex
    now_iso = _now_iso()
    ttl_days = settings.queue.llm_result_ttl_days

    record = {
        "job_id": job_id,
        "request_id": job_id,
        "user_id": normalized_user,
        "session_id": normalized_session,
        "prompt": normalized_prompt,
        "status": _JOB_STATUS_QUEUED,
        "answer": "",
        "error": "",
        "created_at": now_iso,
        "updated_at": now_iso,
        "started_at": "",
        "completed_at": "",
        "attempt_count": 0,
        "queue_message_id": "",
    }
    expires_at = _ttl_epoch_seconds(ttl_days)
    if expires_at > 0:
        record["expires_at"] = expires_at

    _put_initial_job(record)

    body = json.dumps(
        {
            "job_id": job_id,
            "user_id": normalized_user,
            "session_id": normalized_session,
            "prompt": normalized_prompt,
            "submitted_at": now_iso,
        },
        ensure_ascii=False,
    )
    send_kwargs = {
        "QueueUrl": settings.queue.llm_queue_url.strip(),
        "MessageBody": body,
        # Fair-queue tenant key on Standard queue; also valid for FIFO.
        "MessageGroupId": _safe_message_group_id(normalized_session),
    }
    try:
        try:
            response = _sqs_client().send_message(**send_kwargs)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code not in {"InvalidParameterValue", "UnsupportedOperation"}:
                raise
            # Fallback for accounts/regions where MessageGroupId on Standard may not be enabled.
            send_kwargs.pop("MessageGroupId", None)
            response = _sqs_client().send_message(**send_kwargs)
    except Exception as exc:
        mark_job_failed(job_id, f"Queue enqueue failed: {exc}")
        raise

    message_id = str(response.get("MessageId", ""))
    _update_job(
        job_id,
        {
            "queue_message_id": message_id,
            "updated_at": _now_iso(),
        },
    )
    return {
        "job_id": job_id,
        "status": _JOB_STATUS_QUEUED,
        "submitted_at": now_iso,
    }


def get_chat_job(job_id: str) -> dict | None:
    """Fetch one async chat job result from DynamoDB."""
    if not job_id.strip() or not settings.queue.llm_result_table.strip():
        return None
    response = _dynamodb_table().get_item(Key={"job_id": job_id.strip()})
    item = response.get("Item")
    if not isinstance(item, dict):
        return None
    status = str(item.get("status", "")).strip().lower()
    if status not in _VALID_STATUSES:
        item["status"] = _JOB_STATUS_FAILED
    return item


def mark_job_processing(job_id: str) -> None:
    now_iso = _now_iso()
    _dynamodb_table().update_item(
        Key={"job_id": job_id},
        UpdateExpression=(
            "SET #status=:status, started_at=if_not_exists(started_at,:started_at), "
            "updated_at=:updated_at ADD attempt_count :attempt_inc"
        ),
        ConditionExpression="attribute_exists(job_id)",
        ExpressionAttributeNames={"#status": "status"},
        ExpressionAttributeValues={
            ":status": _JOB_STATUS_PROCESSING,
            ":started_at": now_iso,
            ":updated_at": now_iso,
            ":attempt_inc": 1,
        },
    )


def mark_job_completed(job_id: str, answer: str) -> None:
    now_iso = _now_iso()
    _update_job(
        job_id,
        {
            "status": _JOB_STATUS_COMPLETED,
            "answer": str(answer),
            "error": "",
            "completed_at": now_iso,
            "updated_at": now_iso,
        },
    )


def mark_job_failed(job_id: str, error_message: str) -> None:
    _update_job(
        job_id,
        {
            "status": _JOB_STATUS_FAILED,
            "error": str(error_message)[:2000],
            "updated_at": _now_iso(),
        },
    )


def receive_llm_job_messages() -> list[dict]:
    """Poll SQS for async chat jobs using long polling."""
    _require_async_chat_config()
    response = _sqs_client().receive_message(
        QueueUrl=settings.queue.llm_queue_url.strip(),
        MaxNumberOfMessages=settings.queue.llm_max_messages_per_poll,
        WaitTimeSeconds=settings.queue.llm_receive_wait_seconds,
        VisibilityTimeout=settings.queue.llm_visibility_timeout_seconds,
        MessageAttributeNames=["All"],
        AttributeNames=["All"],
    )
    messages = response.get("Messages", [])
    return messages if isinstance(messages, list) else []


def delete_llm_job_message(receipt_handle: str) -> None:
    """Acknowledge one successfully processed message from SQS."""
    if not receipt_handle:
        return
    _sqs_client().delete_message(
        QueueUrl=settings.queue.llm_queue_url.strip(),
        ReceiptHandle=receipt_handle,
    )
