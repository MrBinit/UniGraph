"""Shared SQS helpers for background metrics and evaluation event queues."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from functools import lru_cache

import boto3
from botocore.exceptions import ClientError

from app.core.config import get_settings

settings = get_settings()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _safe_message_group_id(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return "global"
    return text[:128]


def _send_json(queue_url: str, payload: dict, message_group_id: str = "global") -> str:
    body = json.dumps(payload, ensure_ascii=False)
    send_kwargs = {
        "QueueUrl": queue_url.strip(),
        "MessageBody": body,
        # Used for fair queues on Standard and required for FIFO.
        "MessageGroupId": _safe_message_group_id(message_group_id),
    }
    try:
        try:
            response = _sqs_client().send_message(**send_kwargs)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code not in {"InvalidParameterValue", "UnsupportedOperation"}:
                raise
            send_kwargs.pop("MessageGroupId", None)
            response = _sqs_client().send_message(**send_kwargs)
    except Exception:
        raise
    return str(response.get("MessageId", ""))


def enqueue_metrics_aggregation_event(request_id: str) -> str:
    """Publish one metrics-aggregation event keyed by request id."""
    queue_url = settings.queue.metrics_aggregation_queue_url.strip()
    if not settings.queue.metrics_aggregation_queue_enabled or not queue_url:
        return ""
    payload = {
        "type": "metrics_aggregation",
        "request_id": str(request_id).strip(),
        "enqueued_at": _utc_now_iso(),
    }
    return _send_json(queue_url, payload, message_group_id="metrics")


def enqueue_evaluation_event(request_id: str, session_id: str = "") -> str:
    """Publish one evaluation event keyed by request id."""
    queue_url = settings.queue.evaluation_queue_url.strip()
    if not settings.queue.evaluation_queue_enabled or not queue_url:
        return ""
    payload = {
        "type": "evaluation",
        "request_id": str(request_id).strip(),
        "session_id": str(session_id).strip(),
        "enqueued_at": _utc_now_iso(),
    }
    group_id = str(session_id).strip() or "evaluation"
    return _send_json(queue_url, payload, message_group_id=group_id)


def receive_queue_messages(
    *,
    queue_url: str,
    max_messages: int,
    wait_seconds: int,
    visibility_timeout_seconds: int,
) -> list[dict]:
    """Receive SQS messages with long polling for one queue."""
    if not queue_url.strip():
        return []
    response = _sqs_client().receive_message(
        QueueUrl=queue_url.strip(),
        MaxNumberOfMessages=max_messages,
        WaitTimeSeconds=wait_seconds,
        VisibilityTimeout=visibility_timeout_seconds,
        MessageAttributeNames=["All"],
        AttributeNames=["All"],
    )
    messages = response.get("Messages", [])
    return messages if isinstance(messages, list) else []


def delete_queue_message(queue_url: str, receipt_handle: str) -> None:
    """Delete one SQS message by receipt handle."""
    if not queue_url.strip() or not receipt_handle:
        return
    _sqs_client().delete_message(
        QueueUrl=queue_url.strip(),
        ReceiptHandle=receipt_handle,
    )


def parse_message_json(message: dict) -> dict:
    """Decode message body JSON into a dict payload."""
    raw_body = str(message.get("Body", ""))
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
