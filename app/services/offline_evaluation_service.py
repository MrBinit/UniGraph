import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
import boto3
from boto3.dynamodb.types import TypeDeserializer
from app.core.config import get_settings
from app.infra.redis_client import app_redis_client, app_scoped_key
from app.scripts.eval_daily_report import _build_report, _load_eval_rows
from app.scripts.eval_dynamodb_worker import run as run_eval_worker
from redis.exceptions import RedisError

settings = get_settings()
logger = logging.getLogger(__name__)
_deserializer = TypeDeserializer()
_scheduler_task: asyncio.Task | None = None
_scheduler_lock_token: str | None = None
_SCHEDULER_LOCK_KEY = app_scoped_key("evaluation:offline:scheduler:leader")
_SCHEDULER_LOCK_TTL_SECONDS = 120
_SCHEDULER_POLL_SECONDS = 30
_REFRESH_SCHEDULER_LOCK_SCRIPT = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
  return redis.call("EXPIRE", KEYS[1], tonumber(ARGV[2]))
end
return 0
"""
_RELEASE_SCHEDULER_LOCK_SCRIPT = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
  return redis.call("DEL", KEYS[1])
end
return 0
"""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _region_name() -> str | None:
    return (
        os.getenv("AWS_REGION", "").strip()
        or os.getenv("AWS_DEFAULT_REGION", "").strip()
        or os.getenv("AWS_SECRETS_MANAGER_REGION", "").strip()
        or None
    )


def _dynamodb_client():
    kwargs = {"region_name": _region_name()} if _region_name() else {}
    return boto3.client("dynamodb", **kwargs)


def _acquire_scheduler_lock() -> str | None:
    token = uuid.uuid4().hex
    try:
        acquired = bool(
            app_redis_client.set(
                _SCHEDULER_LOCK_KEY,
                token,
                ex=_SCHEDULER_LOCK_TTL_SECONDS,
                nx=True,
            )
        )
    except RedisError as exc:
        logger.warning("OfflineEvalSchedulerLockAcquireFailed | error=%s", exc)
        return None

    if not acquired:
        return None
    return token


def _refresh_scheduler_lock(token: str) -> bool:
    if not token:
        return False
    try:
        result = app_redis_client.eval(
            _REFRESH_SCHEDULER_LOCK_SCRIPT,
            1,
            _SCHEDULER_LOCK_KEY,
            token,
            _SCHEDULER_LOCK_TTL_SECONDS,
        )
    except RedisError as exc:
        logger.warning("OfflineEvalSchedulerLockRefreshFailed | error=%s", exc)
        return False
    return bool(result and int(result) == 1)


def _release_scheduler_lock(token: str):
    if not token:
        return
    try:
        app_redis_client.eval(
            _RELEASE_SCHEDULER_LOCK_SCRIPT,
            1,
            _SCHEDULER_LOCK_KEY,
            token,
        )
    except RedisError as exc:
        logger.warning("OfflineEvalSchedulerLockReleaseFailed | error=%s", exc)


def _deserialize(item: dict) -> dict:
    return {key: _deserializer.deserialize(value) for key, value in item.items()}


def _latest_timestamp_from_table(
    table_name: str,
    *,
    index_name: str,
    status_attr: str,
    status_value: str,
) -> datetime | None:
    if not table_name or not index_name or not status_attr or not status_value:
        return None

    ddb = _dynamodb_client()
    response = ddb.query(
        TableName=table_name,
        IndexName=index_name,
        KeyConditionExpression="#status = :status_value",
        ExpressionAttributeNames={
            "#status": status_attr,
            "#ts": "timestamp",
        },
        ExpressionAttributeValues={":status_value": {"S": status_value}},
        ProjectionExpression="#ts",
        Limit=1,
        ScanIndexForward=False,
    )
    items = response.get("Items", [])
    if not items:
        return None
    row = _deserialize(items[0])
    return _parse_iso(str(row.get("timestamp", "")))


def _new_requests_pending() -> bool:
    requests_table = settings.app.metrics_dynamodb_requests_table.strip()
    index_name = settings.evaluation.request_status_index_name.strip()
    status_attr = settings.evaluation.request_status_attribute.strip() or "eval_status"
    pending = settings.evaluation.request_pending_value.strip() or "pending"
    if not requests_table or not index_name:
        return False
    ddb = _dynamodb_client()
    response = ddb.query(
        TableName=requests_table,
        IndexName=index_name,
        KeyConditionExpression="#status = :status_value",
        ExpressionAttributeNames={"#status": status_attr},
        ExpressionAttributeValues={":status_value": {"S": pending}},
        ProjectionExpression="request_id",
        Limit=1,
    )
    return bool(response.get("Items"))


def get_offline_eval_status() -> dict:
    """Return readiness and scheduling status for offline DynamoDB evaluations."""
    if not settings.evaluation.enabled:
        return {
            "enabled": False,
            "schedule_enabled": settings.evaluation.schedule_enabled,
            "interval_hours": settings.evaluation.schedule_interval_hours,
            "has_new_requests": False,
            "due_by_interval": False,
            "should_auto_run": False,
            "last_request_timestamp": "",
            "last_evaluated_timestamp": "",
            "reason": "evaluation disabled",
        }

    requests_table = settings.app.metrics_dynamodb_requests_table.strip()
    eval_table = settings.evaluation.dynamodb_table.strip()
    if not requests_table or not eval_table:
        return {
            "enabled": True,
            "schedule_enabled": settings.evaluation.schedule_enabled,
            "interval_hours": settings.evaluation.schedule_interval_hours,
            "has_new_requests": False,
            "due_by_interval": False,
            "should_auto_run": False,
            "last_request_timestamp": "",
            "last_evaluated_timestamp": "",
            "reason": "missing DynamoDB table configuration",
        }

    try:
        request_index = settings.evaluation.request_status_index_name.strip()
        request_status_attr = settings.evaluation.request_status_attribute.strip() or "eval_status"
        eval_index = settings.evaluation.eval_status_index_name.strip()
        eval_status_attr = settings.evaluation.eval_status_attribute.strip() or "status"

        pending_requests_ts = _latest_timestamp_from_table(
            requests_table,
            index_name=request_index,
            status_attr=request_status_attr,
            status_value=settings.evaluation.request_pending_value,
        )
        completed_requests_ts = _latest_timestamp_from_table(
            requests_table,
            index_name=request_index,
            status_attr=request_status_attr,
            status_value=settings.evaluation.request_completed_value,
        )
        last_eval_ts = _latest_timestamp_from_table(
            eval_table,
            index_name=eval_index,
            status_attr=eval_status_attr,
            status_value=settings.evaluation.eval_completed_value,
        )
        has_new_requests = _new_requests_pending()
    except Exception as exc:
        return {
            "enabled": True,
            "schedule_enabled": settings.evaluation.schedule_enabled,
            "interval_hours": settings.evaluation.schedule_interval_hours,
            "has_new_requests": False,
            "due_by_interval": False,
            "should_auto_run": False,
            "last_request_timestamp": "",
            "last_evaluated_timestamp": "",
            "reason": f"evaluation status query failed: {exc}",
        }

    latest_candidates = [
        ts for ts in [pending_requests_ts, completed_requests_ts] if ts is not None
    ]
    last_request_ts = max(latest_candidates) if latest_candidates else None

    now = _utc_now()
    interval_seconds = settings.evaluation.schedule_interval_hours * 3600
    if last_eval_ts is None:
        due = True
    else:
        due = (now - last_eval_ts).total_seconds() >= interval_seconds

    should_auto_run = (
        settings.evaluation.schedule_enabled
        and due
        and has_new_requests
        and settings.evaluation.enabled
    )
    return {
        "enabled": True,
        "schedule_enabled": settings.evaluation.schedule_enabled,
        "interval_hours": settings.evaluation.schedule_interval_hours,
        "has_new_requests": has_new_requests,
        "due_by_interval": due,
        "should_auto_run": should_auto_run,
        "last_request_timestamp": last_request_ts.isoformat() if last_request_ts else "",
        "last_evaluated_timestamp": last_eval_ts.isoformat() if last_eval_ts else "",
        "reason": "ok",
    }


async def run_offline_eval(limit: int | None = None, force: bool = False) -> dict:
    """Run offline evaluation now or skip based on schedule/new-data gates."""
    status = get_offline_eval_status()
    if not settings.evaluation.enabled:
        return {
            "ran": False,
            "reason": "evaluation disabled",
            "result": {"evaluated": 0, "skipped": 0},
        }

    if not force and not status.get("should_auto_run", False):
        if not status.get("has_new_requests", False):
            reason = "no new successful requests since last evaluation"
        elif not status.get("due_by_interval", False):
            reason = "interval not reached yet"
        else:
            reason = "schedule disabled"
        return {
            "ran": False,
            "reason": reason,
            "result": {"evaluated": 0, "skipped": 0},
            "status": status,
        }

    result = await run_eval_worker(limit=limit)
    updated_status = get_offline_eval_status()
    return {"ran": True, "reason": "ok", "result": result, "status": updated_status}


def build_offline_eval_report(hours: int, top_bad: int) -> dict:
    """Build an on-demand evaluation report directly from the evaluation DynamoDB table."""
    rows = _load_eval_rows(hours=hours)
    return _build_report(rows=rows, top_bad=top_bad, window_hours=hours)


async def _scheduler_loop(lock_token: str):
    global _scheduler_lock_token
    try:
        while True:
            if not _refresh_scheduler_lock(lock_token):
                logger.info("OfflineEvalSchedulerLockLost | stopping scheduler loop")
                return
            try:
                await run_offline_eval(limit=settings.evaluation.batch_size, force=False)
            except Exception as exc:
                # Scheduler is best-effort and should not crash the app.
                logger.warning("OfflineEvalSchedulerRunFailed | error=%s", exc)
            await asyncio.sleep(_SCHEDULER_POLL_SECONDS)
    finally:
        if _scheduler_lock_token == lock_token:
            _scheduler_lock_token = None


def start_offline_eval_scheduler() -> None:
    """Start background scheduler for periodic offline evaluations."""
    global _scheduler_task, _scheduler_lock_token
    if _scheduler_task is not None and not _scheduler_task.done():
        return
    if not settings.evaluation.enabled or not settings.evaluation.schedule_enabled:
        return
    lock_token = _acquire_scheduler_lock()
    if not lock_token:
        logger.info("OfflineEvalSchedulerLockBusy | scheduler start skipped")
        return
    _scheduler_lock_token = lock_token
    _scheduler_task = asyncio.create_task(_scheduler_loop(lock_token))


async def stop_offline_eval_scheduler() -> None:
    """Stop the background offline evaluation scheduler."""
    global _scheduler_task, _scheduler_lock_token
    lock_token = _scheduler_lock_token
    if _scheduler_task is not None:
        _scheduler_task.cancel()
        try:
            await _scheduler_task
        except asyncio.CancelledError:
            pass
    if lock_token:
        _release_scheduler_lock(lock_token)
        if _scheduler_lock_token == lock_token:
            _scheduler_lock_token = None
    _scheduler_task = None
