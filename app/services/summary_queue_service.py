import json
import logging
import hashlib
from datetime import datetime, timezone
from uuid import uuid4

from redis.exceptions import RedisError, ResponseError

from app.core.config import get_settings
from app.infra.redis_client import worker_redis_client, worker_scoped_key

settings = get_settings()
logger = logging.getLogger(__name__)

_PROCESSING_TTL_SECONDS = 300
_COMPLETED_TTL_SECONDS = 86400


def _stream_key() -> str:
    """Return the worker-scoped Redis stream key for summary jobs."""
    return worker_scoped_key(settings.memory.summary_queue_stream_key)


def _dlq_stream_key() -> str:
    """Return the worker-scoped Redis stream key for summary dead-letter jobs."""
    return worker_scoped_key(settings.memory.summary_queue_dlq_stream_key)


def _processing_key(idempotency_key: str) -> str:
    """Build the Redis key used to mark a summary job as currently processing."""
    return f"{_stream_key()}:processing:{idempotency_key}"


def _completed_key(idempotency_key: str) -> str:
    """Build the Redis key used to mark a summary job as already completed."""
    return f"{_stream_key()}:completed:{idempotency_key}"


def _dlq_alert_key() -> str:
    """Build the Redis key used to throttle DLQ alert emission."""
    return f"{_dlq_stream_key()}:alerted"


def build_summary_job_idempotency_key(
    *,
    user_id: str,
    cutoff_seq: int,
    trigger: str,
    enqueue_version: int,
) -> str:
    """Create a deterministic idempotency key for a logical summary job."""
    raw = f"{user_id}:{cutoff_seq}:{trigger}:{enqueue_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_summary_job_idempotency_key(fields: dict) -> str:
    """Read or reconstruct the idempotency key from a stream entry payload."""
    explicit = fields.get("idempotency_key")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    user_id = fields.get("user_id", "")
    trigger = fields.get("trigger", "summary_trigger")
    try:
        cutoff_seq = int(fields.get("cutoff_seq", "0"))
        enqueue_version = int(fields.get("enqueue_version", "0"))
    except (TypeError, ValueError):
        cutoff_seq = 0
        enqueue_version = 0

    if isinstance(user_id, str) and user_id and cutoff_seq > 0:
        return build_summary_job_idempotency_key(
            user_id=user_id,
            cutoff_seq=cutoff_seq,
            trigger=trigger if isinstance(trigger, str) else "summary_trigger",
            enqueue_version=enqueue_version,
        )

    job_id = fields.get("job_id", "")
    if isinstance(job_id, str) and job_id.strip():
        return f"job:{job_id.strip()}"
    return ""


def ensure_consumer_group():
    """Create the Redis consumer group used by summary workers if it does not exist."""
    try:
        worker_redis_client.xgroup_create(
            name=_stream_key(),
            groupname=settings.memory.summary_queue_group,
            id="0",
            mkstream=True,
        )
    except ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise
    except RedisError as exc:
        logger.warning("Failed to ensure summary queue consumer group. %s", exc)


def get_summary_dlq_state() -> dict:
    """Return the current DLQ depth and the latest failed job metadata."""
    try:
        depth = int(worker_redis_client.xlen(_dlq_stream_key()))
        latest = worker_redis_client.xrevrange(_dlq_stream_key(), count=1)
    except RedisError as exc:
        logger.warning("Failed reading summary DLQ state. %s", exc)
        return {"depth": 0, "latest": None}

    if not latest:
        return {"depth": depth, "latest": None}

    stream_id, fields = latest[0]
    latest_info = {
        "stream_id": stream_id,
        "job_id": fields.get("job_id", ""),
        "user_id": fields.get("user_id", ""),
        "failed_at": fields.get("failed_at", ""),
        "error": fields.get("error", ""),
        "final_attempt": fields.get("final_attempt", ""),
    }
    return {"depth": depth, "latest": latest_info}


def get_summary_queue_state() -> dict:
    """Return the current summary stream depth and pending job count."""
    try:
        stream_depth = int(worker_redis_client.xlen(_stream_key()))
    except RedisError as exc:
        logger.warning("Failed reading summary queue depth. %s", exc)
        stream_depth = 0

    pending_jobs = 0
    try:
        pending = worker_redis_client.xpending(_stream_key(), settings.memory.summary_queue_group)
        if isinstance(pending, dict):
            pending_jobs = int(pending.get("pending", 0))
        elif isinstance(pending, (list, tuple)) and pending:
            pending_jobs = int(pending[0])
    except AttributeError:
        pending_jobs = 0
    except (RedisError, TypeError, ValueError) as exc:
        logger.warning("Failed reading summary queue pending count. %s", exc)

    return {
        "stream_depth": stream_depth,
        "pending_jobs": max(0, pending_jobs),
    }


def monitor_summary_dlq(*, force: bool = False) -> dict:
    """Inspect the DLQ and emit a throttled alert when backlog crosses the threshold."""
    state = get_summary_dlq_state()
    depth = state["depth"]
    threshold = settings.memory.summary_queue_dlq_alert_threshold

    if depth < threshold:
        return {"depth": depth, "latest": state["latest"], "alerted": False}

    should_alert = force
    if not force:
        try:
            should_alert = bool(
                worker_redis_client.set(
                    _dlq_alert_key(),
                    str(depth),
                    ex=settings.memory.summary_queue_dlq_alert_cooldown_seconds,
                    nx=True,
                )
            )
        except RedisError as exc:
            logger.warning("Failed setting summary DLQ alert throttle. %s", exc)
            should_alert = True

    if should_alert:
        logger.error(
            "SummaryJobDLQAlert | %s",
            json.dumps(
                {
                    "depth": depth,
                    "threshold": threshold,
                    "latest": state["latest"],
                },
                sort_keys=True,
            ),
        )

    return {"depth": depth, "latest": state["latest"], "alerted": should_alert}


def enqueue_summary_job(
    *,
    user_id: str,
    cutoff_seq: int,
    trigger: str,
    enqueue_version: int,
    approx_removed_tokens: int,
    attempt: int = 0,
) -> str | None:
    """Push a new summary compaction job onto the Redis stream."""
    job_id = str(uuid4())
    idempotency_key = build_summary_job_idempotency_key(
        user_id=user_id,
        cutoff_seq=cutoff_seq,
        trigger=trigger,
        enqueue_version=enqueue_version,
    )
    payload = {
        "job_id": job_id,
        "idempotency_key": idempotency_key,
        "user_id": user_id,
        "cutoff_seq": str(cutoff_seq),
        "trigger": trigger,
        "enqueue_version": str(enqueue_version),
        "approx_removed_tokens": str(approx_removed_tokens),
        "attempt": str(attempt),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        worker_redis_client.xadd(
            _stream_key(),
            payload,
            maxlen=10000,
            approximate=True,
        )
        logger.info("SummaryJobEnqueued | %s", json.dumps(payload, sort_keys=True))
        return job_id
    except RedisError as exc:
        logger.warning("Failed to enqueue summary job for user=%s. %s", user_id, exc)
        return None


def is_summary_job_processed(idempotency_key: str) -> bool:
    """Check whether a logical summary job has already been completed."""
    if not idempotency_key:
        return False
    try:
        return bool(worker_redis_client.get(_completed_key(idempotency_key)))
    except RedisError as exc:
        logger.warning("Failed checking summary job completion marker. %s", exc)
        return False


def claim_summary_job_processing(idempotency_key: str, stream_id: str) -> bool:
    """Claim the in-progress marker for a summary job to avoid duplicate processing."""
    if not idempotency_key:
        return True
    try:
        claimed = worker_redis_client.set(
            _processing_key(idempotency_key),
            stream_id,
            ex=_PROCESSING_TTL_SECONDS,
            nx=True,
        )
        return bool(claimed)
    except RedisError as exc:
        logger.warning("Failed claiming summary job processing marker. %s", exc)
        return True


def release_summary_job_processing(idempotency_key: str):
    """Remove the in-progress marker so a failed job can be retried safely."""
    if not idempotency_key:
        return
    try:
        worker_redis_client.delete(_processing_key(idempotency_key))
    except RedisError as exc:
        logger.warning("Failed releasing summary job processing marker. %s", exc)


def mark_summary_job_processed(idempotency_key: str, stream_id: str):
    """Mark a summary job as completed and clear any in-progress marker."""
    if not idempotency_key:
        return
    try:
        worker_redis_client.set(
            _completed_key(idempotency_key),
            stream_id,
            ex=_COMPLETED_TTL_SECONDS,
        )
        worker_redis_client.delete(_processing_key(idempotency_key))
    except RedisError as exc:
        logger.warning("Failed marking summary job as processed. %s", exc)


def read_summary_jobs(consumer_name: str) -> list[tuple[str, dict]]:
    """Read the next batch of summary jobs for a worker consumer."""
    ensure_consumer_group()
    try:
        response = worker_redis_client.xreadgroup(
            groupname=settings.memory.summary_queue_group,
            consumername=consumer_name,
            streams={_stream_key(): ">"},
            count=settings.memory.summary_queue_read_count,
            block=settings.memory.summary_queue_block_ms,
        )
    except RedisError as exc:
        logger.warning("Failed reading summary queue. %s", exc)
        return []

    jobs = []
    for _stream, entries in response:
        for stream_id, fields in entries:
            jobs.append((stream_id, fields))
    return jobs


def claim_stale_summary_jobs(consumer_name: str) -> list[tuple[str, dict]]:
    """Claim stale pending jobs abandoned by other consumers."""
    ensure_consumer_group()
    try:
        response = worker_redis_client.xautoclaim(
            _stream_key(),
            settings.memory.summary_queue_group,
            consumer_name,
            settings.memory.summary_queue_claim_idle_ms,
            start_id="0-0",
            count=settings.memory.summary_queue_claim_batch_size,
        )
    except AttributeError:
        logger.warning("Redis client does not support xautoclaim; stale job recovery is disabled.")
        return []
    except ResponseError as exc:
        logger.warning("Failed reclaiming stale summary jobs. %s", exc)
        return []
    except RedisError as exc:
        logger.warning("Failed reclaiming stale summary jobs. %s", exc)
        return []

    if not isinstance(response, (list, tuple)) or len(response) < 2:
        return []

    claimed_entries = response[1]
    if not isinstance(claimed_entries, list):
        return []

    jobs = []
    for entry in claimed_entries:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        stream_id, fields = entry
        if not isinstance(fields, dict):
            continue
        jobs.append((stream_id, fields))
    return jobs


def ack_summary_job(stream_id: str):
    """Acknowledge a processed or discarded summary job in the Redis stream."""
    try:
        worker_redis_client.xack(_stream_key(), settings.memory.summary_queue_group, stream_id)
    except RedisError as exc:
        logger.warning("Failed acknowledging summary job %s. %s", stream_id, exc)


def retry_or_dlq_summary_job(stream_id: str, fields: dict, error: str):
    """Retry a failed summary job or move it to the DLQ after the final attempt."""
    try:
        attempt = int(fields.get("attempt", "0")) + 1
    except (TypeError, ValueError):
        attempt = 1

    if attempt >= settings.memory.summary_queue_max_attempts:
        dlq_payload = dict(fields)
        dlq_payload["failed_at"] = datetime.now(timezone.utc).isoformat()
        dlq_payload["error"] = error[:500]
        dlq_payload["final_attempt"] = str(attempt)
        try:
            worker_redis_client.xadd(
                _dlq_stream_key(),
                dlq_payload,
                maxlen=5000,
                approximate=True,
            )
            logger.error("SummaryJobDLQ | %s", json.dumps(dlq_payload, sort_keys=True))
            monitor_summary_dlq()
        except RedisError as exc:
            logger.warning("Failed writing summary job to DLQ. %s", exc)

        ack_summary_job(stream_id)
        return

    retry_payload = dict(fields)
    retry_payload["attempt"] = str(attempt)
    retry_payload["last_error"] = error[:300]
    retry_payload["retried_at"] = datetime.now(timezone.utc).isoformat()

    try:
        worker_redis_client.xadd(
            _stream_key(),
            retry_payload,
            maxlen=10000,
            approximate=True,
        )
        ack_summary_job(stream_id)
    except RedisError as exc:
        logger.warning("Failed to retry summary job %s. %s", stream_id, exc)
