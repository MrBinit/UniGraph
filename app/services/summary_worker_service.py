import json
import logging

from app.core.token_utils import count_tokens
from app.services.guardrails_service import sanitize_summary_output
from app.services.memory_compaction_service import compose_context, merge_summaries, safe_token_count
from app.services.memory_metrics_service import record_compaction_metrics
from app.services.memory_service import (
    _strip_seq,
    load_memory,
    save_memory_if_version,
    summarize_messages,
)
from app.services.summary_queue_service import (
    ack_summary_job,
    claim_summary_job_processing,
    get_summary_job_idempotency_key,
    is_summary_job_processed,
    mark_summary_job_processed,
    release_summary_job_processing,
    retry_or_dlq_summary_job,
)

logger = logging.getLogger(__name__)


def _to_int(value, default=0) -> int:
    """Convert a stream field value to int and fall back when parsing fails."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_updated_memory(memory: dict, cutoff_seq: int, summary_text: str, job_id: str) -> tuple[dict, list]:
    """Build the next memory snapshot after summarizing messages up to the cutoff."""
    updated = dict(memory)
    old_messages = [m for m in memory["messages"] if m.get("seq", 0) <= cutoff_seq]
    remaining_messages = [m for m in memory["messages"] if m.get("seq", 0) > cutoff_seq]

    updated["summary"] = merge_summaries(memory["summary"], summary_text)
    updated["messages"] = remaining_messages
    updated["last_summarized_seq"] = max(memory["last_summarized_seq"], cutoff_seq)
    if updated.get("last_summary_job_id") == job_id:
        updated["summary_pending"] = False
        updated["last_summary_job_id"] = ""
    updated["version"] = memory["version"] + 1
    updated["next_seq"] = max(
        updated["next_seq"],
        (remaining_messages[-1]["seq"] + 1) if remaining_messages else updated["next_seq"],
    )
    return updated, old_messages


async def process_summary_job(stream_id: str, fields: dict):
    """Process one queued summary job and update short-term memory asynchronously."""
    user_id = fields.get("user_id", "")
    job_id = fields.get("job_id", "")
    trigger = fields.get("trigger", "summary_trigger")
    cutoff_seq = _to_int(fields.get("cutoff_seq"), 0)
    idempotency_key = get_summary_job_idempotency_key(fields)

    if not user_id or cutoff_seq <= 0:
        ack_summary_job(stream_id)
        return

    if is_summary_job_processed(idempotency_key):
        logger.info(
            "SummaryJobSkipped | %s",
            json.dumps(
                {
                    "stream_id": stream_id,
                    "job_id": job_id,
                    "user_id": user_id,
                    "reason": "already_processed",
                },
                sort_keys=True,
            ),
        )
        ack_summary_job(stream_id)
        return

    if not claim_summary_job_processing(idempotency_key, stream_id):
        logger.info(
            "SummaryJobSkipped | %s",
            json.dumps(
                {
                    "stream_id": stream_id,
                    "job_id": job_id,
                    "user_id": user_id,
                    "reason": "already_in_progress",
                },
                sort_keys=True,
            ),
        )
        ack_summary_job(stream_id)
        return

    try:
        for _ in range(4):
            memory = await load_memory(user_id)

            if memory["last_summarized_seq"] >= cutoff_seq:
                if memory.get("last_summary_job_id") == job_id and memory.get("summary_pending"):
                    stale_update = dict(memory)
                    stale_update["summary_pending"] = False
                    stale_update["last_summary_job_id"] = ""
                    stale_update["version"] = memory["version"] + 1
                    save_memory_if_version(user_id, memory["version"], stale_update)
                mark_summary_job_processed(idempotency_key, stream_id)
                ack_summary_job(stream_id)
                return

            old_messages = [m for m in memory["messages"] if m.get("seq", 0) <= cutoff_seq]
            if not old_messages:
                if memory.get("last_summary_job_id") == job_id and memory.get("summary_pending"):
                    stale_update = dict(memory)
                    stale_update["summary_pending"] = False
                    stale_update["last_summary_job_id"] = ""
                    stale_update["version"] = memory["version"] + 1
                    save_memory_if_version(user_id, memory["version"], stale_update)
                mark_summary_job_processed(idempotency_key, stream_id)
                ack_summary_job(stream_id)
                return

            summary_text = await summarize_messages(_strip_seq(old_messages))
            summary_text = sanitize_summary_output(summary_text)
            if not summary_text.strip():
                raise RuntimeError("empty summary generated")

            updated_memory, removed_messages = _build_updated_memory(memory, cutoff_seq, summary_text, job_id)
            updated, latest = save_memory_if_version(user_id, memory["version"], updated_memory)
            if not updated:
                memory = latest
                continue

            before_tokens = safe_token_count(
                count_tokens,
                compose_context(memory["summary"], memory["messages"], ""),
            )
            after_tokens = safe_token_count(
                count_tokens,
                compose_context(updated_memory["summary"], updated_memory["messages"], ""),
            )
            removed_tokens = safe_token_count(count_tokens, removed_messages)

            record_compaction_metrics(
                user_id=user_id,
                trigger=f"async_{trigger}",
                removed_messages=len(removed_messages),
                removed_tokens=removed_tokens,
                before_tokens=before_tokens,
                after_tokens=after_tokens,
                summary_text=summary_text,
                token_counter=count_tokens,
            )

            logger.info(
                "SummaryJobProcessed | %s",
                json.dumps(
                    {
                        "stream_id": stream_id,
                        "job_id": job_id,
                        "user_id": user_id,
                        "cutoff_seq": cutoff_seq,
                    },
                    sort_keys=True,
                ),
            )
            mark_summary_job_processed(idempotency_key, stream_id)
            ack_summary_job(stream_id)
            return

        raise RuntimeError("version conflict while updating memory")
    except Exception as exc:
        release_summary_job_processing(idempotency_key)
        logger.exception("Summary job processing failed stream_id=%s", stream_id)
        retry_or_dlq_summary_job(stream_id, fields, str(exc))
