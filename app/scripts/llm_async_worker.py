"""SQS worker for async chat jobs.

Consumes jobs from SQS, runs the existing `generate_response` pipeline,
and writes completed/failed status to DynamoDB result store.
"""

from __future__ import annotations
import asyncio
import json
import logging
from app.core.config import get_settings
from app.services.llm_async_queue_service import (
    delete_llm_job_message,
    get_chat_job,
    mark_job_completed,
    mark_job_failed,
    mark_job_processing,
    receive_llm_job_messages,
)
from app.services.llm_service import generate_response

settings = get_settings()
logger = logging.getLogger(__name__)


def _parse_message_body(raw_body: str) -> dict:
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


async def _process_message(message: dict) -> None:
    receipt_handle = str(message.get("ReceiptHandle", ""))
    payload = _parse_message_body(str(message.get("Body", "")))
    job_id = str(payload.get("job_id", "")).strip()
    user_id = str(payload.get("user_id", "")).strip()
    prompt = str(payload.get("prompt", "")).strip()

    if not job_id or not user_id or not prompt:
        if job_id:
            mark_job_failed(job_id, "Invalid async job payload.")
        delete_llm_job_message(receipt_handle)
        return

    existing = get_chat_job(job_id)
    if existing and str(existing.get("status", "")).strip().lower() == "completed":
        delete_llm_job_message(receipt_handle)
        return

    mark_job_processing(job_id)
    try:
        answer = await generate_response(user_id, prompt)
        mark_job_completed(job_id, answer)
        delete_llm_job_message(receipt_handle)
    except Exception as exc:
        # Intentionally do not delete SQS message on failure so queue retries can happen.
        mark_job_failed(job_id, str(exc))
        logger.exception("AsyncLLMJobFailed | job_id=%s", job_id)


async def run_forever() -> None:
    if not settings.queue.llm_async_enabled:
        logger.warning("AsyncLLMWorkerDisabled | LLM_ASYNC_ENABLED=false")
    while True:
        if not settings.queue.llm_async_enabled:
            await asyncio.sleep(max(1.0, settings.queue.llm_poll_sleep_seconds))
            continue
        try:
            messages = await asyncio.to_thread(receive_llm_job_messages)
        except Exception:
            logger.exception("AsyncLLMWorkerPollFailed")
            await asyncio.sleep(max(1.0, settings.queue.llm_poll_sleep_seconds))
            continue

        if not messages:
            await asyncio.sleep(max(0.0, settings.queue.llm_poll_sleep_seconds))
            continue

        for message in messages:
            try:
                await _process_message(message)
            except Exception:
                logger.exception("AsyncLLMWorkerMessageProcessingFailed")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(run_forever())
    except KeyboardInterrupt:
        logger.info("AsyncLLMWorkerStopped")


if __name__ == "__main__":
    main()
