"""SQS worker for per-request offline evaluation jobs."""

from __future__ import annotations
import asyncio
import logging
from app.core.config import get_settings
from app.core.security import validate_security_configuration
from app.scripts.eval_dynamodb_worker import run_request_eval
from app.services.sqs_event_queue_service import (
    delete_queue_message,
    parse_message_json,
    receive_queue_messages,
)

settings = get_settings()
logger = logging.getLogger(__name__)


async def _process_message(message: dict) -> None:
    receipt_handle = str(message.get("ReceiptHandle", ""))
    payload = parse_message_json(message)
    queue_url = settings.queue.evaluation_queue_url.strip()

    request_id = str(payload.get("request_id", "")).strip()
    if not request_id:
        delete_queue_message(queue_url, receipt_handle)
        return

    event_type = str(payload.get("type", "")).strip().lower()
    if event_type and event_type != "evaluation":
        delete_queue_message(queue_url, receipt_handle)
        return

    result = await run_request_eval(request_id=request_id)
    if result.get("evaluated") or result.get("skipped"):
        delete_queue_message(queue_url, receipt_handle)


async def run_forever() -> None:
    if not settings.queue.evaluation_queue_enabled:
        logger.warning("EvaluationQueueWorkerDisabled | EVALUATION_QUEUE_ENABLED=false")
    while True:
        if not settings.queue.evaluation_queue_enabled:
            await asyncio.sleep(max(1.0, settings.queue.evaluation_poll_sleep_seconds))
            continue

        queue_url = settings.queue.evaluation_queue_url.strip()
        if not queue_url:
            logger.warning("EvaluationQueueWorkerMisconfigured | EVALUATION_QUEUE_URL missing")
            await asyncio.sleep(max(1.0, settings.queue.evaluation_poll_sleep_seconds))
            continue

        try:
            messages = await asyncio.to_thread(
                receive_queue_messages,
                queue_url=queue_url,
                max_messages=settings.queue.evaluation_max_messages_per_poll,
                wait_seconds=settings.queue.evaluation_receive_wait_seconds,
                visibility_timeout_seconds=(settings.queue.evaluation_visibility_timeout_seconds),
            )
        except Exception:
            logger.exception("EvaluationQueueWorkerPollFailed")
            await asyncio.sleep(max(1.0, settings.queue.evaluation_poll_sleep_seconds))
            continue

        if not messages:
            await asyncio.sleep(max(0.0, settings.queue.evaluation_poll_sleep_seconds))
            continue

        for message in messages:
            try:
                await _process_message(message)
            except Exception:
                logger.exception("EvaluationQueueWorkerMessageProcessingFailed")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    validate_security_configuration()
    try:
        asyncio.run(run_forever())
    except KeyboardInterrupt:
        logger.info("EvaluationQueueWorkerStopped")


if __name__ == "__main__":
    main()
