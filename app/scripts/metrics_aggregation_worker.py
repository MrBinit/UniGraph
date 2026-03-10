"""SQS worker that syncs latest metrics aggregate snapshot to DynamoDB."""

from __future__ import annotations

import asyncio
import logging

from app.core.config import get_settings
from app.services.metrics_dynamodb_service import persist_aggregate_snapshot_dynamodb
from app.services.metrics_json_service import load_chat_metrics_aggregate_json
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
    event_type = str(payload.get("type", "")).strip().lower()
    queue_url = settings.queue.metrics_aggregation_queue_url.strip()

    if event_type and event_type != "metrics_aggregation":
        delete_queue_message(queue_url, receipt_handle)
        return

    aggregate = load_chat_metrics_aggregate_json()
    if not isinstance(aggregate, dict):
        # Nothing to aggregate yet; ack event to avoid retries for empty state.
        delete_queue_message(queue_url, receipt_handle)
        return

    persist_aggregate_snapshot_dynamodb(aggregate)
    delete_queue_message(queue_url, receipt_handle)


async def run_forever() -> None:
    if not settings.queue.metrics_aggregation_queue_enabled:
        logger.warning(
            "MetricsAggregationWorkerDisabled | METRICS_AGGREGATION_QUEUE_ENABLED=false"
        )
    while True:
        if not settings.queue.metrics_aggregation_queue_enabled:
            await asyncio.sleep(
                max(1.0, settings.queue.metrics_aggregation_poll_sleep_seconds)
            )
            continue

        queue_url = settings.queue.metrics_aggregation_queue_url.strip()
        if not queue_url:
            logger.warning(
                (
                    "MetricsAggregationWorkerMisconfigured | "
                    "METRICS_AGGREGATION_QUEUE_URL missing"
                )
            )
            await asyncio.sleep(
                max(1.0, settings.queue.metrics_aggregation_poll_sleep_seconds)
            )
            continue

        try:
            messages = await asyncio.to_thread(
                receive_queue_messages,
                queue_url=queue_url,
                max_messages=settings.queue.metrics_aggregation_max_messages_per_poll,
                wait_seconds=settings.queue.metrics_aggregation_receive_wait_seconds,
                visibility_timeout_seconds=(
                    settings.queue.metrics_aggregation_visibility_timeout_seconds
                ),
            )
        except Exception:
            logger.exception("MetricsAggregationWorkerPollFailed")
            await asyncio.sleep(
                max(1.0, settings.queue.metrics_aggregation_poll_sleep_seconds)
            )
            continue

        if not messages:
            await asyncio.sleep(
                max(0.0, settings.queue.metrics_aggregation_poll_sleep_seconds)
            )
            continue

        for message in messages:
            try:
                await _process_message(message)
            except Exception:
                logger.exception("MetricsAggregationWorkerMessageProcessingFailed")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(run_forever())
    except KeyboardInterrupt:
        logger.info("MetricsAggregationWorkerStopped")


if __name__ == "__main__":
    main()
