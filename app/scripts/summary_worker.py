import asyncio
import logging
import os
import socket
import time

os.environ.setdefault("REDIS_RUNTIME_ROLE", "worker")
from app.core.config import get_settings  # noqa: E402
from app.core.security import validate_security_configuration  # noqa: E402
from app.services.summary_queue_service import (  # noqa: E402
    claim_stale_summary_jobs,
    ensure_consumer_group,
    monitor_summary_dlq,
    read_summary_jobs,
)
from app.services.summary_worker_service import process_summary_job  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()


def _consumer_name() -> str:
    """Build the Redis consumer name for this worker process."""
    env_name = os.getenv("SUMMARY_WORKER_CONSUMER")
    if env_name:
        return env_name
    return f"{socket.gethostname()}-{os.getpid()}"


async def run_worker():
    """Run the long-lived background loop that drains summary jobs."""
    consumer_name = _consumer_name()
    ensure_consumer_group()
    logger.info("Summary worker started consumer=%s", consumer_name)
    last_dlq_monitor_at = 0.0

    while True:
        now = time.monotonic()
        if now - last_dlq_monitor_at >= settings.memory.summary_queue_dlq_monitor_interval_seconds:
            monitor_summary_dlq()
            last_dlq_monitor_at = now

        jobs = claim_stale_summary_jobs(consumer_name)
        if not jobs:
            jobs = read_summary_jobs(consumer_name)
        if not jobs:
            await asyncio.sleep(0.2)
            continue

        for stream_id, fields in jobs:
            await process_summary_job(stream_id, fields)


def main():
    """Start the summary worker event loop from the CLI entrypoint."""
    validate_security_configuration()
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
