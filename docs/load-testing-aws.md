# AWS Full-Stack Load Testing

This document describes how to run repeatable load tests that exercise the production-like AWS integration paths:

- API async stream enqueue path (`POST /api/v1/chat/stream`)
- SQS job queue consumption
- Postgres retrieval path
- DynamoDB result and metrics writes
- Redis memory and summary queue worker

The load profile is YAML-driven and intended to run from your laptop.

## Scope

In this test mode:

- **Included**: API, SQS, DynamoDB, Postgres retrieval, Redis, async workers.
- **Mocked**: external model APIs (`LLM_MOCK_MODE`, `EMBEDDING_MOCK_MODE`, `MEMORY_SUMMARY_MOCK_MODE`).

This gives architecture/scalability signal for the system design without model cost variance.

## Config File

Main file:

- `loadtest/aws_full_stack_load.yaml`

Most-used keys:

- `workload.users`: number of logical users.
- `workload.requests_per_user`: requests per user.
- `workload.concurrency`: concurrent enqueue clients.
- `runtime.api_workers`: API worker count.
- `runtime.llm_async_workers`: async chat worker count.
- `runtime.metrics_workers`: metrics queue worker count.
- `runtime.wait_for_metrics_drain_seconds`: extra drain window before shutdown.
- `runtime.llm_queue_url`: dedicated SQS queue URL for async chat.
- `runtime.metrics_queue_url`: dedicated SQS queue URL for metrics events.

For 1000-user testing, start by increasing:

- `workload.users`
- `workload.concurrency`
- `runtime.llm_async_workers`
- `runtime.metrics_workers`

## Run Command

From repository root:

```bash
POSTGRES_PASSWORD="$(aws secretsmanager get-secret-value --region us-east-1 --secret-id unigraph/prod/app --query SecretString --output text | sed -n 's/.*\"POSTGRES_PASSWORD\":\"\([^\"]*\)\".*/\1/p')" \
./loadtest/run_aws_full_stack_load.sh loadtest/aws_full_stack_load.yaml
```

The runner performs:

1. Redis startup
2. AWS preflight (SQS + DynamoDB + identity)
3. Postgres connectivity check
4. Worker/API startup
5. Load run (`loadtest/aws_async_queue_load.py`)
6. Optional metrics queue drain wait
7. Ops snapshot collection

## How To Read Results

### Async Queue Summary

- `enqueue_success`: accepted async requests (`202`).
- `jobs_completed`: requests fully processed by workers.
- `jobs_failed`: jobs that ended in failed status.
- `jobs_unresolved`: still queued/processing at poll timeout.

Healthy run pattern:

- `enqueue_error_rate` near `0`
- `jobs_failed` near `0`
- `jobs_unresolved` near `0`

### Metrics Drain Section

- `metrics_queue_visible` + `metrics_queue_in_flight` should trend toward `0`.
- If not zero before timeout, increase:
  - `runtime.metrics_workers`
  - `runtime.wait_for_metrics_drain_seconds`

### Ops Snapshot

Key fields:

- `status`: expected `ok`.
- `latency.last_retrieval_strategy`: confirms retrieval path used (`hnsw` in current setup).
- `queue.dlq_depth`: expected `0`.
- `compaction.events`: confirms summary/compaction activity.

## Example Interpretation (Your 200 users x 4 requests run)

Observed:

- `total_enqueue_requests=800`
- `enqueue_success=800`
- `jobs_completed=800`
- `jobs_failed=0`
- `jobs_unresolved=0`
- ops `status=ok`

Meaning:

- Core async architecture sustained this profile end-to-end.
- Main queue path was healthy at this load.
- Metrics queue showed temporary backlog during drain window, which is expected under bursty writes and can be tuned with more metrics workers/drain time.

## Failure Semantics

If the system is stressed or degraded:

- Enqueue may return `429`, `503`, or `500` instead of `202`.
- Accepted jobs may remain `queued`/`processing` longer.
- Some streams may return an error event when jobs move to `failed` status.

## Practical Notes

- Use dedicated load-test SQS queues to avoid interference from shared production traffic.
- Purge load-test queues before each major run if you want clean measurements.
- Keep run artifacts:
  - `/tmp/ai-system-aws-load-api.log`
  - `/tmp/ai-system-aws-load-llm-async-worker.log`
  - `/tmp/ai-system-aws-load-metrics-worker.log`
  - `/tmp/ai-system-aws-load-summary-worker.log`
