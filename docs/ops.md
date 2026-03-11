# Operations And Status

## Status Endpoint

UniGraph exposes an admin-only operational endpoint:

- `GET /api/v1/ops/status`

This is the fastest way to inspect runtime health without opening Redis manually.

## Response Sections

### `status`

Top-level health state:

- `ok`: normal state
- `degraded`: Redis is reachable, but DLQ depth is at or above the alert threshold
- `down`: app-side Redis is not reachable

### `memory`

Current fields:

- `redis_available`
- `ttl_seconds`
- `encryption_enabled`

Meaning:

- tells you whether app Redis is reachable
- shows the effective TTL used for memory persistence
- confirms that short-term memory encryption is active

### `queue`

Current fields:

- `stream_depth`
- `pending_jobs`
- `dlq_depth`
- `consumer_group`
- `last_dlq_error`

Operational meaning:

- `stream_depth` is total stream length, not only unprocessed work
- `pending_jobs` is the more useful live pressure metric
- `dlq_depth` should normally stay at `0`
- `last_dlq_error` gives the most recent failure reason stored in DLQ

### `compaction`

Current fields:

- `events`
- `removed_messages`
- `removed_tokens`

These are global compaction counters written by the short-term memory system.

### `latency`

Current fields:

- `count`
- `average_ms`
- `max_ms`
- `last_ms`
- `last_outcome`
- JSON aggregate also includes `p95` and `p99` for latency series

Latency outcomes currently include:

- `success`
- `cache_hit`
- `blocked_input`
- `blocked_context`

## What To Watch

Primary operational signals:

1. `queue.pending_jobs`
2. `queue.dlq_depth`
3. `latency.average_ms`
4. `latency.max_ms`
5. `compaction.events`

High-risk patterns:

- `pending_jobs` rising continuously: worker is falling behind
- `dlq_depth > 0`: failed jobs are accumulating
- `average_ms` rising sharply: model, Redis, or middleware pressure is increasing
- `compaction.events` exploding: user context may be too large or budgets may be too low

## Backpressure Behavior

Backpressure runs as two gates:

- distributed gate (Redis sorted-set lease) for cross-instance admission control
- local gate (atomic lock+counter) for per-process in-flight limits

Local gate behavior:

- when local in-flight count reaches `max_in_flight_requests`, requests are rejected with `503`
- local admission and release are atomic; there is no private semaphore pre-check
- if a request is rejected by the local gate after distributed admission, the Redis lease token is released immediately

## Current Alerting

DLQ alerting is log-based right now.

The system emits:

- `SummaryJobDLQAlert`

This is useful immediately, but should later be connected to:

- PagerDuty
- Slack
- Datadog
- CloudWatch
- Grafana alerting

## Gradio Session Isolation

Gradio no longer uses one shared memory identity.

- each browser session gets a generated user id: `gradio-session-<uuid>`
- the id is stored in `gr.State` and reused for later submits from that same session
- short-term memory and cache keys are therefore isolated per Gradio session instead of being shared globally

Operational note:

- if memory keys are inspected in Redis, expect multiple session-scoped keys rather than one shared `gradio-user` key

## JSON Metrics Persistence

When `APP_METRICS_JSON_ENABLED=true`, each chat request is persisted to JSON for offline analysis.

Output files under `APP_METRICS_JSON_DIR`:

- `chat_metrics_requests.jsonl`: one event per request (question, answer, latency breakdown, usage, outcome).
- `chat_metrics_aggregate.json`: rolling totals and averages for latency, outcomes, and token usage.

Latency percentile notes:

- `chat_metrics_aggregate.json` computes `p95` and `p99` for each latency series:
  - `overall`
  - `llm_response`
  - `short_term_memory`
  - `long_term_memory`
  - `memory_update`
  - `cache_read`
  - `cache_write`
  - `evaluation_trace`

## DynamoDB Metrics Persistence

When `APP_METRICS_DYNAMODB_ENABLED=true`, metrics are also persisted to DynamoDB.

Tables:

- requests table: `APP_METRICS_DYNAMODB_REQUESTS_TABLE`
  - partition key: `request_id` (String)
- aggregate table: `APP_METRICS_DYNAMODB_AGGREGATE_TABLE`
  - partition key: `id` (String), uses singleton item `id=global`

Request item storage model:

- queryable top-level fields:
  - `request_id`
  - `timestamp`
  - `user_id`
  - `session_id`
  - `outcome`
  - `retrieval_strategy`
  - `latency_overall_ms`
  - `prompt_tokens`
  - `total_tokens`
- full payload:
  - `record_json` (entire metrics record as JSON string)

Aggregate item storage model:

- queryable top-level fields:
  - `id`
  - `updated_at`
  - `total_requests`
  - `overall_avg_latency_ms`
- full payload:
  - `aggregate_json` (entire aggregate snapshot as JSON string)

Optional queue mode (recommended at higher throughput):

- `METRICS_AGGREGATION_QUEUE_ENABLED=true`
- `METRICS_AGGREGATION_QUEUE_URL=<sqs-url>`
- run worker: `python -m app.scripts.metrics_aggregation_worker`
- behavior:
  - live request persists request row immediately
  - aggregate sync is offloaded to SQS worker
  - if queue publish fails, service falls back to inline aggregate write

## Offline Evaluation (Separate Table)

Live request handling does not compute hallucination/clarity/relevance metrics.
Those quality metrics are generated asynchronously and stored by `request_id` in a dedicated evaluation table.

Config (`app/config/evaluation_config.yaml`):

- `evaluation.enabled`
- `evaluation.dynamodb_table`
- `evaluation.judge_model_id`
- `evaluation.lookback_hours`
- `evaluation.max_items_per_run`
- `evaluation.schedule_enabled`
- `evaluation.schedule_interval_hours`
- `evaluation.request_status_attribute`
- `evaluation.request_status_index_name`
- `evaluation.request_pending_value`
- `evaluation.request_in_progress_value`
- `evaluation.request_completed_value`
- `evaluation.eval_status_attribute`
- `evaluation.eval_status_index_name`
- `evaluation.eval_completed_value`
- `evaluation.judge_model_id` is the LLM-as-judge model ID (default Bedrock Nova 2 Lite: `us.amazon.nova-2-lite-v1:0`)

Judge prompt config:

- file: `app/config/evaluation_prompt.yaml`
- keys:
  - `evaluation_judge.clarity_system_prompt`
  - `evaluation_judge.relevance_system_prompt`
  - `evaluation_judge.hallucination_system_prompt`

Evaluation table shape:

- table: `evaluation.dynamodb_table` (for example `unigraph-chat-evaluations`)
- partition key: `request_id` (String)
- one item per evaluated request
- `status=completed` is written for indexed status lookups
- hallucination scoring is grounded against request-time retrieval evidence snapshot

Request table eval status flow:

- `pending`: successful request waiting for evaluation
- `in_progress`: claimed by one worker
- `completed`: evaluation stored
- `not_applicable`: non-success request, excluded from evaluation

Required GSIs (to avoid table scans):

- requests table GSI: `eval-status-timestamp-index` (`eval_status`, `timestamp`)
- evaluations table GSI: `status-timestamp-index` (`status`, `timestamp`)

Scripts:

- worker/backfill run:
  - `python -m app.scripts.eval_dynamodb_worker --limit 200`
- daily report:
  - `python -m app.scripts.eval_daily_report --hours 24 --top-bad 10`
- queue worker (per-request evaluation from SQS):
  - `python -m app.scripts.eval_queue_worker`

API endpoints (admin):

- scheduler status:
  - `GET /api/v1/eval/offline/status`
- run now (bypass interval/new-data guard with `force=true`):
  - `POST /api/v1/eval/offline/run?force=true&limit=50`
- run guarded (only if interval reached and new requests exist):
  - `POST /api/v1/eval/offline/run`
- on-demand report JSON:
  - `GET /api/v1/eval/offline/report?hours=24&top_bad=10`

Scheduling behavior:

- background scheduler runs every `evaluation.schedule_interval_hours` (default 1h)
- each run only evaluates when:
  - there are indexed `eval_status=pending` request records
  - and the interval gate is reached
- manual `force=true` always runs immediately

Optional queue-driven evaluation mode:

- `EVALUATION_QUEUE_ENABLED=true`
- `EVALUATION_QUEUE_URL=<sqs-url>`
- request metrics persistence publishes one eval event per successful request
- worker consumes by `request_id`, checks/claims `pending`, then transitions:
  - `pending -> in_progress -> completed`
- if processing fails, worker sets status back to `pending` for retry
- scheduler can remain enabled as periodic indexed backfill/safety net

Processing guarantee:

- The queue worker processes only requests whose status is `pending`, claims them by updating status to `in_progress`, runs evaluation, and marks them `completed`, so each request is evaluated only once.

Daily report output:

- p50/p95 for `clarity_score`, `relevance_score`, `hallucination_score`, and `overall_score`
- p50/p95 for `evidence_similarity_score`
- failure reason distribution
- top low-score examples

TTL behavior:

- if `APP_METRICS_DYNAMODB_TTL_DAYS > 0`, items include `expires_at` (epoch seconds)
- enable DynamoDB TTL on attribute `expires_at` in both tables

Verification examples:

```bash
aws dynamodb scan \
  --region us-east-1 \
  --table-name unigraph-chat-metrics-requests \
  --projection-expression "request_id,timestamp,outcome,session_id,latency_overall_ms,retrieval_strategy,retrieval_evidence_count,prompt_tokens,total_tokens" \
  --max-items 5

aws dynamodb get-item \
  --region us-east-1 \
  --table-name unigraph-chat-metrics-aggregate \
  --key '{"id":{"S":"global"}}'
```
