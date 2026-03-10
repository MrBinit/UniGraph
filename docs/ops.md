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

## JSON Metrics Persistence

When `APP_METRICS_JSON_ENABLED=true`, each chat request is persisted to JSON for offline analysis.

Output files under `APP_METRICS_JSON_DIR`:

- `chat_metrics_requests.jsonl`: one event per request (question, answer, latency breakdown, quality metrics, usage, outcome).
- `chat_metrics_aggregate.json`: rolling totals and averages for latency, quality, outcomes, and token usage.

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

TTL behavior:

- if `APP_METRICS_DYNAMODB_TTL_DAYS > 0`, items include `expires_at` (epoch seconds)
- enable DynamoDB TTL on attribute `expires_at` in both tables

Verification examples:

```bash
aws dynamodb scan \
  --region us-east-1 \
  --table-name unigraph-chat-metrics-requests \
  --projection-expression "request_id,timestamp,outcome,session_id,latency_overall_ms,retrieval_strategy,prompt_tokens,total_tokens" \
  --max-items 5

aws dynamodb get-item \
  --region us-east-1 \
  --table-name unigraph-chat-metrics-aggregate \
  --key '{"id":{"S":"global"}}'
```
