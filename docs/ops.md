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
