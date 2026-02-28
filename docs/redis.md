# Redis

## Roles

Redis is used for multiple responsibilities:

- short-term memory storage
- chat response caching
- compaction metrics
- summary queue and consumer group
- replay-protection markers
- dead-letter queue
- DLQ alert throttling
- latency counters

## Client Separation

The code defines two Redis clients:

- `app_redis_client`
- `worker_redis_client`

Runtime alias:

- `redis_client` points to app or worker depending on `REDIS_RUNTIME_ROLE`

The summary worker script sets:

- `REDIS_RUNTIME_ROLE=worker`

This allows the worker runtime to use worker credentials by default.

## Namespaces

Two namespaces are configured:

- app namespace, default `app`
- worker namespace, default `worker`

The code uses helper functions to generate scoped keys:

- `app_scoped_key(...)`
- `worker_scoped_key(...)`

## Key Patterns

### App Namespace

Typical app keys:

- `app:memory:chat:{user_id}`
- `app:cache:chat:{user_id}:{sanitized_prompt}`
- `app:metrics:memory:compaction:global`
- `app:metrics:memory:compaction:user:{user_id}`
- `app:metrics:llm:latency`

### Worker Namespace

Typical worker keys:

- `worker:memory:summary:jobs`
- `worker:memory:summary:dlq`
- `worker:memory:summary:jobs:processing:{idempotency_key}`
- `worker:memory:summary:jobs:completed:{idempotency_key}`
- `worker:memory:summary:dlq:alerted`

## Queue Model

The summary queue uses Redis Streams.

Main stream:

- summary jobs are enqueued by the app

Consumer group:

- workers read via a Redis consumer group

Worker behavior:

- read pending work
- process summary job
- write back summary to memory
- acknowledge stream entry

## Idempotency And Replay Protection

Each summary job includes a deterministic `idempotency_key`.

The worker uses Redis markers to prevent duplicate logical work:

- `processing` marker
- `completed` marker

Current TTLs:

- processing marker: `300` seconds
- completed marker: `86400` seconds

## Dead-Letter Queue

Failed jobs that exceed max attempts are pushed into the DLQ stream.

The DLQ is monitored by:

- immediate alert check on DLQ insertion
- periodic worker-side checks

## Operational Notes

- old unprefixed queue entries are separate from the new worker namespace stream
- old short-term memory keys are still readable through a legacy fallback
- if you change namespaces in production, plan migration for historical queue data if needed

## Recommended Redis ACL Shape

At the infrastructure level, define separate ACL users for:

- app runtime
- worker runtime

Goal:

- app user can read and write app keys and enqueue summary jobs
- worker user can read and manage worker queue keys and the worker-side operational keys

If you want strict production ACLs, document exact command categories per runtime before rollout.
