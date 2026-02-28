# Short-Term Memory

## Purpose

Short-term memory keeps recent conversation state so UniGraph can answer with continuity while staying within token limits.

It is designed for:

- recent conversation context
- user continuity across requests
- bounded token growth
- asynchronous summarization under load

It is not designed to be permanent memory.

## Storage Model

Per-user memory is stored in Redis under:

- `app:memory:chat:{user_id}`

Legacy compatibility:

- the loader still checks old keys in the form `chat:{user_id}`

Stored fields:

- `summary`
- `messages`
- `version`
- `next_seq`
- `last_summarized_seq`
- `summary_pending`
- `last_summary_job_id`

The payload is encrypted before Redis write and decrypted on read.

## Memory Lifecycle

### Read

On a new chat request:

1. load memory from Redis
2. decrypt and normalize it
3. fall back to an empty structure if Redis is unavailable or payload is invalid

### Context Build

The app builds context from:

- optional summary
- recent messages
- current user message

The main goal is to fit the context within per-user budgets.

### Write

After a successful model response:

1. append the user message
2. append the assistant reply
3. increment version
4. persist the updated encrypted memory

## Token Budgets

Per-user memory uses:

- soft limit
- hard limit
- minimum recent messages to keep

Resolution order:

1. user-specific override if configured
2. default memory config values otherwise

Behavior:

- soft limit: trims oldest messages while preserving recent context
- hard limit: trims more aggressively and can remove summary text if still oversized

## Compaction And Metrics

When truncation happens, the system records:

- event count
- removed message count
- removed token count
- before and after token counts
- summary quality hints

Metrics are written to Redis and also logged.

## Async Summarization

When pre-compaction token count exceeds `summary_trigger`:

1. the app selects older messages based on `summary_ratio`
2. it enqueues a summary job
3. it marks memory as `summary_pending`
4. a background worker processes the job later
5. the worker writes a merged summary back into memory

This avoids blocking the foreground request on summarization work.

## Queue Safety

The async summarization path includes:

- idempotency keys
- in-progress markers
- completed markers
- retry handling
- dead-letter queue support

This is important because Redis Streams are effectively at-least-once delivery.

## Design Intent

The short-term layer is optimized for:

- bounded recent context
- cost control
- resilience under retries and worker restarts

The long-term layer should later store durable facts, not this raw rolling chat state.
