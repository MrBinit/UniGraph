# UniGraph System Overview

## 1) End-to-End Request Flow
Queue-aware runtime order (API + worker path):

`User Query`
-> `POST /api/v1/chat`
-> `SQS LLM Queue (unigraph-llm-jobs)`
-> `LLM Worker Dequeue`
-> `Input Guardrails`
-> `Sanitized Prompt`
-> `Response Cache Lookup`
-> `Short-Term Memory Build (summary + recent turns + new query)`
-> `Long-Term Retrieval (Bedrock embedding + pgvector top_k=2)`
-> `LLM Message Assembly (chat prompt + retrieved context + short-term context)`
-> `Context Guardrails`
-> `LLM Call (AWS Bedrock primary, fallback on failure)`
-> `Output Guardrails`
-> `Short-Term Memory Update`
-> `Response Cache Write`
-> `Evaluation Trace (background async task, non-blocking)`
-> `Metrics Persistence (JSON + DynamoDB request row)`
-> `SQS Metrics Aggregation Queue (unigraph-metrics-aggregation-jobs, optional)`
-> `SQS Evaluation Queue (unigraph-evaluation-jobs, optional, success-only request_id events)`
-> `Offline Judge Evaluation Worker`

Memory-first view:
`Query` -> `LLM Job Queue` -> `Worker` -> `Short-Term Memory` -> `Long-Term Memory` -> `LLM` -> `Metrics/Evaluation Queues`.

Interactive streaming path (direct, non-queued):
`POST /api/v1/chat/stream` -> `Guardrails + Memory + Retrieval` -> `Bedrock token stream` -> `SSE chunk events`.

## 2) Core Architecture
- Domain: AI backend for university/research discovery.
- Chunking: Recursive markdown chunking with semantic separators and overlap.
- Embeddings: Amazon Bedrock Titan (`amazon.titan-embed-text-v2:0`, 1024 dims).
- Generation: AWS Bedrock Anthropic profiles:
  - primary: `us.anthropic.claude-3-5-sonnet-20240620-v1:0`
  - fallback: `us.anthropic.claude-3-haiku-20240307-v1:0`
- Long-term memory: PostgreSQL + pgvector (`document_chunks`), HNSW default.
- Short-term memory: Redis (encrypted payloads, TTL, recent-window + summary strategy).
- Queue: Redis Streams consumer group for async summary compaction.
- Queue (SQS):
  - `unigraph-llm-jobs` for async chat request admission and worker fan-out
  - `unigraph-metrics-aggregation-jobs` for aggregate sync
  - `unigraph-evaluation-jobs` for per-request offline judge
  - DLQs: `unigraph-llm-jobs-dlq`, `unigraph-metrics-aggregation-jobs-dlq`, `unigraph-evaluation-jobs-dlq`
- Caching:
  - `app:cache:chat:{user_id}:{sanitized_prompt}`
  - embedding cache
- Current model path: retrieval embedding + generation both run on AWS Bedrock.

## 3) DynamoDB Metrics Storage (AWS)
Request and aggregate metrics are persisted to DynamoDB in addition to JSON files.

- Requests table: `unigraph-chat-metrics-requests`
- Aggregate table: `unigraph-chat-metrics-aggregate`
- TTL: `APP_METRICS_DYNAMODB_TTL_DAYS` (currently 30 days)

Per-request top-level attributes include:
- `request_id`, `timestamp`, `user_id`, `session_id`
- `query`, `outcome`
- `latency_overall_ms`, `latency_llm_ms`, `retrieval_strategy`
- `prompt_tokens`, `total_tokens`
- `record_json` (full JSON payload)

Offline evaluation results are stored in a separate table keyed by `request_id`:
- Evaluations table: `unigraph-chat-evaluations`
- Judge model: Amazon Nova 2 Lite (`us.amazon.nova-2-lite-v1:0`)
- Stored fields include `clarity_score`, `relevance_score`, `hallucination_score`, `evidence_similarity_score`, `answered_question`, `failure_reason`, and `overall_score`.
- Queue mode supports event-driven evaluation by `request_id` (instead of only scan scheduling).

## 4) Guardrails, Security, and Scalability
- Input/context/output guardrails enabled.
- JWT authentication (`HS256`) enabled.
- Distributed rate limiting + backpressure via Redis.
- Local backpressure fallback uses an atomic lock+counter gate (no private semaphore internals), so local admission/rejection is race-safe under concurrency.
- Circuit breakers are wired on Bedrock model generation (per model id) and embedding calls to fail fast during repeated downstream outages.
- Timeout middleware enabled for request protection.
- Secrets Manager integration for sensitive runtime values.
- TLS for ElastiCache and hardened production containers.
- API/worker services are stateless and horizontally scalable.
- Startup backend warmup added: Redis ping + Postgres pool query.
- Gradio chat UI isolates short-term memory per browser session using a generated session user id (`gradio-session-<uuid>`), persisted in `gr.State` for that session.

## 5) Token and Context Budgeting (Current)
- Retrieval `top_k=2`.
- `max_context_messages=24`.
- `max_input_chars=6000`.
- Memory budgets:
  - `summary_trigger=2200`
  - `default_soft_token_budget=1800`
  - `default_hard_token_budget=2600`
  - `min_recent_messages_to_keep=3`
  - `max_tokens=3200`

## 6) Data Scale
- Retrieval corpus was stress-tested with duplicated chunk data scaled to ~100,000 records in `document_chunks`.

## 7) Performance Snapshot

### A) Production aggregate snapshot (2026-03-10 UTC, n=20)
- Total requests: 20
- Success: 13
- Blocked input: 1
- Model error: 6 (during Bedrock IAM/access rollout)
- Avg overall latency: 10,757.6 ms
- Avg LLM latency: 11,175.3 ms
- Avg short-term memory latency: 7.9 ms
- Avg long-term memory latency: 105.8 ms
- Avg total tokens/request (with usage): 2,760.9

### B) Latest successful Bedrock request (2026-03-10 UTC)
- Overall latency: 3,159 ms
- LLM latency: 3,130 ms
- Long-term retrieval latency: 6 ms
- Short-term memory latency: 2 ms
- Memory update latency: 8 ms
- Evaluation trace latency in request path: 0 ms (background persistence active)

### C) Migration impact
- Generation runs on AWS Bedrock with Nova models for primary and fallback responses.
- After migration and latency tuning, successful request latency dropped sharply versus earlier baseline runs (for example, previous successful LLM latencies in this project were often in the ~18-30s range; latest successful Bedrock sample is ~3.1s).
- Current remaining risk is access/IAM rollout stability, not pipeline stage latency.

### D) Evaluation strategy shift
- Live request path no longer computes hallucination/clarity/relevance metrics.
- Quality scoring now runs asynchronously from stored request records and writes results to a separate DynamoDB evaluation table by `request_id`.
- Each request now stores a compact retrieval evidence snapshot (top retrieved chunks, ids, metadata, distances, trimmed content) and offline hallucination checks are grounded against that evidence.
- Offline evaluation uses three separate LLM-as-judge prompts (clarity, relevance, hallucination) with Nova 2 Lite.
- Daily reporting computes p50/p95, failure-reason distribution, and top low-score examples.
- Evaluation and metrics aggregation can run in queue-driven mode through dedicated SQS workers.
- Evaluation can be triggered two ways:
  - on-demand API run (`POST /api/v1/eval/offline/run?force=true`)
  - scheduled background run every 1 hour (configurable) that executes only when indexed `eval_status=pending` records exist.

## 8) Evaluation Pipeline (Detailed)

Reference: `docs/evaluation-pipeline.md`

Pipeline:

`request + retrieval evidence snapshot`
-> `DynamoDB requests table`
-> `evaluation event queue (optional) OR offline evaluator indexed pending query`
-> `judge(clarity)`
-> `judge(relevance)`
-> `judge(hallucination vs retrieval evidence)`
-> `aggregate scoring + failure reason`
-> `DynamoDB evaluations table`
-> `p50/p95 report`

Queue evaluation guarantee:
- The queue worker processes only requests with `eval_status=pending`, claims them by moving status to `in_progress`, runs evaluation, then marks them `completed`, so each request is evaluated only once.

Prompts and model:
- prompts file: `app/config/evaluation_prompt.yaml`
- model config: `app/config/evaluation_config.yaml` (`evaluation.judge_model_id`)
