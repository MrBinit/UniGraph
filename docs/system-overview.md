# UniGraph System Overview

## 1) End-to-End Request Flow
Verified runtime order (`app/services/llm_service.py`):

`User Query`
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
-> `Metrics Persistence (JSON + DynamoDB)`

Memory-first view:
`Query` -> `Short-Term Memory` -> `Long-Term Memory` -> `LLM`.

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

## 4) Guardrails, Security, and Scalability
- Input/context/output guardrails enabled.
- JWT authentication (`HS256`) enabled.
- Distributed rate limiting + backpressure via Redis.
- Timeout middleware enabled for request protection.
- Secrets Manager integration for sensitive runtime values.
- TLS for ElastiCache and hardened production containers.
- API/worker services are stateless and horizontally scalable.
- Startup backend warmup added: Redis ping + Postgres pool query.

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
- Generation was previously run on Azure OpenAI (explored first), and was shifted to AWS Bedrock to address high response latency.
- After migration and latency tuning, successful request latency dropped sharply versus earlier baseline runs (for example, previous successful LLM latencies in this project were often in the ~18-30s range; latest successful Bedrock sample is ~3.1s).
- Current remaining risk is access/IAM rollout stability, not pipeline stage latency.
