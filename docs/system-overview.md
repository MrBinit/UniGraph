# Overview UniGraph System

## 1) End-to-End Request Flow
Verified runtime order (`app/services/llm_service.py`):

`User Query`
-> `Input Guardrails`
-> `Sanitized Prompt`
-> `Response Cache Lookup`
-> `Short-Term Memory Build (summary + recent turns + new query)`
-> `Long-Term Retrieval (embedding + pgvector top_k=3)`
-> `LLM Message Assembly (chat system prompt + retrieved context + short-term context)`
-> `Context Guardrails`
-> `LLM Call (primary, then fallback if needed)`
-> `Output Guardrails`
-> `Short-Term Memory Update`
-> `Response Cache Write`
-> `Metrics/Trace Persistence`

Memory-first view (cache miss path):
`Query` -> `Short-Term Memory` -> `Long-Term Memory` -> `LLM`.

## 2) Guardrails (Implemented)
- **Input guardrails**:
  - scope enforcement (universities/labs/courses/professors/research in Germany),
  - blocked safety regexes,
  - `max_input_chars=8000`,
  - sensitive-data redaction.
- **Context guardrails**:
  - removes/sanitizes prompt-injection patterns,
  - prepends policy system message,
  - caps message count at `max_context_messages=60`.
- **Output guardrails**:
  - blocks unsafe output patterns,
  - redacts sensitive data,
  - truncates to `max_output_chars=8000`.

## 3) Core Architecture
- **Domain**: AI backend for university/research discovery.
- **Chunking**: Recursive Markdown chunking with semantic separators (`\n\n`, `\n`, `.`, space), fixed windows, overlap, and tiny-chunk merge.
- **Embedding model**: Amazon Bedrock Titan text embeddings (`amazon.titan-embed-text-v2:0`, `1024` dimensions) with TTL-based embedding cache.
- **Generation models**: Azure OpenAI `gpt-5.2-chat` (primary) and `gpt-4o-mini` (fallback).
- **Long-term memory**: PostgreSQL + `pgvector` (`document_chunks`) with tested strategies:
  - Seq Scan (exact baseline)
  - IVFFlat
  - HNSW (selected default for best latency/quality tradeoff)
  - Runtime retrieval uses **`top_k=3`** (config default).
- **Short-term memory (hybrid)**: Redis chat memory with TTL + encrypted payloads, combining recent-message window and rolling summary (async compaction).
  - Hybrid memory strategy: **recent-window + rolling-summary + token-budget truncation + async summary queue**.
  - Redis key: `app:memory:chat:{user_id}`
  - TTL: `3600s`
  - optimistic concurrency for update safety.
- **Queue**: Redis Streams + consumer group (`memory-summary-workers`) with idempotency keys, retry handling, and DLQ alerting.
- **Caching**:
  - User response cache: `app:cache:chat:{user_id}:{sanitized_prompt}`
  - Embedding cache
  - TTL-based for repeated-query cost reduction
- **Model resilience**:
  - primary/fallback routing (`gpt-5.2-chat` -> `gpt-4o-mini`),
  - circuit-breaker config present (`fail_max=5`, `reset_timeout=60s`).

## 4) Token & Context Budgeting
- Context token estimator uses `cl100k_base` (`tiktoken`) with fallback approximation.
- Short-term memory budgets:
  - `summary_trigger=3000`
  - soft budget `2800`
  - hard budget `3600`
  - `min_recent_messages_to_keep=4`
  - `max_tokens=4000` (memory config ceiling).
- Summary compaction ratio: `0.7` (oldest slice summarized asynchronously).
- Current observed token usage (successful requests):
  - average prompt tokens: `2,489`
  - average completion tokens: `1,169.8`
  - average total tokens: `3,658.8`

## 5) Security & Operations
- JWT authentication (`HS256`, issuer `ai-system`, token expiry `60 min`)
- Input/context/output guardrails
- Request timeout middleware: `35s` (returns `504` on timeout)
- Redis-backed distributed rate limiting: `120 requests / 60s` (with local fallback)
- Distributed backpressure gate: max in-flight `200`, lease `45s` (returns `503` when busy)
- AWS Secrets Manager for sensitive values
- TLS for ElastiCache connections
- Hardened containers in prod (non-root, read-only filesystem, `no-new-privileges`)

## 6) Scalability Notes
- Stateless API/worker containers scale horizontally behind Docker/EC2 orchestration.
- LLM call concurrency is bounded (`max_concurrency=50`) to protect upstream model endpoints.
- Embedding ingestion concurrency is bounded (`max_concurrency=4`) to avoid overload.
- Summary pipeline is decoupled through Redis Streams worker group (`memory-summary-workers`).

## 7) Deployment
Dockerized services -> ECR images -> EC2 runtime
with IAM role, Secrets Manager, ElastiCache, and RDS.

## 8) Data Scale
- Ingestion and retrieval were stress-tested with duplicated corpus/chunk data scaled to **~100,000 records** in `document_chunks` to validate index behavior and latency under larger load.

## 9) Current Metrics Snapshot
- Total requests: **7**
- Success: **6 (85.7%)**
- Blocked input: **1 (14.3%)**
- Avg overall latency: **22,118 ms (22.1s)**
- Max overall latency: **28,805 ms**
- Avg LLM latency: **25,721 ms (25.7s)**
- Avg short-term memory latency: **18.2 ms**
- Avg long-term memory latency: **24.7 ms**
- Avg cache read/write latency: **1.3 ms / 1.0 ms**
- Avg tokens per successful request: **3,658.8**
  (`2,489` prompt + `1,169.8` completion)
