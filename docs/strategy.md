# Current Strategy And What To Build Next

## Current State

The current backend is in a solid pre-long-term-memory state.

It already has:

- scoped API routes
- JWT auth and authorization
- guardrails
- hybrid rate limiting
- backpressure and timeout control
- short-term memory with token budgets
- async summarization queue and worker
- replay protection for summary jobs
- dead-letter queue with alerting
- Redis namespace separation for app and worker
- ops visibility through `/api/v1/ops/status`

This is enough to move to the next stage.

## Recommended Next Priority

Move to long-term memory now.

The main reason:

- short-term memory is already bounded and operationally visible
- the next architectural value comes from durable retrieval, not from adding more complexity to the short-term layer

## Recommended Order

### 1. Build Long-Term Memory

Recommended first implementation:

- Postgres + `pgvector`

Why:

- you want to learn scalable systems
- you want to learn secure systems
- you want to learn connection pooling
- Postgres teaches more operationally than jumping directly to a standalone vector store

What to store:

- durable user preferences
- target universities
- desired countries or regions
- degree type
- lab interests
- professor interests
- confirmed constraints

What not to store:

- raw rolling chat history
- transient prompt text that has not been normalized into facts

### 2. Add Fact Extraction

Write durable memory only from:

- trusted summary output
- explicit user confirmations
- structured preference extraction

This keeps long-term memory cleaner than raw conversation dumps.

### 3. Add Retrieval Merge

Before model call, merge:

- system prompt
- retrieved long-term facts
- short-term summary
- recent short-term message window
- current user prompt

### 4. Evaluate Retrieval Quality

Measure:

- whether relevant facts are retrieved
- whether irrelevant facts pollute context
- whether answers improve with retrieval

## What Not To Do Yet

Do not move to tool calling or MCP before long-term retrieval is reliable.

Reason:

- tool calling on top of weak memory only increases complexity
- retrieval quality is the stronger foundation for the next stage

## Immediate Practical Roadmap

Recommended next implementation sequence:

1. add Postgres configuration and connection pooling
2. add a long-term memory repository abstraction
3. add a fact extraction service
4. add `pgvector` embeddings and similarity retrieval
5. merge top-K long-term memories into the request context
6. add tests for write rules, retrieval ranking, and context merge behavior

## Documentation Use

Use this docs folder as the baseline for the current short-term-memory architecture.

When long-term memory lands, update:

- `openapi.md` if new endpoints are added
- `redis.md` if queue or caching behavior changes
- `strategy.md` with the new retrieval flow
- add a new `long-term-memory.md` document
