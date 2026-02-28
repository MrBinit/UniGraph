# UniGraph Documentation

This folder documents the current UniGraph backend as it exists in the codebase.

Files in this folder:

- `openapi.md`: HTTP API surface, routes, request and response contracts, and OpenAPI usage.
- `security.md`: authentication, authorization, guardrails, rate limiting, and data protection.
- `caching.md`: response cache behavior, key structure, and operational considerations.
- `short-term-memory.md`: short-term memory lifecycle, compaction, summarization, and async updates.
- `redis.md`: Redis topology, namespaces, clients, key patterns, and queue usage.
- `ops.md`: operational status endpoint and the metrics it exposes.
- `strategy.md`: current architecture state and the recommended next build steps.

Recommended reading order:

1. `openapi.md`
2. `security.md`
3. `short-term-memory.md`
4. `redis.md`
5. `ops.md`
6. `strategy.md`
