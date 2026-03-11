# UniGraph Documentation

This folder documents the current UniGraph backend as it exists in the codebase.

Files in this folder:

- `deployment-ec2.md`: production deployment runbook for Docker Compose on EC2 (security, scaling, health checks).
- `cicd-github-actions.md`: GitHub Actions CI/CD setup (OIDC, ECR push, EC2 deploy, required secrets/vars).
- `openapi.md`: HTTP API surface, routes, request and response contracts, and OpenAPI usage.
- `security.md`: authentication, authorization, guardrails, rate limiting, and data protection.
- `caching.md`: response cache behavior, key structure, and operational considerations.
- `short-term-memory.md`: short-term memory lifecycle, compaction, summarization, and async updates.
- `long-term-memory.md`: long-term retrieval architecture, pgvector experiments, and HNSW rationale.
- `redis.md`: Redis topology, namespaces, clients, key patterns, and queue usage.
- `async-io-path.md`: hot-path async I/O architecture, dependency limiters, async Redis/Postgres, and Bedrock executor isolation.
- `ops.md`: operational status endpoint and the metrics it exposes.
- `evaluation-pipeline.md`: end-to-end offline evaluation architecture (evidence capture, judge prompts, scoring, storage, reports).
- `async-chat-sqs.md`: async chat queue architecture (SQS enqueue, worker consumption, DynamoDB result store, APIs).
- `system-overview.md`: concise end-to-end architecture, Bedrock-based generation/retrieval flow, security/scalability, and latest performance snapshot.
- `strategy.md`: current architecture state and the recommended next build steps.

Recommended reading order:

1. `deployment-ec2.md`
2. `cicd-github-actions.md`
3. `openapi.md`
4. `security.md`
5. `short-term-memory.md`
6. `long-term-memory.md`
7. `redis.md`
8. `ops.md`
9. `async-io-path.md`
10. `evaluation-pipeline.md`
11. `system-overview.md`
12. `strategy.md`
