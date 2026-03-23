# UniGraph Documentation

This folder documents the current UniGraph backend as it exists in the codebase.

Files in this folder:

- `deployment-ec2.md`: production deployment runbook for Docker Compose on EC2 (security, scaling, health checks).
- `cicd-github-actions.md`: GitHub Actions CI/CD setup (OIDC, ECR push, EC2 deploy, required secrets/vars).
- `sonarqube.md`: local SonarQube setup, scan commands, CI quality-gate integration, and troubleshooting.
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
- `fastapi-architecture.md`: FastAPI component diagram (middleware, routers, services, workers, and data stores).
- `load-testing-aws.md`: YAML-driven AWS full-stack load testing (SQS, Postgres, DynamoDB, Redis, workers, result interpretation).
- `system-overview.md`: concise end-to-end architecture, Bedrock-based generation/retrieval flow, security/scalability, and latest performance snapshot.
- `website-search.md`: SerpAPI-based website fallback architecture, citation-grounded answering, and web-fallback quality tracking.
- `strategy.md`: current architecture state and the recommended next build steps.

Recommended reading order:

1. `deployment-ec2.md`
2. `cicd-github-actions.md`
3. `sonarqube.md`
4. `openapi.md`
5. `security.md`
6. `short-term-memory.md`
7. `long-term-memory.md`
8. `redis.md`
9. `ops.md`
10. `async-io-path.md`
11. `evaluation-pipeline.md`
12. `async-chat-sqs.md`
13. `fastapi-architecture.md`
14. `load-testing-aws.md`
15. `system-overview.md`
16. `website-search.md`
17. `strategy.md`
