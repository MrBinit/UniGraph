# OpenAPI And API Surface

## Framework

UniGraph is a FastAPI application.

FastAPI automatically exposes:

- `/openapi.json`: machine-readable OpenAPI schema
- `/docs`: Swagger UI
- `/redoc`: ReDoc UI

The app title is loaded from `app/config/app_config.yaml`.

## Base Path

All application routes are mounted under:

- `/api/v1`

## Endpoints

### `POST /api/v1/auth/login`

Password login endpoint that returns a bearer token.
Users must be configured through `SECURITY_LOGIN_USERS_JSON`.

Request body:

```json
{
  "username": "alice",
  "password": "alice-pass"
}
```

Response body (example):

```json
{
  "access_token": "<jwt>",
  "token_type": "bearer",
  "user_id": "alice@example.com",
  "roles": ["user", "admin"],
  "expires_in_seconds": 3600
}
```

### `POST /api/v1/chat/stream`

Queue-backed streaming endpoint over Server-Sent Events (SSE).

Purpose:

- Accept a user prompt
- Enforce authentication and authorization
- Enqueue an async chat job to SQS
- Poll DynamoDB job status
- Stream queue status plus final response as SSE events

Request body:

```json
{
  "user_id": "user-1",
  "prompt": "Find AI research labs at Stanford University"
}
```

Response content type:

- `text/event-stream`

SSE event examples:

```text
data: {"type":"queued","job_id":"4d7a9b6d6a5b4cf7b3ef31e3f3468b0b","status":"queued","submitted_at":"2026-03-10T10:30:00+00:00"}

data: {"type":"status","job_id":"4d7a9b6d6a5b4cf7b3ef31e3f3468b0b","status":"processing"}

data: {"type":"status","job_id":"4d7a9b6d6a5b4cf7b3ef31e3f3468b0b","status":"completed"}

data: {"type":"chunk","text":"Find AI research labs at Stanford University..."}

data: {"type":"done"}
```

### `GET /api/v1/chat/{job_id}`

Read async job status and final result.

Response body (example):

```json
{
  "job_id": "4d7a9b6d6a5b4cf7b3ef31e3f3468b0b",
  "user_id": "user-1",
  "session_id": "user-1",
  "status": "completed",
  "submitted_at": "2026-03-10T10:30:00+00:00",
  "started_at": "2026-03-10T10:30:01+00:00",
  "completed_at": "2026-03-10T10:30:04+00:00",
  "response": "...",
  "error": ""
}
```

Status values:

- `queued`
- `processing`
- `completed`
- `failed`

### `GET /api/v1/ops/status`

Admin-only operational health endpoint.

Purpose:

- Report app Redis availability
- Report queue depth and pending jobs
- Report DLQ depth and latest DLQ error
- Report memory compaction counters
- Report request latency counters

Security:

- Requires a bearer token
- Requires an admin role

Response shape:

```json
{
  "status": "ok",
  "memory": {
    "redis_available": true,
    "ttl_seconds": 3600,
    "encryption_enabled": true
  },
  "queue": {
    "stream_depth": 0,
    "pending_jobs": 0,
    "dlq_depth": 0,
    "consumer_group": "memory-summary-workers",
    "last_dlq_error": ""
  },
  "compaction": {
    "events": 0,
    "removed_messages": 0,
    "removed_tokens": 0
  },
  "latency": {
    "count": 0,
    "average_ms": 0.0,
    "max_ms": 0,
    "last_ms": 0,
    "last_outcome": ""
  }
}
```

## Middleware Stack

The app can enable or disable middleware from config.

Current middleware responsibilities:

- route matching: returns a structured 404 for unknown paths
- backpressure: limits in-flight concurrent requests
- timeout: aborts long-running requests
- rate limit: limits requests by `user_id + IP + path`
- request logging: logs request-level visibility

## OpenAPI Notes

FastAPI generates the OpenAPI schema from:

- route decorators
- Pydantic request and response schemas
- dependency declarations

If you add a new endpoint or change a schema, `/openapi.json` updates automatically.
