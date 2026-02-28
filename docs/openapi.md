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

### `POST /api/v1/chat`

Primary user-facing endpoint for scoped academic discovery requests.

Purpose:

- Accept a user prompt
- Enforce authentication and authorization
- Apply input and context guardrails
- Read short-term memory
- Call the LLM
- Update short-term memory
- Queue async summarization when memory exceeds the threshold

Request body:

```json
{
  "user_id": "user-1",
  "prompt": "Find AI research labs at Stanford University"
}
```

Request validation:

- `user_id`
  - required
  - length `3..128`
  - pattern: `^[A-Za-z0-9_.:@\\-]+$`
- `prompt`
  - required
  - length `1..8000`

Response body:

```json
{
  "response": "..."
}
```

Response validation:

- `response`
  - required
  - length `1..12000`

Security:

- Requires a bearer token unless `auth_enabled` is disabled
- Caller can access only their own `user_id`, unless they have an admin role

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
