# Local Load Testing (No Model Cost)

This load-test setup runs on your laptop and skips external model calls.

## What is mocked

- Chat model calls are mocked via `LLM_MOCK_MODE=true`.
- Retrieval/embedding path is skipped via `RETRIEVAL_DISABLED=true`.

All other app paths still run:
- auth/JWT
- request validation
- guardrails
- memory read/write in Redis
- caching
- middleware
- metrics bookkeeping

## Recommended settings for MacBook Air M2 (8GB RAM)

- `API_WORKERS=1`
- `CONCURRENCY=16` to `24`
- `REQUESTS=200` to `500`
- Increase slowly after baseline.

## One-command run

From repo root:

```bash
chmod +x loadtest/run_local_mock_chat_load.sh
LOCAL_REDIS_PORT=6380 REQUESTS=300 CONCURRENCY=24 API_WORKERS=1 ./loadtest/run_local_mock_chat_load.sh
```

## Custom run (manual)

1. Start Redis:

```bash
docker compose -f docker-compose.yml --profile local-redis up -d redis
```

2. Start API with safe local flags:

```bash
export SECURITY_JWT_SECRET='01234567890123456789012345678901'
export MEMORY_ENCRYPTION_KEY='abcdefghijklmnopqrstuvwxyz1234567890AB'
export POSTGRES_ENABLED=false
export LLM_ASYNC_ENABLED=false
export METRICS_AGGREGATION_QUEUE_ENABLED=false
export EVALUATION_QUEUE_ENABLED=false
export EVALUATION_ENABLED=false
export EVALUATION_SCHEDULE_ENABLED=false
export LLM_MOCK_MODE=true
export RETRIEVAL_DISABLED=true
export LLM_MOCK_DELAY_MS=80
export LLM_MOCK_STREAM_CHUNK_CHARS=16
export REDIS_APP_HOST=127.0.0.1
export REDIS_WORKER_HOST=127.0.0.1
export REDIS_APP_TLS=false
export REDIS_WORKER_TLS=false
export REDIS_APP_SSL_CERT_REQS=none
export REDIS_WORKER_SSL_CERT_REQS=none

./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

3. In another terminal, generate token:

```bash
TOKEN="$(./venv/bin/python - <<'PY'
from app.core.security import create_access_token
print(create_access_token(user_id='load-user', roles=['admin']))
PY
)"
```

4. Run load:

```bash
./venv/bin/python loadtest/http_load.py \
  --url http://127.0.0.1:8000/api/v1/chat/stream \
  --mode chat-stream \
  --token "$TOKEN" \
  --requests 300 \
  --concurrency 24 \
  --timeout 20
```

The tool prints:
- throughput (RPS)
- p50/p95/p99 latency
- status counts
- error rate

## AWS Full-Stack Mode (SQS + Postgres + DynamoDB, No LLM/Embedding API)

Use this mode when you want AWS-backed services involved during load:
- `/api/v1/chat/stream` queue-backed SSE path
- SQS workers (`llm_async_worker`, `metrics_aggregation_worker`)
- Postgres retrieval path
- DynamoDB job/metrics writes
- Redis memory + summary queue

Model API calls stay mocked via:
- `LLM_MOCK_MODE=true`
- `EMBEDDING_MOCK_MODE=true`
- `MEMORY_SUMMARY_MOCK_MODE=true`

Files:
- `loadtest/aws_full_stack_load.yaml`
- `loadtest/run_aws_full_stack_load.sh`
- `loadtest/aws_async_queue_load.py`

Change user count in YAML:

```yaml
workload:
  users: 100
```

Scale async workers in YAML:

```yaml
runtime:
  llm_async_workers: 6
  metrics_workers: 3
  wait_for_metrics_drain_seconds: 120
```

Run:

```bash
chmod +x loadtest/run_aws_full_stack_load.sh
./loadtest/run_aws_full_stack_load.sh loadtest/aws_full_stack_load.yaml
```

Requirements:
- Valid AWS credentials in shell.
- Working Postgres credentials for your configured DB (`POSTGRES_PASSWORD` or Secrets Manager path).
- Existing configured AWS resources (SQS queues + DynamoDB tables).

Tip: if your shared queue already has traffic, set dedicated queue URLs in YAML (`runtime.llm_queue_url`, `runtime.metrics_queue_url`) to isolate load-test results.

Note: `evaluation_queue_enabled` is `false` in the YAML because that worker uses an LLM-judge model.
