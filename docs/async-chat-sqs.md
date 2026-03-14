# Async Chat With SQS

This document describes the new async chat path:

1. `POST /api/v1/chat/stream` enqueues a job to SQS and opens an SSE stream.
2. Worker (`python -m app.scripts.llm_async_worker`) consumes SQS messages.
3. Worker runs the existing LLM pipeline and writes job status/result to DynamoDB.
4. Stream emits queued/processing/completed events and final chunk.
5. `GET /api/v1/chat/{job_id}` remains available for direct status polling.

## Required AWS resources

- SQS main queue (Standard): `unigraph-llm-jobs`
- SQS DLQ (Standard): `unigraph-llm-jobs-dlq`
- DynamoDB table for results: `unigraph-llm-results`
  - Partition key: `job_id` (String)
  - Billing mode: On-demand
  - TTL attribute: `expires_at` (optional but recommended)

## Required config/env

Set these values (via env or Secrets Manager):

- `LLM_ASYNC_ENABLED=true`
- `LLM_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/<account-id>/unigraph-llm-jobs`
- `LLM_DLQ_URL=https://sqs.us-east-1.amazonaws.com/<account-id>/unigraph-llm-jobs-dlq`
- `LLM_RESULT_TABLE=unigraph-llm-results`
- `LLM_RESULT_TTL_DAYS=7`
- `LLM_RECEIVE_WAIT_SECONDS=20`
- `LLM_MAX_MESSAGES_PER_POLL=5`
- `LLM_VISIBILITY_TIMEOUT_SECONDS=300`

## IAM permissions (EC2 role / worker role)

Allow SQS read/write on main + DLQ:

- `sqs:SendMessage`
- `sqs:ReceiveMessage`
- `sqs:DeleteMessage`
- `sqs:ChangeMessageVisibility`
- `sqs:GetQueueAttributes`
- `sqs:GetQueueUrl`

Allow DynamoDB job status operations on result table:

- `dynamodb:PutItem`
- `dynamodb:GetItem`
- `dynamodb:UpdateItem`

## Local/prod run

- stream enqueue + result: `POST /api/v1/chat/stream`
- optional status polling: `GET /api/v1/chat/{job_id}`

Run async worker (compose profile):

```bash
docker compose --profile llm-async up -d llm-worker
```

Or directly:

```bash
python -m app.scripts.llm_async_worker
```
