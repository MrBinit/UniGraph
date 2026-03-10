# EC2 Deployment Guide (Docker Compose)

This is the production runbook for deploying UniGraph on a single EC2 host with Docker Compose.

It covers:

- secure production image usage
- required environment variables
- API/worker scaling strategy
- Redis TLS and distributed middleware controls
- health checks and post-deploy verification
- safe local Redis profile behavior

## Deployment Topology

Runtime services:

- `api`: FastAPI + Uvicorn
- `worker`: summary queue worker

Data services (recommended AWS managed):

- Redis: ElastiCache Valkey/Redis (TLS enabled)
- Postgres: RDS PostgreSQL

## 1) Host Preparation (EC2)

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
```

Clone the repository, then run all commands from project root.

## 2) Image and Dependency Model

The container image is production-hardened to install only runtime dependencies.

- `requirements-prod.txt`: runtime packages only
- `requirements-dev.txt`: test/lint/dev tooling
- `requirements.txt`: points to dev requirements for local development workflows

Docker builds use `requirements-prod.txt` only.

## 3) Required Environment Variables

This deployment flow does not rely on `.env` files. Export runtime variables in shell.

```bash
# Runtime process count (optional, defaults to 2)
export API_WORKERS=2

# ECR images
export APP_IMAGE=<aws-account-id>.dkr.ecr.us-east-1.amazonaws.com/unigraph-app:latest
export GRADIO_IMAGE=<aws-account-id>.dkr.ecr.us-east-1.amazonaws.com/unigraph-gradio:latest

# Secrets Manager source for sensitive keys
export AWS_SECRETS_MANAGER_SECRET_ID=unigraph/prod/app
export AWS_SECRETS_MANAGER_REGION=us-east-1
```

Secrets Manager option:

- Sensitive keys are loaded from AWS Secrets Manager.
- Set:
  - `AWS_SECRETS_MANAGER_SECRET_ID=unigraph/prod/app`
  - `AWS_SECRETS_MANAGER_REGION=us-east-1` (optional if region is already configured)
- Secret value must be JSON object keys matching expected env names (for example `AZURE_OPENAI_API_KEY`, `POSTGRES_PASSWORD`, `SECURITY_JWT_SECRET`, `MEMORY_ENCRYPTION_KEY`).
- App startup loads that secret once and maps keys into environment only when those env vars are not already set.

Metrics JSON notes:

- `APP_METRICS_JSON_ENABLED=true` writes per-request chat metrics and rolling aggregates to disk.
- `APP_METRICS_JSON_DIR` is resolved relative to project root unless absolute.
- Current files:
  - `<dir>/chat_metrics_requests.jsonl`
  - `<dir>/chat_metrics_aggregate.json`
- `docker-compose.prod.yml` mounts `./data/metrics:/app/data/metrics` so metrics are visible on host and survive container restarts/recreates.

Metrics DynamoDB notes:

- `APP_METRICS_DYNAMODB_ENABLED=true` enables DynamoDB metrics persistence.
- required config keys:
  - `APP_METRICS_DYNAMODB_REQUESTS_TABLE`
  - `APP_METRICS_DYNAMODB_AGGREGATE_TABLE`
  - `APP_METRICS_DYNAMODB_TTL_DAYS` (optional; set `0` to disable TTL stamping)
- recommended table setup:
  - requests table partition key: `request_id` (String)
  - aggregate table partition key: `id` (String), singleton item `id=global`
- enable DynamoDB TTL on `expires_at` when TTL days > 0.

Required IAM permission for instance role:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowDynamoMetricsWrites",
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem"
      ],
      "Resource": [
        "arn:aws:dynamodb:us-east-1:<account-id>:table/<requests-table>",
        "arn:aws:dynamodb:us-east-1:<account-id>:table/<aggregate-table>"
      ]
    }
  ]
}
```

Optional SQS queue mode (metrics aggregation + per-request evaluations):

- config keys:
  - `METRICS_AGGREGATION_QUEUE_ENABLED`
  - `METRICS_AGGREGATION_QUEUE_URL`
  - `EVALUATION_QUEUE_ENABLED`
  - `EVALUATION_QUEUE_URL`
- workers:
  - `python -m app.scripts.metrics_aggregation_worker`
  - `python -m app.scripts.eval_queue_worker`
- in compose:
  - `metrics-worker` profile: `metrics-queue`
  - `eval-worker` profile: `eval-queue`

Required SQS IAM permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowMetricsAndEvaluationQueues",
      "Effect": "Allow",
      "Action": [
        "sqs:SendMessage",
        "sqs:ReceiveMessage",
        "sqs:DeleteMessage",
        "sqs:ChangeMessageVisibility",
        "sqs:GetQueueAttributes",
        "sqs:GetQueueUrl"
      ],
      "Resource": [
        "arn:aws:sqs:us-east-1:<account-id>:<metrics-aggregation-queue>",
        "arn:aws:sqs:us-east-1:<account-id>:<evaluation-queue>"
      ]
    }
  ]
}
```

Gradio streaming notes:

- Gradio response streaming is enabled by default via chunked progressive rendering.
- optional tuning env vars:
  - `GRADIO_STREAM_CHUNK_SIZE` (default `120`)
  - `GRADIO_STREAM_CHUNK_DELAY_MS` (default `12`)

Proxy trust note:

- set `MIDDLEWARE_TRUSTED_PROXY_CIDRS` only to known proxy/load-balancer CIDRs
- if unset, `X-Forwarded-For` is ignored to prevent spoofing

## 4) Pre-Deploy Validation

Validate production Compose:

```bash
docker compose -f docker-compose.prod.yml config
```

## 5) Deploy

Pull and start core services:

```bash
docker compose -f docker-compose.prod.yml pull api worker
docker compose -f docker-compose.prod.yml up -d api worker
```

Tail logs:

```bash
docker compose -f docker-compose.prod.yml logs -f api worker
```

## 6) Scaling Guidance

Single EC2 host API scaling (vertical process scaling):

- increase `API_WORKERS` (for example `2`, `4`, `6` depending on vCPU and memory)

Worker scaling (horizontal on same host):

```bash
docker compose -f docker-compose.prod.yml up -d --scale worker=2
```

API horizontal scaling:

- use multiple EC2 instances behind an ALB target group
- keep per-instance `API_WORKERS` tuned to host size

## 7) Health and Smoke Tests

Health endpoint (unauthenticated):

```bash
curl -fsS http://127.0.0.1:8000/healthz
```

Expected response:

```json
{"status":"ok"}
```

Redis connectivity from API container:

```bash
docker compose -f docker-compose.prod.yml exec api python -c "from app.infra.redis_client import app_redis_client; print(app_redis_client.ping())"
```

Expected output: `True`

## 8) Production Security Checklist

- `APP_DOCS_ENABLED=false` in production
- `SECURITY_JWT_SECRET` set to strong random value
- `MEMORY_ENCRYPTION_KEY` set and different from JWT secret
- no plaintext secrets in repo or compose YAML
- security groups restrict:
  - inbound `8000` to ALB or trusted CIDR only
  - Redis/Postgres access only from app nodes

## 9) Optional Local Redis Profile (Non-Production)

Start local Redis + app stack:

```bash
docker compose --profile local-redis up -d redis api worker
```

Local profile safety:

- Redis port binds only to localhost: `127.0.0.1:6379:6379`
- do not use `local-redis` profile on production EC2

Suggested local overrides:

```bash
REDIS_APP_HOST=redis
REDIS_WORKER_HOST=redis
REDIS_APP_TLS=false
REDIS_WORKER_TLS=false
APP_DOCS_ENABLED=true
```

## 10) Rollback

If deployment is unhealthy:

```bash
docker compose -f docker-compose.prod.yml logs --tail 200 api worker
docker compose -f docker-compose.prod.yml down
```

Revert to previous version and redeploy:

```bash
docker compose -f docker-compose.prod.yml up -d api worker
```
