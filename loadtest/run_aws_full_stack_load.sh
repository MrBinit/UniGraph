#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${1:-loadtest/aws_full_stack_load.yaml}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[aws-load] config not found: ${CONFIG_PATH}"
  exit 1
fi

yaml_get() {
  ./venv/bin/python - "$CONFIG_PATH" "$1" <<'PY'
import sys
import yaml

config_path, dotted = sys.argv[1], sys.argv[2]
with open(config_path, "r", encoding="utf-8") as handle:
    payload = yaml.safe_load(handle) or {}

node = payload
for part in dotted.split("."):
    if not isinstance(node, dict):
        node = None
        break
    node = node.get(part)

if node is None:
    print("")
elif isinstance(node, bool):
    print("true" if node else "false")
else:
    print(str(node))
PY
}

USERS="$(yaml_get workload.users)"
CONCURRENCY="$(yaml_get workload.concurrency)"
REQUESTS_PER_USER="$(yaml_get workload.requests_per_user)"
API_PORT="$(yaml_get runtime.api_port)"
API_WORKERS="$(yaml_get runtime.api_workers)"
LLM_MOCK_DELAY_MS="$(yaml_get runtime.llm_mock_delay_ms)"
EMBEDDING_MOCK_DELAY_MS="$(yaml_get runtime.embedding_mock_delay_ms)"
MEMORY_SUMMARY_MOCK_DELAY_MS="$(yaml_get runtime.memory_summary_mock_delay_ms)"
EVALUATION_QUEUE_ENABLED="$(yaml_get runtime.evaluation_queue_enabled)"
LLM_ASYNC_WORKERS="$(yaml_get runtime.llm_async_workers)"
METRICS_WORKERS="$(yaml_get runtime.metrics_workers)"
WAIT_FOR_METRICS_DRAIN_SECONDS="$(yaml_get runtime.wait_for_metrics_drain_seconds)"
LLM_QUEUE_URL_OVERRIDE="$(yaml_get runtime.llm_queue_url)"
METRICS_QUEUE_URL_OVERRIDE="$(yaml_get runtime.metrics_queue_url)"
EVALUATION_QUEUE_URL_OVERRIDE="$(yaml_get runtime.evaluation_queue_url)"

USERS="${USERS:-100}"
CONCURRENCY="${CONCURRENCY:-25}"
REQUESTS_PER_USER="${REQUESTS_PER_USER:-1}"
API_PORT="${API_PORT:-18000}"
API_WORKERS="${API_WORKERS:-1}"
LLM_MOCK_DELAY_MS="${LLM_MOCK_DELAY_MS:-80}"
EMBEDDING_MOCK_DELAY_MS="${EMBEDDING_MOCK_DELAY_MS:-12}"
MEMORY_SUMMARY_MOCK_DELAY_MS="${MEMORY_SUMMARY_MOCK_DELAY_MS:-10}"
EVALUATION_QUEUE_ENABLED="${EVALUATION_QUEUE_ENABLED:-false}"
LLM_ASYNC_WORKERS="${LLM_ASYNC_WORKERS:-1}"
METRICS_WORKERS="${METRICS_WORKERS:-2}"
WAIT_FOR_METRICS_DRAIN_SECONDS="${WAIT_FOR_METRICS_DRAIN_SECONDS:-0}"

API_LOG_FILE="/tmp/ai-system-aws-load-api.log"
SUMMARY_WORKER_LOG_FILE="/tmp/ai-system-aws-load-summary-worker.log"
LLM_ASYNC_WORKER_LOG_FILE="/tmp/ai-system-aws-load-llm-async-worker.log"
METRICS_WORKER_LOG_FILE="/tmp/ai-system-aws-load-metrics-worker.log"
EVAL_WORKER_LOG_FILE="/tmp/ai-system-aws-load-eval-worker.log"

echo "[aws-load] config=${CONFIG_PATH}"
echo "[aws-load] users=${USERS} requests_per_user=${REQUESTS_PER_USER} concurrency=${CONCURRENCY}"
echo "[aws-load] llm_async_workers=${LLM_ASYNC_WORKERS}"
echo "[aws-load] metrics_workers=${METRICS_WORKERS}"
echo "[aws-load] wait_for_metrics_drain_seconds=${WAIT_FOR_METRICS_DRAIN_SECONDS}"

export SECURITY_JWT_SECRET="${SECURITY_JWT_SECRET:-01234567890123456789012345678901}"
export MEMORY_ENCRYPTION_KEY="${MEMORY_ENCRYPTION_KEY:-abcdefghijklmnopqrstuvwxyz1234567890AB}"
export REDIS_APP_TLS="${REDIS_APP_TLS:-true}"
export REDIS_WORKER_TLS="${REDIS_WORKER_TLS:-true}"
export REDIS_APP_SSL_CERT_REQS="${REDIS_APP_SSL_CERT_REQS:-required}"
export REDIS_WORKER_SSL_CERT_REQS="${REDIS_WORKER_SSL_CERT_REQS:-required}"

export POSTGRES_ENABLED="${POSTGRES_ENABLED:-false}"
export RETRIEVAL_DISABLED="${RETRIEVAL_DISABLED:-false}"
export LLM_MOCK_MODE="${LLM_MOCK_MODE:-true}"
export EMBEDDING_MOCK_MODE="${EMBEDDING_MOCK_MODE:-true}"
export MEMORY_SUMMARY_MOCK_MODE="${MEMORY_SUMMARY_MOCK_MODE:-true}"
export LLM_MOCK_DELAY_MS
export EMBEDDING_MOCK_DELAY_MS
export MEMORY_SUMMARY_MOCK_DELAY_MS

export LLM_ASYNC_ENABLED="${LLM_ASYNC_ENABLED:-true}"
export METRICS_AGGREGATION_QUEUE_ENABLED="${METRICS_AGGREGATION_QUEUE_ENABLED:-true}"
export EVALUATION_QUEUE_ENABLED
export APP_METRICS_DYNAMODB_ENABLED="${APP_METRICS_DYNAMODB_ENABLED:-true}"
export APP_METRICS_JSON_ENABLED="${APP_METRICS_JSON_ENABLED:-true}"
export EVALUATION_ENABLED="${EVALUATION_ENABLED:-false}"
export EVALUATION_SCHEDULE_ENABLED="${EVALUATION_SCHEDULE_ENABLED:-false}"

# Avoid load-test failures driven only by middleware throttles.
export MIDDLEWARE_ENABLE_RATE_LIMIT="${MIDDLEWARE_ENABLE_RATE_LIMIT:-false}"
export MIDDLEWARE_ENABLE_DISTRIBUTED_RATE_LIMIT="${MIDDLEWARE_ENABLE_DISTRIBUTED_RATE_LIMIT:-false}"

if [[ -n "${LLM_QUEUE_URL_OVERRIDE}" ]]; then
  export LLM_QUEUE_URL="${LLM_QUEUE_URL_OVERRIDE}"
fi
if [[ -n "${METRICS_QUEUE_URL_OVERRIDE}" ]]; then
  export METRICS_AGGREGATION_QUEUE_URL="${METRICS_QUEUE_URL_OVERRIDE}"
fi
if [[ -n "${EVALUATION_QUEUE_URL_OVERRIDE}" ]]; then
  export EVALUATION_QUEUE_URL="${EVALUATION_QUEUE_URL_OVERRIDE}"
fi

echo "[aws-load] preflight: AWS identity + queues/tables + postgres"
if ! ./venv/bin/python - <<'PY' >/tmp/ai-system-aws-load-preflight.log 2>&1
import os
import boto3
from app.core.config import get_settings

settings = get_settings()
region = (
    os.getenv("AWS_REGION", "").strip()
    or os.getenv("AWS_DEFAULT_REGION", "").strip()
    or os.getenv("AWS_SECRETS_MANAGER_REGION", "").strip()
    or None
)
kwargs = {"region_name": region} if region else {}

sts = boto3.client("sts", **kwargs)
identity = sts.get_caller_identity()
print("aws_account=", identity.get("Account"))

sqs = boto3.client("sqs", **kwargs)
for queue_url in [
    settings.queue.llm_queue_url.strip(),
    settings.queue.metrics_aggregation_queue_url.strip(),
]:
    if not queue_url:
        raise RuntimeError("Missing required queue url.")
    sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["QueueArn"])
    print("queue_ok=", queue_url)

ddb = boto3.client("dynamodb", **kwargs)
for table_name in [
    settings.queue.llm_result_table.strip(),
    settings.app.metrics_dynamodb_requests_table.strip(),
    settings.app.metrics_dynamodb_aggregate_table.strip(),
]:
    if not table_name:
        raise RuntimeError("Missing required DynamoDB table name.")
    ddb.describe_table(TableName=table_name)
    print("table_ok=", table_name)
PY
then
  echo "[aws-load] preflight failed:"
  tail -n 120 /tmp/ai-system-aws-load-preflight.log || true
  exit 1
fi
tail -n 40 /tmp/ai-system-aws-load-preflight.log || true

if [[ "$(printf '%s' "${POSTGRES_ENABLED}" | tr '[:upper:]' '[:lower:]')" == "true" ]]; then
  if ! ./venv/bin/python -m app.scripts.check_postgres >/tmp/ai-system-aws-load-postgres.log 2>&1; then
    echo "[aws-load] postgres connectivity failed:"
    tail -n 120 /tmp/ai-system-aws-load-postgres.log || true
    exit 1
  fi
  tail -n 10 /tmp/ai-system-aws-load-postgres.log || true
else
  echo "[aws-load] postgres connectivity check skipped (POSTGRES_ENABLED=false)"
fi

if lsof -nP -iTCP:"${API_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[aws-load] port ${API_PORT} is already in use; choose another API_PORT."
  exit 1
fi

echo "[aws-load] starting workers..."
REDIS_RUNTIME_ROLE=worker ./venv/bin/python -m app.scripts.summary_worker >"${SUMMARY_WORKER_LOG_FILE}" 2>&1 &
SUMMARY_WORKER_PID=$!
: >"${LLM_ASYNC_WORKER_LOG_FILE}"
LLM_ASYNC_WORKER_PIDS=()
for worker_index in $(seq 1 "${LLM_ASYNC_WORKERS}"); do
  REDIS_RUNTIME_ROLE=app ./venv/bin/python -m app.scripts.llm_async_worker \
    >>"${LLM_ASYNC_WORKER_LOG_FILE}" 2>&1 &
  LLM_ASYNC_WORKER_PIDS+=("$!")
done
: >"${METRICS_WORKER_LOG_FILE}"
METRICS_WORKER_PIDS=()
for worker_index in $(seq 1 "${METRICS_WORKERS}"); do
  REDIS_RUNTIME_ROLE=app ./venv/bin/python -m app.scripts.metrics_aggregation_worker \
    >>"${METRICS_WORKER_LOG_FILE}" 2>&1 &
  METRICS_WORKER_PIDS+=("$!")
done

EVAL_WORKER_PID=""
if [[ "$(printf '%s' "${EVALUATION_QUEUE_ENABLED}" | tr '[:upper:]' '[:lower:]')" == "true" ]]; then
  REDIS_RUNTIME_ROLE=app ./venv/bin/python -m app.scripts.eval_queue_worker >"${EVAL_WORKER_LOG_FILE}" 2>&1 &
  EVAL_WORKER_PID=$!
fi

echo "[aws-load] starting api on :${API_PORT} (workers=${API_WORKERS})..."
REDIS_RUNTIME_ROLE=app ./venv/bin/uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "${API_PORT}" \
  --workers "${API_WORKERS}" \
  >"${API_LOG_FILE}" 2>&1 &
API_PID=$!

cleanup() {
  if [[ -n "${API_PID:-}" ]] && kill -0 "${API_PID}" 2>/dev/null; then
    kill "${API_PID}" >/dev/null 2>&1 || true
    wait "${API_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SUMMARY_WORKER_PID:-}" ]] && kill -0 "${SUMMARY_WORKER_PID}" 2>/dev/null; then
    kill "${SUMMARY_WORKER_PID}" >/dev/null 2>&1 || true
    wait "${SUMMARY_WORKER_PID}" 2>/dev/null || true
  fi
  for llm_worker_pid in "${LLM_ASYNC_WORKER_PIDS[@]:-}"; do
    if [[ -n "${llm_worker_pid}" ]] && kill -0 "${llm_worker_pid}" 2>/dev/null; then
      kill "${llm_worker_pid}" >/dev/null 2>&1 || true
      wait "${llm_worker_pid}" 2>/dev/null || true
    fi
  done
  for metrics_worker_pid in "${METRICS_WORKER_PIDS[@]:-}"; do
    if [[ -n "${metrics_worker_pid}" ]] && kill -0 "${metrics_worker_pid}" 2>/dev/null; then
      kill "${metrics_worker_pid}" >/dev/null 2>&1 || true
      wait "${metrics_worker_pid}" 2>/dev/null || true
    fi
  done
  if [[ -n "${EVAL_WORKER_PID:-}" ]] && kill -0 "${EVAL_WORKER_PID}" 2>/dev/null; then
    kill "${EVAL_WORKER_PID}" >/dev/null 2>&1 || true
    wait "${EVAL_WORKER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[aws-load] waiting for api health..."
for _ in $(seq 1 45); do
  if ! kill -0 "${API_PID}" 2>/dev/null; then
    echo "[aws-load] api exited during startup."
    tail -n 120 "${API_LOG_FILE}" || true
    exit 1
  fi
  if curl --max-time 2 -fsS "http://127.0.0.1:${API_PORT}/healthz" >/dev/null; then
    break
  fi
  sleep 1
done
curl --max-time 2 -fsS "http://127.0.0.1:${API_PORT}/healthz" >/dev/null

TOKEN="$(
  ./venv/bin/python - <<'PY'
from app.core.security import create_access_token
print(create_access_token(user_id="load-admin", roles=["admin"]))
PY
)"

echo "[aws-load] running async queue load..."
./venv/bin/python loadtest/aws_async_queue_load.py --config "${CONFIG_PATH}" --token "${TOKEN}"

if [[ "${METRICS_AGGREGATION_QUEUE_ENABLED}" == "true" && "${WAIT_FOR_METRICS_DRAIN_SECONDS}" -gt 0 ]]; then
  echo "[aws-load] waiting for metrics queue drain..."
  ./venv/bin/python - "${WAIT_FOR_METRICS_DRAIN_SECONDS}" <<'PY'
import os
import sys
import time
import boto3
from app.core.config import get_settings

settings = get_settings()
deadline_seconds = int(sys.argv[1])
queue_url = settings.queue.metrics_aggregation_queue_url.strip()
if not queue_url:
    print("metrics_queue_url_missing=true")
    raise SystemExit(0)

region = (
    os.getenv("AWS_REGION", "").strip()
    or os.getenv("AWS_DEFAULT_REGION", "").strip()
    or os.getenv("AWS_SECRETS_MANAGER_REGION", "").strip()
    or None
)
kwargs = {"region_name": region} if region else {}
sqs = boto3.client("sqs", **kwargs)

deadline = time.time() + max(0, deadline_seconds)
while True:
    attrs = sqs.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"],
    )["Attributes"]
    visible = int(attrs.get("ApproximateNumberOfMessages", "0"))
    in_flight = int(attrs.get("ApproximateNumberOfMessagesNotVisible", "0"))
    print(f"metrics_queue_visible={visible} metrics_queue_in_flight={in_flight}")
    if visible == 0 and in_flight == 0:
        break
    if time.time() >= deadline:
        break
    time.sleep(5)
PY
fi

echo "[aws-load] collecting ops status..."
OPS_PAYLOAD="$(curl -fsS "http://127.0.0.1:${API_PORT}/api/v1/ops/status" -H "Authorization: Bearer ${TOKEN}")"
./venv/bin/python - "${OPS_PAYLOAD}" <<'PY'
import json
import sys
payload = json.loads(sys.argv[1])
print("=== Ops Snapshot ===")
print(json.dumps(payload, indent=2))
PY

echo "[aws-load] done."
echo "[aws-load] api log: ${API_LOG_FILE}"
echo "[aws-load] summary worker log: ${SUMMARY_WORKER_LOG_FILE}"
echo "[aws-load] llm async worker log: ${LLM_ASYNC_WORKER_LOG_FILE}"
echo "[aws-load] metrics worker log: ${METRICS_WORKER_LOG_FILE}"
if [[ -n "${EVAL_WORKER_PID}" ]]; then
  echo "[aws-load] eval worker log: ${EVAL_WORKER_LOG_FILE}"
fi
