#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

REQUESTS="${REQUESTS:-300}"
CONCURRENCY="${CONCURRENCY:-24}"
API_WORKERS="${API_WORKERS:-1}"
API_PORT="${API_PORT:-18000}"
LLM_MOCK_DELAY_MS="${LLM_MOCK_DELAY_MS:-80}"
LLM_MOCK_STREAM_CHUNK_CHARS="${LLM_MOCK_STREAM_CHUNK_CHARS:-16}"

echo "[loadtest] starting local redis..."
docker compose -f docker-compose.yml --profile local-redis up -d redis >/dev/null

export SECURITY_JWT_SECRET="${SECURITY_JWT_SECRET:-01234567890123456789012345678901}"
export MEMORY_ENCRYPTION_KEY="${MEMORY_ENCRYPTION_KEY:-abcdefghijklmnopqrstuvwxyz1234567890AB}"
export REDIS_APP_HOST="${REDIS_APP_HOST:-127.0.0.1}"
export REDIS_APP_PORT="${REDIS_APP_PORT:-6379}"
export REDIS_WORKER_HOST="${REDIS_WORKER_HOST:-127.0.0.1}"
export REDIS_WORKER_PORT="${REDIS_WORKER_PORT:-6379}"
export REDIS_APP_TLS="${REDIS_APP_TLS:-false}"
export REDIS_WORKER_TLS="${REDIS_WORKER_TLS:-false}"
export REDIS_APP_SSL_CERT_REQS="${REDIS_APP_SSL_CERT_REQS:-none}"
export REDIS_WORKER_SSL_CERT_REQS="${REDIS_WORKER_SSL_CERT_REQS:-none}"
export REDIS_APP_SSL_CA_CERTS="${REDIS_APP_SSL_CA_CERTS:-}"
export REDIS_WORKER_SSL_CA_CERTS="${REDIS_WORKER_SSL_CA_CERTS:-}"
export POSTGRES_ENABLED="${POSTGRES_ENABLED:-false}"
export LLM_ASYNC_ENABLED="${LLM_ASYNC_ENABLED:-false}"
export METRICS_AGGREGATION_QUEUE_ENABLED="${METRICS_AGGREGATION_QUEUE_ENABLED:-false}"
export EVALUATION_QUEUE_ENABLED="${EVALUATION_QUEUE_ENABLED:-false}"
export EVALUATION_ENABLED="${EVALUATION_ENABLED:-false}"
export EVALUATION_SCHEDULE_ENABLED="${EVALUATION_SCHEDULE_ENABLED:-false}"
export LLM_MOCK_MODE="${LLM_MOCK_MODE:-true}"
export RETRIEVAL_DISABLED="${RETRIEVAL_DISABLED:-true}"
export LLM_MOCK_DELAY_MS
export LLM_MOCK_STREAM_CHUNK_CHARS

echo "[loadtest] starting api on :${API_PORT} (workers=${API_WORKERS})..."
if lsof -nP -iTCP:"${API_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[loadtest] port ${API_PORT} is already in use; choose another API_PORT"
  exit 1
fi

./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "${API_PORT}" --workers "${API_WORKERS}" \
  >/tmp/ai-system-loadtest-api.log 2>&1 &
API_PID=$!

cleanup() {
  if kill -0 "${API_PID}" 2>/dev/null; then
    kill "${API_PID}" >/dev/null 2>&1 || true
    wait "${API_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[loadtest] waiting for api health..."
for _ in $(seq 1 30); do
  if ! kill -0 "${API_PID}" 2>/dev/null; then
    echo "[loadtest] api process exited before becoming healthy"
    tail -n 80 /tmp/ai-system-loadtest-api.log || true
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
print(create_access_token(user_id="load-user", roles=["admin"]))
PY
)"

echo "[loadtest] running chat-stream load test..."
./venv/bin/python loadtest/http_load.py \
  --url "http://127.0.0.1:${API_PORT}/api/v1/chat/stream" \
  --mode chat-stream \
  --token "${TOKEN}" \
  --requests "${REQUESTS}" \
  --concurrency "${CONCURRENCY}" \
  --timeout 20 \
  --max-error-rate 0.02

echo "[loadtest] done. api logs: /tmp/ai-system-loadtest-api.log"
