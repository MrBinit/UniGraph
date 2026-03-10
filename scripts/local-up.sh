#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export LOCAL_REDIS_PORT="${LOCAL_REDIS_PORT:-6380}"
export GRADIO_PORT="${GRADIO_PORT:-7860}"
export ENV_FILE="${ENV_FILE:-.env.local}"
mkdir -p "${ROOT_DIR}/data/metrics"

if [[ ! -f "${ROOT_DIR}/${ENV_FILE}" ]]; then
  if [[ -f "${ROOT_DIR}/.env" ]]; then
    echo "[local-up] ${ENV_FILE} not found; falling back to .env"
    ENV_FILE=".env"
  else
    echo "[local-up] Missing env file: ${ENV_FILE}"
    exit 1
  fi
fi

COMPOSE_ARGS=(
  -f docker-compose.yml
  -f docker-compose.local.yml
  --profile local-redis
  --profile llm-async
  --profile eval-queue
  --profile metrics-queue
)

echo "[local-up] Building images..."
docker compose --env-file "${ENV_FILE}" "${COMPOSE_ARGS[@]}" build api gradio

echo "[local-up] Starting api/worker/llm-worker/eval-worker/metrics-worker/redis/gradio (redis:${LOCAL_REDIS_PORT}, gradio:${GRADIO_PORT})..."
docker compose --env-file "${ENV_FILE}" "${COMPOSE_ARGS[@]}" up -d --no-build api worker llm-worker eval-worker metrics-worker redis gradio
echo "[local-up] Started."
echo "[local-up] API:    http://127.0.0.1:${API_PORT:-8000}"
echo "[local-up] Gradio: http://127.0.0.1:${GRADIO_PORT}"
echo "[local-up] Metrics: ${ROOT_DIR}/data/metrics"
echo "[local-up] ENV:    ${ENV_FILE}"
echo "[local-up] Next:   ./scripts/local-smoke.sh --chat"
