#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export LOCAL_REDIS_PORT="${LOCAL_REDIS_PORT:-6380}"
export GRADIO_PORT="${GRADIO_PORT:-7860}"
export ENV_FILE="${ENV_FILE:-.env.local}"

if [[ ! -f "${ROOT_DIR}/${ENV_FILE}" ]]; then
  if [[ -f "${ROOT_DIR}/.env" ]]; then
    echo "[local-down] ${ENV_FILE} not found; falling back to .env"
    ENV_FILE=".env"
  else
    echo "[local-down] Missing env file: ${ENV_FILE}"
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

echo "[local-down] Stopping local stack..."
docker compose --env-file "${ENV_FILE}" "${COMPOSE_ARGS[@]}" down
echo "[local-down] Done."
