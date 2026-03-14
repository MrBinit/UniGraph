#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export LOCAL_REDIS_PORT="${LOCAL_REDIS_PORT:-6380}"
export GRADIO_PORT="${GRADIO_PORT:-7860}"
export ENV_FILE="${ENV_FILE:-.env.local}"

if [[ ! -f "${ROOT_DIR}/${ENV_FILE}" ]]; then
  if [[ -f "${ROOT_DIR}/.env" ]]; then
    echo "[local-smoke] ${ENV_FILE} not found; falling back to .env"
    ENV_FILE=".env"
  else
    echo "[local-smoke] Missing env file: ${ENV_FILE}"
    exit 1
  fi
fi

RUN_CHAT=0
if [[ "${1:-}" == "--chat" ]]; then
  RUN_CHAT=1
fi

COMPOSE_ARGS=(
  -f docker-compose.yml
  -f docker-compose.local.yml
  --profile local-redis
)

echo "[local-smoke] Checking API health..."
if curl --noproxy "*" --max-time 10 -fsS "http://127.0.0.1:${API_PORT:-8000}/healthz" >/tmp/local-healthz.out 2>/tmp/local-healthz.err; then
  cat /tmp/local-healthz.out
  echo
else
  echo "[local-smoke] Host curl failed; falling back to in-container health check."
  docker compose --env-file "${ENV_FILE}" "${COMPOSE_ARGS[@]}" exec -T api python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=5).read().decode())"
fi

echo "[local-smoke] Checking Redis connectivity from API container..."
docker compose --env-file "${ENV_FILE}" "${COMPOSE_ARGS[@]}" exec -T api python -c "from app.infra.redis_client import app_redis_client as c; print(c.ping())"

echo "[local-smoke] Checking Gradio endpoint..."
if curl --noproxy "*" --max-time 10 -fsS "http://127.0.0.1:${GRADIO_PORT}" >/dev/null 2>/tmp/local-gradio.err; then
  echo "gradio_ok=true"
else
  echo "[local-smoke] Host Gradio check failed; running in-container check."
  docker compose --env-file "${ENV_FILE}" "${COMPOSE_ARGS[@]}" exec -T gradio python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860', timeout=5); print('gradio_ok=true')"
fi

if [[ "${RUN_CHAT}" -eq 1 ]]; then
  echo "[local-smoke] Running one authenticated /chat/stream call..."
  TOKEN="$(docker compose --env-file "${ENV_FILE}" "${COMPOSE_ARGS[@]}" run --rm --no-deps api python -m app.scripts.generate_jwt --user-id local-user --roles admin | tail -n 1)"
  if ! curl --noproxy "*" --max-time 30 -fsS "http://127.0.0.1:${API_PORT:-8000}/api/v1/chat/stream" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{"user_id":"local-user","prompt":"Say hello in one short line."}' >/tmp/local-chat.out 2>/tmp/local-chat.err; then
    echo "[local-smoke] Host /chat/stream call failed; running in-container /chat/stream check."
    docker compose --env-file "${ENV_FILE}" "${COMPOSE_ARGS[@]}" exec -T -e TOKEN="${TOKEN}" api python - <<'PY'
import json
import os
import urllib.request

token = os.environ["TOKEN"]
body = json.dumps({"user_id": "local-user", "prompt": "Say hello in one short line."}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:8000/api/v1/chat/stream",
    data=body,
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    },
    method="POST",
)
with urllib.request.urlopen(req, timeout=30) as response:
    print(response.read().decode())
PY
  else
    cat /tmp/local-chat.out
  fi
  echo
fi

echo "[local-smoke] OK"
