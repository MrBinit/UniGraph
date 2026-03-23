#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[sonarqube-up] Starting SonarQube + Postgres..."
docker compose -f docker-compose.sonarqube.yml up -d

echo "[sonarqube-up] Waiting for SonarQube on http://localhost:9000 ..."
for i in {1..60}; do
  if curl -fsS "http://localhost:9000/api/system/status" >/dev/null 2>&1; then
    echo "[sonarqube-up] SonarQube is reachable."
    exit 0
  fi
  sleep 2
done

echo "[sonarqube-up] SonarQube is not ready yet. Check logs:"
echo "  docker compose -f docker-compose.sonarqube.yml logs -f sonarqube"
exit 1
