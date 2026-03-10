#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

METRICS_DIR="${METRICS_DIR:-data/metrics}"
REQUESTS_FILE="${METRICS_DIR}/chat_metrics_requests.jsonl"
AGG_FILE="${METRICS_DIR}/chat_metrics_aggregate.json"
TAIL_COUNT="${1:-3}"

echo "[prod-metrics] Metrics dir: ${METRICS_DIR}"

if [[ ! -f "${REQUESTS_FILE}" || ! -f "${AGG_FILE}" ]]; then
  echo "[prod-metrics] Metrics files not found yet."
  echo "[prod-metrics] Send one in-scope chat request, then run again."
  exit 1
fi

echo "[prod-metrics] Last ${TAIL_COUNT} request metrics:"
tail -n "${TAIL_COUNT}" "${REQUESTS_FILE}"
echo

echo "[prod-metrics] Aggregate snapshot:"
if command -v jq >/dev/null 2>&1; then
  jq '{
    total_requests,
    outcomes,
    latest_request,
    overall_latency: (.latency_ms.overall | {count,total,average,max,p95,p99}),
    llm_latency: (.latency_ms.llm_response | {count,total,average,max,p95,p99}),
    short_term_memory_latency: (.latency_ms.short_term_memory | {count,total,average,max,p95,p99}),
    long_term_memory_latency: (.latency_ms.long_term_memory | {count,total,average,max,p95,p99}),
    cache_read_latency: (.latency_ms.cache_read | {count,total,average,max,p95,p99}),
    cache_write_latency: (.latency_ms.cache_write | {count,total,average,max,p95,p99}),
    token_usage
  }' "${AGG_FILE}"
else
  cat "${AGG_FILE}"
fi
