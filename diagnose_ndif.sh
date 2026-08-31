#!/usr/bin/env bash
set -euo pipefail

API_URL="${NDIF_API_URL:-http://localhost:5001}"
RAY_LOG_DIR="${RAY_LOG_DIR:-$PWD/ray-tmp}"
CONTAINER_LOG="${CONTAINER_LOG:-$PWD/ndif-container.log}"

echo "== NDIF API =="
echo "URL: $API_URL"
if command -v curl >/dev/null 2>&1; then
    curl -fsS "$API_URL/status" || true
    echo
else
    echo "curl not found; skipping API status check"
fi

echo
echo "== Container Log Errors =="
if [ -f "$CONTAINER_LOG" ]; then
    grep -nEi 'error|exception|traceback|failed|aborted|address already in use|port .* in use' "$CONTAINER_LOG" | tail -n 120 || true
else
    echo "No container log found at $CONTAINER_LOG"
fi

echo
echo "== Ray Log Errors =="
if [ -d "$RAY_LOG_DIR" ]; then
    find "$RAY_LOG_DIR" -type f \( -name '*.err' -o -name '*.out' -o -name '*.log' \) \
        -exec grep -nH -Ei 'error|exception|traceback|failed|aborted|address already in use|port .* in use|disconnecting client due to connection error' {} + \
        | tail -n 160 || true
else
    echo "No Ray log directory found at $RAY_LOG_DIR"
fi
