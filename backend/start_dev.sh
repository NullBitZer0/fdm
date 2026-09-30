#!/usr/bin/env bash
# Start the Stage 9 API in the background, wait for /health to answer, print the URL.
# Usage: bash backend/start_dev.sh [port]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-${FDM_PORT:-8002}}"
LOG="${FDM_LOG:-/tmp/fdm-api.log}"
PIDFILE="/tmp/fdm-api.pid"

cd "$ROOT"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill "$(cat "$PIDFILE")" 2>/dev/null
    sleep 2
fi

: > "$LOG"
setsid nohup python -m uvicorn app.main:app --app-dir backend \
    --host 127.0.0.1 --port "$PORT" --log-level info >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
disown

for _ in $(seq 1 60); do
    if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        echo "API ready at http://127.0.0.1:${PORT}  (pid $(cat "$PIDFILE"), log $LOG)"
        echo "Interactive docs: http://127.0.0.1:${PORT}/docs"
        exit 0
    fi
    sleep 1
done

echo "API did not come up. Last log lines:" >&2
tail -20 "$LOG" >&2
exit 1
