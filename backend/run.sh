#!/usr/bin/env bash
# Start the Stage 9 API on a free port, wait until it answers, and report the URL.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${FDM_PORT:-8002}"
LOG="${FDM_LOG:-/tmp/fdm-api.log}"

cd "$ROOT"
exec python -m uvicorn app.main:app --app-dir backend \
    --host "${FDM_HOST:-127.0.0.1}" --port "$PORT" --log-level info
