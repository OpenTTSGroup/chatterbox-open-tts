#!/usr/bin/env bash
set -euo pipefail

# Engine defaults
: "${CHATTERBOX_VARIANT:=multilingual}"
: "${CHATTERBOX_DEVICE:=auto}"
: "${CHATTERBOX_DTYPE:=float16}"
: "${CHATTERBOX_DEFAULT_LANGUAGE:=en}"

# Service-level defaults
: "${VOICES_DIR:=/voices}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${LOG_LEVEL:=info}"
: "${CORS_ENABLED:=false}"

export CHATTERBOX_VARIANT CHATTERBOX_DEVICE CHATTERBOX_DTYPE \
       CHATTERBOX_DEFAULT_LANGUAGE VOICES_DIR HOST PORT LOG_LEVEL CORS_ENABLED

if [ "$#" -eq 0 ]; then
  exec uvicorn app.server:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
fi
exec "$@"
