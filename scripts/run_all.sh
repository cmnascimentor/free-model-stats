#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

# Rotate cron log: keep last 5 rotated files, cap live log at ~1 MB.
LOG=logs/cron.log
if [ -f "$LOG" ] && [ "$(stat -c%s "$LOG" 2>/dev/null || echo 0)" -gt 1048576 ]; then
  for i in 4 3 2 1; do
    [ -f "$LOG.$i" ] && mv -f "$LOG.$i" "$LOG.$((i+1))"
  done
  mv -f "$LOG" "$LOG.1"
fi

if [ -z "${DRY_RUN:-}" ] && [ -f .env ]; then
  set -a
  source .env
  set +a
fi

CORE_PROBE="${CORE_PROBE:-hermes_triage}"
OPENROUTER_MODEL_LIMIT="${OPENROUTER_MODEL_LIMIT:-20}"

echo "[$(date -Is)] Starting NIM + OpenRouter benchmark run probe=$CORE_PROBE"

DRY_RUN_FLAG=""
if [ -n "${DRY_RUN:-}" ]; then
  DRY_RUN_FLAG="--dry-run"
fi

if [ -n "${NIM_API_KEY:-}" ] || [ -n "$DRY_RUN_FLAG" ]; then
  echo "[$(date -Is)] Running NIM benchmark... ${DRY_RUN_FLAG}"
  python3 scripts/nim/test_models.py --probe "$CORE_PROBE" $DRY_RUN_FLAG
else
  echo "[$(date -Is)] Skipping NIM: NIM_API_KEY is not set"
fi

if [ -n "${OPENROUTER_API_KEY:-}" ] || [ -n "$DRY_RUN_FLAG" ]; then
  echo "[$(date -Is)] Running OpenRouter benchmark (limit=$OPENROUTER_MODEL_LIMIT)... ${DRY_RUN_FLAG}"
  MARK_INACTIVE="--mark-missing-inactive"
  if [ -n "$DRY_RUN_FLAG" ]; then
    MARK_INACTIVE=""
  fi
  python3 scripts/openrouter/discover_models.py $MARK_INACTIVE $DRY_RUN_FLAG
  python3 scripts/openrouter/test_models.py --probe "$CORE_PROBE" --limit "$OPENROUTER_MODEL_LIMIT" $DRY_RUN_FLAG
  python3 scripts/openrouter/test_router.py --probe "$CORE_PROBE" --runs 2 $DRY_RUN_FLAG
else
  echo "[$(date -Is)] Skipping OpenRouter: OPENROUTER_API_KEY is not set"
fi

echo "[$(date -Is)] Finished benchmark run"
