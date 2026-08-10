#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

echo "[$(date -Is)] Starting merged NIM + OpenRouter benchmark run"

DRY_RUN_FLAG=""
if [ -n "${DRY_RUN:-}" ]; then
  DRY_RUN_FLAG="--dry-run"
fi

if [ -n "${NIM_API_KEY:-}" ] || [ -n "$DRY_RUN_FLAG" ]; then
  echo "[$(date -Is)] Running NIM benchmarks... ${DRY_RUN_FLAG}"
  python3 scripts/nim/test_models.py $DRY_RUN_FLAG
else
  echo "[$(date -Is)] Skipping NIM benchmarks: NIM_API_KEY is not set"
fi

if [ -n "${OPENROUTER_API_KEY:-}" ] || [ -n "$DRY_RUN_FLAG" ]; then
  echo "[$(date -Is)] Running OpenRouter benchmarks... ${DRY_RUN_FLAG}"
  python3 scripts/openrouter/discover_models.py --mark-missing-inactive $DRY_RUN_FLAG
  python3 scripts/openrouter/test_models.py --probe hermes_triage $DRY_RUN_FLAG
  python3 scripts/openrouter/test_router.py --probe hermes_triage --runs 2 $DRY_RUN_FLAG
else
  echo "[$(date -Is)] Skipping OpenRouter benchmarks: OPENROUTER_API_KEY is not set"
fi

echo "[$(date -Is)] Finished merged benchmark run"
