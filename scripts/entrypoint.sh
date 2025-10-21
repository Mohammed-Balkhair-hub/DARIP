#!/usr/bin/env bash
set -euo pipefail

echo "[orchestrator] starting run-once pipeline…"
python -V
echo "TOPIC=${TOPIC}  TIMEZONE=${TIMEZONE}"
echo "FEEDS_FILE=${FEEDS_FILE}"
echo "BACKGROUND_BRIEFS=${BACKGROUND_BRIEFS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "USE_FAKE_TTS=${USE_FAKE_TTS}"

python run_daily.py

echo "[orchestrator] done."