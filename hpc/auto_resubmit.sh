#!/usr/bin/env bash
set -euo pipefail

INTERVAL=${1:-300} # seconds between retries
MAX_ATTEMPTS=${2:-0} # 0 = unlimited
ATTEMPT=0

LOGFILE="$(pwd)/logs/auto_submit.log"
mkdir -p "$(dirname "$LOGFILE")"

echo "Auto resubmit started at $(date). Interval=${INTERVAL}s" | tee -a "$LOGFILE"

while true; do
  ATTEMPT=$((ATTEMPT+1))
  if [ "$MAX_ATTEMPTS" -ne 0 ] && [ "$ATTEMPT" -gt "$MAX_ATTEMPTS" ]; then
    echo "Reached max attempts ($MAX_ATTEMPTS). Exiting." | tee -a "$LOGFILE"
    exit 1
  fi

  # If a generation job already exists, do not submit another.
  existing=$(squeue -u "$USER" -h -o "%i %j" | awk '$2=="pubmedqa-mistral"{print $1; exit}') || true
  if [ -n "${existing-}" ]; then
    echo "Found existing generation job: $existing — no submission needed. Exiting." | tee -a "$LOGFILE"
    exit 0
  fi

  echo "Attempt $ATTEMPT: running submit_pipeline.sh at $(date)" | tee -a "$LOGFILE"
  if bash hpc/submit_pipeline.sh 2>&1 | tee -a "$LOGFILE"; then
    echo "Submission succeeded on attempt $ATTEMPT" | tee -a "$LOGFILE"
    exit 0
  else
    # Check last lines for the node-configuration error
    tail -n 20 "$LOGFILE" | tee -a "$LOGFILE" | grep -q "Requested node configuration is not available" && rc=2 || rc=3
    if [ "$rc" -eq 2 ]; then
      echo "No suitable nodes available; will retry after ${INTERVAL}s" | tee -a "$LOGFILE"
    else
      echo "Submission failed (unknown reason); will retry after ${INTERVAL}s" | tee -a "$LOGFILE"
    fi
    sleep "$INTERVAL"
  fi
done
