#!/usr/bin/env bash
set -euo pipefail

RETRIEVAL_JOB=${1:-5465600}
INTERVAL=${2:-300} # 5 minutes default
OPENAI_API_KEY=${3:-}

LOGFILE="$(pwd)/logs/wait_and_retry.log"
mkdir -p "$(dirname "$LOGFILE")"

echo "Waiting for retrieval job $RETRIEVAL_JOB to complete before retrying generation..." | tee "$LOGFILE"
if [ -n "$OPENAI_API_KEY" ]; then
  echo "RAGAS evaluation will be included (OpenAI API key detected)" | tee -a "$LOGFILE"
fi

# Wait for retrieval job to finish (poll every 30s)
while true; do
  state=$(squeue -u "$USER" -h -o "%T" -j "$RETRIEVAL_JOB" 2>/dev/null || echo "COMPLETED")
  if [ "$state" != "RUNNING" ]; then
    echo "Retrieval job $RETRIEVAL_JOB finished (state: $state) at $(date)" | tee -a "$LOGFILE"
    break
  fi
  echo "$(date): Retrieval still running (state: $state)..." | tee -a "$LOGFILE"
  sleep 30
done

# Now retry generation submission only (not retrieval) every INTERVAL seconds
ATTEMPT=0
while true; do
  ATTEMPT=$((ATTEMPT+1))
  
  # Check if generation already exists
  existing=$(squeue -u "$USER" -h -o "%i %j" | awk '$2=="pubmedqa-mistral"{print $1; exit}') || true
  if [ -n "${existing-}" ]; then
    echo "Found existing generation job: $existing — no submission needed. Exiting." | tee -a "$LOGFILE"
    exit 0
  fi

  echo "Attempt $ATTEMPT at $(date): Submitting generation job (depends on retrieval $RETRIEVAL_JOB)..." | tee -a "$LOGFILE"
  if generation_job=$(sbatch --parsable --dependency=afterok:"$RETRIEVAL_JOB" hpc/02_generation.slurm 2>&1); then
    echo "Generation submission succeeded: $generation_job at $(date)" | tee -a "$LOGFILE"
    # Submit downstream jobs: metrics, report
    metrics_job=$(sbatch --parsable --dependency=afterok:"$generation_job" hpc/03_generation_metrics.slurm 2>&1)
    echo "Submitted metrics job: $metrics_job" | tee -a "$LOGFILE"
    report_job=$(sbatch --parsable --dependency=afterok:"$metrics_job" hpc/05_report.slurm 2>&1)
    echo "Submitted report job: $report_job" | tee -a "$LOGFILE"
    echo "Pipeline complete at $(date)" | tee -a "$LOGFILE"
    exit 0
  else
    tail -n 5 "$LOGFILE" | grep -q "Requested node configuration is not available" && {
      echo "No suitable nodes available; will retry after ${INTERVAL}s" | tee -a "$LOGFILE"
    } || {
      echo "Submission failed (unknown reason); will retry after ${INTERVAL}s" | tee -a "$LOGFILE"
    }
    sleep "$INTERVAL"
  fi
done
