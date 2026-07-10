#!/usr/bin/env bash
set -euo pipefail

# Avoid submitting duplicate retrieval jobs: if a pubmedqa-retrieval job already
# exists for this user, reuse its job id instead of creating a new one.
existing_ret=$(squeue -u "$USER" -h -o "%i %j" | awk '$2=="pubmedqa-retrieval"{print $1; exit}')
if [ -n "${existing_ret-}" ]; then
  echo "Found existing retrieval job: $existing_ret"
  retrieval_job=$existing_ret
else
  retrieval_job=$(sbatch --parsable hpc/01_retrieval.slurm)
  echo "Submitted retrieval job: $retrieval_job"
fi

# Avoid submitting duplicate generation jobs: if a pubmedqa-mistral job already
# exists for this user, reuse its job id as the generation job so downstream
# steps attach to the existing run instead of creating duplicates.
retrieval_job=$(sbatch --parsable hpc/01_retrieval.slurm)

echo "Submitted retrieval job: $retrieval_job"

generation_job=$(sbatch --parsable --dependency=afterok:"$retrieval_job" hpc/02_generation.slurm)

echo "Submitted generation job: $generation_job"

metrics_job=$(sbatch --parsable --dependency=afterok:"$generation_job" hpc/03_generation_metrics.slurm)

echo "Submitted generation metrics job: $metrics_job"

if [ -n "${OPENAI_API_KEY:-}" ]; then
  ragas_job=$(sbatch --parsable --dependency=afterok:"$generation_job" hpc/04_ragas.slurm)
  echo "Submitted RAGAS evaluation job: $ragas_job"
  report_job=$(sbatch --parsable --dependency=afterok:"$ragas_job" hpc/05_report.slurm)
else
  report_job=$(sbatch --parsable --dependency=afterok:"$metrics_job" hpc/05_report.slurm)
fi

echo "Pipeline submitted"
  generation_job=$(sbatch --parsable --dependency=afterok:"$retrieval_job" hpc/02_generation.slurm)
  echo "Submitted generation job: $generation_job"
fi

metrics_job=$(sbatch --parsable --dependency=afterok:"$generation_job" hpc/03_generation_metrics.slurm)
echo "Submitted generation metrics job: $metrics_job"

# Include RAGAS if API key is available
if [ -n "${OPENAI_API_KEY:-}" ]; then
  ragas_job=$(sbatch --export=ALL --dependency=afterok:"$generation_job" hpc/04_ragas.slurm)
  echo "Submitted RAGAS evaluation job: $ragas_job"
  report_job=$(sbatch --parsable --dependency=afterok:"$ragas_job" hpc/05_report.slurm)
  echo "Submitted report job (depends on RAGAS): $report_job"
else
  echo "NOTE: OPENAI_API_KEY not set — skipping RAGAS. To enable, export OPENAI_API_KEY and resubmit."
  report_job=$(sbatch --parsable --dependency=afterok:"$metrics_job" hpc/05_report.slurm)
  echo "Submitted report job (depends on metrics): $report_job"
fi

cat <<EOF

Pipeline submitted.

EOF
