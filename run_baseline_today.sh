#!/bin/bash
#SBATCH --job-name=base_today
#SBATCH --output=logs/base_today_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --requeue

# Re-run the BASELINE under today's environment, with the seed, so it is comparable
# with the advanced run that finished today.
#
# Why: diag_stored_prompt.py replayed the baseline's OWN stored prompts through today's
# model and got 0/20 "Conclusion:" lines, where the stored answers have 20/20. The
# environment changed since 2 July. The stored baseline (0.618) and today's advanced run
# (0.471) therefore came from different model states and were never comparable.
#
# Writes to outputs_baseline_today/ -- the original outputs/answers.jsonl is NOT touched.

source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis
export TOKENIZERS_PARALLELISM=false
export PUBMEDQA_OUTPUT_DIR="$SLURM_SUBMIT_DIR/outputs_baseline_today"
cd "$SLURM_SUBMIT_DIR"

date +"=== start: %F %T ==="
python -u src/run_pipeline.py --full-dataset --no-resume
RC=$?
date +"=== end: %F %T ==="

if [ $RC -eq 0 ]; then
  echo "=== eliciting decisions (same greedy procedure as the advanced run) ==="
  python -u src/elicit_decisions.py \
      --answers outputs_baseline_today/answers.jsonl \
      --text-field generated_answer \
      --out outputs_baseline_today/answers.elicited.jsonl

  echo "=== BASELINE (today, same environment) decision score ==="
  python src/evaluate_decision.py \
      --answers outputs_baseline_today/answers.elicited.jsonl \
      --field elicited_decision \
      --out outputs_baseline_today/decision_metrics_baseline_today.json
fi
echo "=== exit: $RC ==="
exit $RC
