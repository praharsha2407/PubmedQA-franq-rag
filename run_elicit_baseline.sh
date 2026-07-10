#!/bin/bash
#SBATCH --job-name=elicit_base
#SBATCH --output=logs/elicit_base_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue

# Apply the SAME greedy decision-elicitation procedure to the baseline's stored
# answers (outputs/answers.jsonl, generated_answer field) that the advanced pipeline
# uses on its own output. This is what makes the yes/no/maybe comparison fair: both
# systems' decisions come from the identical follow-up prompt over each system's own
# reasoning, instead of depending on whether the model happened to volunteer a
# "Conclusion:" line. Does NOT regenerate baseline answers -- read-only post-step.

source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis
export TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"

echo "=== node: $(hostname) ==="
date +"=== start: %F %T ==="

python -u src/elicit_decisions.py \
    --answers outputs/answers.jsonl \
    --text-field generated_answer \
    --out outputs/answers.elicited.jsonl
RC=$?

if [ $RC -eq 0 ]; then
  echo "=== BASELINE decision score (elicited) ==="
  python src/evaluate_decision.py \
      --answers outputs/answers.elicited.jsonl \
      --field elicited_decision \
      --out outputs/decision_metrics_baseline_elicited.json
fi

date +"=== end: %F %T ==="
echo "=== exit: $RC ==="
exit $RC
