#!/bin/bash
#SBATCH --job-name=adv_cot_full
#SBATCH --output=logs/adv_cot_full_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --requeue

# THE CONTROLLED RUN. Advanced pipeline with the SAME chain-of-thought prompt the
# baseline uses, so any difference is attributable to the architecture rather than
# the prompt.
#
# Why this run exists:
#   - With minimal_prompt, 26% of advanced answers contain no yes/no/maybe at all,
#     and "maybe" is predicted zero times. Strict PubMedQA accuracy is therefore
#     0.470 vs the baseline's 0.576 -- the pipeline loses the task on a formatting
#     failure, not on reasoning. On the answers that DO commit, it scores 0.634.
#   - Prompt alone moved keyword faithfulness 0.706 -> 0.313 on 5 identical
#     examples, i.e. far more than any architectural change measured so far.
#
# Writes outputs_cot/advanced_answers_cot.jsonl + full_evaluation_report_cot.json,
# and now stores raw_answer + final_decision per row so the decision metric compares
# raw output against the baseline's raw output.
#
# Resume-safe: rerun this exact script after an interruption.

source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis

export PUBMEDQA_OUTPUT_DIR="$SLURM_SUBMIT_DIR/outputs_cot"
export TOKENIZERS_PARALLELISM=false

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date +"=== start: %F %T ==="

python -u src/run_advanced_pipeline.py --strict --resume --prompt cot
RC=$?

date +"=== end: %F %T ==="
echo "=== exit: $RC ==="

if [ $RC -eq 0 ]; then
  echo "=== scoring the PubMedQA decision (volunteered Conclusion line, raw vs raw) ==="
  python src/evaluate_decision.py \
      --answers outputs_cot/advanced_answers_cot.jsonl \
      --field raw_answer \
      --out outputs_cot/decision_metrics_cot.json

  echo "=== explicit decision elicitation (greedy, one word per answer) ==="
  python -u src/elicit_decisions.py \
      --answers outputs_cot/advanced_answers_cot.jsonl \
      --text-field raw_answer \
      --out outputs_cot/advanced_answers_cot.elicited.jsonl

  echo "=== scoring the PubMedQA decision (elicited, the headline number) ==="
  python src/evaluate_decision.py \
      --answers outputs_cot/advanced_answers_cot.elicited.jsonl \
      --field elicited_decision \
      --out outputs_cot/decision_metrics_cot_elicited.json
fi
exit $RC
