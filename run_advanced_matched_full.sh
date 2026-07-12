#!/bin/bash
#SBATCH --job-name=adv_matched
#SBATCH --output=logs/adv_matched_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --requeue

# THE CONTROLLED RUN -- the first true architecture comparison.
#
# Every previous baseline-vs-advanced number was confounded: run_pipeline.py defaults to
# mistralai/Mistral-7B-Instruct-v0.3 while GenerationConfig defaults to BioMistral/BioMistral-7B,
# so the two systems ran DIFFERENT LLMs. BioMistral emitted the required "Conclusion:" line in
# 0/1000 answers (Mistral-Instruct: 999/1000), which alone produced the entire accuracy gap.
#
# This holds the generator constant at the baseline's model. Prompt (cot), seed (42), retrieval
# corpus and evidence are already identical, so the ONLY remaining difference is the architecture.
#
# Compare against: outputs_baseline_today/  (baseline, same model, same day, same seed)
# The BioMistral run in outputs_cot/ is retained as the domain-adaptation ablation row.

source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis
export TOKENIZERS_PARALLELISM=false
export PUBMEDQA_OUTPUT_DIR="$SLURM_SUBMIT_DIR/outputs_matched"
cd "$SLURM_SUBMIT_DIR"

echo "=== node: $(hostname) ==="
date +"=== start: %F %T ==="

python -u src/run_advanced_pipeline.py --strict --resume --prompt cot \
       --model-name mistralai/Mistral-7B-Instruct-v0.3
RC=$?
date +"=== end: %F %T ==="

if [ $RC -eq 0 ]; then
  echo "=== decision elicitation (same greedy procedure as the baseline) ==="
  python -u src/elicit_decisions.py \
      --answers outputs_matched/advanced_answers_cot.jsonl \
      --text-field raw_answer \
      --out outputs_matched/advanced_answers_cot.elicited.jsonl

  echo "=== ADVANCED (matched model) decision score -- THE HEADLINE NUMBER ==="
  python src/evaluate_decision.py \
      --answers outputs_matched/advanced_answers_cot.elicited.jsonl \
      --field elicited_decision \
      --out outputs_matched/decision_metrics_matched.json
fi
echo "=== exit: $RC ==="
exit $RC
