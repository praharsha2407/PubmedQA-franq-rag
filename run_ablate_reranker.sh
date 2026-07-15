#!/bin/bash
#SBATCH --job-name=abl_rr
#SBATCH --output=logs/abl_rr_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
# ABLATION: re-enable the cross-encoder reranker in the FULL pipeline.
source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis
export TOKENIZERS_PARALLELISM=false
export PUBMEDQA_OUTPUT_DIR="$SLURM_SUBMIT_DIR/outputs_ablation_reranker"
cd "$SLURM_SUBMIT_DIR"
date +"=== start : %F %T ==="
python -u src/run_advanced_pipeline.py --strict --resume --prompt cot \
       --model-name mistralai/Mistral-7B-Instruct-v0.3 --rerank
RC=$?
date +"=== end : %F %T ==="
if [ $RC -eq 0 ]; then
  D="outputs_ablation_reranker"
  python -u src/elicit_decisions.py --answers $D/advanced_answers_cot.jsonl \
      --text-field raw_answer --out $D/advanced_answers_cot.elicited.jsonl
  python src/evaluate_decision.py --answers $D/advanced_answers_cot.elicited.jsonl \
      --field elicited_decision --out $D/decision_metrics.json
fi
echo "=== exit: $RC ==="; exit $RC
