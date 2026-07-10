#!/bin/bash
#SBATCH --job-name=final_smoke
#SBATCH --output=logs/final_smoke_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:40:00
source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis
export TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"

echo "########## RUN 1 (seed reproducibility check) ##########"
export PUBMEDQA_OUTPUT_DIR="$SLURM_SUBMIT_DIR/outputs_final_sample"
python -u src/run_advanced_pipeline.py --sample-size 5 --strict --prompt cot
echo "=== run1 exit: $? ==="

echo "########## RUN 2 (same seed, must reproduce RUN 1) ##########"
export PUBMEDQA_OUTPUT_DIR="$SLURM_SUBMIT_DIR/outputs_final_sample_rep"
python -u src/run_advanced_pipeline.py --sample-size 5 --strict --prompt cot
echo "=== run2 exit: $? ==="

echo "########## SEED CHECK: raw answers identical across the two runs? ##########"
python - <<'PY'
import json
def raws(p):
    return [json.loads(l)["raw_answer"] for l in open(p) if l.strip()]
a = raws("outputs_final_sample/advanced_answers_cot.jsonl")
b = raws("outputs_final_sample_rep/advanced_answers_cot.jsonl")
same = a == b
print(f"run1 n={len(a)} run2 n={len(b)}  IDENTICAL={same}")
if not same:
    for i,(x,y) in enumerate(zip(a,b)):
        if x != y:
            print(f"  row {i} DIFFERS")
            print(f"   run1: {x[:120]!r}")
            print(f"   run2: {y[:120]!r}")
PY

echo "########## DECISION ELICITATION (pipeline) ##########"
export PUBMEDQA_OUTPUT_DIR="$SLURM_SUBMIT_DIR/outputs_final_sample"
python -u src/elicit_decisions.py \
    --answers outputs_final_sample/advanced_answers_cot.jsonl \
    --text-field raw_answer \
    --out outputs_final_sample/advanced_answers_cot.elicited.jsonl

echo "########## SCORE on elicited decision ##########"
python src/evaluate_decision.py \
    --answers outputs_final_sample/advanced_answers_cot.elicited.jsonl \
    --field elicited_decision

echo "########## (for reference) SCORE on volunteered Conclusion line ##########"
python src/evaluate_decision.py \
    --answers outputs_final_sample/advanced_answers_cot.jsonl \
    --field raw_answer 2>/dev/null | head -20
echo "=== DONE ==="
