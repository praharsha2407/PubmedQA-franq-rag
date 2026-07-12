#!/bin/bash
#SBATCH --job-name=prelim
#SBATCH --output=logs/prelim_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis
export TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"
mkdir -p scratch

# Snapshot the in-progress matched run so we read a stable file while it keeps writing.
cp outputs_matched/advanced_answers_cot.jsonl scratch/matched_partial.jsonl
echo "=== snapshot rows: $(wc -l < scratch/matched_partial.jsonl) ==="

python -u src/elicit_decisions.py \
    --answers scratch/matched_partial.jsonl \
    --text-field raw_answer \
    --out scratch/matched_partial.elicited.jsonl > /dev/null 2>&1

echo "=== PRELIMINARY: ADVANCED (matched generator), partial ==="
python src/evaluate_decision.py \
    --answers scratch/matched_partial.elicited.jsonl \
    --field elicited_decision

echo "=== BASELINE restricted to the SAME pubids (exact like-for-like) ==="
python3 - <<'PY'
import json
adv={json.loads(l)["pubid"] for l in open("scratch/matched_partial.elicited.jsonl") if l.strip()}
rows=[json.loads(l) for l in open("outputs_baseline_today/answers.elicited.jsonl") if l.strip()]
sub=[r for r in rows if r["pubid"] in adv]
with open("scratch/baseline_same_subset.jsonl","w") as f:
    for r in sub: f.write(json.dumps(r)+"\n")
print(f"  baseline rows on the same {len(sub)} pubids")
PY
python src/evaluate_decision.py \
    --answers scratch/baseline_same_subset.jsonl \
    --field elicited_decision
echo "=== exit: $? ==="
