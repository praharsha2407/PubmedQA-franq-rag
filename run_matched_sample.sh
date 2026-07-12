#!/bin/bash
#SBATCH --job-name=matched_smoke
#SBATCH --output=logs/matched_smoke_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:40:00
source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis
export TOKENIZERS_PARALLELISM=false
export PUBMEDQA_OUTPUT_DIR="$SLURM_SUBMIT_DIR/outputs_matched_sample"
cd "$SLURM_SUBMIT_DIR"

# Advanced pipeline with the BASELINE'S generator, holding the LLM constant.
python -u src/run_advanced_pipeline.py --sample-size 5 --strict --prompt cot \
       --model-name mistralai/Mistral-7B-Instruct-v0.3
echo "=== exit: $? ==="

echo "=== does the Conclusion come back with the matched model? ==="
python3 -c "
import json
rows=[json.loads(l) for l in open('outputs_matched_sample/advanced_answers_cot.jsonl') if l.strip()]
c=sum('Conclusion' in r['raw_answer'] for r in rows)
print(f'  Conclusion present: {c}/{len(rows)}')
print(f'  generator recorded: {rows[0][\"run_config\"].get(\"generator_model\")}')
for r in rows: print(f'   {r[\"pubid\"]}: {len(r[\"raw_answer\"].split())}w | {r[\"raw_answer\"][:70]!r}')
"
