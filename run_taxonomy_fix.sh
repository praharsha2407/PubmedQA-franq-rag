#!/bin/bash
#SBATCH --job-name=tax_fix
#SBATCH --output=logs/tax_fix_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:30:00
source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis
export TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"

echo "########## corrected keyword overlap -> corrected taxonomy ##########"
python -u src/recompute_taxonomy.py \
    --answers outputs_matched/advanced_answers_cot.jsonl \
    --out outputs_matched/taxonomy_corrected.json

echo
echo "########## threshold sweep: is 82% just an artifact of the 0.70 cutoff? ##########"
python -u src/threshold_sweep.py --answers outputs_matched/advanced_answers_cot.jsonl
echo "=== exit: $? ==="
