#!/bin/bash
#SBATCH --job-name=diag_stored
#SBATCH --output=logs/diag_stored_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis
export TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"
python -u src/diag_stored_prompt.py --answers outputs/answers.jsonl --n 20
echo "=== exit: $? ==="
