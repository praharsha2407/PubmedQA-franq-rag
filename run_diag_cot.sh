#!/bin/bash
#SBATCH --job-name=diag_cot
#SBATCH --output=logs/diag_cot_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis
export TOKENIZERS_PARALLELISM=false
python -u src/diag_cot_conclusion.py --n 6
echo "=== exit: $? ==="
