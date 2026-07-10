#!/bin/bash
#SBATCH --job-name=diag_prior
#SBATCH --output=logs/diag_prior_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis
export TOKENIZERS_PARALLELISM=false
python -u src/diag_faithful_prior.py --answers scratch/diag_in.jsonl --limit 40
echo "=== exit: $? ==="
