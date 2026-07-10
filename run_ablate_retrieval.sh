#!/bin/bash
#SBATCH --job-name=ablate_ret
#SBATCH --output=logs/ablate_ret_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:35:00
source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis
export TOKENIZERS_PARALLELISM=false
date +"=== start %T ==="
python -u src/ablate_retrieval.py --dev-size 700 --offset 300 --out outputs_v2/retrieval_ablation_heldout.json
echo "=== exit: $? ==="
date +"=== end %T ==="
