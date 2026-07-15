#!/bin/bash
#SBATCH --job-name=abl_rrfk
#SBATCH --output=logs/abl_rrfk_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis
export TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"

echo "################ DEV (n=300) ################"
python -u src/ablate_rrf_k.py --dev-size 300 --offset 0 \
    --out results/ablation_rrf_k_dev.json

echo
echo "################ HELD-OUT (n=700) ################"
python -u src/ablate_rrf_k.py --dev-size 700 --offset 300 \
    --out results/ablation_rrf_k_heldout.json
echo "=== exit: $? ==="
