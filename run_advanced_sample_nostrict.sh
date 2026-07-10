#!/bin/bash
#SBATCH --job-name=adv_smoke
#SBATCH --output=logs/adv_smoke_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:50:00
source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis
export PUBMEDQA_OUTPUT_DIR="$SLURM_SUBMIT_DIR/outputs_sample"
export TOKENIZERS_PARALLELISM=false
echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date +"=== start: %T ==="
python -u src/run_advanced_pipeline.py --sample-size 5
echo "=== exit: $? ==="
date +"=== end: %T ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader
