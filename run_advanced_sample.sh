#!/bin/bash
#SBATCH --job-name=adv_sample
#SBATCH --output=logs_adv_sample_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:45:00

# 5-example GPU validation of the FRANQ advanced pipeline (stages 1-12).
# --strict aborts if any stage falls back to a substitute component, so a
# green run here means the real components actually loaded on a GPU node.
#
# Writes to outputs_sample/ so it cannot clobber the baseline's outputs/.

source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis

export PUBMEDQA_OUTPUT_DIR="$SLURM_SUBMIT_DIR/outputs_sample"
export TOKENIZERS_PARALLELISM=false

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== starting 5-example strict run ==="

python -u src/run_advanced_pipeline.py --sample-size 5 --strict

echo "=== exit code: $? ==="
