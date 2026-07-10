#!/bin/bash
#SBATCH --job-name=adv_v2_smoke
#SBATCH --output=logs/adv_v2_smoke_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:40:00
source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis
export PUBMEDQA_OUTPUT_DIR="$SLURM_SUBMIT_DIR/outputs_v2_sample"
export TOKENIZERS_PARALLELISM=false
date +"=== start %T ==="
python -u src/run_advanced_pipeline.py --sample-size 5 --strict
echo "=== exit: $? ==="
date +"=== end %T ==="
