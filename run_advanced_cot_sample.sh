#!/bin/bash
#SBATCH --job-name=adv_cot_smoke
#SBATCH --output=logs/adv_cot_smoke_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:40:00
source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis
export PUBMEDQA_OUTPUT_DIR="$SLURM_SUBMIT_DIR/outputs_cot_sample"
export TOKENIZERS_PARALLELISM=false
echo "=== node: $(hostname) ==="
date +"=== start %T ==="
python -u src/run_advanced_pipeline.py --sample-size 5 --strict --prompt cot
echo "=== exit: $? ==="
date +"=== end %T ==="
