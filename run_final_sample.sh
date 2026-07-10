#!/bin/bash
#SBATCH --job-name=final_smoke
#SBATCH --output=logs/final_smoke_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis
export PUBMEDQA_OUTPUT_DIR="$SLURM_SUBMIT_DIR/outputs_final_sample"
export TOKENIZERS_PARALLELISM=false
python -u src/run_advanced_pipeline.py --sample-size 5 --strict --prompt cot
echo "=== exit: $? ==="
python src/evaluate_decision.py --answers outputs_final_sample/advanced_answers_cot.jsonl --field raw_answer 2>/dev/null | head -12
