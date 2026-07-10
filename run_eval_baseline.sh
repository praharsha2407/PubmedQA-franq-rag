#!/bin/bash
#SBATCH --job-name=eval_base
#SBATCH --output=logs_eval_base_%j.out
#SBATCH --partition=gpu     
#SBATCH --gres=gpu:1        
#SBATCH --mem=32G           
#SBATCH --time=12:00:00     

source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis

python src/evaluate_hallucinations_baseline.py
