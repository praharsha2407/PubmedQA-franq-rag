#!/usr/bin/env bash
#SBATCH --job-name=pubmedqa-genmetrics
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs data/cache

# module purge
# module load python/3.11

if [ -f ".venv/bin/activate" ]; then
	source .venv/bin/activate
else
	if [ ! -d ".venv" ]; then
		echo ".venv not found — creating virtualenv (this may take a few minutes)..." >&2
		bash hpc/setup_env.sh || true
	fi
	if [ -f ".venv/bin/activate" ]; then
		source .venv/bin/activate
	elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
		source "$HOME/miniconda3/etc/profile.d/conda.sh"
		conda activate rag
	else
		echo "Warning: no .venv and conda not found; proceeding without a virtual environment" >&2
	fi
fi
export PUBMEDQA_OUTPUT_DIR="$SLURM_SUBMIT_DIR/outputs"
export PUBMEDQA_CACHE_DIR="$SLURM_SUBMIT_DIR/data/cache"
export HF_HOME="$PUBMEDQA_CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

# Merge chunk outputs if generation was run as a SLURM array
if ls outputs/answers.*.jsonl 1> /dev/null 2>&1; then
    echo "Merging chunked answer files..."
    cat outputs/answers.*.jsonl > outputs/answers.jsonl
fi

python src/evaluate_generation.py
python src/error_analysis.py