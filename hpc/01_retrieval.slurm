#!/usr/bin/env bash
#SBATCH --job-name=pubmedqa-retrieval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs data/faiss data/cache

# Edit these module commands for your university HPC if needed.
# module purge
# module load python/3.11

if [ -f ".venv/bin/activate" ]; then
	source .venv/bin/activate
else
	# If no virtualenv exists, try to create it (may take time) or fallback to conda
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
export PUBMEDQA_INDEX_DIR="$SLURM_SUBMIT_DIR/data/faiss"
export PUBMEDQA_CACHE_DIR="$SLURM_SUBMIT_DIR/data/cache"
export HF_HOME="$PUBMEDQA_CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

python src/evaluate_retrieval.py --top-k 5 --save-index
