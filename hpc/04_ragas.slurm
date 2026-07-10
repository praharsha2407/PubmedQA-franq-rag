#!/usr/bin/env bash
#SBATCH --job-name=pubmedqa-ragas
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs data/cache

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

# Run evaluation on the full 1000-sample dataset
python src/evaluate_custom.py --full-dataset
