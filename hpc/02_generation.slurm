#!/usr/bin/env bash
#SBATCH --job-name=pubmedqa-mistral
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs data/faiss data/cache

# Edit these module commands for your university HPC if needed.
# module purge
# module load python/3.11
# module load cuda/12.1

if [ -f ".venv/bin/activate" ]; then
	source .venv/bin/activate
	export PYTHONUNBUFFERED=1
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
export PUBMEDQA_INDEX_DIR="$SLURM_SUBMIT_DIR/data/faiss"
export PUBMEDQA_CACHE_DIR="$SLURM_SUBMIT_DIR/data/cache"
export HF_HOME="$PUBMEDQA_CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

# If you have a HF token set in your environment, it will be exported
export HF_TOKEN="${HF_TOKEN:-}"

# Ensure required packages are available on the compute node (best-effort)
python -m pip install -U "bitsandbytes>=0.46.1" ragas || true

# Use cached model files under data/cache when possible (we predownloaded them).
export TRANSFORMERS_CACHE="$SLURM_SUBMIT_DIR/data/cache/models"

# Try running with 4-bit quantization first; if it fails, retry without 4-bit.
# We temporarily disable 'set -e' around the trial run so we can retry.
set +e
# Support SLURM array chunking: when SLURM_ARRAY_TASK_ID is set we compute
# start/end indices using CHUNK_SIZE (default 500) so the dataset can be split
CHUNK_SIZE=${CHUNK_SIZE:-500}
if [ -n "${SLURM_ARRAY_TASK_ID-}" ]; then
	TASK_ID=$SLURM_ARRAY_TASK_ID
	START_INDEX=$(( (TASK_ID - 1) * CHUNK_SIZE ))
	END_INDEX=$(( START_INDEX + CHUNK_SIZE ))
	OUTFILE="answers_chunk_${TASK_ID}.jsonl"
	echo "Output file for this task: $OUTFILE" >&2	
	echo "Running chunk task $TASK_ID -> start=$START_INDEX end=$END_INDEX out=$OUTFILE" >&2
	# try with 4-bit first
	python src/run_pipeline.py --top-k 5 --start-index $START_INDEX --end-index $END_INDEX --output-file $OUTFILE --load-in-4bit
	rc=$?
	if [ $rc -ne 0 ]; then
		echo "Mistral chunk run failed with 4-bit quantization (rc=$rc). Retrying without 4-bit..." >&2
		python src/run_pipeline.py --top-k 5 --start-index $START_INDEX --end-index $END_INDEX --output-file $OUTFILE --no-load-in-4bit
	fi
else
	# full-dataset mode (legacy)
	python src/run_pipeline.py --full-dataset --top-k 5 --load-in-4bit
	rc=$?
	set -e
	if [ $rc -ne 0 ]; then
		echo "Mistral run failed with 4-bit quantization (rc=$rc). Retrying without 4-bit..." >&2
		python src/run_pipeline.py --full-dataset --top-k 5 --no-load-in-4bit
	fi
fi
