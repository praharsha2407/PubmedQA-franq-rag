#!/bin/bash
#SBATCH --job-name=adv_full_v2
#SBATCH --output=logs/adv_full_v2_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --requeue

# v2 of the advanced pipeline. Changes vs v1 (job 5522230):
#   1. Faithfulness = max NLI entailment over ALL top-k chunks (was: one chunk
#      picked by keyword overlap, often an unrelated one).
#   2. Keyword-overlap tie-break uses '>' not '>=' (the '>=' bug made a sentence
#      with zero overlap select the LAST, lowest-ranked chunk as its premise).
#   3. HybridRetrievalConfig.final_top_k 20 -> 40, so the cross-encoder scores the
#      whole dense-union-sparse pool instead of half of it.
#
# Writes to outputs_v2/ so v1 results in outputs/ are untouched and comparable.

source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis

export PUBMEDQA_OUTPUT_DIR="$SLURM_SUBMIT_DIR/outputs_v2"
export TOKENIZERS_PARALLELISM=false

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date +"=== start: %F %T ==="

python -u src/run_advanced_pipeline.py --strict --resume
RC=$?

date +"=== end: %F %T ==="
echo "=== exit: $RC ==="
exit $RC
