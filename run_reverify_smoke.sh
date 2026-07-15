#!/bin/bash
#SBATCH --job-name=rv_smoke
#SBATCH --output=logs/rv_smoke_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:40:00
source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis
export TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/ablation
# SMOKE TEST ONLY (10 answers) -- checks each component swap loads and runs.
# The real runs are on the full 1000 and are submitted after this passes.
for CFG in "sat keybert facebook/bart-large-mnli" \
           "nltk keybert facebook/bart-large-mnli" \
           "sat tfidf facebook/bart-large-mnli" \
           "sat keybert roberta-large-mnli"; do
  set -- $CFG
  echo "######## smoke: splitter=$1 keywords=$2 nli=$3 ########"
  python -u src/reverify.py --limit 10 --splitter "$1" --keywords "$2" --nli "$3" \
      --out "results/ablation/smoke_$1_$2.json" 2>&1 | tail -14
  echo "  rc=$?"
done
echo "=== SMOKE DONE ==="
