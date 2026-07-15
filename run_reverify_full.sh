#!/bin/bash
#SBATCH --job-name=rv_full
#SBATCH --output=logs/rv_full_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --requeue
# Re-verification ablations on the FULL 1000 stored answers. No generation.
# Usage: sbatch run_reverify_full.sh <group>     group = splitter | keynli
source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis
export TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/ablation
GROUP=$1

run() {  # splitter keywords nli tag
  echo "############ $4 ############"
  date +"  start %T"
  python -u src/reverify.py \
      --answers outputs_matched/advanced_answers_cot.jsonl \
      --splitter "$1" --keywords "$2" --nli "$3" \
      --out "results/ablation/$4.json" | tail -16
  date +"  end   %T"
}

if [ "$GROUP" = "splitter" ]; then
  run sat   keybert facebook/bart-large-mnli  REF_sat_keybert_bart
  run nltk  keybert facebook/bart-large-mnli  SPLIT_nltk
  run spacy keybert facebook/bart-large-mnli  SPLIT_spacy
else
  run sat tfidf   facebook/bart-large-mnli    KW_tfidf
  run sat keybert roberta-large-mnli          NLI_roberta
  run sat keybert microsoft/deberta-large-mnli NLI_deberta
fi
echo "=== DONE $GROUP ==="
