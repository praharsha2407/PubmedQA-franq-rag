#!/bin/bash
#SBATCH --job-name=adv_full
#SBATCH --output=logs/adv_full_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --requeue

# Full-dataset run of the FRANQ advanced pipeline (stages 1-12).
# PubMedQA pqa_labeled: 1000 questions, 3358 unique context chunks.
#
# --strict : abort rather than silently substitute a component (SPLADE->BM25,
#            SaT->NLTK, UMLS skipped, BioMistral->stub). Never report numbers
#            from a run that did not use --strict.
# --resume : skip pubids already in outputs/advanced_answers.jsonl and append.
#            At ~18 s/example this run takes roughly 5 h, so it will very likely
#            be interrupted by the maintenance window. Resubmitting this exact
#            script afterwards continues from where it stopped; the consolidated
#            report is rebuilt from the answers file, so partial runs still score.
#
# Writes outputs/advanced_answers.jsonl + outputs/full_evaluation_report.json.
# Does NOT touch the baseline's outputs/answers.jsonl.

source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis

export TOKENIZERS_PARALLELISM=false

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date +"=== start: %F %T ==="

python -u src/run_advanced_pipeline.py --strict --resume
RC=$?

date +"=== end: %F %T ==="
echo "=== exit: $RC ==="
if [ $RC -ne 0 ]; then
  echo "NOTE: non-zero exit. If this was a walltime/maintenance kill, just resubmit"
  echo "      this same script -- --resume will continue from the completed pubids."
fi
exit $RC
