#!/bin/bash
#SBATCH --job-name=diag_base
#SBATCH --output=logs/diag_base_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:25:00
source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh
conda activate rag_thesis
export TOKENIZERS_PARALLELISM=false
python - <<'PY'
import sys; sys.path.insert(0,'src')
import transformers, torch
print("transformers:", transformers.__version__, "torch:", torch.__version__)
from config import PipelineConfig
from data import load_pubmedqa, build_context_corpus
from retrieval import DenseRetriever
from generation import MistralGenerator
from prompts import build_cot_rag_prompt

cfg = PipelineConfig()
ds = load_pubmedqa(cfg.dataset)
texts, meta = build_context_corpus(ds)
r = DenseRetriever(cfg.retrieval); r.build(texts, meta)
g = MistralGenerator(cfg.generation)

print("\n=== Reproducing the BASELINE exactly (run_pipeline.py path) ===")
for ex in ds[:4]:
    chunks = r.search(ex.question, top_k=cfg.retrieval.top_k)
    out = g.generate(build_cot_rag_prompt(ex.question, chunks))
    print(f"  {ex.pubid}  Conclusion={'YES' if 'Conclusion' in out else 'no'}  words={len(out.split())}")
    print(f"     tail: ...{out[-90:].strip()}")
PY
echo "=== exit: $? ==="
