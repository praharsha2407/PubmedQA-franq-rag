# PubMedQA Baseline RAG with Chain-of-Thought Prompting

This project implements a university-level baseline Retrieval-Augmented Generation
(RAG) system for the PubMedQA `pqa_labeled` training split.

## Scope

- Dataset: `qiaojin/PubMedQA`, subset `pqa_labeled`, split `train`
- Retrieval: dense embeddings + FAISS cosine search
- Generation: Mistral instruction model
- Prompting: retrieval-augmented Chain-of-Thought style reasoning
- Evaluation:
  - Retrieval: Precision@K, Recall@K, MRR, MAP
  - Generation: BLEU, ROUGE-1/2/L, BERTScore
  - RAG: RAGAS faithfulness, answer relevancy, context precision, context recall
- Output:
  - generated answers
  - metric files
  - error analysis
  - report-ready architecture and improvement proposal

## Quick Start

Use Python 3.10 or 3.11 for the smoothest compatibility with PyTorch, FAISS,
RAGAS, and transformer tooling.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Windows, use the lighter test requirements because FAISS, RAGAS, and
bitsandbytes can require extra system compilers:

```powershell
pip install -r requirements-windows.txt
```

On HPC, install the full baseline requirements and optional RAGAS/4-bit extras:

```bash
pip install -r requirements.txt
pip install -r requirements-hpc-extra.txt
```

Set your Hugging Face token if you use a gated Mistral model:

```powershell
$env:HF_TOKEN="your_huggingface_token"
```

RAGAS metrics require an evaluator LLM and embeddings. By default, RAGAS commonly
uses OpenAI-compatible settings if available, so set the relevant API key before
running `src/evaluate_ragas.py`, or adapt that script to your university-approved
local evaluator model.

Run the full baseline on a small sample first:

```powershell
python src/run_pipeline.py --sample-size 50 --top-k 5
```

Mistral-7B generation is hardware intensive. If your local machine does not have
enough GPU memory, run the retrieval and metric code locally, then execute
generation on Google Colab/Kaggle/HPC with a GPU, or change `--model-name` to a
smaller Mistral-compatible checkpoint approved by your supervisor.

Run retrieval-only evaluation:

```powershell
python src/evaluate_retrieval.py --sample-size 200 --top-k 5
```

Run generation and RAGAS evaluation after answers have been generated:

```powershell
python src/evaluate_generation.py
python src/evaluate_ragas.py
python src/error_analysis.py
python src/format_answers.py
```

## HPC / SLURM Run

For your HPC, use the scripts in `hpc/`.

```bash
bash hpc/setup_env.sh
bash hpc/submit_pipeline.sh
```

This submits retrieval, Mistral generation, generation metrics, error analysis,
and final report creation with SLURM dependencies. See `hpc/README_HPC.md` for
manual `sbatch` commands and RAGAS instructions.

## Important Notes

The baseline uses the PubMedQA long answer as the reference answer. For retrieval
evaluation, the gold evidence for each question is the set of context snippets
provided in the same PubMedQA record. The FAISS corpus is built from all provided
context snippets in the selected split.

The prompt asks the model to reason step by step internally and return a concise,
evidence-grounded explanation. This keeps the research design aligned with
Chain-of-Thought prompting while reducing unsupported free-form speculation.

## Outputs

Generated files are written under `outputs/`:

- `answers.jsonl`
- `answers_readable.md`
- `retrieval_metrics.json`
- `generation_metrics.json`
- `ragas_metrics.json`
- `error_analysis.md`

## Report Material

See `reports/architecture_and_error_analysis.md` for the baseline architecture,
Mermaid diagrams, error analysis methodology, and proposed improved architecture.

## Troubleshooting

See `TROUBLESHOOTING.md` if Windows package installation or PyTorch/torchvision
errors occur.
