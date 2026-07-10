# File-by-File Setup Guide

This guide is for creating the PubMedQA baseline RAG project calmly, one file at
a time. The full project already exists in this folder:

```text
C:\Users\bvenk\Documents\baseline rag
```

If you are uploading to HPC, the easiest and safest method is to upload this
whole folder. If you must recreate it manually, follow the order below.

## 1. Create the Main Project Folder

On Windows PowerShell:

```powershell
mkdir pubmedqa_baseline_rag
cd pubmedqa_baseline_rag
```

On HPC/Linux:

```bash
mkdir pubmedqa_baseline_rag
cd pubmedqa_baseline_rag
```

## 2. Create the Folder Structure

Create these folders:

```text
src
hpc
reports
outputs
data
data/faiss
data/cache
logs
```

Windows PowerShell:

```powershell
mkdir src
mkdir hpc
mkdir reports
mkdir outputs
mkdir data
mkdir data\faiss
mkdir data\cache
mkdir logs
```

HPC/Linux:

```bash
mkdir -p src hpc reports outputs data/faiss data/cache logs
```

## 3. Create Root Files First

Create these files in the main project folder:

```text
README.md
requirements.txt
.gitignore
FILE_BY_FILE_SETUP_GUIDE.md
```

Purpose:

| File | Purpose |
| --- | --- |
| `README.md` | Main explanation and run commands |
| `requirements.txt` | Python packages to install |
| `.gitignore` | Prevents generated/cache files from being committed |
| `FILE_BY_FILE_SETUP_GUIDE.md` | This guide |

## 4. Create Python Source Files

Create these inside the `src/` folder in this exact order:

```text
src/__init__.py
src/config.py
src/data.py
src/retrieval.py
src/prompts.py
src/generation.py
src/metrics.py
src/evaluate_retrieval.py
src/run_pipeline.py
src/evaluate_generation.py
src/evaluate_ragas.py
src/error_analysis.py
src/format_answers.py
src/generate_report.py
```

Purpose:

| File | Purpose |
| --- | --- |
| `src/__init__.py` | Marks `src` as a Python package |
| `src/config.py` | Stores dataset/model/output configuration |
| `src/data.py` | Loads PubMedQA and extracts context snippets |
| `src/retrieval.py` | Builds dense embeddings and FAISS retrieval |
| `src/prompts.py` | Contains the Chain-of-Thought RAG prompt |
| `src/generation.py` | Loads Mistral and generates answers |
| `src/metrics.py` | Implements retrieval metrics |
| `src/evaluate_retrieval.py` | Runs Precision@K, Recall@K, MRR, MAP |
| `src/run_pipeline.py` | Runs retrieval plus Mistral answer generation |
| `src/evaluate_generation.py` | Runs BLEU, ROUGE, and BERTScore |
| `src/evaluate_ragas.py` | Runs RAGAS metrics |
| `src/error_analysis.py` | Creates automatic error analysis summary |
| `src/format_answers.py` | Converts machine-readable JSONL answers into readable Markdown |
| `src/generate_report.py` | Creates final report draft from results |

## 5. Create HPC Files

Create these inside the `hpc/` folder:

```text
hpc/README_HPC.md
hpc/setup_env.sh
hpc/submit_pipeline.sh
hpc/01_retrieval.slurm
hpc/02_generation.slurm
hpc/03_generation_metrics.slurm
hpc/04_ragas.slurm
hpc/05_report.slurm
```

Purpose:

| File | Purpose |
| --- | --- |
| `hpc/README_HPC.md` | HPC instructions |
| `hpc/setup_env.sh` | Creates Python environment on HPC |
| `hpc/submit_pipeline.sh` | Submits main SLURM jobs in order |
| `hpc/01_retrieval.slurm` | Retrieval evaluation job |
| `hpc/02_generation.slurm` | GPU Mistral generation job |
| `hpc/03_generation_metrics.slurm` | BLEU/ROUGE/BERTScore and error analysis |
| `hpc/04_ragas.slurm` | RAGAS evaluation job |
| `hpc/05_report.slurm` | Final report generation job |

## 6. Create Report File

Create this inside the `reports/` folder:

```text
reports/architecture_and_error_analysis.md
```

This file contains the baseline architecture, Mermaid diagram, CoT prompt
rationale, error analysis methodology, improved architecture, and research
justification.

## 7. Final Folder Should Look Like This

```text
pubmedqa_baseline_rag/
├── README.md
├── requirements.txt
├── .gitignore
├── FILE_BY_FILE_SETUP_GUIDE.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── retrieval.py
│   ├── prompts.py
│   ├── generation.py
│   ├── metrics.py
│   ├── evaluate_retrieval.py
│   ├── run_pipeline.py
│   ├── evaluate_generation.py
│   ├── evaluate_ragas.py
│   ├── error_analysis.py
│   └── generate_report.py
├── hpc/
│   ├── README_HPC.md
│   ├── setup_env.sh
│   ├── submit_pipeline.sh
│   ├── 01_retrieval.slurm
│   ├── 02_generation.slurm
│   ├── 03_generation_metrics.slurm
│   ├── 04_ragas.slurm
│   └── 05_report.slurm
├── reports/
│   └── architecture_and_error_analysis.md
├── outputs/
├── data/
│   ├── faiss/
│   └── cache/
└── logs/
```

## 8. How to Run on HPC

After uploading the folder to HPC:

```bash
cd pubmedqa_baseline_rag
bash hpc/setup_env.sh
bash hpc/submit_pipeline.sh
```

The generated results will appear in:

```text
outputs/
```

The final report draft will appear at:

```text
outputs/final_report_draft.md
```

## 9. Important Advice

Do not type the code manually if you can avoid it. Copy the existing files from:

```text
C:\Users\bvenk\Documents\baseline rag
```

Manual typing often creates small errors such as wrong indentation, missing
quotes, or changed filenames. Python is very sensitive to these details.
