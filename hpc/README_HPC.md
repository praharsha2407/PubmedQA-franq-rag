# HPC Running Guide

These scripts assume a SLURM-based HPC cluster. They are intentionally generic:
you may only need to change the module names at the top of each `.slurm` file to
match your university cluster.

## 1. Create the Python Environment

Run this once on a login node:

```bash
bash hpc/setup_env.sh
```

If your HPC provides a specific Python module, edit `hpc/setup_env.sh` and load
that module before creating the virtual environment.

## 2. Submit Jobs

To submit the main pipeline with dependencies:

```bash
bash hpc/submit_pipeline.sh
```

Or submit each step manually:

Run retrieval evaluation first:

```bash
sbatch hpc/01_retrieval.slurm
```

Run generation on a GPU node:

```bash
sbatch hpc/02_generation.slurm
```

Run generation and automatic metrics:

```bash
sbatch hpc/03_generation_metrics.slurm
```

Run RAGAS only if you have configured an evaluator LLM key:

```bash
export OPENAI_API_KEY="your_key"
sbatch --export=ALL hpc/04_ragas.slurm
```

Create the final markdown report draft:

```bash
sbatch hpc/05_report.slurm
```

## Output Locations

By default, outputs are written to:

- `outputs/`
- `data/faiss/`
- `data/cache/`

On HPC, the SLURM files set:

- `PUBMEDQA_OUTPUT_DIR=$SLURM_SUBMIT_DIR/outputs`
- `PUBMEDQA_INDEX_DIR=$SLURM_SUBMIT_DIR/data/faiss`
- `PUBMEDQA_CACHE_DIR=$SLURM_SUBMIT_DIR/data/cache`

If your HPC has a scratch directory, change those variables to scratch paths.

## Recommended Full Run

For the final project, use the full PubMedQA training split:

```bash
python src/evaluate_retrieval.py --top-k 5 --save-index
python src/run_pipeline.py --full-dataset --top-k 5 --load-in-4bit
python src/evaluate_generation.py
python src/error_analysis.py
python src/generate_report.py
```

RAGAS may be run separately because it requires an evaluator LLM.
