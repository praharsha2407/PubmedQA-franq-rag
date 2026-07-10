# Resuming the advanced (FRANQ) pipeline after HPC maintenance

## TL;DR

```bash
cd "/mnt/aiongpfs/users/pkonda/pubmedqa_baseline_rag/baseline rag"
wc -l outputs/advanced_answers.jsonl     # how many examples survived
sbatch run_advanced_full.sh              # continues from there
```

**Do not delete `outputs/advanced_answers.jsonl`.** That file *is* the checkpoint.
`run_advanced_full.sh` passes `--resume`, which reads the completed `pubid`s out of
that file, skips them, and appends only new rows.

## Before resubmitting, check three things

**1. Is the HF token still valid?** SPLADE-v3 is a gated repo. Without a working
token, `--strict` aborts the job rather than silently running BM25.

```bash
python -c "from huggingface_hub import whoami; print(whoami()['name'])"
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('naver/splade-v3','config.json'); print('SPLADE OK')"
```

If either fails: `huggingface-cli login` and paste a **Read** token.
(Type it at the prompt. Never paste a token into a chat or a script.)

**2. Is the conda env intact?**

```bash
source /home/users/pkonda/miniconda3/etc/profile.d/conda.sh && conda activate rag_thesis
python -c "import torch, transformers, wtpsplit, scispacy, keybert; print('env OK')"
```

**3. Are the models still cached?** (`~/.cache/huggingface/hub`)
Needed: `naver/splade-v3`, `facebook/bart-large-mnli`, `pritamdeka/S-PubMedBert-MS-MARCO`,
`cross-encoder/ms-marco-MiniLM-L-6-v2`, `BioMistral/BioMistral-7B`, `roberta-large`.
Compute nodes have internet, so a missing one will re-download; it just costs time.

## What the flags mean

| Flag | Effect |
|---|---|
| `--strict` | Abort if any stage substitutes a component (SPLADE→BM25, SaT→NLTK, UMLS skipped, BioMistral→stub). **Never report numbers from a run without this.** |
| `--resume` | Skip pubids already in `advanced_answers.jsonl`, append new ones. |
| `--sample-size N` | Only the first N examples. Use for testing, not for results. |

## How partial runs still produce valid metrics

The consolidated report is rebuilt **from the answers file** at the end of every run,
not accumulated in memory. Each row stores its own `hallucination_report` and
`mean_franq_score`. So a run that processes examples 500-1000 still emits a report
covering all 1000, because rows 1-499 are read back off disk.

## Outputs

- `outputs/advanced_answers.jsonl` — one row per example (the checkpoint)
- `outputs/full_evaluation_report.json` — retrieval + generation + FRANQ metrics

Neither touches the baseline's `outputs/answers.jsonl`.

## Known issues (not blockers)

- `run_advanced_pipeline.py` hardcodes `p_true_given_faithful = 0.95` and the
  `raw_franq < 0.7` threshold instead of reading `config.franq`. The values match
  `FranqConfig`'s defaults, so results are correct; it's a reproducibility nicety.
- `build_final_answer` renders citations as `[Context 21645374:0]` (a chunk_id),
  though its docstring says an integer index. Cosmetic; affects no metric.
- Retrieval searches the shared 3358-chunk corpus, so an answer can cite a chunk
  from a *different* PubMed article. That is real cross-document retrieval, and the
  FRANQ verifier is what catches the resulting extrinsic hallucinations. Worth a
  paragraph in the error analysis rather than a code change.

## Licensing to cite in the methodology

- **SPLADE-v3** (`naver/splade-v3`): CC-BY-NC-SA, non-commercial. Fine for a thesis.
- **UMLS** (via scispaCy linker): requires a free UTS account and license acceptance.
