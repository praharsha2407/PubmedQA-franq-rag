# Troubleshooting

## Windows Error: `RuntimeError: operator torchvision::nms does not exist`

This project does not use images or `torchvision`. This error happens when
`torchvision` is installed but does not match the installed PyTorch version.
`transformers` detects `torchvision`, imports it, and then fails.

Fix on Windows:

```powershell
pip uninstall -y torchvision
pip install -r requirements-windows.txt
```

Then rerun:

```powershell
python src/evaluate_retrieval.py --sample-size 10 --top-k 5 --save-index
```

The code also contains a local safety fallback: if `sentence-transformers` cannot
load because of a Windows PyTorch/torchvision issue, retrieval will continue with
a lightweight hashing encoder. This is only for checking that the pipeline runs.
For final academic results, run on HPC with the real biomedical embedding model.

## Windows Error While Loading Mistral

If `run_pipeline.py` fails while importing `MistralForCausalLM`, it is usually
the same PyTorch/torchvision environment issue. The code now catches this and
uses a simple evidence-based fallback generator so you can create
`outputs/answers.jsonl` locally and test the metric/report scripts.

The fallback generator is only for local pipeline testing. Do not report those
fallback answers as final Mistral results. The official experiment should be run
on HPC with:

```bash
bash hpc/setup_env.sh
bash hpc/submit_pipeline.sh
```

## Python 3.14 Warning

Your traceback shows Python 3.14:

```text
Python314
```

For NLP/ML packages, Python 3.10 or 3.11 is much safer. If package errors keep
happening on Windows, create a Python 3.11 environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-windows.txt
```

Then test:

```powershell
python src/evaluate_retrieval.py --sample-size 10 --top-k 5 --save-index
```

## If Windows Still Gives Package Errors

Use Windows only for editing files and small checks. Run the full project on HPC:

```bash
bash hpc/setup_env.sh
bash hpc/submit_pipeline.sh
```
