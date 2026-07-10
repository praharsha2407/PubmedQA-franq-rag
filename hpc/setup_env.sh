#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Edit these module commands for your university HPC if needed.
# module purge
# module load python/3.11
# module load cuda/12.1

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
python -m pip install -r requirements-hpc-extra.txt

echo "Environment ready. Activate with: source .venv/bin/activate"
