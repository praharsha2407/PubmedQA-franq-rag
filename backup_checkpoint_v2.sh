#!/bin/bash
cd "$(dirname "$0")"
SRC=outputs_v2/advanced_answers.jsonl
[ -f "$SRC" ] || exit 0
mkdir -p archive/checkpoints_v2
N=$(wc -l < "$SRC")
cp "$SRC" "archive/checkpoints_v2/v2_$(date +%H%M%S)_${N}rows.jsonl"
echo "  backed up $N rows"
