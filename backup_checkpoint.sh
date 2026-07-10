#!/bin/bash
# Snapshot the advanced-pipeline checkpoint. Cheap insurance: this file is the only
# thing standing between a maintenance kill and re-running 1000 GPU examples.
cd "$(dirname "$0")"
SRC=outputs/advanced_answers.jsonl
[ -f "$SRC" ] || { echo "no checkpoint at $SRC"; exit 1; }
mkdir -p archive/checkpoints
DST="archive/checkpoints/advanced_answers_$(date +%Y%m%d_%H%M%S)_$(wc -l < "$SRC")rows.jsonl"
cp "$SRC" "$DST"
echo "saved $(wc -l < "$SRC") rows -> $DST"
