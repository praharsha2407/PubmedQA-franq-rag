#!/usr/bin/env python3
import json
from pathlib import Path

OUT_DIR = Path(".") / "outputs"
pattern = "answers.*.jsonl"
files = sorted(OUT_DIR.glob(pattern))
if not files:
    print("No per-task output files found in outputs/ matching", pattern)
    raise SystemExit(1)

merged = OUT_DIR / "answers.merged.jsonl"
count_in = 0
count_ok = 0
count_bad = 0
with merged.open("w", encoding="utf-8") as outf:
    for f in files:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), start=1):
            count_in += 1
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                outf.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count_ok += 1
            except Exception:
                count_bad += 1
                print(f"Skipping malformed JSON in {f} line {i}")

print(f"Merged {count_ok} valid lines ({count_bad} malformed) from {len(files)} files into {merged}")
