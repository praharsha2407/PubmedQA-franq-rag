"""Is the baseline still reproducible today?

The baseline's answers.jsonl stores the EXACT prompt string it used, and every one of its
1000 stored answers contains a "Conclusion:" line. The advanced pipeline, using the same
prompt template and the same model name, produces one in only 1.2% of answers.

This removes every variable except the generator itself: it replays the baseline's OWN
stored prompts, verbatim, through today's model, and asks whether the Conclusion comes back.

  - If Conclusion reappears  -> the generator is fine; the difference is upstream
                                (context/chunks), and the baseline is reproducible.
  - If Conclusion stays gone -> the environment has changed since the baseline was run.
                                The stored baseline answers came from a different model
                                state, and comparing them against a fresh advanced run is
                                comparing two different systems.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import AdvancedPipelineConfig
from generation import MistralGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", default="outputs/answers.jsonl")
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    rows = []
    for line in Path(args.answers).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
            if len(rows) >= args.n:
                break

    generator = MistralGenerator(AdvancedPipelineConfig().generation)
    if generator.using_fallback_generator:
        raise SystemExit("needs the real model on a GPU")

    stored_hits = fresh_hits = 0
    for i, row in enumerate(rows, 1):
        stored_answer = row["generated_answer"]
        stored_has = "Conclusion" in stored_answer
        stored_hits += stored_has

        # Replay the stored prompt verbatim -- no rebuilding, no chunk differences.
        fresh = generator.generate(row["prompt"])
        fresh_has = "Conclusion" in fresh
        fresh_hits += fresh_has

        print(f"[{i:>3}] pubid={row['pubid']:<10} stored_Conclusion={'YES' if stored_has else 'no ':<3} "
              f"fresh_Conclusion={'YES' if fresh_has else 'no ':<3} fresh_words={len(fresh.split()):>3}",
              flush=True)
        if i <= 2:
            print(f"      stored: {stored_answer[:160]!r}")
            print(f"      fresh : {fresh[:160]!r}", flush=True)

    n = len(rows)
    print("\n" + "=" * 66)
    print(f"stored answers with 'Conclusion': {stored_hits}/{n}")
    print(f"fresh  answers with 'Conclusion': {fresh_hits}/{n}   (same prompts, today's model)")
    print("=" * 66)
    if stored_hits > 0 and fresh_hits == 0:
        print("VERDICT: the generator no longer reproduces the baseline on its OWN prompts.")
        print("The environment changed. The stored baseline and a fresh advanced run are")
        print("NOT comparable -- the baseline must be re-run before any accuracy claim.")
    elif fresh_hits >= 0.8 * stored_hits and stored_hits:
        print("VERDICT: the generator reproduces the baseline. The Conclusion loss in the")
        print("advanced pipeline is caused UPSTREAM (retrieved context), not by the model.")
    else:
        print("VERDICT: partial reproduction -- inspect the per-row output above.")


if __name__ == "__main__":
    main()
