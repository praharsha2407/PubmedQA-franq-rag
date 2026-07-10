"""Sensitivity of the hallucination rates to the FRANQ `verified_threshold`.

Every hallucination number the pipeline reports depends on one asserted constant:
a sentence counts as verified iff raw_franq_score >= 0.70. That 0.70 is not derived
from anything. This script re-derives the rates at a range of thresholds directly
from the per-sentence scores already stored in the answers file -- no GPU, no rerun.

Report the curve, not a single point. If the ranking of two systems flips across
plausible thresholds, neither system is meaningfully better.

    python src/threshold_sweep.py --answers outputs_v2/advanced_answers.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_sentences(path: Path) -> list[dict]:
    sentences = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        sentences.extend(row["hallucination_report"]["details"])
    return sentences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True)
    parser.add_argument("--compare", default=None, help="second answers file to overlay")
    args = parser.parse_args()

    sets = [("A", load_sentences(Path(args.answers)))]
    if args.compare:
        sets.append(("B", load_sentences(Path(args.compare))))

    print(f"  A = {args.answers}")
    if args.compare:
        print(f"  B = {args.compare}")
    for tag, sents in sets:
        scores = [s["raw_franq_score"] for s in sents]
        scores.sort()
        n = len(scores)
        print(f"\n  {tag}: {n} sentences   "
              f"franq min={scores[0]:.3f} p25={scores[n//4]:.3f} median={scores[n//2]:.3f} "
              f"p75={scores[3*n//4]:.3f} max={scores[-1]:.3f}")

    print(f"\n{'threshold':>10} " + " ".join(f"{tag+' verified%':>14}" for tag, _ in sets))
    print("  " + "-" * (10 + 15 * len(sets)))
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        cells = []
        for _, sents in sets:
            verified = sum(1 for s in sents if s["raw_franq_score"] >= thresh)
            cells.append(f"{verified / len(sents):>13.1%}")
        marker = "  <- current" if abs(thresh - 0.70) < 1e-9 else ""
        print(f"{thresh:>10.2f} " + " ".join(cells) + marker)

    if len(sets) == 2:
        print("\n  Does the A-vs-B ranking hold across thresholds?")
        flips = []
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            rates = [sum(1 for s in sents if s["raw_franq_score"] >= thresh) / len(sents) for _, sents in sets]
            flips.append(rates[1] > rates[0])
        print("    B better than A at each of 0.5/0.6/0.7/0.8/0.9:", flips)
        print("    STABLE" if len(set(flips)) == 1 else "    UNSTABLE -- the ranking depends on the threshold")


if __name__ == "__main__":
    main()
