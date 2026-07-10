"""PubMedQA task metrics: accuracy and macro-F1 over the yes/no/maybe decision.

This is the metric PubMedQA is actually benchmarked on, and it was missing from
the pipeline entirely. BLEU/ROUGE/BERTScore measure how the answer is phrased;
this measures whether it is right.

Gold labels come from the dataset (`final_decision`). The prediction is parsed out
of the generated answer text. Any answer with no parseable decision is counted
separately as `unparseable` and, in the strict scoring, as WRONG -- silently
dropping them would inflate accuracy for a model that simply refuses to commit.
Both scorings are reported.

    python src/evaluate_decision.py --answers outputs_v2/advanced_answers.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from config import DatasetConfig
from data import load_pubmedqa

LABELS = ("yes", "no", "maybe")

# Ordered most-explicit first: an explicit "Conclusion: no" must beat a stray
# "yes" appearing earlier in the reasoning.
_PATTERNS = (
    re.compile(r"conclusion\s*:?\s*\**\s*(yes|no|maybe)\b", re.I),
    re.compile(r"final answer\s*:?\s*\**\s*(yes|no|maybe)\b", re.I),
    re.compile(r"the answer (?:to the question )?is\s*[\"']?\s*(yes|no|maybe)\b", re.I),
    re.compile(r"^\s*\**\s*(yes|no|maybe)\b", re.I),
)


def extract_decision(answer: str) -> str | None:
    for pattern in _PATTERNS:
        match = pattern.search(answer)
        if match:
            return match.group(1).lower()
    # Last resort: a bare label in the opening clause, where a direct answer lives.
    match = re.search(r"\b(yes|no|maybe)\b", answer[:150], re.I)
    return match.group(1).lower() if match else None


def macro_f1(pairs: list[tuple[str, str]]) -> tuple[float, dict[str, float]]:
    """pairs of (gold, pred). Returns macro-F1 and per-label F1."""
    per_label: dict[str, float] = {}
    for label in LABELS:
        tp = sum(1 for g, p in pairs if g == label and p == label)
        fp = sum(1 for g, p in pairs if g != label and p == label)
        fn = sum(1 for g, p in pairs if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_label[label] = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return sum(per_label.values()) / len(LABELS), per_label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--field", default="generated_answer",
        help="Which text to parse the decision from. The baseline stores its raw LLM output in "
             "generated_answer; the advanced pipeline stores post-verification text there and the "
             "raw output in raw_answer. Compare raw against raw, or state which you used.",
    )
    args = parser.parse_args()

    gold_by_pubid = {ex.pubid: ex.final_decision.lower() for ex in load_pubmedqa(DatasetConfig())}

    rows = []
    for line in Path(args.answers).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    graded: list[tuple[str, str]] = []   # strict: unparseable -> "unparseable" (never matches gold)
    parsed_only: list[tuple[str, str]] = []
    unparseable = 0
    for row in rows:
        gold = row.get("final_decision") or gold_by_pubid.get(str(row["pubid"]))
        if gold is None:
            continue
        text = row.get(args.field)
        if text is None:  # e.g. --field raw_answer on rows written before it was stored
            text = row["generated_answer"]
        pred = extract_decision(str(text))
        if pred is None:
            unparseable += 1
            graded.append((gold, "unparseable"))
        else:
            graded.append((gold, pred))
            parsed_only.append((gold, pred))

    n = len(graded)
    strict_acc = sum(g == p for g, p in graded) / n if n else 0.0
    strict_f1, strict_per = macro_f1(graded)
    parsed_acc = sum(g == p for g, p in parsed_only) / len(parsed_only) if parsed_only else 0.0
    parsed_f1, _ = macro_f1(parsed_only)

    report = {
        "answers_file": args.answers,
        "sample_size": n,
        "unparseable": unparseable,
        "unparseable_rate": unparseable / n if n else 0.0,
        "strict": {  # unparseable counted as wrong -- the honest number
            "accuracy": strict_acc,
            "macro_f1": strict_f1,
            "per_label_f1": strict_per,
        },
        "parsed_only": {  # ignores unparseable -- optimistic, report alongside
            "accuracy": parsed_acc,
            "macro_f1": parsed_f1,
            "n": len(parsed_only),
        },
        "gold_distribution": dict(Counter(g for g, _ in graded)),
        "pred_distribution": dict(Counter(p for _, p in graded)),
    }

    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
