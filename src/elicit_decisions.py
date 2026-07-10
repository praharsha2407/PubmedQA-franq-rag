"""Explicit yes/no/maybe decision elicitation, applied identically to any answers file.

Why this exists
---------------
PubMedQA is scored on a yes/no/maybe decision. Both the baseline and the pipeline get
that decision only if the model *volunteers* a "Conclusion: <label>" line -- and whether
it does is unstable: the same (prompt, model) under sampled decoding wrote a parseable
verdict in the baseline's stored answers.jsonl (200/200) but frequently omitted it on
re-runs. Accuracy that depends on the model choosing to format its answer a certain way
measures formatting, not correctness.

This decouples the decision from the free-text answer. For every row it runs one short,
deterministic (greedy) follow-up generation that asks the model to commit to a single word
given the question and its own reasoning. Because it reads only `question` and a reasoning
text field that BOTH schemas have, the exact same procedure scores the baseline and the
pipeline -- a fair comparison, and no change to either runner.

    python src/elicit_decisions.py --answers outputs/answers.jsonl \
        --text-field generated_answer --out outputs/answers.elicited.jsonl
    python src/evaluate_decision.py --answers outputs/answers.elicited.jsonl \
        --field elicited_decision
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from config import AdvancedPipelineConfig
from generation import MistralGenerator

LABELS = ("yes", "no", "maybe")
_LABEL_RE = re.compile(r"\b(yes|no|maybe)\b", re.I)


def build_elicitation_prompt(question: str, reasoning: str) -> str:
    # Kept deliberately minimal and identical for both systems. The reasoning already
    # summarises the retrieved evidence, so we do not re-inject context here -- that keeps
    # the prompt schema-independent and the two systems strictly comparable.
    reasoning = (reasoning or "").strip()
    return (
        f"Question:\n{question}\n\n"
        f"Reasoning:\n{reasoning}\n\n"
        "Given the reasoning above, what is the answer to the question? "
        "Reply with exactly one word: yes, no, or maybe. "
        "Use 'maybe' if the evidence is mixed or inconclusive."
    )


def parse_label(text: str) -> str:
    match = _LABEL_RE.search(text or "")
    return match.group(1).lower() if match else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--text-field", default="raw_answer",
        help="Field holding the model's reasoning text. Baseline: generated_answer. "
             "Pipeline: raw_answer (the pre-verification output).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    config = AdvancedPipelineConfig()
    generator = MistralGenerator(config.generation)
    if generator.using_fallback_generator:
        raise SystemExit("elicitation needs the real model on a GPU; the fallback cannot decide.")

    rows = []
    for line in Path(args.answers).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    counts = {"yes": 0, "no": 0, "maybe": 0, "": 0}
    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as out_file:
        for i, row in enumerate(rows, 1):
            reasoning = row.get(args.text_field)
            if reasoning is None:
                reasoning = row.get("generated_answer", "")
            prompt = build_elicitation_prompt(row["question"], reasoning)
            out = generator.generate(prompt, max_new_tokens=args.max_new_tokens, greedy=True)
            label = parse_label(out)
            row["elicited_decision"] = label
            row["elicited_raw"] = out.strip()
            counts[label] = counts.get(label, 0) + 1
            out_file.write(json.dumps(row) + "\n")
            out_file.flush()
            print(f"  [{i}/{len(rows)}] pubid={row['pubid']} -> {label or '<none>':<6} "
                  f"(raw: {out.strip()[:40]!r})", flush=True)

    print(f"\nwrote {len(rows)} rows -> {out_path}")
    print(f"elicited distribution: {counts}")


if __name__ == "__main__":
    main()
