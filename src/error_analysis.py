from __future__ import annotations

import json

from config import OUTPUT_DIR


FAITHFULNESS_THRESHOLD = 0.70
ANSWER_RELEVANCY_THRESHOLD = 0.70
CONTEXT_RECALL_THRESHOLD = 0.70
CONTEXT_PRECISION_THRESHOLD = 0.70


def classify_error(row: dict, ragas_scores: dict) -> list[str]:
    labels = []

    faithfulness = ragas_scores.get("faithfulness")
    answer_relevancy = ragas_scores.get("answer_relevancy")
    context_precision = ragas_scores.get("context_precision")
    context_recall = ragas_scores.get("context_recall")

    if context_recall is not None and context_recall < CONTEXT_RECALL_THRESHOLD:
        labels.append("retrieval_failure")

    if context_precision is not None and context_precision < CONTEXT_PRECISION_THRESHOLD:
        labels.append("poor_context_grounding")

    if faithfulness is not None and faithfulness < FAITHFULNESS_THRESHOLD:
        labels.append("possible_hallucination")

    if answer_relevancy is not None and answer_relevancy < ANSWER_RELEVANCY_THRESHOLD:
        labels.append("weak_biomedical_reasoning")

    if not labels:
        labels.append("no_obvious_error")

    return labels


def main():
    ragas_details_path = OUTPUT_DIR / "ragas_details.json"

    if not ragas_details_path.exists():
        print(
            "RAGAS details not found. "
            "Run evaluate_ragas.py before error analysis."
        )
        return

    ragas_details = json.loads(
        ragas_details_path.read_text(encoding="utf-8")
    )

    # Map pubid to its specific ragas scores for row-level error analysis
    pubid_to_scores = {str(item["pubid"]): item for item in ragas_details}

    rows = [
        json.loads(line)
        for line in (
            OUTPUT_DIR / "answers.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    counts = {}
    examples = {}

    for row in rows:
        scores = pubid_to_scores.get(str(row["pubid"]), {})
        labels = classify_error(row, scores)

        for label in labels:
            counts[label] = counts.get(label, 0) + 1

            examples.setdefault(label, [])

            if len(examples[label]) < 5:
                examples[label].append(row["pubid"])

    lines = [
        "# Error Analysis",
        "",
        "## Error Counts",
        "",
    ]

    for label, count in sorted(
        counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        lines.append(f"- {label}: {count}")

    lines.extend(
        [
            "",
            "## Example PubMed IDs",
            "",
        ]
    )

    for label, ids in examples.items():
        lines.append(
            f"- {label}: {', '.join(map(str, ids))}"
        )

    output_path = OUTPUT_DIR / "error_analysis.md"

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\n".join(lines))


if __name__ == "__main__":
    main()