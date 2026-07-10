from __future__ import annotations

import json
from pathlib import Path

from config import OUTPUT_DIR, PROJECT_ROOT


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_table(title: str, metrics: dict) -> list[str]:
    lines = [f"## {title}", ""]
    if not metrics:
        lines.extend(["Not available yet.", ""])
        return lines
    lines.extend(["| Metric | Value |", "| --- | ---: |"])
    for key, value in metrics.items():
        if isinstance(value, dict):
            continue
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines.append("")
    return lines


def main() -> None:
    retrieval = _load_json(OUTPUT_DIR / "retrieval_metrics.json")
    generation = _load_json(OUTPUT_DIR / "generation_metrics.json")
    ragas = _load_json(OUTPUT_DIR / "ragas_metrics.json")
    architecture = (PROJECT_ROOT / "reports" / "architecture_and_error_analysis.md").read_text(encoding="utf-8")

    lines = [
        "# PubMedQA Baseline RAG Final Report Draft",
        "",
        "This draft combines the implemented baseline, evaluation metrics, error analysis framework, and proposed improved architecture.",
        "",
    ]
    lines.extend(_metric_table("Retrieval Metrics", retrieval.get("metrics", {})))
    lines.extend(_metric_table("Generation Metrics", generation))
    lines.extend(_metric_table("RAGAS Metrics", ragas))
    lines.extend(["---", ""])
    lines.append(architecture)

    error_path = OUTPUT_DIR / "error_analysis.md"
    if error_path.exists():
        lines.extend(["", "---", "", error_path.read_text(encoding="utf-8")])

    answers_path = OUTPUT_DIR / "answers_readable.md"
    if answers_path.exists():
        lines.extend(
            [
                "",
                "---",
                "",
                "# Appendix: Sample Generated Answers",
                "",
                "See `outputs/answers_readable.md` for the full readable answer file.",
            ]
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "final_report_draft.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
