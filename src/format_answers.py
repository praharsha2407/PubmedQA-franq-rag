from __future__ import annotations

import argparse
import json
from textwrap import shorten

from config import OUTPUT_DIR


def _load_rows() -> list[dict[str, object]]:
    path = OUTPUT_DIR / "answers.jsonl"
    if not path.exists():
        raise FileNotFoundError("outputs/answers.jsonl was not found. Run src/run_pipeline.py first.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _context_block(context: dict[str, object], rank: int) -> list[str]:
    metadata = context.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    pubid = metadata.get("pubid", "unknown")
    chunk_id = metadata.get("chunk_id", "unknown")
    score = float(context.get("score", 0.0))
    text = str(context.get("text", "")).strip()
    return [
        f"### Retrieved Context {rank}",
        "",
        f"- PubMed ID: `{pubid}`",
        f"- Chunk ID: `{chunk_id}`",
        f"- Similarity Score: `{score:.4f}`",
        "",
        text,
        "",
    ]


def write_markdown(max_context_chars: int | None = None) -> None:
    rows = _load_rows()
    lines = [
        "# PubMedQA RAG Generated Answers",
        "",
        "This file is a human-readable version of `outputs/answers.jsonl`.",
        "The JSONL file is for metric scripts; this Markdown file is for reading and report inspection.",
        "",
    ]

    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                "---",
                "",
                f"## Question {index}",
                "",
                f"**PubMed ID:** `{row.get('pubid', '')}`",
                "",
                f"**Question:** {row.get('question', '')}",
                "",
                f"**Gold Final Decision:** `{row.get('final_decision', '')}`",
                "",
                "### Reference Long Answer",
                "",
                str(row.get("reference_answer", "")).strip(),
                "",
                "### Generated Answer",
                "",
                str(row.get("generated_answer", "")).strip(),
                "",
                "### Retrieved Evidence",
                "",
            ]
        )

        contexts = row.get("retrieved_contexts", [])
        if not isinstance(contexts, list):
            contexts = []
        for rank, context in enumerate(contexts, start=1):
            if not isinstance(context, dict):
                continue
            if max_context_chars is not None:
                context = dict(context)
                context["text"] = shorten(str(context.get("text", "")), width=max_context_chars, placeholder="...")
            lines.extend(_context_block(context, rank))

    output_path = OUTPUT_DIR / "answers_readable.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote readable answers to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-context-chars", type=int, default=None)
    args = parser.parse_args()
    write_markdown(max_context_chars=args.max_context_chars)


if __name__ == "__main__":
    main()
