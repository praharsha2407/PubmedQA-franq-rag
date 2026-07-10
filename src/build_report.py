"""Rebuild the consolidated evaluation report from an existing answers file.

run_advanced_pipeline.py only writes full_evaluation_report.json after the whole
dataset is processed. A job killed by walltime or maintenance therefore leaves a
perfectly good advanced_answers.jsonl and no report. This script scores whatever
is in that file, so a partial run still yields numbers.

Needs no GPU and none of the pipeline models -- everything it scores is already
recorded per row. (BERTScore loads roberta-large on CPU; pass --no-bertscore to
skip it if that is too slow.)

    python src/build_report.py --answers outputs_v2/advanced_answers.jsonl \
                              --out     outputs_v2/full_evaluation_report_partial.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import AdvancedPipelineConfig, DatasetConfig
from data import load_pubmedqa
from full_evaluation import FullEvaluationCollector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True, help="path to advanced_answers*.jsonl")
    parser.add_argument("--out", default=None, help="where to write the report")
    args = parser.parse_args()

    answers_path = Path(args.answers)
    if not answers_path.exists():
        raise SystemExit(f"no such answers file: {answers_path}")

    config = AdvancedPipelineConfig()

    # The gold chunk ids for a question are derivable from the dataset alone.
    dataset = load_pubmedqa(DatasetConfig())
    relevant_by_pubid = {
        ex.pubid: {f"{ex.pubid}:{i}" for i, _ in enumerate(ex.contexts)} for ex in dataset
    }

    evaluator = FullEvaluationCollector()
    scored = 0
    for line in answers_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print("  skipping truncated final row (expected if the job was killed mid-write)")
            continue
        relevant = relevant_by_pubid.get(str(row["pubid"]))
        if relevant is None:
            continue
        evaluator.add_retrieval_result(row["retrieved_chunk_ids"], relevant, k=config.reranker.rerank_top_k)
        evaluator.add_generation_result(row["generated_answer"], row["reference_answer"])
        if row.get("hallucination_report") and row.get("mean_franq_score") is not None:
            evaluator.add_hallucination_result(row["hallucination_report"], row["mean_franq_score"])
        scored += 1

    print(f"Scoring {scored} examples from {answers_path}")
    out = Path(args.out) if args.out else answers_path.with_name("full_evaluation_report_partial.json")
    evaluator.compute_and_save(out)


if __name__ == "__main__":
    main()
