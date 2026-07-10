"""
Stage 9: Consolidated evaluation.

Combines all three metric families the advanced pipeline can now report into
ONE JSON file, so you have a single source of truth for your results table:

  1. Retrieval quality  -- Precision@K, Recall@K, MRR, MAP  (metrics.py)
     Computed on the reranked top-K (post-Stage-5), against the gold
     "pubid:index" chunks for each question.
  2. Generation quality -- BLEU, ROUGE-1/2/L, BERTScore     (evaluate_generation.py)
     Computed on the final (post-verification) answer text vs. the
     PubMedQA long_answer reference.
  3. Faithfulness / hallucination -- Keyword Faithfulness Score,
     Hallucination Rate by Type, FRANQ Factuality Score    (stage6_evaluation.py)

Output: outputs/full_evaluation_report.json
"""
from __future__ import annotations

import json
from pathlib import Path

from config import OUTPUT_DIR
from metrics import aggregate_retrieval_metrics
from evaluate_generation import compute_generation_metrics
from stage6_evaluation import AdvancedEvaluator


class FullEvaluationCollector:
    """
    Accumulate results across all questions during a single run of
    run_advanced_pipeline.py, then produce one consolidated report.
    """

    def __init__(self):
        self.retrieval_rows: list[dict[str, float]] = []
        self.predictions: list[str] = []
        self.references: list[str] = []
        self.hallucination_evaluator = AdvancedEvaluator()

    def add_retrieval_result(
        self,
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int,
    ) -> None:
        from metrics import precision_at_k, recall_at_k, reciprocal_rank, average_precision

        self.retrieval_rows.append({
            "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k),
            "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k),
            "mrr": reciprocal_rank(retrieved_ids, relevant_ids),
            "map": average_precision(retrieved_ids, relevant_ids, k),
        })

    def add_generation_result(self, prediction: str, reference: str) -> None:
        self.predictions.append(prediction)
        self.references.append(reference)

    def add_hallucination_result(self, hallucination_report: dict, avg_franq_score: float) -> None:
        self.hallucination_evaluator.add_result(hallucination_report, avg_franq_score)

    def compute_and_save(self, save_path: Path | None = None) -> dict:
        save_path = save_path or (OUTPUT_DIR / "full_evaluation_report.json")

        report: dict[str, object] = {
            "sample_size": len(self.predictions),
        }

        report["retrieval_metrics"] = (
            aggregate_retrieval_metrics(self.retrieval_rows) if self.retrieval_rows else {}
        )

        if self.predictions:
            report["generation_metrics"] = compute_generation_metrics(self.predictions, self.references)
        else:
            report["generation_metrics"] = {}

        report["hallucination_and_franq_metrics"] = self.hallucination_evaluator.calculate_metrics()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nFull evaluation report saved to {save_path}")
        print(json.dumps(report, indent=2))
        return report
