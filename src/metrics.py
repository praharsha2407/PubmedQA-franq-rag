#Purpose: Mathematical functions for retrieval evaluation.
#Core Functions:
#precision_at_k(): Fraction of retrieved Top-K documents that are correct.
#recall_at_k(): Fraction of correct documents found in the Top-K.
#reciprocal_rank(): Reciprocal of the rank of the first correct document.
#average_precision(): Area under the Precision-Recall curve, evaluating rank order.

from __future__ import annotations

from statistics import mean


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(doc_id in relevant_ids for doc_id in retrieved_ids[:k]) / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return sum(doc_id in relevant_ids for doc_id in retrieved_ids[:k]) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = 0
    precisions = []
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            hits += 1
            precisions.append(hits / rank)
    return sum(precisions) / len(relevant_ids) if precisions else 0.0


def aggregate_retrieval_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "map": 0.0}
    return {
        "precision_at_k": mean(row["precision_at_k"] for row in rows),
        "recall_at_k": mean(row["recall_at_k"] for row in rows),
        "mrr": mean(row["mrr"] for row in rows),
        "map": mean(row["map"] for row in rows),
    }
