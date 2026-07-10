#Purpose: Orchestrates retrieval-stage testing.
#How it works: Loads the dataset and index, queries FAISS for each question, maps retrieval keys against gold context keys, runs the mathematical functions in metrics.py, and writes results to outputs/retrieval_metrics.json.from __future__ import annotations

import argparse
import json

from config import INDEX_DIR, OUTPUT_DIR, PipelineConfig, RetrievalConfig
from data import build_context_corpus, load_pubmedqa
from metrics import (
    aggregate_retrieval_metrics,
    average_precision,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from retrieval import DenseRetriever


def evaluate(sample_size: int | None, top_k: int, save_index: bool) -> dict[str, object]:
    config = PipelineConfig(retrieval=RetrievalConfig(top_k=top_k))
    examples = load_pubmedqa(config.dataset, sample_size=sample_size)
    corpus, metadata = build_context_corpus(examples)

    retriever = DenseRetriever(config.retrieval)
    retriever.build(corpus, metadata)
    if save_index:
        retriever.save(INDEX_DIR)

    rows = []
    for example in examples:
        retrieved = retriever.search(example.question, top_k=top_k)
        retrieved_ids = [chunk.metadata["chunk_id"] for chunk in retrieved]
        relevant_ids = {f"{example.pubid}:{idx}" for idx, _ in enumerate(example.contexts)}
        rows.append(
            {
                "pubid": example.pubid,
                "question": example.question,
                "retrieved_ids": retrieved_ids,
                "relevant_ids": sorted(relevant_ids),
                "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, top_k),
                "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, top_k),
                "mrr": reciprocal_rank(retrieved_ids, relevant_ids),
                "map": average_precision(retrieved_ids, relevant_ids, top_k),
            }
        )

    metrics = aggregate_retrieval_metrics(rows)
    result = {"top_k": top_k, "sample_size": len(examples), "metrics": metrics, "per_question": rows}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "retrieval_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--save-index", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.sample_size, args.top_k, args.save_index)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
