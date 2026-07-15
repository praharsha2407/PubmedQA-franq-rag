"""Ablation: does the RRF constant k=60 matter, and what is the right evidence depth?

Two questions this answers, neither of which was previously justified:

  1. WHY k = 60?  It was taken verbatim from Cormack et al. (2009) and never swept.
     This sweeps k over {1, 10, 20, 40, 60, 80, 100} and reports MAP.

  2. HOW MANY CHUNKS should reach the generator?  The pipeline uses top-5. This scores
     retrieval quality at top-k in {5, 10, 20, 30, 40} for dense, sparse and RRF.

Retrieval only -- no LLM, no generation. Cheap. Runs on a compute node via Slurm.

Dev/held-out split so the choice can be made on dev and confirmed on data it never saw.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import AdvancedPipelineConfig, DatasetConfig
from data import build_context_corpus, load_pubmedqa
from retrieval import DenseRetriever, RetrievedChunk
from splade_retrieval import SpladeRetriever


def average_precision(retrieved: list[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    hits = 0
    total = 0.0
    for i, cid in enumerate(retrieved, start=1):
        if cid in gold:
            hits += 1
            total += hits / i
    return total / min(len(gold), len(retrieved)) if hits else 0.0


def score(runs: list[tuple[list[str], set[str]]]) -> dict[str, float]:
    n = len(runs)
    if n == 0:
        return {"map": 0.0, "recall": 0.0, "precision": 0.0}
    maps = [average_precision(r, g) for r, g in runs]
    recs = [len(set(r) & g) / len(g) if g else 0.0 for r, g in runs]
    precs = [len(set(r) & g) / len(r) if r else 0.0 for r, g in runs]
    return {
        "map": sum(maps) / n,
        "recall": sum(recs) / n,
        "precision": sum(precs) / n,
    }


def rrf_fuse(dense_ids: list[str], sparse_ids: list[str], k: int) -> list[str]:
    """Reciprocal Rank Fusion over two ranked id lists, with constant k."""
    scores: dict[str, float] = {}
    for rank, cid in enumerate(dense_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(sparse_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda c: scores[c], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-size", type=int, default=300)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--pool", type=int, default=50, help="candidates from EACH retriever")
    parser.add_argument("--out", default="results/ablation_rrf_k.json")
    args = parser.parse_args()

    config = AdvancedPipelineConfig()
    dataset = load_pubmedqa(DatasetConfig())
    texts, meta = build_context_corpus(dataset)
    all_chunks = [RetrievedChunk(text=t, score=0.0, metadata=m) for t, m in zip(texts, meta)]

    dense = DenseRetriever(config.retrieval)
    dense.build(texts, meta)
    sparse = SpladeRetriever(config.splade)
    sparse.build(all_chunks)

    examples = dataset[args.offset : args.offset + args.dev_size]
    print(f"scoring {len(examples)} questions (offset {args.offset})\n", flush=True)

    K_VALUES = [1, 10, 20, 40, 60, 80, 100]
    TOP_KS = [5, 10, 20, 30, 40]

    # Retrieve once per question, deep enough for every variant below.
    cached = []
    for i, ex in enumerate(examples, 1):
        gold = {m["chunk_id"] for m in meta if str(m["pubid"]) == str(ex.pubid)}
        d = [c.metadata["chunk_id"] for c in dense.search(ex.question, top_k=args.pool)]
        s = [c.metadata["chunk_id"] for c in sparse.search(ex.question, top_k=args.pool)]
        cached.append((d, s, gold))
        if i % 50 == 0:
            print(f"  retrieved {i}/{len(examples)}", flush=True)

    report: dict = {"dev_size": len(examples), "offset": args.offset, "pool": args.pool}

    # ---- Q1: the RRF constant k -------------------------------------------------
    print("\n" + "=" * 58)
    print("Q1.  Does the RRF constant k matter?   (evidence depth = 5)")
    print("=" * 58)
    print(f"{'k':>6}{'MAP':>10}{'Recall@5':>11}{'P@5':>9}")
    print("-" * 58)
    k_results = {}
    for k in K_VALUES:
        runs = [(rrf_fuse(d, s, k)[:5], g) for d, s, g in cached]
        m = score(runs)
        k_results[k] = m
        print(f"{k:>6}{m['map']:>10.4f}{m['recall']:>11.4f}{m['precision']:>9.4f}", flush=True)
    best_k = max(k_results, key=lambda k: k_results[k]["map"])
    print("-" * 58)
    print(f"best k by MAP: {best_k}   (the pipeline uses 60, taken from Cormack et al.)")
    report["rrf_k_sweep"] = {str(k): v for k, v in k_results.items()}
    report["best_k"] = best_k

    # ---- Q2: how many chunks should reach the generator? ------------------------
    print("\n" + "=" * 58)
    print("Q2.  How deep should the evidence be?   (RRF k = 60)")
    print("=" * 58)
    print(f"{'top-k':>7}{'method':>10}{'MAP':>10}{'Recall':>10}{'Precision':>12}")
    print("-" * 58)
    depth_results: dict = {}
    for top_k in TOP_KS:
        row = {}
        for name in ("dense", "sparse", "rrf"):
            if name == "dense":
                runs = [(d[:top_k], g) for d, s, g in cached]
            elif name == "sparse":
                runs = [(s[:top_k], g) for d, s, g in cached]
            else:
                runs = [(rrf_fuse(d, s, 60)[:top_k], g) for d, s, g in cached]
            m = score(runs)
            row[name] = m
            print(f"{top_k:>7}{name:>10}{m['map']:>10.4f}{m['recall']:>10.4f}{m['precision']:>12.4f}",
                  flush=True)
        depth_results[str(top_k)] = row
        print("-" * 58)
    report["evidence_depth"] = depth_results

    print("\nNOTE: recall rises with depth by construction, precision falls. The question is not")
    print("'which top-k has the best retrieval score' -- it is 'how much evidence helps the")
    print("GENERATOR', and only a full pipeline run can answer that. This table sizes the")
    print("trade-off; the generation runs test it.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
