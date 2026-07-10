"""Retrieval ablation: which stage of stages 2-5 actually helps?

The full v1 run scored MAP 0.749 against the dense-only baseline's 0.755, i.e. the
SPLADE + RRF + cross-encoder stack did not beat plain dense retrieval. This isolates
each component so we know which one to keep, rather than guessing.

Variants (all scored with the same gold definition and the same metrics.py):
    dense_top5        the baseline: dense retrieval, take 5
    sparse_top5       SPLADE alone, take 5
    rrf_top5          dense(20) + SPLADE(20) fused by RRF, take 5   (no reranker)
    rrf40_rerank5     fused pool of 40, cross-encoder reranks to 5  (the pipeline)
    dense20_rerank5   dense(20) reranked to 5                       (no sparse at all)

IMPORTANT: run this on a DEV slice (--dev-size), pick the winner there, then report
final numbers on the held-out remainder. Choosing a retrieval config on the same
examples you report is overfitting, and it is the first thing a reviewer checks.

    python src/ablate_retrieval.py --dev-size 300
"""
from __future__ import annotations

import argparse
import json

from config import AdvancedPipelineConfig
from data import build_context_corpus, load_pubmedqa
from hybrid_retrieval import HybridRetriever
from metrics import average_precision, precision_at_k, recall_at_k, reciprocal_rank
from reranker import CrossEncoderReranker
from retrieval import DenseRetriever, RetrievedChunk
from splade_retrieval import SpladeRetriever


def score(runs: list[tuple[list[str], set[str]]], k: int = 5) -> dict[str, float]:
    if not runs:
        return {}
    return {
        "precision_at_k": sum(precision_at_k(r, g, k) for r, g in runs) / len(runs),
        "recall_at_k": sum(recall_at_k(r, g, k) for r, g in runs) / len(runs),
        "mrr": sum(reciprocal_rank(r, g) for r, g in runs) / len(runs),
        "map": sum(average_precision(r, g, k) for r, g in runs) / len(runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-size", type=int, default=300, help="examples to evaluate on")
    parser.add_argument("--offset", type=int, default=0, help="skip this many examples first (for a held-out slice)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = AdvancedPipelineConfig()

    print("Loading dataset + building the shared corpus (all 1000 questions' chunks)...")
    dataset = load_pubmedqa(config.dataset)
    texts, metadata = build_context_corpus(dataset)
    all_chunks = [RetrievedChunk(text=t, score=0.0, metadata=m) for t, m in zip(texts, metadata)]

    dense = DenseRetriever(config.retrieval)
    dense.build(texts, metadata)
    sparse = SpladeRetriever(config.splade)
    sparse.build(all_chunks)
    if getattr(sparse, "using_fallback", False):
        raise SystemExit("SPLADE fell back to BM25; ablation would be mislabelled. Check the HF token.")
    hybrid = HybridRetriever(dense, sparse, config.hybrid_retrieval)
    reranker = CrossEncoderReranker(config.reranker)
    if getattr(reranker, "using_fallback", False):
        raise SystemExit("Cross-encoder failed to load; ablation would be mislabelled.")

    eval_set = dataset[args.offset : args.offset + args.dev_size]
    print(f"Evaluating {len(eval_set)} examples (offset {args.offset})\n")

    variants: dict[str, list] = {k: [] for k in
                                ["dense_top5", "sparse_top5", "rrf_top5", "rrf40_rerank5", "dense20_rerank5"]}

    for i, ex in enumerate(eval_set):
        if i % 25 == 0:
            print(f"  {i}/{len(eval_set)}", flush=True)
        q = ex.question              # no query expansion: isolates retrieval itself
        gold = {f"{ex.pubid}:{j}" for j, _ in enumerate(ex.contexts)}
        cid = lambda chunks: [c.metadata["chunk_id"] for c in chunks]

        d20 = dense.search(q, top_k=20)
        s20 = sparse.search(q, top_k=20)
        fused = hybrid.search(q)                       # RRF over d20 + s20, pool = final_top_k

        variants["dense_top5"].append((cid(d20[:5]), gold))
        variants["sparse_top5"].append((cid(s20[:5]), gold))
        variants["rrf_top5"].append((cid(fused[:5]), gold))
        variants["rrf40_rerank5"].append((cid(reranker.rerank(q, fused, top_k=5)), gold))
        variants["dense20_rerank5"].append((cid(reranker.rerank(q, d20, top_k=5)), gold))

    results = {name: score(runs) for name, runs in variants.items()}

    print(f"\n{'variant':<20}{'P@5':>9}{'R@5':>9}{'MRR':>9}{'MAP':>9}")
    print("-" * 56)
    for name, m in results.items():
        print(f"{name:<20}{m['precision_at_k']:>9.4f}{m['recall_at_k']:>9.4f}{m['mrr']:>9.4f}{m['map']:>9.4f}")

    best = max(results, key=lambda n: results[n]["map"])
    print(f"\nbest by MAP: {best}")
    print("Pick on this dev slice, then report on a held-out slice (--offset).")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"dev_size": len(eval_set), "offset": args.offset, "results": results, "best_by_map": best}, f, indent=2)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
