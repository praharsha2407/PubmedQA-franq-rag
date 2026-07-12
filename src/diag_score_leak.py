"""Does the raw RRF score printed into the prompt cause the advanced pipeline's collapse?

format_contexts() writes `score={chunk.score:.4f}` into the prompt. The baseline's scores are
dense cosine similarities (~0.94, "strong evidence"). The advanced pipeline's are RRF scores,
1/(k+rank) summed over two retrievers, which max out around 0.033 -- so the prompt tells the
model its (correct!) evidence is ~3% relevant. The model then hedges, declines to emit the
required Conclusion, and answers "no".

This replays the advanced pipeline's OWN retrieved chunks for real examples and generates three
ways, changing nothing but the score shown in the prompt:

    A. rrf_raw     -- the score as the pipeline currently prints it   (expect: no Conclusion)
    B. normalized  -- RRF scores min-max scaled into [0,1]
    C. no_score    -- the score field removed from the context header

If Conclusion returns in B/C, the retrieval scale leaking into the prompt is the cause, and the
architecture is exonerated.
"""
from __future__ import annotations

import argparse
import json
import re

from config import AdvancedPipelineConfig, DatasetConfig
from data import build_context_corpus, load_pubmedqa
from generation import MistralGenerator
from prompts import build_cot_rag_prompt
from retrieval import RetrievedChunk

_SCORE_RE = re.compile(r" \| score=[-0-9.]+")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", default="outputs_cot/advanced_answers_cot.jsonl")
    parser.add_argument("--n", type=int, default=15)
    args = parser.parse_args()

    texts, meta = build_context_corpus(load_pubmedqa(DatasetConfig()))
    cid2text = {m["chunk_id"]: t for t, m in zip(texts, meta)}
    cid2meta = {m["chunk_id"]: m for m in meta}

    rows = []
    with open(args.answers, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= args.n:
                    break

    generator = MistralGenerator(AdvancedPipelineConfig().generation)
    if generator.using_fallback_generator:
        raise SystemExit("needs the real model on a GPU")

    # Reconstruct the RRF scores the pipeline actually used: RRF is rank-based, so
    # position i in the stored top-k had a score on the 1/(k+rank) scale.
    k = AdvancedPipelineConfig().hybrid_retrieval.rrf_k
    hits = {"rrf_raw": 0, "normalized": 0, "no_score": 0}
    words = {"rrf_raw": [], "normalized": [], "no_score": []}

    for i, row in enumerate(rows, 1):
        cids = [c for c in row["retrieved_chunk_ids"] if c in cid2text]
        if not cids:
            continue
        rrf = [1.0 / (k + r) + 1.0 / (k + r) for r in range(1, len(cids) + 1)]
        lo, hi = min(rrf), max(rrf)
        norm = [(s - lo) / (hi - lo) if hi > lo else 1.0 for s in rrf]

        def chunks_with(scores):
            return [
                RetrievedChunk(text=cid2text[c], score=s, metadata=cid2meta[c])
                for c, s in zip(cids, scores)
            ]

        variants = {
            "rrf_raw": build_cot_rag_prompt(row["question"], chunks_with(rrf)),
            "normalized": build_cot_rag_prompt(row["question"], chunks_with(norm)),
        }
        variants["no_score"] = _SCORE_RE.sub("", variants["rrf_raw"])

        line_out = [f"[{i:>3}] pubid={row['pubid']:<10}"]
        for name, prompt in variants.items():
            out = generator.generate(prompt)
            has = "Conclusion" in out
            hits[name] += has
            words[name].append(len(out.split()))
            line_out.append(f"{name}={'YES' if has else 'no ':<3}({len(out.split()):>3}w)")
        print("  ".join(line_out), flush=True)

    n = len(rows)
    print("\n" + "=" * 70)
    print(f"{'variant':<14}{'Conclusion':>14}{'median words':>16}")
    for name in ("rrf_raw", "normalized", "no_score"):
        w = sorted(words[name])
        med = w[len(w) // 2] if w else 0
        print(f"{name:<14}{hits[name]:>8}/{n:<5}{med:>16}")
    print("=" * 70)
    if hits["rrf_raw"] == 0 and (hits["normalized"] or hits["no_score"]):
        print("CONFIRMED: the raw RRF score printed into the prompt suppresses the Conclusion.")
        print("The retrieval scale is leaking into generation. The architecture is fine.")
    else:
        print("NOT confirmed by this test -- inspect the per-row output above.")


if __name__ == "__main__":
    main()
