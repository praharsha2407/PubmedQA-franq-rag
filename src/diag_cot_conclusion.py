"""Why does the CoT prompt yield 'Conclusion:' for the baseline but never for the pipeline?

Baseline: 200/200 answers contain "Conclusion". Advanced pipeline with the SAME prompt
template and the same model: 0/5. Answers are 118-281 tokens against a 384 ceiling, so it
is not truncation. The prompt template is identical. The only remaining difference is which
chunks are placed in the context block.

This holds the prompt and the model fixed and varies only the retrieved chunks:
    A. dense top-5              (what the baseline feeds it)
    B. RRF-fused top-5          (what the pipeline feeds it)
    C. dense top-5, but with the expanded query used for retrieval
"""
from __future__ import annotations

import argparse

from config import AdvancedPipelineConfig, DatasetConfig
from data import build_context_corpus, load_pubmedqa
from generation import MistralGenerator
from hybrid_retrieval import HybridRetriever
from prompts import build_cot_rag_prompt
from retrieval import DenseRetriever, RetrievedChunk
from splade_retrieval import SpladeRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    args = parser.parse_args()

    config = AdvancedPipelineConfig()
    dataset = load_pubmedqa(DatasetConfig())
    texts, metadata = build_context_corpus(dataset)
    all_chunks = [RetrievedChunk(text=t, score=0.0, metadata=m) for t, m in zip(texts, metadata)]

    dense = DenseRetriever(config.retrieval)
    dense.build(texts, metadata)
    sparse = SpladeRetriever(config.splade)
    sparse.build(all_chunks)
    hybrid = HybridRetriever(dense, sparse, config.hybrid_retrieval)
    gen = MistralGenerator(config.generation)

    rows = []
    for ex in dataset[: args.n]:
        q = ex.question
        variants = {
            "dense_top5": dense.search(q, top_k=5),
            "rrf_top5": hybrid.search(q)[:5],
        }
        for name, chunks in variants.items():
            prompt = build_cot_rag_prompt(q, chunks)
            out = gen.generate(prompt)
            ctx_words = sum(len(c.text.split()) for c in chunks)
            own = sum(1 for c in chunks if c.metadata["pubid"] == ex.pubid)
            rows.append({
                "pubid": ex.pubid, "variant": name,
                "prompt_words": len(prompt.split()),
                "ctx_words": ctx_words,
                "own_pubid_chunks": f"{own}/5",
                "has_conclusion": "Conclusion" in out,
                "answer_words": len(out.split()),
            })
            print(f"  {ex.pubid}  {name:<11} promptW={len(prompt.split()):>4} ctxW={ctx_words:>4} "
                  f"own={own}/5  Conclusion={'YES' if 'Conclusion' in out else 'no ':<3} "
                  f"ansW={len(out.split()):>3}", flush=True)

    print("\n" + "=" * 62)
    for name in ("dense_top5", "rrf_top5"):
        sub = [r for r in rows if r["variant"] == name]
        hits = sum(r["has_conclusion"] for r in sub)
        pw = sum(r["prompt_words"] for r in sub) / len(sub)
        cw = sum(r["ctx_words"] for r in sub) / len(sub)
        own = sum(int(r["own_pubid_chunks"].split("/")[0]) for r in sub) / len(sub)
        print(f"{name:<12} Conclusion in {hits}/{len(sub)}   mean prompt {pw:.0f}w  "
              f"mean context {cw:.0f}w  mean own-pubid chunks {own:.1f}/5")


if __name__ == "__main__":
    main()
