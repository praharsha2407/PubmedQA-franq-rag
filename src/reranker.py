"""
Stage 5: Cross-encoder reranking (new stage, sits between fusion and
keyword extraction).

Grounded in: Reimers & Gurevych (2019), "Sentence-BERT: Sentence Embeddings
using Siamese BERT-Networks", EMNLP 2019 -- the cross-encoder variant of
this architecture (query and candidate encoded TOGETHER, not separately)
is what's used here, via the pretrained MS MARCO cross-encoder checkpoint.

Dense retrieval and SPLADE both score (query, chunk) pairs by comparing two
SEPARATELY computed representations (a bi-encoder setup) -- fast, but less
accurate, because the model never actually reads the query and chunk
together. A cross-encoder feeds both into one BERT pass and outputs a
single relevance score, which is substantially more accurate but much
slower -- which is why it only reranks the already-small fused top-K
(e.g. top-20), not the whole corpus.
"""
from __future__ import annotations

from config import RerankerConfig
from retrieval import RetrievedChunk


class CrossEncoderReranker:
    def __init__(self, config: RerankerConfig):
        self.config = config
        self.model = None
        self.using_fallback = False
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(config.model_name)
        except Exception as exc:
            self.using_fallback = True
            print(
                "WARNING: Cross-encoder reranker could not be loaded -- falling back "
                f"to passing through the fused ranking unchanged. Original error: {exc}"
            )

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self.config.rerank_top_k
        if self.using_fallback or not candidates:
            return candidates[:k]

        pairs = [(query, chunk.text) for chunk in candidates]
        scores = self.model.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda pair: pair[1], reverse=True)

        reranked = []
        for chunk, score in scored[:k]:
            reranked.append(RetrievedChunk(text=chunk.text, score=float(score), metadata=chunk.metadata))
        return reranked
