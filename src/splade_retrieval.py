"""
Stage 3: SPLADE -- neural sparse retrieval, replacing BM25.

Grounded in:
  Formal, Piwowarski, Clinchant (2021), "SPLADE: Sparse Lexical and
  Expansion Model for First Stage Ranking", SIGIR'21, arXiv:2107.05720.
  Lassance, Dejean, Formal, Clinchant (2024), "SPLADE-v3: New baselines
  for SPLADE", arXiv:2403.06789.

BM25 (sparse_retrieval.py) is a statistical formula with no learned
parameters. SPLADE is a BERT-based model, trained end-to-end, that predicts
a sparse (mostly-zero) importance weight for every term in the vocabulary --
including terms that don't literally appear in the text (this is the
"expansion" in its name). Retrieval is then a dot product between the
query's and each document's sparse weight vectors, same efficiency class as
BM25 (inverted-index-friendly), but the weights themselves are learned.

This class exposes the SAME .build() / .search() interface as
BM25Retriever, so it's a drop-in replacement inside HybridRetriever
(hybrid_retrieval.py) -- no changes needed there.
"""
from __future__ import annotations

from config import SpladeConfig
from retrieval import RetrievedChunk


class SpladeRetriever:
    def __init__(self, config: SpladeConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.using_fallback = False
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(config.model_name)
            self.model.eval()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        except Exception as exc:
            self.using_fallback = True
            print(
                "WARNING: SPLADE model could not be loaded (this needs internet access "
                "to Hugging Face and enough memory/GPU) -- falling back to BM25 for "
                f"this run. Original error: {exc}"
            )
            from sparse_retrieval import BM25Retriever
            from config import SparseRetrievalConfig
            self._bm25_fallback = BM25Retriever(SparseRetrievalConfig(top_k=config.top_k))

        self.corpus_chunks: list[RetrievedChunk] = []
        self._doc_vectors = None  # scipy sparse matrix, one row per chunk
        self._vocab_size = None

    def _encode_sparse(self, texts: list[str]):
        """
        Runs the SPLADE forward pass and returns a scipy sparse CSR matrix,
        one row per input text, one column per vocabulary token.
        SPLADE's pooling: log(1 + relu(logits)) then max-pool over tokens --
        this is the standard SPLADE pooling from Formal et al. (2021).
        """
        import scipy.sparse as sp

        all_rows = []
        batch_size = self.config.batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=self.config.max_seq_length,
            ).to(self.device)
            with self.torch.no_grad():
                logits = self.model(**inputs).logits  # (batch, seq_len, vocab)
            weights, _ = self.torch.max(
                self.torch.log1p(self.torch.relu(logits)) * inputs["attention_mask"].unsqueeze(-1),
                dim=1,
            )  # (batch, vocab) -- max-pool over sequence
            weights = weights.cpu().numpy()
            self._vocab_size = weights.shape[1]
            all_rows.append(sp.csr_matrix(weights))
        return sp.vstack(all_rows)

    def build(self, chunks: list[RetrievedChunk]) -> None:
        if self.using_fallback:
            self._bm25_fallback.build(chunks)
            return
        self.corpus_chunks = chunks
        texts = [c.text for c in chunks]
        print(f"Encoding {len(texts)} chunks with SPLADE ({self.config.model_name})...")
        self._doc_vectors = self._encode_sparse(texts)
        print(f"Built SPLADE index with {len(chunks)} chunks.")

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if self.using_fallback:
            return self._bm25_fallback.search(query, top_k)
        if self._doc_vectors is None:
            raise ValueError("SPLADE index not built. Call build() first.")

        k = top_k or self.config.top_k
        query_vector = self._encode_sparse([query])  # (1, vocab)
        scores = (self._doc_vectors @ query_vector.T).toarray().ravel()  # dot product

        top_indices = scores.argsort()[::-1][:k]
        results = []
        for i in top_indices:
            chunk = self.corpus_chunks[i]
            results.append(RetrievedChunk(text=chunk.text, score=float(scores[i]), metadata=chunk.metadata))
        return results
