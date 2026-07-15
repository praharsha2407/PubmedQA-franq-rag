"""TF-IDF keyword extraction -- the classical alternative to KeyBERT (Stage 6 ablation).

KeyBERT is *semantic*: it embeds the document and each candidate n-gram with a transformer and
keeps the n-grams whose embeddings are closest to the document embedding. TF-IDF is *statistical*:
it keeps the terms that are frequent in THIS chunk but rare across the corpus.

They should disagree, and that is the point of the ablation:

  - TF-IDF favours rare, discriminative tokens (gene names, acronyms) -- it cannot recognise a
    paraphrase, but it never invents relevance.
  - KeyBERT favours phrases that "feel like" the document's topic -- it captures paraphrase, but
    it needs a GPU and can drift toward generic phrasing.

The IDF is fitted over the FULL corpus (all 3,358 chunks), which is what makes the "inverse
document frequency" meaningful. Fitting it on only the 5 retrieved chunks would make almost every
term look rare and the weights would be close to meaningless.

Same interface as KeywordExtractor, so it is a drop-in swap in the verification pipeline.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from retrieval import RetrievedChunk


class TfidfKeywordExtractor:
    """Drop-in replacement for KeywordExtractor, backed by TF-IDF instead of KeyBERT."""

    def __init__(self, corpus_texts: list[str], top_n: int = 10, ngram_range=(1, 2)):
        # Fit the IDF statistics ONCE over the entire corpus. This is the whole point:
        # "inverse document frequency" is only meaningful relative to a real document collection.
        self.top_n = top_n
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            stop_words="english",
            lowercase=True,
            min_df=1,
        )
        self.vectorizer.fit(corpus_texts)
        self.vocab = self.vectorizer.get_feature_names_out()
        print(f"TF-IDF fitted on {len(corpus_texts)} chunks; vocabulary = {len(self.vocab)} terms")

    def extract_keywords_from_chunk(self, chunk: RetrievedChunk) -> list[str]:
        """Top-N terms of this text by TF-IDF weight, using corpus-wide IDF."""
        text = chunk.text if hasattr(chunk, "text") else str(chunk)
        if not text.strip():
            return []
        row = self.vectorizer.transform([text])
        if row.nnz == 0:
            return []
        # row is 1 x V sparse; take the highest-weighted terms.
        pairs = sorted(
            zip(row.indices, row.data), key=lambda p: p[1], reverse=True
        )[: self.top_n]
        return [str(self.vocab[i]) for i, _ in pairs]

    def build_keyword_corpus(self, chunks: list[RetrievedChunk]) -> dict:
        corpus: dict = {}
        all_keywords: set[str] = set()
        for chunk in chunks:
            kws = self.extract_keywords_from_chunk(chunk)
            corpus[chunk.metadata["chunk_id"]] = kws
            all_keywords.update(kws)
        corpus["all_keywords"] = list(all_keywords)
        return corpus
