import string
from rank_bm25 import BM25Okapi
from config import SparseRetrievalConfig
from retrieval import RetrievedChunk

def tokenize(text: str) -> list[str]:
    """Basic tokenizer for BM25: lowercase and remove punctuation."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

class BM25Retriever:
    def __init__(self, config: SparseRetrievalConfig):
        self.config = config
        self.bm25 = None
        self.corpus_chunks = []

    def build(self, chunks: list[RetrievedChunk]):
        """Builds the BM25 index from a list of RetrievedChunk objects."""
        self.corpus_chunks = chunks
        tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.config.k1, b=self.config.b)
        print(f"Built BM25 index with {len(chunks)} chunks.")

    def search(self, query: str, top_k: int = None) -> list[RetrievedChunk]:
        """Searches the BM25 index and returns top_k RetrievedChunk objects."""
        if not self.bm25:
            raise ValueError("BM25 index not built. Call build() first.")

        k = top_k or self.config.top_k
        tokenized_query = tokenize(query)

        # Get raw BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Sort indices by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for i in top_indices:
            chunk = self.corpus_chunks[i]
            results.append(RetrievedChunk(
                text=chunk.text,
                score=float(scores[i]),
                metadata=chunk.metadata
            ))

        return results
