from keybert import KeyBERT
from retrieval import RetrievedChunk
from config import KeywordConfig

class KeywordExtractor:
    def __init__(self, config: KeywordConfig):
        self.config = config
        # Initialize KeyBERT with the specified biomedical model
        # Using the same PubMedBert model as retrieval ensures consistent domain understanding
        self.kw_model = KeyBERT(model=self.config.keybert_model)

    def extract_keywords_from_chunk(self, chunk: RetrievedChunk) -> list[str]:
        # Extract keywords with KeyBERT
        # Allow 1-gram to 2-gram to capture phrases like "cardiovascular risk"
        keywords_with_scores = self.kw_model.extract_keywords(
            chunk.text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=self.config.top_n_keywords
        )
        # We only need the text, not the score for our overlap metric
        return [kw for kw, score in keywords_with_scores]

    def build_keyword_corpus(self, chunks: list[RetrievedChunk]) -> dict:
        """
        Builds the reference K_ctx (Keyword Corpus) from retrieved chunks.

        Returns:
            A dictionary mapping chunk_id to its extracted keywords,
            plus a unified set of 'all_keywords'.
        """
        corpus = {}
        all_keywords = set()

        for chunk in chunks:
            kws = self.extract_keywords_from_chunk(chunk)
            corpus[chunk.metadata["chunk_id"]] = kws
            all_keywords.update(kws)

        # Store union of all keywords as a quick check for extrinsic hallucination
        corpus["all_keywords"] = list(all_keywords)
        return corpus
