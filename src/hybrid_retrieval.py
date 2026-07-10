from config import HybridRetrievalConfig
from retrieval import DenseRetriever, RetrievedChunk
from sparse_retrieval import BM25Retriever

class HybridRetriever:
    def __init__(self, dense_retriever: DenseRetriever, sparse_retriever: BM25Retriever, config: HybridRetrievalConfig):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.config = config

    def search(self, query: str) -> list[RetrievedChunk]:
        """
        Executes Dense and Sparse retrieval in parallel (conceptually),
        then fuses their results using Reciprocal Rank Fusion (RRF).
        """
        # 1. Retrieve from both
        dense_results = self.dense.search(query, top_k=self.config.dense_top_k)
        sparse_results = self.sparse.search(query, top_k=self.config.sparse_top_k)

        # 2. Map chunk_id to its rank in each list
        # Ranks are 1-indexed for the RRF formula
        dense_ranks = {chunk.metadata["chunk_id"]: rank for rank, chunk in enumerate(dense_results, 1)}
        sparse_ranks = {chunk.metadata["chunk_id"]: rank for rank, chunk in enumerate(sparse_results, 1)}

        # 3. Collect all unique chunks
        all_chunks_map = {}
        for chunk in dense_results + sparse_results:
            if chunk.metadata["chunk_id"] not in all_chunks_map:
                all_chunks_map[chunk.metadata["chunk_id"]] = chunk

        # 4. Compute RRF score for each unique chunk
        rrf_scores = {}
        k = self.config.rrf_k

        for chunk_id in all_chunks_map:
            dense_rank = dense_ranks.get(chunk_id, None)
            sparse_rank = sparse_ranks.get(chunk_id, None)

            score = 0.0
            if dense_rank is not None:
                score += 1.0 / (k + dense_rank)
            if sparse_rank is not None:
                score += 1.0 / (k + sparse_rank)

            rrf_scores[chunk_id] = score

        # 5. Sort chunks by RRF score descending
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        # 6. Return the final top_k
        final_top_k = self.config.final_top_k
        fused_results = []
        for cid in sorted_chunk_ids[:final_top_k]:
            original_chunk = all_chunks_map[cid]
            # Create a new chunk object representing the fused result
            fused_results.append(RetrievedChunk(
                text=original_chunk.text,
                score=rrf_scores[cid],  # Store the RRF score
                metadata=original_chunk.metadata
            ))

        return fused_results
