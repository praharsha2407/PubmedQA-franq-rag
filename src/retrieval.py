#Purpose: Encodes text and manages the FAISS vector index.
#Core Classes & Functions:
#DenseRetriever:
#.encode(): Uses SentenceTransformer(config.embedding_model) to convert texts to numpy arrays and normalizes them.
#.build(): Generates embeddings for the entire corpus and loads them into faiss.IndexFlatIP (flat Inner Product index).
#.save() / .load(): Persists the FAISS index and corresponding text/metadata files to the local disk.
#.search(): Takes a question, embeds it, queries FAISS for nearest neighbors, and returns a list of RetrievedChunk instances.
#Includes a fallback hashing-based vectorizer (_hashing_embedding) for local testing.

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from config import RetrievalConfig

try:
    import faiss
except ImportError:
    faiss = None


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    metadata: dict[str, str]


class DenseRetriever:
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.encoder = None
        self.using_fallback_encoder = False
        try:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(config.embedding_model)
            self.encoder.max_seq_length = config.max_seq_length
        except Exception as exc:
            self.using_fallback_encoder = True
            print(
                "WARNING: sentence-transformers could not be loaded. "
                "Using a lightweight hashing encoder for local testing only. "
                f"Original error: {exc}"
            )
        self.index = None
        self.embeddings: np.ndarray | None = None
        self.texts: list[str] = []
        self.metadata: list[dict[str, str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.encoder is not None:
            embeddings = self.encoder.encode(
                texts,
                batch_size=self.config.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            return embeddings.astype("float32")

        embeddings = np.vstack([self._hashing_embedding(text) for text in texts])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
        return embeddings.astype("float32")

    @staticmethod
    def _hashing_embedding(text: str, dim: int = 768) -> np.ndarray:
        vector = np.zeros(dim, dtype="float32")
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        for token in tokens:
            vector[hash(token) % dim] += 1.0
        return vector

    def build(self, texts: list[str], metadata: list[dict[str, str]]) -> None:
        if len(texts) != len(metadata):
            raise ValueError("texts and metadata must have the same length")
        self.texts = texts
        self.metadata = metadata
        embeddings = self.encode(texts)
        self.embeddings = embeddings
        # Defensive checks: ensure embeddings is a 2D array with shape (n_items, dim)
        if embeddings is None or embeddings.size == 0:
            raise RuntimeError("Encoder returned empty embeddings array; check input texts and encoder configuration")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
            self.embeddings = embeddings
        if embeddings.ndim != 2:
            raise RuntimeError(f"Unexpected embeddings shape: {embeddings.shape}")
        if faiss is not None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)

    def save(self, directory: Path) -> None:
        if self.embeddings is None:
            raise RuntimeError("Cannot save before building the retrieval index")
        directory.mkdir(parents=True, exist_ok=True)
        if faiss is not None and self.index is not None:
            faiss.write_index(self.index, str(directory / "index.faiss"))
        np.save(directory / "embeddings.npy", self.embeddings)
        (directory / "texts.json").write_text(json.dumps(self.texts, ensure_ascii=False), encoding="utf-8")
        (directory / "metadata.json").write_text(json.dumps(self.metadata, ensure_ascii=False), encoding="utf-8")

    def load(self, directory: Path) -> None:
        faiss_path = directory / "index.faiss"
        if faiss is not None and faiss_path.exists():
            self.index = faiss.read_index(str(faiss_path))
        self.embeddings = np.load(directory / "embeddings.npy")
        self.texts = json.loads((directory / "texts.json").read_text(encoding="utf-8"))
        self.metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))

    def search(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if self.index is None and self.embeddings is None:
            raise RuntimeError("Retrieval index has not been built")
        k = top_k or self.config.top_k
        query_embedding = self.encode([question])

        if self.index is not None:
            scores, indices = self.index.search(query_embedding, k)
            score_values = scores[0]
            index_values = indices[0]
        else:
            similarities = np.dot(self.embeddings, query_embedding[0])
            index_values = np.argsort(similarities)[::-1][:k]
            score_values = similarities[index_values]

        results: list[RetrievedChunk] = []
        for score, idx in zip(score_values, index_values):
            if idx < 0:
                continue
            results.append(
                RetrievedChunk(
                    text=self.texts[idx],
                    score=float(score),
                    metadata=self.metadata[idx],
                )
            )
        return results
