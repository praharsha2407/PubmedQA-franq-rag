from dataclasses import dataclass
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ.get("PUBMEDQA_OUTPUT_DIR", PROJECT_ROOT / "outputs")).resolve()
INDEX_DIR = Path(os.environ.get("PUBMEDQA_INDEX_DIR", PROJECT_ROOT / "data" / "faiss")).resolve()
CACHE_DIR = Path(os.environ.get("PUBMEDQA_CACHE_DIR", PROJECT_ROOT / "data" / "cache")).resolve()

@dataclass(frozen=True)
class DatasetConfig:
    name: str = "qiaojin/PubMedQA"
    subset: str = "pqa_labeled"
    split: str = "train"

@dataclass(frozen=True)
class RetrievalConfig:
    embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    top_k: int = 5
    batch_size: int = 32
    max_seq_length: int = 512

@dataclass(frozen=True)
class GenerationConfig:
    model_name: str = "BioMistral/BioMistral-7B"
    max_new_tokens: int = 384
    temperature: float = 0.2
    top_p: float = 0.9
    load_in_4bit: bool = True
    # Seed for reproducible sampled decoding. With do_sample=True and no seed the
    # same (prompt, model) produced different answers across runs -- that is why the
    # baseline's stored answers.jsonl (Conclusion in 200/200) could not be reproduced
    # by re-running the same code path. Fixed here so runs are comparable.
    seed: int = 42

@dataclass(frozen=True)
class PipelineConfig:
    dataset: DatasetConfig = DatasetConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()

@dataclass(frozen=True)
class QueryExpansionConfig:
    dictionary_path: str = ""
    use_mesh_lookup: bool = False
    # LLM-based expansion (Jagerman et al., 2023, arXiv 2305.03653) -- reuses
    # the same generator model already loaded for Stage 7, no new dependency.
    use_llm_expansion: bool = True
    max_new_tokens: int = 48

@dataclass(frozen=True)
class SparseRetrievalConfig:
    # Kept for the legacy BM25Retriever (sparse_retrieval.py), which SpladeRetriever
    # falls back to when the SPLADE checkpoint cannot be loaded.
    top_k: int = 20
    k1: float = 1.5
    b: float = 0.75

@dataclass(frozen=True)
class SpladeConfig:
    # SPLADE-v3 (Lassance et al., 2024), building on SPLADE (Formal et al.,
    # 2021, SIGIR). Replaces BM25 as the sparse retriever with a trained
    # neural sparse-lexical model instead of a statistical formula.
    model_name: str = "naver/splade-v3"
    top_k: int = 20
    batch_size: int = 16
    max_seq_length: int = 512

@dataclass(frozen=True)
class HybridRetrievalConfig:
    dense_top_k: int = 20
    sparse_top_k: int = 20
    # Fused pool handed to the reranker. dense (20) and sparse (20) can union to 40
    # distinct chunks; truncating to 20 here discarded up to half the candidates
    # before the cross-encoder ever scored them.
    final_top_k: int = 40
    rrf_k: int = 60

@dataclass(frozen=True)
class RerankerConfig:
    # Cross-encoder reranking (Reimers & Gurevych, 2019, Sentence-BERT --
    # cross-encoder variant). Reads (query, chunk) jointly for a more
    # accurate relevance score than the fusion step alone provides.
    #
    # DISABLED BY DEFAULT on the evidence of src/ablate_retrieval.py. This
    # checkpoint is trained on MS MARCO web-search relevance and degrades
    # retrieval on PubMedQA abstracts. Measured MAP, dev(n=300)/held-out(n=700):
    #     dense only            0.7420 / 0.7608   (the baseline)
    #     RRF fusion, no rerank 0.7692 / 0.7768   <- best, and what we now use
    #     RRF + this reranker   0.7308 / 0.7503   <- worse than the baseline
    # Re-enable with --rerank to reproduce the ablation. A biomedical
    # cross-encoder (e.g. ncbi/MedCPT-Cross-Encoder) may well win; untested.
    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 5

@dataclass(frozen=True)
class SegmentationConfig:
    # Segment Any Text / SaT (Frohmann et al., 2024, arXiv 2406.16678).
    # Neural sentence segmentation, replacing NLTK's rule-based tokenizer
    # (which produced the "1." / "2." splitter artifacts noted earlier).
    model_name: str = "sat-3l-sm"
    fallback_to_nltk: bool = True

@dataclass(frozen=True)
class VerificationConfig:
    nli_model: str = "facebook/bart-large-mnli"
    entailment_threshold: float = 0.5
    overlap_threshold: float = 0.3

@dataclass(frozen=True)
class KeywordConfig:
    method: str = "keybert"
    top_n_keywords: int = 10
    keybert_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"

@dataclass(frozen=True)
class FranqConfig:
    # FRANQ verification (Stage 5). Step G (isotonic calibration) is intentionally omitted:
    # it requires a labeled calibration set to do real work, so the raw FRANQ score is used directly.
    nli_model: str = "facebook/bart-large-mnli"
    p_true_given_faithful: float = 0.95   # prior used in the FRANQ marginalization (Step F)
    verified_threshold: float = 0.70      # FRANQ score at/above which a sentence counts as verified

@dataclass(frozen=True)
class AdvancedPipelineConfig:
    dataset: DatasetConfig = DatasetConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    query_expansion: QueryExpansionConfig = QueryExpansionConfig()
    sparse_retrieval: SparseRetrievalConfig = SparseRetrievalConfig()
    splade: SpladeConfig = SpladeConfig()
    hybrid_retrieval: HybridRetrievalConfig = HybridRetrievalConfig()
    reranker: RerankerConfig = RerankerConfig()
    segmentation: SegmentationConfig = SegmentationConfig()
    verification: VerificationConfig = VerificationConfig()
    keyword: KeywordConfig = KeywordConfig()
    franq: FranqConfig = FranqConfig()