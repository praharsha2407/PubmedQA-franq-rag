"""
Advanced pipeline -- model-at-every-stage architecture.

Stage 1: LLM-based query expansion       (BioMistral, reused)      -- llm_query_expansion.py
Stage 2: Dense retrieval                 (S-PubMedBERT)            -- retrieval.py
Stage 3: Sparse retrieval                (SPLADE-v3)                -- splade_retrieval.py
Stage 4: Fusion                          (Reciprocal Rank Fusion)   -- hybrid_retrieval.py
Stage 5: Cross-encoder reranking         (MS MARCO MiniLM)          -- reranker.py
Stage 6: Keyword extraction              (KeyBERT / PubMedBERT)     -- keyword_extraction.py
Stage 7: Generation                      (BioMistral-7B)            -- generation.py
Stage 8: Sentence segmentation           (SaT)                      -- sentence_segmentation.py
Stage 9: Verification (NLI + parametric knowledge + FRANQ formula)  -- stage4_verification.py
Stage 10: Final answer assembly + hallucination typing              -- stage5_final_output.py
Stage 11: Consolidated evaluation (retrieval + generation + FRANQ)  -- full_evaluation.py
"""
import argparse
import json
import sys
from dataclasses import replace

from config import AdvancedPipelineConfig, OUTPUT_DIR
from query_expansion_final import expand_query_synonyms_and_related_terms
from splade_retrieval import SpladeRetriever
from hybrid_retrieval import HybridRetriever
from reranker import CrossEncoderReranker
from keyword_extraction import KeywordExtractor
from minimal_prompt import build_minimal_prompt
from prompts import build_cot_rag_prompt
from sentence_segmentation import segment_sentences
from stage4_verification import (
    step_b_keyword_overlap_text,
    StepD_NLI,
    StepE_ParametricKnowledge,
    step_f_franq_formula,
)
from stage5_final_output import build_final_answer, generate_hallucination_report
from full_evaluation import FullEvaluationCollector

from data import load_pubmedqa, build_context_corpus
from retrieval import DenseRetriever, RetrievedChunk
from generation import MistralGenerator


def _enforce_strict_mode(config, sparse_retriever, reranker, generator) -> None:
    """Fail loudly if any stage silently fell back to a substitute component.

    Every optional dependency in this pipeline degrades with a WARNING and keeps
    running. That is right for a debug run and wrong for an experimental one: a
    Slurm log scrolls past, and the results table ends up claiming SPLADE-v3 when
    BM25 actually ran. Under --strict, any such substitution aborts the run.
    """
    import sentence_segmentation
    import umls_synonym_expansion

    problems: list[str] = []

    if getattr(sparse_retriever, "using_fallback", False):
        problems.append("Stage 3: SPLADE-v3 unavailable, fell back to BM25")
    # reranker is None when deliberately disabled, which is not a silent fallback.
    if reranker is not None and getattr(reranker, "using_fallback", False):
        problems.append("Stage 5: cross-encoder reranker unavailable, ranking passed through unchanged")
    if getattr(generator, "using_fallback_generator", False):
        problems.append("Stage 7: BioMistral-7B unavailable, using the stub generator")

    # Stages 1a and 8 load lazily, so probe them before the run rather than
    # discovering the fallback halfway through the dataset.
    umls_synonym_expansion._load_umls_pipeline()
    if umls_synonym_expansion._using_fallback:
        problems.append("Stage 1a: scispaCy/UMLS unavailable, synonym expansion skipped")

    segment_sentences("A probe sentence. And another one.", config.segmentation)
    if sentence_segmentation._using_fallback:
        problems.append("Stage 8: SaT unavailable, fell back to NLTK segmentation")

    if problems:
        print("\n" + "!" * 72, file=sys.stderr)
        print("STRICT MODE: refusing to run a degraded pipeline.", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("!" * 72 + "\n", file=sys.stderr)
        raise SystemExit(2)

    print("Strict mode: all stages using their intended components.")


def run_advanced_pipeline(
    sample_size: int | None = None,
    strict: bool = False,
    resume: bool = False,
    prompt_style: str = "minimal",
    use_reranker: bool = False,
    constant_faithful_prior: bool = False,
    model_name: str | None = None,
):
    # The baseline runs the CoT prompt; this pipeline defaults to the minimal one.
    # Comparing the two as-is confounds "better architecture" with "different prompt",
    # so --prompt cot exists to hold the prompt constant and vary only the architecture.
    # Each style writes its own answers/report files so the two runs never collide.
    build_prompt = build_minimal_prompt if prompt_style == "minimal" else build_cot_rag_prompt
    suffix = "" if prompt_style == "minimal" else f"_{prompt_style}"

    config = AdvancedPipelineConfig()

    # The generator LLM must be selectable, and must be recorded. It was neither.
    # run_pipeline.py (the baseline) defaults to mistralai/Mistral-7B-Instruct-v0.3 via its
    # own argparse, while GenerationConfig defaults to BioMistral/BioMistral-7B -- so the
    # baseline and this pipeline silently ran DIFFERENT LLMs, and every baseline-vs-advanced
    # number was a model comparison, not an architecture comparison. Pass --model-name to hold
    # the generator constant.
    if model_name:
        config = replace(config, generation=replace(config.generation, model_name=model_name))
    print(f"Stage 7 generator: {config.generation.model_name}")

    print("Initializing Pipeline Models...")
    dense_retriever = DenseRetriever(config.retrieval)
    sparse_retriever = SpladeRetriever(config.splade)  # Stage 3: SPLADE, not BM25
    hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever, config.hybrid_retrieval)
    if config.reranker.enabled or use_reranker:
        reranker = CrossEncoderReranker(config.reranker)  # Stage 5
    else:
        reranker = None
        print("Stage 5: cross-encoder reranking DISABLED (RerankerConfig.enabled=False); "
              "using the RRF-fused ranking directly.")
    keyword_extractor = KeywordExtractor(config.keyword)
    generator = MistralGenerator(config.generation)

    if strict:
        _enforce_strict_mode(config, sparse_retriever, reranker, generator)

    # Stage 9 models
    nli_model = StepD_NLI()
    parametric_knowledge = StepE_ParametricKnowledge(generator.model, generator.tokenizer)
    # NOTE: Step G (isotonic calibration) intentionally removed -- see CHANGES.md.
    # We report the raw FRANQ score directly ("FRANQ no calibration" variant,
    # a legitimate baseline reported in the FRANQ paper itself).

    evaluator = FullEvaluationCollector()  # Stage 11: consolidated metrics

    if sample_size:
        print(f"Loading PubMedQA (first {sample_size} examples)...")
    else:
        print("Loading full PubMedQA dataset...")
    dataset = load_pubmedqa(config.dataset, sample_size=sample_size)

    print("Building shared corpus (one chunk_id convention for BOTH retrievers)...")
    texts, metadata = build_context_corpus(dataset)
    all_chunks = [RetrievedChunk(text=t, score=0.0, metadata=m) for t, m in zip(texts, metadata)]

    sparse_retriever.build(all_chunks)
    dense_retriever.build(texts, metadata)

    print("Starting pipeline processing...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    answers_path = OUTPUT_DIR / f"advanced_answers{suffix}.jsonl"
    print(f"Prompt style: {prompt_style}  ->  {answers_path.name}")

    # A full run takes hours. Without --resume a killed job (walltime, maintenance,
    # preemption) loses every example. With it, completed pubids are read back and
    # skipped, and new rows are appended.
    run_fingerprint = {
        "prompt_style": prompt_style,
        # The generator was NOT recorded before. Its absence is what allowed the baseline
        # (Mistral-7B-Instruct-v0.3) and this pipeline (BioMistral-7B) to run different LLMs
        # unnoticed across every comparison. Never omit it again.
        "generator_model": config.generation.model_name,
        "seed": config.generation.seed,
        "reranker_enabled": bool(config.reranker.enabled or use_reranker),
        "reranker_model": config.reranker.model_name,
        "sparse_model": config.splade.model_name,
        "dense_model": config.retrieval.embedding_model,
        "final_top_k": config.hybrid_retrieval.final_top_k,
        "rerank_top_k": config.reranker.rerank_top_k,
        "faithfulness": "max_entailment_over_all_chunks",
        "p_true_given_faithful": (
            f"constant_{config.franq.p_true_given_faithful}" if constant_faithful_prior
            else "max_claim_probability"
        ),
    }

    done_pubids: set[str] = set()
    if resume and answers_path.exists():
        prior_fingerprints: list[dict] = []
        with open(answers_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A truncated final line is expected if the job was killed mid-write.
                    print("  (skipping one malformed/truncated row while resuming)")
                    continue
                done_pubids.add(str(row["pubid"]))
                if "run_config" in row:
                    prior_fingerprints.append(row["run_config"])

        mismatched = [fp for fp in prior_fingerprints if fp != run_fingerprint]
        if mismatched:
            print("\n" + "!" * 72, file=sys.stderr)
            print("REFUSING TO RESUME: the existing rows were produced by a different", file=sys.stderr)
            print("configuration. Appending would mix two systems into one results file.", file=sys.stderr)
            print(f"\n  existing: {mismatched[0]}", file=sys.stderr)
            print(f"  current : {run_fingerprint}", file=sys.stderr)
            print("\nEither match the old settings (e.g. --rerank / --prompt), or start a fresh", file=sys.stderr)
            print("run into a new PUBMEDQA_OUTPUT_DIR.", file=sys.stderr)
            print("!" * 72 + "\n", file=sys.stderr)
            raise SystemExit(3)

        if done_pubids and not prior_fingerprints:
            print("  WARNING: existing rows predate config fingerprinting; their settings are unknown.")
        print(f"Resuming: {len(done_pubids)} examples already complete, skipping them.")

    answers_file = open(answers_path, "a" if resume else "w", encoding="utf-8")

    for i, example in enumerate(dataset):
        if example.pubid in done_pubids:
            continue
        print(f"\nProcessing example {i+1}/{len(dataset)}: {example.pubid}")
        question = example.question

        # STAGE 1: synonym (UMLS) + related-term (LLM) query expansion
        expanded_query = expand_query_synonyms_and_related_terms(question, generator, config.query_expansion)

        # STAGES 2-4: dense + SPLADE retrieval, fused by RRF -> top-20 pool
        fused_chunks = hybrid_retriever.search(expanded_query)

        # STAGE 5: cross-encoder reranking -> top-5, or straight off the fused ranking
        # when disabled (see RerankerConfig.enabled: the MS MARCO cross-encoder
        # measurably degrades retrieval on this corpus).
        if reranker is not None:
            top_k_chunks = reranker.rerank(question, fused_chunks)
        else:
            top_k_chunks = fused_chunks[: config.reranker.rerank_top_k]

        # Retrieval metrics are scored later, from the answers file (see below), on the
        # FINAL reranked top-K since that is what actually reaches generation.
        retrieved_ids = [chunk.metadata["chunk_id"] for chunk in top_k_chunks]

        # STAGE 6: keyword corpus
        keyword_corpus = keyword_extractor.build_keyword_corpus(top_k_chunks)

        # STAGE 7: generation (prompt style selected by --prompt)
        prompt = build_prompt(question, top_k_chunks)
        raw_answer = generator.generate(prompt)

        # STAGE 8: neural sentence segmentation (SaT, replaces NLTK)
        sentences = segment_sentences(raw_answer, config.segmentation)

        # STAGE 9: verification
        # The evidence string for P(true | faithful); built once, not once per sentence.
        context_for_franq = "\n".join(chunk.text for chunk in top_k_chunks)
        sentence_results = []
        answer_franq_scores = []

        for sentence in sentences:
            s_keywords = keyword_extractor.extract_keywords_from_chunk(
                RetrievedChunk(text=sentence, score=0.0, metadata={"chunk_id": "", "doc_id": ""})
            )

            # Keyword overlap is used ONLY as the extrinsic signal: does this sentence
            # share any lexical anchor with ANY retrieved chunk? Strict '>' matters --
            # with '>=' and max_overlap starting at 0.0, a sentence overlapping nothing
            # made every chunk "win" in turn, leaving the LAST (lowest-ranked) chunk.
            # Matched against the chunk's FULL TEXT, not its top-10 keyphrases, and at the
            # token level rather than by exact phrase equality. The old metric compared two
            # independently-extracted KeyBERT phrase sets by string identity against a 10-item
            # summary of the chunk, so it returned 0 for sentences whose content was plainly
            # present in the evidence -- branding them "extrinsic" (invented) when they were not.
            max_overlap = 0.0
            for chunk in top_k_chunks:
                overlap = step_b_keyword_overlap_text(s_keywords, chunk.text)
                if overlap > max_overlap:
                    max_overlap = overlap

            # Faithfulness: a sentence is faithful if ANY retrieved chunk entails it, so
            # score it against all of them and keep the strongest. Selecting a single
            # premise by keyword overlap frequently handed the NLI model an unrelated
            # chunk, which it (correctly) refused to entail -- inflating "logical".
            best_chunk_id = None
            nli_class = "neutral"
            p_faithful = 0.0
            for chunk in top_k_chunks:
                cls, prob = nli_model.classify_and_score(chunk.text, sentence)
                if prob > p_faithful:
                    p_faithful = prob
                    nli_class = cls
                    best_chunk_id = chunk.metadata["chunk_id"]

            p_true_given_unfaithful = parametric_knowledge.compute_without_context(question, sentence)

            # P(true | faithful). The FRANQ paper estimates this per claim (Max Claim
            # Probability, p(c|x,r)) because a claim can be faithful to the evidence and
            # still answer the question wrongly. The original constant 0.95 is kept behind
            # a flag so that variant remains reproducible as an ablation row.
            if constant_faithful_prior:
                p_true_given_faithful = config.franq.p_true_given_faithful
            else:
                p_true_given_faithful = parametric_knowledge.compute_with_context(
                    question, context_for_franq, sentence
                )

            raw_franq = step_f_franq_formula(p_faithful, p_true_given_faithful, p_true_given_unfaithful)
            answer_franq_scores.append(raw_franq)

            verdict = "verified"
            if raw_franq < 0.7:
                if nli_class == "contradiction":
                    verdict = "intrinsic"
                elif max_overlap == 0:
                    verdict = "extrinsic"
                else:
                    verdict = "logical"

            sentence_results.append({
                # Persisted so the keyword-grounding metric can be computed from the answers
                # file instead of being conflated with the verified rate (see stage6_evaluation).
                "keyword_overlap": max_overlap,
                "nli_class": nli_class,
                "sentence": sentence,
                "cited_chunk": best_chunk_id if verdict == "verified" else None,
                "verdict": verdict,
                "raw_franq_score": raw_franq,
            })

        # STAGE 10: final output
        final_answer = build_final_answer(sentence_results)
        report = generate_hallucination_report(sentence_results)

        mean_franq = (sum(answer_franq_scores) / len(answer_franq_scores)) if answer_franq_scores else None

        # The verification results are persisted per row, not just accumulated in
        # memory, so a resumed run can rebuild the consolidated report from the
        # file rather than only scoring the examples processed in this process.
        answer_record = {
            "pubid": example.pubid,
            "question": example.question,
            "expanded_query": expanded_query,
            "generated_answer": final_answer,
            # The raw pre-verification output. The baseline stores its raw output as
            # generated_answer, so scoring the PubMedQA yes/no/maybe decision against
            # our post-verification text compares two different things: verification can
            # drop or reshape the "Conclusion: ..." sentence before it is ever read.
            "raw_answer": raw_answer,
            # Gold label, so the decision metric never has to re-load the dataset.
            "final_decision": example.final_decision,
            "reference_answer": example.long_answer,
            "retrieved_chunk_ids": retrieved_ids,
            "hallucination_report": report,
            "mean_franq_score": mean_franq,
            # Provenance. A --resume run picks up whatever config.py currently says, which
            # is not necessarily what produced the earlier rows. Recording the settings that
            # actually generated each row makes a mixed-configuration file detectable instead
            # of silently averaging two different systems into one results table.
            "run_config": run_fingerprint,
        }
        answers_file.write(json.dumps(answer_record) + "\n")
        answers_file.flush()

        print(f"Generated raw answer:\n{raw_answer}")
        print(f"Verified final answer:\n{final_answer}")

    answers_file.close()

    # STAGE 11: consolidated evaluation -> outputs/full_evaluation_report.json
    # Rebuild from the answers file so the report covers every completed example,
    # including ones written by an earlier (killed) run when --resume is used.
    print("\nBuilding consolidated report from", answers_path)
    relevant_by_pubid = {
        ex.pubid: {f"{ex.pubid}:{idx}" for idx, _ in enumerate(ex.contexts)} for ex in dataset
    }
    scored = 0
    with open(answers_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print("  (skipping malformed row)")
                continue
            relevant = relevant_by_pubid.get(str(row["pubid"]))
            if relevant is None:
                continue  # row from a different sample_size slice
            evaluator.add_retrieval_result(row["retrieved_chunk_ids"], relevant, k=config.reranker.rerank_top_k)
            evaluator.add_generation_result(row["generated_answer"], row["reference_answer"])
            if row.get("hallucination_report") and row.get("mean_franq_score") is not None:
                evaluator.add_hallucination_result(row["hallucination_report"], row["mean_franq_score"])
            scored += 1
    print(f"Scoring {scored} completed examples.")

    evaluator.compute_and_save(OUTPUT_DIR / f"full_evaluation_report{suffix}.json")

    print("\nPipeline execution complete. Results saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FRANQ advanced RAG pipeline (stages 1-12).")
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Run on only the first N examples (default: the full dataset).",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Abort if any stage falls back to a substitute component (SPLADE->BM25, SaT->NLTK, "
             "UMLS skipped, BioMistral->stub). Use this for any run whose numbers you intend to report.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip examples already present in outputs/advanced_answers.jsonl and append new ones. "
             "Use this to continue a run that was killed by walltime or maintenance.",
    )
    parser.add_argument(
        "--prompt", choices=["minimal", "cot"], default="minimal",
        help="Which prompt Stage 7 uses. 'minimal' (default) is the unconstrained prompt; "
             "'cot' is the same chain-of-thought prompt the baseline uses. Run with 'cot' to "
             "compare against the baseline with the prompt held constant, so any difference is "
             "attributable to the architecture rather than the prompt. Writes "
             "advanced_answers_cot.jsonl / full_evaluation_report_cot.json.",
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help="Re-enable Stage 5 cross-encoder reranking. Off by default: on PubMedQA the MS MARCO "
             "cross-encoder lowers MAP below the dense-only baseline (0.750 vs 0.761 held-out), "
             "while RRF fusion alone raises it to 0.777. See src/ablate_retrieval.py.",
    )
    parser.add_argument(
        "--constant-faithful-prior", action="store_true",
        help="Use the fixed P(true|faithful)=0.95 instead of the paper's per-claim Max Claim "
             "Probability p(c|x,r). Reproduces the earlier runs as an ablation row; not the "
             "faithful reading of Fadeeva et al. (2025).",
    )
    parser.add_argument(
        "--model-name", default=None,
        help="Stage 7 generator LLM. Defaults to GenerationConfig (BioMistral/BioMistral-7B). "
             "The baseline (run_pipeline.py) defaults to mistralai/Mistral-7B-Instruct-v0.3, so "
             "the two systems ran DIFFERENT models and no past baseline-vs-advanced number was a "
             "clean architecture comparison. Pass the baseline's model here to hold it constant.",
    )
    args = parser.parse_args()
    run_advanced_pipeline(
        sample_size=args.sample_size, strict=args.strict, resume=args.resume,
        prompt_style=args.prompt, use_reranker=args.rerank,
        constant_faithful_prior=args.constant_faithful_prior,
        model_name=args.model_name,
    )
