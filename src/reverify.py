"""Re-run the VERIFICATION half of the pipeline (stages 8-12) with components swapped.

Why this exists
---------------
The pipeline splits cleanly in two:

    stages 1-7   produce the answer      -> needs the 7B generator, ~4 hours for 1000 questions
    stages 8-12  verify the answer       -> reads the STORED answer, no generation at all

So any ablation that only touches verification -- the sentence splitter, the keyword method, the
NLI model -- does NOT require regenerating 1000 answers. It re-reads `raw_answer` and
`retrieved_chunk_ids` from a finished run and recomputes everything downstream.

That turns three 4-hour runs into one ~1-hour job, which is what makes the deadline reachable.

What it varies
--------------
    --splitter  {sat, nltk, spacy}          Stage 8
    --keywords  {keybert, tfidf}            Stage 6 signal used by Stage 11
    --nli       <any HF sequence-classification MNLI checkpoint>   Stage 9

Everything else is held constant. The generator is still needed (Stage 10 reads its token
probabilities) but only for FORWARD passes -- no sampling, no autoregressive decoding.

Runs on the FULL stored run (all 1000 answers) unless --limit is given for a smoke test.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from config import AdvancedPipelineConfig, DatasetConfig
from data import build_context_corpus, load_pubmedqa
from generation import MistralGenerator
from retrieval import RetrievedChunk
from stage4_verification import (
    StepD_NLI,
    StepE_ParametricKnowledge,
    step_b_keyword_overlap_text,
    step_f_franq_formula,
)

THRESHOLD = 0.70


def split_sentences(text: str, method: str, config) -> list[str]:
    if method == "sat":
        from sentence_segmentation import segment_sentences
        return segment_sentences(text, config.segmentation)
    if method == "nltk":
        from nltk.tokenize import sent_tokenize
        return [s for s in sent_tokenize(text) if s.strip()]
    if method == "spacy":
        import spacy
        if not hasattr(split_sentences, "_nlp"):
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            split_sentences._nlp = nlp
        return [s.text.strip() for s in split_sentences._nlp(text).sents if s.text.strip()]
    raise ValueError(f"unknown splitter: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", default="outputs_matched/advanced_answers_cot.jsonl")
    parser.add_argument("--out", required=True, help="where to write the report JSON")
    parser.add_argument("--splitter", default="sat", choices=["sat", "nltk", "spacy"])
    parser.add_argument("--keywords", default="keybert", choices=["keybert", "tfidf"])
    parser.add_argument("--nli", default="facebook/bart-large-mnli")
    parser.add_argument("--limit", type=int, default=None, help="smoke test only; omit for the full run")
    args = parser.parse_args()

    config = AdvancedPipelineConfig()
    dataset = load_pubmedqa(DatasetConfig())
    texts, meta = build_context_corpus(dataset)
    cid2text = {m["chunk_id"]: t for t, m in zip(texts, meta)}

    print(f"CONFIG  splitter={args.splitter}  keywords={args.keywords}  nli={args.nli}", flush=True)

    # Stage 6 -- keyword extractor
    if args.keywords == "keybert":
        from keyword_extraction import KeywordExtractor
        extractor = KeywordExtractor(config.keyword)
    else:
        from tfidf_keywords import TfidfKeywordExtractor
        # IDF fitted over the FULL corpus, not the 5 retrieved chunks.
        extractor = TfidfKeywordExtractor(texts, top_n=config.keyword.top_n_keywords)

    # Stage 9 -- NLI
    nli = StepD_NLI(model_name=args.nli)

    # Stage 10 -- the generator, for token probabilities only (forward passes, no sampling)
    generator = MistralGenerator(config.generation)
    if generator.using_fallback_generator:
        raise SystemExit("needs the real generator on a GPU (Stage 10 reads its logits)")
    parametric = StepE_ParametricKnowledge(generator.model, generator.tokenizer)

    rows = []
    for line in Path(args.answers).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break
    print(f"re-verifying {len(rows)} stored answers\n", flush=True)

    verdicts = Counter()
    franq_scores: list[float] = []
    overlaps: list[float] = []
    per_answer: list[dict] = []

    for i, row in enumerate(rows, 1):
        chunks = [
            RetrievedChunk(text=cid2text[c], score=0.0,
                           metadata={"chunk_id": c, "doc_id": c.split(":")[0]})
            for c in row["retrieved_chunk_ids"] if c in cid2text
        ]
        if not chunks:
            continue
        context = "\n".join(c.text for c in chunks)
        question = row["question"]

        sentences = split_sentences(row["raw_answer"], args.splitter, config)
        answer_scores = []

        for sentence in sentences:
            s_kw = extractor.extract_keywords_from_chunk(
                RetrievedChunk(text=sentence, score=0.0, metadata={"chunk_id": "", "doc_id": ""})
            )
            max_overlap = max(
                (step_b_keyword_overlap_text(s_kw, c.text) for c in chunks), default=0.0
            )
            overlaps.append(max_overlap)

            nli_class, p_faithful = "neutral", 0.0
            for c in chunks:
                cls, prob = nli.classify_and_score(c.text, sentence)
                if prob > p_faithful:
                    p_faithful, nli_class = prob, cls

            p_tf = parametric.compute_with_context(question, context, sentence)
            p_tu = parametric.compute_without_context(question, sentence)
            franq = step_f_franq_formula(p_faithful, p_tf, p_tu)
            franq_scores.append(franq)
            answer_scores.append(franq)

            if franq >= THRESHOLD:
                verdict = "verified"
            elif nli_class == "contradiction":
                verdict = "intrinsic"
            elif max_overlap == 0:
                verdict = "extrinsic"
            else:
                verdict = "logical"
            verdicts[verdict] += 1

        per_answer.append({
            "pubid": row["pubid"],
            "mean_franq": sum(answer_scores) / len(answer_scores) if answer_scores else None,
            "sentences": len(sentences),
        })

        if i % 50 == 0:
            print(f"  {i}/{len(rows)} answers, {sum(verdicts.values())} sentences", flush=True)

    n = sum(verdicts.values())
    rates = {k: verdicts[k] / n for k in ("verified", "intrinsic", "extrinsic", "logical")}
    report = {
        "config": {"splitter": args.splitter, "keywords": args.keywords, "nli": args.nli},
        "answers": len(per_answer),
        "sentences": n,
        "mean_sentences_per_answer": n / len(per_answer) if per_answer else 0,
        "verified_rate": rates["verified"],
        "hallucination_rate": 1 - rates["verified"],
        "rates": rates,
        "franq_factuality": sum(franq_scores) / len(franq_scores) if franq_scores else 0.0,
        "keyword_grounding": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        "per_answer": per_answer,
    }

    print("\n" + "=" * 62)
    print(f"splitter={args.splitter}  keywords={args.keywords}")
    print(f"nli={args.nli}")
    print("-" * 62)
    print(f"  sentences            {n}   ({report['mean_sentences_per_answer']:.2f} per answer)")
    print(f"  verified             {rates['verified']:.1%}")
    print(f"  intrinsic            {rates['intrinsic']:.1%}")
    print(f"  extrinsic            {rates['extrinsic']:.1%}")
    print(f"  logical              {rates['logical']:.1%}")
    print(f"  hallucination rate   {1 - rates['verified']:.1%}")
    print(f"  FRANQ factuality     {report['franq_factuality']:.4f}")
    print(f"  keyword grounding    {report['keyword_grounding']:.4f}")
    print("=" * 62)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
