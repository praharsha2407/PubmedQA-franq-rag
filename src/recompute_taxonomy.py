"""Recompute the hallucination taxonomy on a finished run, with the corrected keyword overlap.

Why this can be done without re-running the pipeline
----------------------------------------------------
The verdict depends on three things:

    raw_franq  >= 0.70                      -> verified
    else nli_class == "contradiction"       -> intrinsic
    else keyword_overlap == 0               -> extrinsic
    else                                    -> logical

`raw_franq` is already stored per sentence, and `nli_class == contradiction` is recoverable
(it is exactly the set of sentences the old run labelled "intrinsic" -- that branch is tested
before keyword overlap and does not depend on it). So the ONLY quantity that needs recomputing
is the keyword overlap, which needs KeyBERT but not the 7B generator or the NLI model.

What this does and does not change
----------------------------------
The keyword overlap is consulted ONLY to choose between "extrinsic" and "logical", and only
for sentences already below the FRANQ threshold. So:

    verified rate / hallucination rate  -> UNCHANGED (decided solely by raw_franq >= 0.70)
    FRANQ factuality score              -> UNCHANGED
    extrinsic vs logical split          -> this is what moves
    keyword grounding score             -> computed properly for the first time

That is stated plainly because it would be easy to imply this fixes the 82% figure. It does not.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from config import AdvancedPipelineConfig, DatasetConfig
from data import build_context_corpus, load_pubmedqa
from keyword_extraction import KeywordExtractor
from retrieval import RetrievedChunk
from stage4_verification import step_b_keyword_overlap_text

THRESHOLD = 0.70


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", default="outputs_matched/advanced_answers_cot.jsonl")
    parser.add_argument("--out", default="outputs_matched/taxonomy_corrected.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = AdvancedPipelineConfig()
    texts, meta = build_context_corpus(load_pubmedqa(DatasetConfig()))
    cid2text = {m["chunk_id"]: t for t, m in zip(texts, meta)}
    extractor = KeywordExtractor(config.keyword)

    rows = []
    for line in Path(args.answers).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break

    old = Counter()
    new = Counter()
    overlaps = []
    n_sent = 0

    for i, row in enumerate(rows, 1):
        chunk_texts = [cid2text[c] for c in row["retrieved_chunk_ids"] if c in cid2text]
        for detail in row["hallucination_report"]["details"]:
            sentence = detail["sentence"]
            franq = detail["raw_franq_score"]
            old_verdict = detail["verdict"]
            old[old_verdict] += 1
            n_sent += 1

            keywords = extractor.extract_keywords_from_chunk(
                RetrievedChunk(text=sentence, score=0.0, metadata={"chunk_id": "", "doc_id": ""})
            )
            overlap = max(
                (step_b_keyword_overlap_text(keywords, text) for text in chunk_texts),
                default=0.0,
            )
            overlaps.append(overlap)

            if franq >= THRESHOLD:
                verdict = "verified"
            elif old_verdict == "intrinsic":
                # NLI said contradiction; that branch is tested before keyword overlap,
                # so the corrected overlap cannot change it.
                verdict = "intrinsic"
            elif overlap == 0:
                verdict = "extrinsic"
            else:
                verdict = "logical"
            new[verdict] += 1

        if i % 100 == 0:
            print(f"  {i}/{len(rows)} answers, {n_sent} sentences", flush=True)

    def rates(counter):
        return {k: counter[k] / n_sent for k in ("verified", "intrinsic", "extrinsic", "logical")}

    old_r, new_r = rates(old), rates(new)
    grounding = sum(overlaps) / len(overlaps) if overlaps else 0.0
    zero = sum(1 for o in overlaps if o == 0)

    print("\n" + "=" * 68)
    print(f"{'verdict':<12}{'OLD (exact phrase)':>22}{'NEW (token vs text)':>24}")
    print("-" * 68)
    for k in ("verified", "intrinsic", "extrinsic", "logical"):
        print(f"{k:<12}{old[k]:>8}  {old_r[k]:>9.1%}{new[k]:>12}  {new_r[k]:>9.1%}")
    print("-" * 68)
    old_h = 1 - old_r["verified"]
    new_h = 1 - new_r["verified"]
    print(f"{'hallucinated':<12}{old_h:>19.1%}{new_h:>23.1%}   <- UNCHANGED by design")
    print("=" * 68)
    print(f"\nKeyword grounding (mean overlap, corrected) : {grounding:.4f}")
    print(f"  sentences with ZERO overlap               : {zero}/{n_sent} = {zero/n_sent:.1%}")
    print(f"  (the old exact-phrase metric reported 0.1782, which was NOT a keyword metric")
    print(f"   at all -- it was total_verified / total_sentences.)")

    report = {
        "answers_file": args.answers,
        "sentences": n_sent,
        "threshold": THRESHOLD,
        "old_taxonomy": {k: old[k] for k in old},
        "old_rates": old_r,
        "corrected_taxonomy": {k: new[k] for k in new},
        "corrected_rates": new_r,
        "hallucination_rate": new_h,
        "keyword_grounding_score": grounding,
        "zero_overlap_sentences": zero,
        "note": (
            "Keyword overlap only selects extrinsic vs logical among sentences already below "
            "the FRANQ threshold. The verified/hallucinated split is decided solely by "
            "raw_franq >= 0.70 and is therefore identical in both columns."
        ),
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
