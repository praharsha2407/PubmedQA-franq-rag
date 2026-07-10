# Meeting Talking Points — v2 RAG Pipeline

Speaking notes, not a document to hand over. Links: `reports/v2_architecture.md`,
`reports/v2_worked_example.md`, and the two live artifacts if presenting on
screen.

## Opening line

"The baseline is solid and untouched. I found a real bug in my first advanced
pipeline that was making the comparison invalid, retired that version, and
rebuilt it correctly — it's running on the full dataset right now, live."

## 1. Baseline — one line

Reviewed every baseline file end to end: dense FAISS retrieval, CoT prompt,
Mistral generation, standard IR/NLG metrics. No bugs. No changes made.

## 2. Why the first "advanced pipeline" was retired (be direct, not defensive)

- **The core bug:** dense and BM25 indexes were built with two different
  chunk-ID formats (`"pubid:0"` vs `"pubid_0"`). That silently broke fusion's
  ability to recognize the same chunk found by both retrievers, and made any
  BM25-sourced result unmatchable against gold labels during evaluation.
  This is almost certainly why hybrid retrieval scored *worse* than plain
  dense retrieval — an artifact, not a real finding.
- Two smaller issues: the generation-quality script was silently scoring the
  wrong answers file, and one hallucination-scoring component was a hardcoded
  placeholder that was never actually called.
- Nothing was deleted — it's archived at `archive/advanced_pipeline_v1/` for
  reference.

If asked "why didn't you just fix v1 in place": the ID-format fix required
rebuilding how the corpus feeds both retrievers, which touched the same code
the cross-encoder/reranking improvement needed anyway — cleaner to rebuild
once than patch twice.

## 3. What v2 actually is (elevator pitch)

"Same idea as the roadmap I already proposed — query expansion, hybrid
retrieval, reranking, grounded generation, verification — but every stage
now uses a real model doing real work, nothing mocked, and retrieval is
fixed at the root cause."

Stage list, in order, one clause each:
1. Query expansion (biomedical abbreviation dictionary)
2. One shared corpus feeds both retrievers (the fix)
3. Dense (FAISS) + sparse (BM25) retrieval, independently
4. Reciprocal Rank Fusion combines them
5. Cross-encoder reranks the fused pool down to top-5
6. Same Chain-of-Thought prompt and Mistral model as baseline (isolates the
   comparison to retrieval quality, not prompt style)
7. Sentence-level NLI faithfulness check on the output (real entailment
   model, not a formula)

## 4. The live proof point (use this if they push on "does it actually work")

Walked one real question end to end (`reports/v2_worked_example.md`):
*"Do mitochondria play a role in remodelling lace plant leaves during
programmed cell death?"*

- Both dense and BM25 correctly surfaced the source paper's own two chunks
  first — and RRF correctly promoted a chunk found by *both* retrievers over
  one only dense liked strongly. That specific behavior was impossible under
  the old bug.
- Final answer: correct "yes", grounded in the right evidence.
- Retrieval metrics for this example: Recall@5 = 1.0, MRR = 1.0.
- Caught and disclosed a real limitation myself: the sentence-splitter
  mis-flags bare list markers ("1.", "2.") as unverifiable — not a real
  hallucination, a splitter quirk, with a one-line fix identified.

Lead with this if asked to prove it's not just architecture on paper.

## 5. Current status — say this plainly

"Job is running on GPU right now — as of this check, 128/1000 examples done
(~21 minutes in), no errors, roughly 2 hours left. I have not seen final
aggregate numbers yet, so I'm not going to claim v2 beats baseline today.
What I can show is that the comparison itself is now fair and the pipeline
works correctly end to end."

*(Re-check `squeue -j 5513293` and `wc -l outputs/answers_v2.jsonl` right
before the meeting to refresh these numbers — they'll be higher by then.)*

## 6. Likely questions and short answers

**"So is v2 better than baseline?"**
Don't know yet — that's the honest answer, and the right one. The generation
and retrieval quality numbers will be directly comparable once the run
finishes (`retrieval_metrics_v2.json` vs `retrieval_metrics.json`, etc., same
metric code both times).

**"Why trust the new numbers more than the old ones?"**
Because both pipelines are now scored by identical code on identical metric
definitions — `evaluate_generation.py`/`evaluate_custom.py` take an
`--answers-file` flag so baseline and v2 go through the exact same scoring
path. The old v1 comparison used a script that was silently pointed at the
wrong answers file.

**"What if v2 doesn't beat baseline?"**
Then that's the honest result and worth reporting as-is — a thesis showing a
well-reasoned improvement that didn't pan out, with a clear bug-fixed
methodology, is stronger than one presenting numbers that turn out to rest on
a retrieval-evaluation bug.

**"What's left / future work?"**
Scoped out this iteration, named explicitly in the architecture doc:
metadata/decision-aware filtering, explicit context-compression as its own
stage (reranking's top-5 cutoff currently serves that role), and a
biomedical-specific cross-encoder (currently using a general-purpose one).

## 7. If time is short, the 30-second version

"Baseline's correct. Found and fixed a real bug in the first advanced
pipeline — mismatched chunk IDs were silently breaking hybrid retrieval's
evaluation. Retired that version, rebuilt it with real models throughout, no
mocks. It's running on the full dataset right now. I already traced one
question through every stage end to end and it's working correctly — full
comparison numbers by [time]."
