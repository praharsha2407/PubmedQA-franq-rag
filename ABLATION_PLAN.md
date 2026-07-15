# Ablation study — plan, changes, and results

Requested by the supervisor: *"it does not matter about the results — we need to know the importance
of each method separately."* That framing is followed exactly. Every component is switched off or
swapped, and the effect is measured.

**Every run below is on the FULL dataset (all 1,000 PubMedQA questions).** Where a dev/held-out split
appears, it is still all 1,000 — split so a choice can be made on 300 and confirmed on the 700 it
never saw. No number in this document comes from a subsample.

**The baseline is untouched.** `run_pipeline.py`, `prompts.py`, `retrieval.py`, `data.py` and
`outputs/answers.jsonl` are unchanged since the first commit.

---

## Code changes made for this study

| # | File | Change | Baseline affected? |
|---|---|---|---|
| 1 | `src/ablate_rrf_k.py` | **NEW.** Sweeps the RRF constant `k` and the evidence depth. Retrieval only, no generation. | No — new file |
| 2 | `src/tfidf_keywords.py` | **NEW.** TF-IDF keyword extractor, drop-in replacement for KeyBERT. IDF fitted on the **full 3,358-chunk corpus**. | No — new file |
| 3 | `src/reverify.py` | **NEW.** Re-runs stages 8–12 on *stored* answers with the splitter / keyword method / NLI model swapped. Avoids regenerating 1,000 answers per variant. | No — new file |
| 4 | `src/run_advanced_pipeline.py` | Added `--evidence-k` (how many chunks reach the generator **and** the keyword corpus; default 5 = old behaviour). Added `rrf_k` to the run fingerprint. | No — advanced runner only |

All ablation outputs go to **new** directories: `outputs_ablation_*/` and `results/ablation/`.

---

## The design decision that made this feasible

The pipeline splits in two:

- **Stages 1–7** produce the answer → needs the 7B generator → **~4 hours per full run**
- **Stages 8–12** verify the answer → reads the **stored** answer → **no generation needed**

So an ablation that only touches *verification* (splitter, keyword method, NLI model) does **not**
require regenerating 1,000 answers. `reverify.py` re-reads `raw_answer` and `retrieved_chunk_ids`
from a finished run and recomputes everything downstream. Stage 10 still needs the generator, but
only for **forward passes** (reading token probabilities) — no sampling, no autoregressive decoding.

**Three 4-hour runs become one ~1-hour job.** Only prompt-changing variants (evidence depth,
reranker, generator) need a full re-run.

---

## Ablation matrix

### A. Retrieval — which retriever matters? *(already complete)*

MAP, dev (n=300) / held-out (n=700):

| Variant | dev | held-out |
|---|---|---|
| Dense only | 0.7420 | 0.7608 |
| Sparse only (SPLADE) | 0.7492 | 0.7660 |
| **Both, fused by RRF** | **0.7692** | **0.7768** |
| RRF + cross-encoder reranker | 0.7308 | 0.7503 |
| Dense + cross-encoder reranker | 0.7380 | 0.7583 |

**Reading:** neither retriever alone matches the fusion. SPLADE alone slightly beats dense alone —
the exact-term matching matters on biomedical text. The cross-encoder **hurts** on both splits.

*Evidence:* `src/ablate_retrieval.py`, `results/retrieval_ablation_{dev,heldout}.json`

### B. RRF constant `k` — why 60? **COMPLETE**

`k = 60` was taken verbatim from Cormack et al. (2009) and **never justified**. Swept over
{1, 10, 20, 40, 60, 80, 100}. Candidate pool = 50 per retriever, evidence depth 5.

MAP:

| k | dev (n=300) | held-out (n=700) |
|---|---|---|
| **1** | **0.7880** | **0.7947** |
| 10 | 0.7843 | 0.7890 |
| 20 | 0.7834 | 0.7871 |
| 40 | 0.7819 | 0.7860 |
| 60 *(the pipeline's value)* | 0.7818 | 0.7854 |
| 80 | 0.7811 | 0.7854 |
| 100 | 0.7811 | 0.7854 |

**Finding: MAP decreases monotonically as k increases, on both splits. k = 60 is not optimal;
k = 1 is best.** The effect is small (~0.01 MAP) but consistent and it replicates on held-out data.

**Why.** `k` controls how steeply the top ranks are weighted in `1 / (k + rank)`:

- `k = 1`: rank 1 → `1/2 = 0.500`, rank 2 → `1/3 = 0.333` — **steep. Rank 1 dominates.**
- `k = 60`: rank 1 → `1/61 = 0.0164`, rank 2 → `1/62 = 0.0161` — **almost flat. All ranks weigh
  nearly the same.**

In PubMedQA the gold evidence is the question's own abstract, and it is *usually ranked #1 by both
retrievers*. So a small `k`, which aggressively trusts rank 1, wins; a large `k` flattens the
weighting and lets lower-ranked noise compete with the correct chunk.

**How to report it honestly.** The pipeline's results were produced with `k = 60`. This sweep shows
that choice was inherited, not optimised, and that a better value exists. The gain is small, and the
headline results are *not* re-run with `k = 1` — that is stated, not hidden. Cormack et al. tuned
`k = 60` on TREC web-search collections, where relevant documents are spread across ranks; PubMedQA's
rank-1-dominant structure is different, which is exactly why the optimum differs.

*Evidence:* `src/ablate_rrf_k.py`, `results/ablation_rrf_k_{dev,heldout}.json`

### B2. Retrieval depth — how deep should the candidate pool be? **COMPLETE**

Held-out (n=700), RRF with k=60:

| top-k | MAP | Recall | Precision |
|---|---|---|---|
| 5 | 0.7854 | 0.805 | 0.516 |
| 10 | 0.8024 | 0.868 | 0.280 |
| 20 | 0.8095 | 0.901 | 0.146 |
| 30 | 0.8117 | 0.918 | 0.099 |
| 40 | 0.8124 | 0.925 | 0.075 |

**Retrieval quality keeps rising with depth, but precision collapses** (0.52 → 0.08). Both are true
by construction: deeper lists find more of the gold chunks, but the fraction of retrieved chunks that
are relevant falls.

**This table cannot decide the evidence depth.** It measures the *retriever*, not the *generator*.
Whether more evidence helps the LLM — or simply floods the prompt with noise — is settled only by the
full generation runs in section C.

*Note:* these MAP values are higher than section A's (0.7854 vs 0.7768 at depth 5) because the
candidate pool here is 50 per retriever rather than 20. A deeper pool gives RRF more to fuse.

### C. Evidence depth — how many chunks reach the generator? *(jobs 5539402/03/04)*

The pipeline sends **top-5**. Never justified. Full runs at **k = 10, 20, 30**.

Two effects pull against each other: more evidence raises recall, but also adds noise and dilutes
the prompt. Retrieval metrics alone cannot settle it — only a full generation run can.

*Results: pending.*

### D. Cross-encoder reranker, enabled *(job 5539405)*

Disabled by default on retrieval evidence (it lowers MAP). This runs the **full pipeline** with it
on, to measure the end-to-end effect on accuracy and hallucination, not just retrieval.

*Results: pending.*

### E. Generator — BioMistral vs Mistral-Instruct *(complete)*

| Generator | Accuracy | Emits required `Conclusion:` |
|---|---|---|
| Mistral-7B-Instruct-v0.3 | **0.630** | 999 / 1000 |
| BioMistral-7B | 0.471 | **0 / 1000** |

**Reading:** the biomedical model is *worse*. Continued pretraining on PubMed appears to have eroded
its instruction-following — it never produces the required output format. Domain knowledge and
instruction-following do not combine for free.

*Evidence:* `outputs_matched/`, `outputs_cot/`

### F. Keyword extraction — KeyBERT vs TF-IDF *(reverify)*

KeyBERT is *semantic* (embeds document and n-grams, keeps the closest). TF-IDF is *statistical*
(frequent here, rare across the corpus). IDF is fitted on the **full corpus**, since "inverse
document frequency" is only meaningful against a real collection.

Affects the **extrinsic** hallucination signal only.

*Results: pending.*

### G. Sentence segmentation — SaT vs NLTK vs spaCy *(reverify)*

Verification is per-sentence, so a bad split becomes a fake hallucination. NLTK keys on punctuation
and slices `"1."` off a numbered list as its own sentence; the NLI model then refuses to entail a
bare digit and it is recorded as a hallucination.

*Results: pending.*

### H. NLI model for P(faithful) — BART vs RoBERTa vs DeBERTa *(reverify)*

How sensitive is the faithfulness estimate to the entailment model?

**Note on AlignScore:** the published FRANQ method uses AlignScore, not NLI. It was **deliberately
not attempted**: it requires new packages whose dependencies risk breaking the shared conda
environment the baseline depends on, and there was no time to recover from a broken environment.
This remains the single most valuable piece of future work, and is stated as such.

*Results: pending.*

### I. Extrinsic vs intrinsic signal — **not pursued**

The supervisor raised a point about the extrinsic/intrinsic signals that could not be reconstructed
precisely enough to turn into an experiment, and clarification was not available in time. It is
recorded here rather than quietly dropped.

For reference, the current definitions are:

- **intrinsic** — the NLI model returns *contradiction*: the sentence **fights** the evidence.
- **extrinsic** — keyword overlap is zero: the sentence shares **no lexical anchor** with any
  retrieved chunk, i.e. it was invented from nowhere.

A plausible future experiment would be to detect *extrinsic* with the NLI model as well (e.g. a very
low entailment probability against every chunk) rather than with a lexical signal, and compare the
two definitions. Not run.

---

## Results table — ALL COMPLETE

### Generation-side ablations (full pipeline, n=1000, task accuracy)

| Stage | Variant | Accuracy | macro-F1 | Reading |
|---|---|---|---|---|
| 7 | Generator = Mistral-Instruct *(reference)* | 0.629 | 0.470 | the controlled system |
| 7 | Generator = BioMistral-7B | 0.471 | 0.329 | domain model **fails** (0/1000 format) |
| 5+7 | **Evidence depth = 5** *(current)* | 0.629 | 0.470 | |
| 5+7 | **Evidence depth = 10** | **0.642** | **0.485** | **best — the peak** |
| 5+7 | Evidence depth = 20 | 0.631 | 0.483 | past the peak |
| 5+7 | Evidence depth = 30 | 0.630 | 0.475 | noise dilutes the prompt |
| 5 | Reranker **enabled** | 0.608 | 0.461 | **hurts accuracy too**, not just MAP |

**Findings.**
- **Evidence depth has an optimum at 10**, not 5. Accuracy rises 0.629 → 0.642 (5 → 10), then falls
  back to 0.630 by depth 30. More evidence helps until it starts flooding the prompt with noise. This
  is the trade-off section B2's retrieval table *could not* resolve — the generation runs resolve it.
- **The reranker degrades end-to-end accuracy** (0.629 → 0.608), strengthening the retrieval-only
  finding: it fails on the metric that actually matters, not just on MAP.

### Verification-side ablations (reverify on the stored 1000 answers)

| Stage | Variant | Verified % | FRANQ | Intrinsic % | Keyword grounding |
|---|---|---|---|---|---|
| 6/8/9 | SaT · KeyBERT · BART *(reference)* | 12.2 | 0.451 | 6.5 | 0.508 |
| 8 | Splitter → NLTK | 12.9 | 0.449 | 6.5 | 0.526 |
| 8 | Splitter → spaCy | 12.9 | 0.449 | 6.5 | 0.526 |
| 6 | Keywords → **TF-IDF** | 12.2 | 0.451 | 6.5 | 0.507 |
| 9 | NLI → RoBERTa-MNLI | 11.5 | 0.454 | 3.8 | 0.508 |
| 9 | NLI → DeBERTa-MNLI | 9.0 | 0.435 | 4.2 | 0.508 |

**Findings.**
- **TF-IDF ≡ KeyBERT** (12.2% both, FRANQ 0.451 both). The extrinsic signal is robust to the keyword
  method — the simpler, GPU-free TF-IDF is as good as semantic KeyBERT. A simpler design is justified.
- **The sentence splitter barely matters here** (12.2 vs 12.9%). SaT's advantage over NLTK is real but
  *model-dependent*: it protects against the numbered-list mis-splits that BioMistral produced, and
  Mistral-Instruct produces far fewer of them. Report SaT's benefit as conditional, not universal.
- **The NLI model is the most influential verification component**: verified rate spans 9.0–12.9%.
  DeBERTa is strictest; RoBERTa finds the fewest contradictions (intrinsic 3.8% vs BART's 6.5%). This
  sensitivity is precisely why the paper's AlignScore choice matters, and why it is named as the
  highest-value future work.

*Evidence:* `outputs_ablation_*/decision_metrics*.json`, `results/ablation/*.json`

### One line per component — "the importance of each method"

| Component | If removed / swapped | Verdict |
|---|---|---|
| Dense retrieval | −MAP vs fusion (0.761 vs 0.777) | contributes |
| Sparse (SPLADE) | −MAP vs fusion (0.766 vs 0.777) | contributes (slightly > dense alone) |
| RRF fusion | best of the three | **essential** |
| Cross-encoder reranker | −MAP **and** −accuracy | **remove** |
| RRF k = 60 | k = 1 is better (small) | inherited, not optimal |
| Evidence depth 5 | depth 10 is better | **raise to 10** |
| Generator (BioMistral) | −17 accuracy points | **use Mistral-Instruct** |
| KeyBERT | = TF-IDF | either; TF-IDF is simpler |
| SaT splitter | ≈ NLTK/spaCy on this model | keep, but benefit is conditional |
| BART NLI | swings verified 9–13% | most sensitive knob; AlignScore = future work |
