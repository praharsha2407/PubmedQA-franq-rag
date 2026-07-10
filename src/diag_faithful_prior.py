"""Compare the constant P(true|faithful)=0.95 against the paper's Max Claim Probability.

FRANQ(c) = P_faithful · P(true|faithful) + (1 - P_faithful) · P(true|unfaithful)

The pipeline froze the middle term at 0.95. Fadeeva et al. (2025, Sec 2.2) estimate it
per claim as p(c | x, r) -- the model's own probability of the claim GIVEN the retrieved
evidence -- precisely because a claim can be faithful to the evidence and still answer the
question wrongly. That is the "logical" hallucination category.

This reads real sentences out of a completed run, recomputes both variants, and reports
how much the verdicts move. No pipeline rerun; it only needs the generator.
"""
from __future__ import annotations

import argparse
import json
import sys

from config import AdvancedPipelineConfig, DatasetConfig
from data import build_context_corpus, load_pubmedqa
from generation import MistralGenerator
from stage4_verification import step_f_franq_formula, StepE_ParametricKnowledge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True)
    parser.add_argument("--limit", type=int, default=40, help="sentences to examine")
    args = parser.parse_args()

    config = AdvancedPipelineConfig()
    dataset = load_pubmedqa(DatasetConfig())
    texts, meta = build_context_corpus(dataset)
    cid2text = {m["chunk_id"]: t for t, m in zip(texts, meta)}

    generator = MistralGenerator(config.generation)
    if generator.using_fallback_generator:
        sys.exit("generator fell back; this diagnostic needs the real model on a GPU")
    pk = StepE_ParametricKnowledge(generator.model, generator.tokenizer)

    cases = []
    with open(args.answers, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            context = "\n".join(cid2text[c] for c in row["retrieved_chunk_ids"] if c in cid2text)
            for d in row["hallucination_report"]["details"]:
                cases.append((row["question"], context, d["sentence"], d["verdict"], d["raw_franq_score"]))
            if len(cases) >= args.limit:
                break
    cases = cases[: args.limit]
    print(f"examining {len(cases)} sentences\n", flush=True)

    thresh = config.franq.verified_threshold
    mcp_vals, flips_up, flips_down = [], 0, 0
    by_verdict: dict[str, list[float]] = {}

    print(f"{'verdict':<11}{'P(f)':>8}{'MCP':>8}{'FRANQ old':>11}{'FRANQ new':>11}  change")
    print("-" * 62)
    for q, ctx, sent, verdict, old_franq in cases:
        p_unf = pk.compute_without_context(q, sent)
        mcp = pk.compute_with_context(q, ctx, sent)
        mcp_vals.append(mcp)
        by_verdict.setdefault(verdict, []).append(mcp)

        # Recover P(faithful) from the stored score: old = pf*0.95 + (1-pf)*p_unf
        denom = 0.95 - p_unf
        pf = (old_franq - p_unf) / denom if abs(denom) > 1e-6 else 0.0
        pf = min(max(pf, 0.0), 1.0)

        new_franq = step_f_franq_formula(pf, mcp, p_unf)
        moved = ""
        if old_franq >= thresh and new_franq < thresh:
            flips_down += 1; moved = "-> now UNVERIFIED"
        elif old_franq < thresh and new_franq >= thresh:
            flips_up += 1; moved = "-> now VERIFIED"
        print(f"{verdict:<11}{pf:>8.3f}{mcp:>8.3f}{old_franq:>11.3f}{new_franq:>11.3f}  {moved}", flush=True)

    n = len(mcp_vals)
    print(f"\nMax Claim Probability p(c|x,r) over {n} sentences:")
    mcp_vals.sort()
    print(f"  min {mcp_vals[0]:.3f}  p25 {mcp_vals[n//4]:.3f}  median {mcp_vals[n//2]:.3f} "
          f"p75 {mcp_vals[3*n//4]:.3f}  max {mcp_vals[-1]:.3f}")
    print(f"  mean {sum(mcp_vals)/n:.3f}   (the hardcoded constant was 0.950)")
    print(f"\nverdict flips at threshold {thresh}: {flips_up} newly verified, {flips_down} newly unverified")
    print("\nmean MCP by original verdict:")
    for v, vals in sorted(by_verdict.items()):
        print(f"  {v:<11} n={len(vals):<4} mean p(c|x,r) = {sum(vals)/len(vals):.3f}")
    print("\nIf 'logical' sentences show a markedly lower MCP than 'verified' ones, the constant")
    print("was masking exactly the failure mode the term exists to detect.")


if __name__ == "__main__":
    main()
