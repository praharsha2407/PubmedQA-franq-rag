#Purpose: Evaluates the linguistic and semantic quality of answers.
#How it works: Loads generated answers from the pipeline, runs them through sacrebleu (BLEU) and rouge_score (ROUGE-1/2/L), initializes bert_score library (loading roberta-large), computes contextual embedding similarities, and saves the output.

from __future__ import annotations

import argparse
import json

import sacrebleu
from rouge_score import rouge_scorer

from config import OUTPUT_DIR


def load_answer_rows(answers_file: str) -> list[dict[str, object]]:
    path = OUTPUT_DIR / answers_file
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_generation_metrics(predictions: list[str], references: list[str]) -> dict[str, object]:
    """BLEU, ROUGE-1/2/L and BERTScore for a set of prediction/reference pairs.

    BERTScore is optional: it needs roberta-large, so it is skipped with a warning
    (and reported as None) in environments where it cannot be loaded.
    """
    if not predictions:
        raise ValueError("compute_generation_metrics requires at least one prediction")

    bleu_score = sacrebleu.corpus_bleu(predictions, [references]).score
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(reference, prediction) for prediction, reference in zip(predictions, references)]
    rouge_1 = sum(score["rouge1"].fmeasure for score in rouge_scores) / len(rouge_scores)
    rouge_2 = sum(score["rouge2"].fmeasure for score in rouge_scores) / len(rouge_scores)
    rouge_l = sum(score["rougeL"].fmeasure for score in rouge_scores) / len(rouge_scores)

    bertscore_precision = None
    bertscore_recall = None
    bertscore_f1 = None
    try:
        from bert_score import score as bert_score

        precision, recall, f1 = bert_score(predictions, references, lang="en", verbose=True)
        bertscore_precision = float(precision.mean())
        bertscore_recall = float(recall.mean())
        bertscore_f1 = float(f1.mean())
    except Exception as exc:
        print(
            "WARNING: BERTScore could not be computed in this environment. "
            "Run on HPC for final BERTScore. "
            f"Original error: {exc}"
        )

    return {
        "bleu": bleu_score,
        "rouge_1": rouge_1,
        "rouge_2": rouge_2,
        "rouge_l": rouge_l,
        "bertscore_precision": bertscore_precision,
        "bertscore_recall": bertscore_recall,
        "bertscore_f1": bertscore_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--answers-file", default="answers.jsonl",
        help="Filename under outputs/ with generated answers (default: answers.jsonl, the baseline run)",
    )
    parser.add_argument(
        "--output-file", default=None,
        help="Filename under outputs/ to write metrics to (default: generation_metrics.json for the "
        "baseline answers file, generation_metrics_<stem>.json otherwise)",
    )
    args = parser.parse_args()

    rows = load_answer_rows(args.answers_file)
    if not rows:
        raise RuntimeError(f"No generated answers found in {args.answers_file}. Run the pipeline first.")
    predictions = [str(row["generated_answer"]) for row in rows]
    references = [str(row["reference_answer"]) for row in rows]

    result = {
        "answers_file": args.answers_file,
        "sample_size": len(rows),
        **compute_generation_metrics(predictions, references),
    }

    if args.output_file:
        output_name = args.output_file
    elif args.answers_file == "answers.jsonl":
        output_name = "generation_metrics.json"
    else:
        stem = args.answers_file[:-len(".jsonl")] if args.answers_file.endswith(".jsonl") else args.answers_file
        output_name = f"generation_metrics_{stem}.json"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / output_name).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
