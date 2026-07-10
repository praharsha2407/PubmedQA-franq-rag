#Purpose: Main execution entrypoint for running the RAG pipeline.
#How it works: Initialized with command-line arguments (e.g. --sample-size, --top-k). Loads data, builds the retriever corpus, constructs the FAISS index, initializes Mistral, runs retrieval and generation loops for each sample, and appends outputs to outputs/answers.jsonl. Features a progress-resuming check.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from config import OUTPUT_DIR, PipelineConfig, GenerationConfig, RetrievalConfig
from data import build_context_corpus, load_pubmedqa
from generation import MistralGenerator
from prompts import build_cot_rag_prompt
from retrieval import DenseRetriever


def _existing_pubids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    pubids: set[str] = set()
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            pubids.add(str(json.loads(line)["pubid"]))
        except Exception:
            # Skip malformed lines to make resume robust
            print(f"Warning: skipping malformed JSON line {i} in {path}", flush=True)
            continue
    return pubids


def run(sample_size: int | None, top_k: int, model_name: str, load_in_4bit: bool, resume: bool, start_index: int | None = None, end_index: int | None = None, output_file: str | None = None) -> None:
    config = PipelineConfig(
        retrieval=RetrievalConfig(top_k=top_k),
        generation=GenerationConfig(model_name=model_name, load_in_4bit=load_in_4bit),
    )
    examples = load_pubmedqa(config.dataset, sample_size=sample_size)
    # apply optional slicing for chunked runs
    if start_index is not None or end_index is not None:
        s = start_index or 0
        e = end_index or None
        examples = examples[s:e]
    corpus, metadata = build_context_corpus(examples)
    if not corpus:
        total_contexts = sum(len(e.contexts) for e in examples)
        raise RuntimeError(
            f"Retrieval corpus is empty after building contexts. "
            f"examples_count={len(examples)}, total_contexts_in_examples={total_contexts}"
        )

    retriever = DenseRetriever(config.retrieval)
    retriever.build(corpus, metadata)
    generator = MistralGenerator(config.generation)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_name = output_file or "answers.jsonl"
    output_path = OUTPUT_DIR / output_name

    print(f"[INFO] Writing results to: {output_path}", flush=True)

    completed = _existing_pubids(output_path) if resume else set()
    mode = "a" if (resume and output_path.exists()) else "w"

    with output_path.open(mode, encoding="utf-8") as file:
        for example in tqdm(examples, desc="Generating answers"):
            if example.pubid in completed:
                continue
            chunks = retriever.search(example.question, top_k=top_k)
            prompt = build_cot_rag_prompt(example.question, chunks)
            answer = generator.generate(prompt)
            if len(completed) % 10 == 0:
                print(f"[INFO] Completed {len(completed)} examples", flush=True)
            row = {
                "pubid": example.pubid,
                "question": example.question,
                "reference_answer": example.long_answer,
                "final_decision": example.final_decision,
                "retrieved_contexts": [
                    {
                        "text": chunk.text,
                        "score": chunk.score,
                        "metadata": chunk.metadata,
                    }
                    for chunk in chunks
                ],
                "prompt": prompt,
                "generated_answer": answer,
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--full-dataset", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-name", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--output-file", type=str, default=None,
                        help="Filename under outputs/ to write results (default answers.jsonl)")
    args = parser.parse_args()
    sample_size = None if args.full_dataset else args.sample_size
    run(sample_size, args.top_k, args.model_name, args.load_in_4bit, resume=not args.no_resume, start_index=args.start_index, end_index=args.end_index, output_file=args.output_file)


if __name__ == "__main__":
    main()