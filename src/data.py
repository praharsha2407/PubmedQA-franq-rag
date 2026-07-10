#Purpose: Handles loading and structuring of raw biomedical data.
#Core Classes & Functions:
#PubMedQAExample: Dataclass that structures each sample's ID, question, context list, human long answer, and yes/no/maybe decision.
#load_pubmedqa(): Downloads the dataset from Hugging Face, slices it to a sample size, and returns a list of parsed examples.
#build_context_corpus(): Deduplicates context abstracts across all queries, preparing the unique text snippets and metadata (PubID, rank) to build the retrieval index.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_dataset

from config import DatasetConfig


@dataclass(frozen=True)
class PubMedQAExample:
    pubid: str
    question: str
    contexts: list[str]
    long_answer: str
    final_decision: str


def _extract_contexts(context_field: Any) -> list[str]:
    if isinstance(context_field, dict):
        contexts = context_field.get("contexts", [])
    else:
        contexts = context_field
    return [str(context).strip() for context in contexts if str(context).strip()]


def load_pubmedqa(config: DatasetConfig, sample_size: int | None = None) -> list[PubMedQAExample]:
    dataset: Dataset = load_dataset(config.name, config.subset, split=config.split)
    if sample_size is not None:
        dataset = dataset.select(range(min(sample_size, len(dataset))))

    examples: list[PubMedQAExample] = []
    for row in dataset:
        examples.append(
            PubMedQAExample(
                pubid=str(row["pubid"]),
                question=str(row["question"]).strip(),
                contexts=_extract_contexts(row["context"]),
                long_answer=str(row["long_answer"]).strip(),
                final_decision=str(row["final_decision"]).strip(),
            )
        )
    return examples


def build_context_corpus(examples: list[PubMedQAExample]) -> tuple[list[str], list[dict[str, str]]]:
    texts: list[str] = []
    metadata: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for example in examples:
        for index, context in enumerate(example.contexts):
            key = (example.pubid, context)
            if key in seen:
                continue
            seen.add(key)
            texts.append(context)
            metadata.append(
                {
                    "pubid": example.pubid,
                    "chunk_id": f"{example.pubid}:{index}",
                    "question": example.question,
                }
            )
    return texts, metadata
