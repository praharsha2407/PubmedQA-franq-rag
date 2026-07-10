#Purpose: Houses prompt templates and structures retrieved contexts.
#Core Functions:
#format_contexts(): Formats retrieved chunks with indicators for rank, score, and PubMed IDs, creating a readable block for the LLM.
#build_cot_rag_prompt(): Dynamically inserts the question and formatted contexts into COT_RAG_PROMPT_TEMPLATE to enforce step-by-step reasoning.

from __future__ import annotations

from retrieval import RetrievedChunk


COT_RAG_PROMPT_TEMPLATE = """You are a biomedical question-answering assistant. Your task is to answer the question using ONLY the retrieved context provided below. 

Retrieved Context:
{retrieved_context}

Question:
{question}

Instructions:
- Base your entire response ONLY on the retrieved evidence provided above.
- Do NOT use any external biomedical knowledge, personal opinions, or unsupported facts.
- Every claim in your reasoning must be directly and explicitly supported by the text in the Retrieved Context.
- Reason step by step from the biomedical findings in the retrieved context.
- Identify the key entities, intervention/exposure, outcome, and study conclusion when available.
- If the evidence is insufficient or conflicting to answer the question, state that clearly.

Final Answer:
Provide a concise evidence-grounded answer containing ONLY:
1. Reasoning: a short step-by-step explanation grounded strictly and exclusively in the retrieved context.
2. Conclusion: yes, no, or maybe, followed by one sentence directly supported by the context.
"""


def format_contexts(chunks: list[RetrievedChunk]) -> str:
    formatted = []
    for rank, chunk in enumerate(chunks, start=1):
        pubid = chunk.metadata.get("pubid", "unknown")
        chunk_id = chunk.metadata.get("chunk_id", f"rank-{rank}")
        formatted.append(
            f"[Context {rank} | pubid={pubid} | chunk_id={chunk_id} | score={chunk.score:.4f}]\n"
            f"{chunk.text}"
        )
    return "\n\n".join(formatted)


def build_cot_rag_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return COT_RAG_PROMPT_TEMPLATE.format(
        retrieved_context=format_contexts(chunks),
        question=question,
    )
