from retrieval import RetrievedChunk

MINIMAL_PROMPT_TEMPLATE = """Context:
{context}

Question:
{question}

Answer:"""

def build_minimal_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """
    Builds a minimal, instruction-free prompt.
    Unlike the baseline CoT prompt, this prompt contains ONLY the raw context
    chunks and the question. The LLM is completely unconstrained and free
    to generate its answer without prompt-engineered self-reflection rules.
    """
    context_text = ""
    for i, chunk in enumerate(chunks):
        # Format the context chunks exactly as they were retrieved
        context_text += f"[Context {i+1}]: {chunk.text}\n"

    return MINIMAL_PROMPT_TEMPLATE.format(context=context_text.strip(), question=question)
