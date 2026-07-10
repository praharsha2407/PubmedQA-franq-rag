"""
Stage 1: LLM-based query expansion.

Grounded in: Jagerman, Zhuang, Qin, Wang, Bendersky (2023),
"Query Expansion by Prompting Large Language Models", arXiv:2305.03653.

Unlike the dictionary-based expand_query() in query_expansion.py, this asks
a language model to generate related biomedical terms for the question,
using its trained knowledge rather than a fixed lookup table. It reuses the
SAME BioMistral model already loaded for Stage 7 generation -- no new model
dependency.

We use a few-shot prompt (Jagerman et al. found few-shot / CoT prompts more
effective than zero-shot for this task).
"""
from __future__ import annotations

from generation import MistralGenerator

_FEW_SHOT_EXPANSION_PROMPT = """You expand short medical research questions with related clinical and biomedical terms, to help a search engine find relevant evidence. Only output the extra terms, comma-separated, nothing else.

Question: Does aspirin reduce the risk of stroke in diabetic patients?
Terms: acetylsalicylic acid, cerebrovascular accident, diabetes mellitus, antiplatelet therapy, cardiovascular risk

Question: Is metformin associated with vitamin B12 deficiency?
Terms: biguanide, cobalamin deficiency, type 2 diabetes, malabsorption, peripheral neuropathy

Question: {question}
Terms:"""


def expand_query_with_llm(question: str, generator: MistralGenerator, max_new_tokens: int = 48) -> str:
    """
    Returns the original question with LLM-generated related terms appended.
    Falls back to returning the original question unchanged if generation
    fails or produces something unusable (e.g. empty output) -- expansion
    should never be allowed to break retrieval.
    """
    prompt = _FEW_SHOT_EXPANSION_PROMPT.format(question=question.strip())
    try:
        raw_terms = generator.generate(prompt, max_new_tokens=max_new_tokens)
    except Exception as exc:
        print(f"WARNING: LLM query expansion failed ({exc}); using original question only.")
        return question

    # Keep only the first line -- models sometimes continue with a new
    # "Question:" turn of their own, which we don't want appended.
    first_line = raw_terms.strip().splitlines()[0] if raw_terms.strip() else ""
    terms = [t.strip() for t in first_line.split(",") if t.strip()]

    if not terms:
        return question

    return f"{question} {' '.join(terms)}"
