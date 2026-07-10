"""
Stage 1 (final version): Synonym + related-term query expansion.

Grounded in: "Ontology-Guided Query Expansion for Biomedical Document
Retrieval using Large Language Models" (BMQExpander), arXiv:2508.11784, 2025.
This paper combines UMLS Metathesaurus knowledge (synonyms/related concepts)
with an LLM's generative ability, specifically for biomedical retrieval --
exactly the "synonym and related term addition" approach requested.

This module combines the two expansion sources already built:
  - umls_synonym_expansion.py  -> SYNONYMS (from the UMLS ontology)
  - llm_query_expansion.py     -> RELATED TERMS (from BioMistral's own
                                    knowledge, via few-shot prompting,
                                    Jagerman et al., 2023)

IMPORTANT HONESTY NOTE: this is a simplified two-source combination, not a
full reimplementation of BMQExpander's exact pipeline. The paper's method
feeds UMLS concept definitions INTO the LLM's prompt so its suggestions are
grounded in that ontology context (a tighter integration than running the
two independently and concatenating results). If you have time later, the
upgrade path is: look up UMLS definitions first, then pass them into the
LLM prompt as grounding context, instead of two separate calls. For now,
this simpler version is a legitimate, citable approximation and is much
faster to get running before your deadline.
"""
from __future__ import annotations

from generation import MistralGenerator
from config import QueryExpansionConfig
from llm_query_expansion import expand_query_with_llm
from umls_synonym_expansion import expand_query_with_umls_synonyms


def expand_query_synonyms_and_related_terms(
    question: str,
    generator: MistralGenerator,
    config: QueryExpansionConfig,
) -> str:
    """
    Combines UMLS synonym injection with LLM-generated related terms into
    one expanded query. Each source is wrapped in its own try/except (both
    umls_synonym_expansion.py and llm_query_expansion.py already degrade
    gracefully on their own), so if one source fails, the other still runs
    rather than the whole expansion silently returning nothing.
    """
    umls_expanded_question = question
    llm_terms_only = ""

    try:
        umls_expanded_question = expand_query_with_umls_synonyms(question)
    except Exception as exc:
        print(f"WARNING: UMLS synonym expansion failed ({exc}); skipping synonyms for this query.")

    try:
        # Run LLM expansion on the ORIGINAL question (not the UMLS-expanded
        # one) to avoid feeding an already-long query back into the LLM
        # prompt, which can dilute the few-shot pattern it's following.
        llm_expanded = expand_query_with_llm(question, generator, config.max_new_tokens)
        # expand_query_with_llm returns "question + terms"; we only want the
        # appended terms here, since the question itself is already present
        # in umls_expanded_question above.
        llm_terms_only = llm_expanded[len(question):].strip()
    except Exception as exc:
        print(f"WARNING: LLM related-term expansion failed ({exc}); skipping related terms for this query.")

    if llm_terms_only:
        return f"{umls_expanded_question} {llm_terms_only}"
    return umls_expanded_question
