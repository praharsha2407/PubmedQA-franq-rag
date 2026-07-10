"""
Optional extension to query_expansion.py: real synonym injection using UMLS,
via scispaCy (Neumann et al., 2019) linked to the UMLS Metathesaurus
(Bodenreider, 2004). This goes beyond the hand-typed abbreviation dictionary
in query_expansion.py -- it recognizes synonym pairs that aren't abbreviations
at all (e.g. "heart attack" <-> "myocardial infarction").

SETUP (do this on your own machine/HPC node, not something I can run here):
    pip install scispacy
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz  # optional, better entity recall

NOTE ON LICENSING: UMLS itself is distributed under a license from the US
National Library of Medicine. Using it (even via scispaCy's redistributed
linker) requires you to have accepted the UMLS Metathesaurus License via a
free UTS account (https://uts.nlm.nih.gov/uts/signup-login). This is free
and just takes a short online form -- do this and mention it in your
methodology section, since using UMLS without agreeing to its license terms
is a real compliance issue for a thesis, not just a technicality.

The linker download itself is ~1GB and will be slow the first time it runs.
"""
from __future__ import annotations

_nlp = None
_linker = None
_using_fallback = False


def _load_umls_pipeline():
    """Lazily loads the scispaCy model + UMLS linker (expensive, do once)."""
    global _nlp, _linker, _using_fallback
    if _nlp is not None or _using_fallback:
        return
    try:
        import spacy
        import scispacy  # noqa: F401
        # Importing scispacy alone does NOT register the "scispacy_linker" factory;
        # importing the linking submodule is what runs the @Language.factory decorator.
        from scispacy.linking import EntityLinker  # noqa: F401

        _nlp = spacy.load("en_core_sci_sm")
        _nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        _linker = _nlp.get_pipe("scispacy_linker")
    except Exception as exc:
        _using_fallback = True
        print(
            "WARNING: scispaCy/UMLS pipeline could not be loaded (needs "
            "scispacy, en_core_sci_sm, and the ~1GB UMLS linker download) -- "
            f"UMLS synonym expansion will be skipped for this run. Original error: {exc}"
        )


def expand_query_with_umls_synonyms(
    question: str,
    max_entities: int = 5,
    max_synonyms_per_entity: int = 2,
    min_link_score: float = 0.85,
) -> str:
    """
    Real synonym injection: finds biomedical entities in the question, links
    each to its best-matching UMLS concept, and appends a few of that
    concept's known aliases (synonyms) to the query text.

    min_link_score filters out low-confidence entity links (scispaCy's linker
    score is a string-overlap based similarity, 0-1) -- keep this high to
    avoid injecting wrong/noisy synonyms, which would hurt retrieval rather
    than help it.
    """
    _load_umls_pipeline()
    if _using_fallback:
        return question
    doc = _nlp(question)

    synonyms: list[str] = []
    for ent in doc.ents[:max_entities]:
        if not ent._.kb_ents:
            continue
        best_cui, score = ent._.kb_ents[0]
        if score < min_link_score:
            continue
        concept = _linker.kb.cui_to_entity[best_cui]
        # concept.aliases is the UMLS synonym list for this concept
        for alias in concept.aliases[:max_synonyms_per_entity]:
            if alias.lower() != ent.text.lower():
                synonyms.append(alias)

    if not synonyms:
        return question

    unique_synonyms = list(dict.fromkeys(synonyms))  # de-dupe, keep order
    return f"{question} {' '.join(unique_synonyms)}"


def expand_query_full(question: str, custom_dict: dict | None = None) -> str:
    """
    Combines BOTH expansion strategies:
    1. The existing fast abbreviation dictionary (query_expansion.expand_query)
    2. Real UMLS-based synonym injection (this module)

    Run the cheap dictionary pass first, then the heavier UMLS pass on top.
    """
    from query_expansion import expand_query

    abbreviation_expanded = expand_query(question, custom_dict)
    return expand_query_with_umls_synonyms(abbreviation_expanded)
