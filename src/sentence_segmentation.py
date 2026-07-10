"""
Stage 8: Neural sentence segmentation, replacing NLTK's rule-based tokenizer.

Grounded in: Frohmann, Sterner, Vulic, Minixhofer, Schedl (2024),
"Segment Any Text: A Universal Approach for Robust, Efficient and
Adaptable Sentence Segmentation", arXiv:2406.16678. Package: wtpsplit.

Why this matters for THIS pipeline specifically: NLTK's sentence tokenizer
relies on punctuation + capitalization heuristics. In our earlier worked
example, it split "1. Reasoning: ..." and "2. Conclusion: ..." into FOUR
sentences instead of two (treating "1." and "2." as their own sentences),
which then got scored by the NLI faithfulness model as unsupported
(P(entail) ~0.1-0.3 for bare digit tokens) -- a segmentation artifact, not
a real hallucination. SaT was trained to be robust to exactly this kind of
non-standard punctuation pattern.
"""
from __future__ import annotations

from config import SegmentationConfig

_sat_model = None
_using_fallback = False


def _load_sat(config: SegmentationConfig):
    global _sat_model, _using_fallback
    if _sat_model is not None or _using_fallback:
        return
    try:
        from wtpsplit import SaT

        _sat_model = SaT(config.model_name)
    except Exception as exc:
        _using_fallback = True
        if config.fallback_to_nltk:
            print(
                "WARNING: SaT model could not be loaded -- falling back to NLTK "
                f"sentence tokenization (less robust to formatting artifacts). "
                f"Original error: {exc}"
            )
        else:
            raise


def segment_sentences(text: str, config: SegmentationConfig) -> list[str]:
    _load_sat(config)
    if _sat_model is not None:
        return [s.strip() for s in _sat_model.split(text) if s.strip()]

    # Fallback: the original NLTK-based approach (stage4_verification.py's
    # step_a_sentence_decomposition), kept available so the pipeline never
    # hard-fails if wtpsplit/SaT isn't installed in a given environment.
    from stage4_verification import step_a_sentence_decomposition
    return step_a_sentence_decomposition(text)
