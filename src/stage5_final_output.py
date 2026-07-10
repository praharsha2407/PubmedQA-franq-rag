# ==========================================
# STAGE 5: FINAL OUTPUT
# ==========================================

def build_final_answer(verified_sentences: list[dict]) -> str:
    """
    Assembles the verified sentences into a final answer with inline citations.

    Args:
        verified_sentences: List of dicts, e.g.,
            [{"sentence": "Aspirin reduces risk.", "cited_chunk": 1}, ...]

    Returns:
        The final formatted string.
    """
    final_text_parts = []

    for item in verified_sentences:
        sentence = item["sentence"].strip()
        citation = item.get("cited_chunk", None)

        # Append [Context j] if citation exists
        if citation is not None:
            # Remove trailing period if exists to insert citation properly
            if sentence.endswith("."):
                sentence = sentence[:-1] + f" [Context {citation}]."
            else:
                sentence = sentence + f" [Context {citation}]"

        final_text_parts.append(sentence)

    return " ".join(final_text_parts)

def generate_hallucination_report(all_sentence_results: list[dict]) -> dict:
    """
    Generates a structured hallucination report based on the 3-Way Classifier.

    Args:
        all_sentence_results: List of dicts containing the verdict for each sentence.

    Returns:
        A dictionary report summarizing the error types.
    """
    total = len(all_sentence_results)
    verified = 0
    intrinsic = 0
    extrinsic = 0
    logical = 0

    for res in all_sentence_results:
        verdict = res.get("verdict", "unknown")
        if verdict == "verified":
            verified += 1
        elif verdict == "intrinsic":
            intrinsic += 1
        elif verdict == "extrinsic":
            extrinsic += 1
        elif verdict == "logical":
            logical += 1

    return {
        "total_sentences": total,
        "verified": verified,
        "hallucinations": {
            "intrinsic": intrinsic,
            "extrinsic": extrinsic,
            "logical": logical
        },
        "details": all_sentence_results
    }
