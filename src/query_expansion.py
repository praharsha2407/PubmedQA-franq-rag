import re

# A basic built-in dictionary for common PubMedQA abbreviations
DEFAULT_MEDICAL_ABBREVIATIONS = {
    "cimt": "carotid intima-media thickness",
    "copd": "chronic obstructive pulmonary disease",
    "hba1c": "glycated hemoglobin",
    "bmi": "body mass index",
    "rct": "randomized controlled trial",
    "hiv": "human immunodeficiency virus",
    "aids": "acquired immunodeficiency syndrome",
    "cvd": "cardiovascular disease",
    "cad": "coronary artery disease",
    "bp": "blood pressure",
    "hr": "heart rate",
    "icu": "intensive care unit",
    "mri": "magnetic resonance imaging",
    "ct": "computed tomography",
    "nsaid": "nonsteroidal anti-inflammatory drug",
    "ssri": "selective serotonin reuptake inhibitor",
    "t2dm": "type 2 diabetes mellitus",
    "t1dm": "type 1 diabetes mellitus",
    "cabg": "coronary artery bypass grafting",
    "ecg": "electrocardiogram"
}

def expand_query(question: str, custom_dict: dict = None) -> str:
    """
    Scans tokens in the question against a biomedical abbreviation dictionary.
    Appends full-form expansions to the query string to aid sparse and dense retrieval.

    Args:
        question: The raw user query.
        custom_dict: Optional override or addition to the default abbreviation dictionary.

    Returns:
        The expanded query.
    """
    abbreviations = DEFAULT_MEDICAL_ABBREVIATIONS.copy()
    if custom_dict:
        abbreviations.update(custom_dict)

    # Tokenize by splitting on non-alphanumeric boundaries
    # We want to match exact words, so we can use regex
    tokens = re.findall(r'\b\w+\b', question.lower())

    expansions = []
    for token in tokens:
        if token in abbreviations:
            expansions.append(abbreviations[token])

    if not expansions:
        return question

    # Append the unique expansions to the query
    unique_expansions = list(set(expansions))
    expansion_str = " ".join(unique_expansions)

    return f"{question} {expansion_str}"
