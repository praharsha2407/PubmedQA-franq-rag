import nltk
from nltk.tokenize import sent_tokenize
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

# Ensure NLTK punkt is available for sentence splitting
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# ==========================================
# STAGE 4: MODEL-BASED VERIFICATION
# ==========================================

# ------------------------------------------
# Step A: Sentence Decomposition (NLTK)
# ------------------------------------------
def step_a_sentence_decomposition(raw_answer: str) -> list[str]:
    """Splits the raw generated answer into atomic sentences."""
    return sent_tokenize(raw_answer)

# ------------------------------------------
# Step B: Keyword Overlap (ALGORITHM)
# ------------------------------------------
def step_b_keyword_overlap(sentence_keywords: list[str], chunk_keywords: list[str]) -> float:
    """Computes |K_sent ∩ K_ctx| / |K_sent|"""
    if not sentence_keywords:
        return 0.0
    intersection = set(sentence_keywords).intersection(set(chunk_keywords))
    return len(intersection) / len(sentence_keywords)

# ------------------------------------------
# Step C: AlignScore MODEL (from FRANQ)
#
# NOTE: not currently called by run_advanced_pipeline.py. Your architecture
# diagram defines faithfulness (P(faithful)) via NLI (StepD_NLI below) instead,
# which is a valid, simpler design choice, different from the original FRANQ
# paper's use of AlignScore for this step. This class is left here in case you
# want to swap it in later for closer alignment with the published method
# (Zha et al., 2023, ACL) -- see AlignScore's official checkpoint/repo:
# https://github.com/yuh-zha/AlignScore
# ------------------------------------------
class StepC_AlignScore:
    def __init__(self, model_path="roberta-large"):
        # Note: In a real environment, you would load the official AlignScore model.
        # This acts as a wrapper for the factual alignment model.
        # Placeholder for actual AlignScore initialization
        pass

    def score(self, premise: str, hypothesis: str) -> float:
        """
        Scores structural alignment (faithfulness).
        Returns P(faithful) in [0, 1].
        """
        # TODO: Replace with actual AlignScore forward pass
        # For now, returning a mock probability so the pipeline runs
        return 0.85

# ------------------------------------------
# Step D: NLI MODEL (BART-Large-MNLI)
# ------------------------------------------
class StepD_NLI:
    def __init__(self, model_name="facebook/bart-large-mnli", device=None):
        # Auto-detect CUDA if no device is explicitly given -- the previous
        # default of "cpu" meant this ran on CPU for every sentence of every
        # example even on a GPU node, dominating runtime.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def classify_and_score(self, premise: str, hypothesis: str) -> tuple[str, float]:
        """Classifies entailment / contradiction / neutral and returns entailment probability."""
        inputs = self.tokenizer(f"{premise} </s></s> {hypothesis}", return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)
        entailment_prob = probs[0, 2].item() # 2 is entailment in bart-large-mnli

        # BART-MNLI labels: 0: contradiction, 1: neutral, 2: entailment
        predicted_class_id = logits.argmax().item()
        mapping = {0: "contradiction", 1: "neutral", 2: "entailment"}
        return mapping[predicted_class_id], entailment_prob

# ------------------------------------------
# Step E: Parametric Knowledge (Mistral)
# ------------------------------------------
class StepE_ParametricKnowledge:
    def __init__(self, generator_model, generator_tokenizer):
        """Reuses the Mistral-7B model from generation."""
        self.model = generator_model
        self.tokenizer = generator_tokenizer

    def compute_without_context(self, question: str, generated_sentence: str) -> float:
        """
        Re-runs Mistral forward pass WITHOUT context to read token logits.
        Returns P(true | unfaithful) = average token probability.
        """
        if self.model is None or self.tokenizer is None:
            # Fallback if Mistral failed to load (e.g., testing on CPU)
            return 0.5

        prompt = f"Question:\n{question}\n\nAnswer:\n"
        # Tokenize prompt and generated sentence separately
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        sentence_ids = self.tokenizer(generated_sentence, return_tensors="pt", add_special_tokens=False).input_ids

        # We concatenate them to get the logits for the sentence part
        input_ids = torch.cat([prompt_ids, sentence_ids], dim=-1).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        # Shift logits to align with tokens (predicting next token)
        shift_logits = logits[0, prompt_ids.shape[1]-1:-1, :]
        shift_labels = sentence_ids[0]

        probs = torch.softmax(shift_logits, dim=-1)
        token_probs = [probs[i, token_id].item() for i, token_id in enumerate(shift_labels)]

        # Average token probability
        return sum(token_probs) / len(token_probs) if token_probs else 0.0

# ------------------------------------------
# Step F: FRANQ Formula (MATH)
# ------------------------------------------
def step_f_franq_formula(p_faithful: float, p_true_given_faithful: float, p_true_given_unfaithful: float) -> float:
    """
    P(true) = P(faithful) × P(true|faithful) + P(unfaithful) × P(true|unfaithful)
    """
    p_unfaithful = 1.0 - p_faithful
    p_true = (p_faithful * p_true_given_faithful) + (p_unfaithful * p_true_given_unfaithful)
    return p_true

# NOTE: Step G (isotonic regression calibration) was removed. Calibration is a
# supervised step that needs hand-labeled (raw_score, true/false) pairs to fit
# a meaningful curve, and none were available. We report the raw FRANQ score
# directly, matching the "FRANQ no calibration" variant reported as a
# legitimate baseline in the original paper this architecture is based on.
