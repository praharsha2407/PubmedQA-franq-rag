from __future__ import annotations

import json
import re
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import OUTPUT_DIR


def split_sentences(text: str) -> list[str]:
    # Split text by period/question/exclamation followed by space and capital letter or end of line
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return [s.strip() for s in sentences if s.strip()]


def main() -> None:
    # 1. Load the answers from answers.jsonl
    path = OUTPUT_DIR / "answers.jsonl"
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise RuntimeError("No generated answers found. Run src/run_pipeline.py first.")
    
    print(f"Loaded {len(rows)} samples. Initializing evaluation models...", flush=True)
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}", flush=True)
    
    # 2. Initialize NLI Model (facebook/bart-large-mnli)
    print("Loading NLI model (facebook/bart-large-mnli) for Faithfulness and Recall...", flush=True)
    nli_name = "facebook/bart-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(nli_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_name).to(device)
    
    # 3. Initialize Embeddings (MiniLM-L6-v2) for Relevancy and Precision
    print("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2) for Relevancy and Precision...", flush=True)
    emb_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': str(device)}
    )
    
    # Entailment check helper (uses BART-MNLI logits)
    def check_entailment(premise: str, hypothesis: str) -> bool:
        # BART classes: 0: contradiction, 1: neutral, 2: entailment
        inputs = tokenizer.encode(premise, hypothesis, return_tensors='pt', truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            logits = nli_model(inputs).logits
        probs = logits.softmax(dim=1)
        # Return True if Entailment probability > 0.5
        return bool(probs[0][2].item() > 0.5)

    # Cosine similarity helper
    def cosine_similarity(v1: list[float], v2: list[float]) -> float:
        v1_arr = np.array(v1)
        v2_arr = np.array(v2)
        dot = np.dot(v1_arr, v2_arr)
        norm_v1 = np.linalg.norm(v1_arr)
        norm_v2 = np.linalg.norm(v2_arr)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot / (norm_v1 * norm_v2))

    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []
    recall_scores = []
    
    print("Starting evaluation loop...", flush=True)
    # Loop over all samples (runs very fast, so we can evaluate the full dataset!)
    for row in tqdm(rows):
        question = str(row["question"])
        answer = str(row["generated_answer"])
        ground_truth = str(row["reference_answer"])
        contexts = [str(ctx["text"]) for ctx in row["retrieved_contexts"]]
        combined_context = "\n".join(contexts)
        
        # A. Faithfulness
        ans_sentences = split_sentences(answer)
        if not ans_sentences:
            f_score = 1.0
        else:
            f_entailed = sum(1 for sent in ans_sentences if check_entailment(combined_context, sent))
            f_score = f_entailed / len(ans_sentences)
        faithfulness_scores.append(f_score)
        
        # B. Context Recall
        gt_sentences = split_sentences(ground_truth)
        if not gt_sentences:
            r_score = 1.0
        else:
            r_entailed = sum(1 for sent in gt_sentences if check_entailment(combined_context, sent))
            r_score = r_entailed / len(gt_sentences)
        recall_scores.append(r_score)
        
        # C. Answer Relevancy
        q_emb = emb_model.embed_query(question)
        a_emb = emb_model.embed_query(answer)
        relevancy_scores.append(cosine_similarity(q_emb, a_emb))
        
        # D. Context Precision
        if not contexts:
            p_score = 0.0
        else:
            c_similarities = []
            for ctx in contexts:
                ctx_emb = emb_model.embed_query(ctx)
                c_similarities.append(cosine_similarity(q_emb, ctx_emb))
            p_score = float(np.mean(c_similarities))
        precision_scores.append(p_score)

    output = {
        "faithfulness": float(np.mean(faithfulness_scores)),
        "answer_relevancy": float(np.mean(relevancy_scores)),
        "context_precision": float(np.mean(precision_scores)),
        "context_recall": float(np.mean(recall_scores)),
    }
    
    (OUTPUT_DIR / "ragas_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\n=== Evaluation Results ===")
    print(json.dumps(output, indent=2))
    print(f"Metrics successfully saved to {OUTPUT_DIR / 'ragas_metrics.json'}")


if __name__ == "__main__":
    main()