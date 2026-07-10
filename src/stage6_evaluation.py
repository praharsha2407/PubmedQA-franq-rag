# ==========================================
# STAGE 6: EVALUATION
# ==========================================
import json

class AdvancedEvaluator:
    def __init__(self):
        self.all_reports = []
        self.all_franq_scores = []
        
    def add_result(self, hallucination_report: dict, average_franq_score: float):
        """Accumulates results from a single generated answer."""
        self.all_reports.append(hallucination_report)
        self.all_franq_scores.append(average_franq_score)
        
    def calculate_metrics(self) -> dict:
        """
        Calculates the 3 NEW metrics defined in the Stage 6 flowchart.
        (Baseline metrics like BLEU/ROUGE are handled by existing scripts).
        """
        if not self.all_reports:
            return {}
            
        total_sentences = sum(report["total_sentences"] for report in self.all_reports)
        total_verified = sum(report["verified"] for report in self.all_reports)
        
        total_intrinsic = sum(report["hallucinations"]["intrinsic"] for report in self.all_reports)
        total_extrinsic = sum(report["hallucinations"]["extrinsic"] for report in self.all_reports)
        total_logical = sum(report["hallucinations"]["logical"] for report in self.all_reports)
        
        # 1. Keyword Faithfulness Score (% of sentences verified against context)
        keyword_faithfulness_score = total_verified / total_sentences if total_sentences > 0 else 0.0
        
        # 2. Hallucination Rate by Type
        hallucination_rates = {
            "intrinsic_rate": total_intrinsic / total_sentences if total_sentences > 0 else 0.0,
            "extrinsic_rate": total_extrinsic / total_sentences if total_sentences > 0 else 0.0,
            "logical_rate": total_logical / total_sentences if total_sentences > 0 else 0.0,
            "overall_hallucination_rate": (total_intrinsic + total_extrinsic + total_logical) / total_sentences if total_sentences > 0 else 0.0
        }
        
        # 3. FRANQ Factuality Score (Average of all calibrated probabilities)
        franq_factuality_score = sum(self.all_franq_scores) / len(self.all_franq_scores) if self.all_franq_scores else 0.0
        
        return {
            "NEW_Keyword_Faithfulness_Score": keyword_faithfulness_score,
            "NEW_Hallucination_Rate_By_Type": hallucination_rates,
            "NEW_FRANQ_Factuality_Score": franq_factuality_score
        }

    def save_report(self, filepath: str):
        metrics = self.calculate_metrics()
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Advanced evaluation metrics saved to {filepath}")
