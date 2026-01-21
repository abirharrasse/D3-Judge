"""
Evaluation Metrics for D3 Framework.

Implements the metrics from Section 4.3 of the D3 paper:
- Accuracy: Agreement with human judgments
- Cohen's Kappa: Agreement corrected for chance
- Positional Swap Consistency: Robustness to answer order
- Self-Enhancement Rate: Preference for own model family
"""

from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
import random
import statistics
from collections import Counter


@dataclass
class EvaluationResult:
    """Result of a single evaluation."""
    question_id: str
    human_label: int  # 0 or 1 (which answer is better)
    model_label: int  # 0 or 1
    model_scores: Tuple[float, float]  # (score1, score2)
    answer1_model: Optional[str] = None  # Model that generated answer1
    answer2_model: Optional[str] = None  # Model that generated answer2
    evaluator_model: Optional[str] = None  # Model used for evaluation


@dataclass
class BiasAuditResults:
    """Results from bias auditing."""
    positional_swap_consistency: float
    self_enhancement_rate: float
    swap_test_samples: int
    self_enhancement_samples: int
    details: Dict = field(default_factory=dict)


def calculate_accuracy(
    human_labels: List[int], 
    model_labels: List[int]
) -> float:
    """
    Calculate simple agreement accuracy.
    
    Args:
        human_labels: List of human preference labels (0 or 1)
        model_labels: List of model preference labels (0 or 1)
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    if len(human_labels) != len(model_labels):
        raise ValueError("Lists must have same length")
    if not human_labels:
        return 0.0
    
    correct = sum(h == m for h, m in zip(human_labels, model_labels))
    return correct / len(human_labels)


def calculate_cohens_kappa(
    human_labels: List[int], 
    model_labels: List[int]
) -> float:
    """
    Calculate Cohen's Kappa coefficient for inter-rater agreement.
    
    Per the paper: "Cohen's Kappa (κ)" corrects for chance agreement on 
    skewed distributions.
    
    Args:
        human_labels: List of human preference labels (0 or 1)
        model_labels: List of model preference labels (0 or 1)
        
    Returns:
        Cohen's Kappa coefficient (-1 to 1)
    """
    if len(human_labels) != len(model_labels):
        raise ValueError("Lists must have same length")
    if not human_labels:
        return 0.0
    
    n = len(human_labels)
    
    # Build confusion matrix
    # True positives (both say 1)
    tp = sum(1 for h, m in zip(human_labels, model_labels) if h == 1 and m == 1)
    # True negatives (both say 0)
    tn = sum(1 for h, m in zip(human_labels, model_labels) if h == 0 and m == 0)
    # False positives (human says 0, model says 1)
    fp = sum(1 for h, m in zip(human_labels, model_labels) if h == 0 and m == 1)
    # False negatives (human says 1, model says 0)
    fn = sum(1 for h, m in zip(human_labels, model_labels) if h == 1 and m == 0)
    
    # Observed agreement
    po = (tp + tn) / n
    
    # Expected agreement by chance
    human_pos = (tp + fn) / n  # P(human = 1)
    human_neg = (tn + fp) / n  # P(human = 0)
    model_pos = (tp + fp) / n  # P(model = 1)
    model_neg = (tn + fn) / n  # P(model = 0)
    
    pe = (human_pos * model_pos) + (human_neg * model_neg)
    
    # Cohen's Kappa
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    
    kappa = (po - pe) / (1 - pe)
    return kappa


def calculate_positional_swap_consistency(
    evaluate_fn: Callable,
    questions: List[str],
    answers1: List[str],
    answers2: List[str],
    sample_size: Optional[int] = None
) -> Tuple[float, Dict]:
    """
    Calculate positional swap consistency.
    
    Per the paper (Section 4.3):
    "Positional Swap Consistency: each evaluation performed twice with answers 
    in order (A, B) and (B, A), measuring consistency of verdicts"
    
    Args:
        evaluate_fn: Function that takes (question, answer1, answer2) and returns label
        questions: List of questions
        answers1: List of first answers
        answers2: List of second answers
        sample_size: Number of samples to test (None = all)
        
    Returns:
        Tuple of (consistency_rate, details_dict)
    """
    if sample_size is None:
        sample_size = len(questions)
    else:
        sample_size = min(sample_size, len(questions))
    
    indices = random.sample(range(len(questions)), sample_size)
    
    consistent = 0
    inconsistent_cases = []
    
    for idx in indices:
        q = questions[idx]
        a1 = answers1[idx]
        a2 = answers2[idx]
        
        # Normal order
        label_normal = evaluate_fn(q, a1, a2)
        
        # Swapped order
        label_swapped = evaluate_fn(q, a2, a1)
        
        # Check consistency (labels should be opposite when swapped)
        # If normal says 1 (answer1 better), swapped should say 0 (answer1 better, now in position 2)
        is_consistent = (label_normal == 1 and label_swapped == 0) or \
                       (label_normal == 0 and label_swapped == 1)
        
        if is_consistent:
            consistent += 1
        else:
            inconsistent_cases.append({
                "question_idx": int(idx),
                "label_normal": label_normal,
                "label_swapped": label_swapped
            })
    
    consistency_rate = consistent / sample_size if sample_size > 0 else 0.0
    
    details = {
        "sample_size": sample_size,
        "consistent_count": consistent,
        "inconsistent_count": len(inconsistent_cases),
        "inconsistent_cases": inconsistent_cases[:10]  # Sample of inconsistencies
    }
    
    return consistency_rate, details


def calculate_self_enhancement_rate(
    results: List[EvaluationResult],
    evaluator_model_family: str
) -> Tuple[float, Dict]:
    """
    Calculate self-enhancement rate.
    
    Per the paper (Section 4.3):
    "Self-Enhancement Rate: percentage of cases where evaluator prefers its 
    own model family despite human labels indicating otherwise"
    
    Args:
        results: List of evaluation results with model provenance
        evaluator_model_family: Model family of the evaluator (e.g., "gpt", "claude")
        
    Returns:
        Tuple of (self_enhancement_rate, details_dict)
    """
    def get_model_family(model_name: str) -> str:
        """Extract model family from model name."""
        if not model_name:
            return "unknown"
        model_lower = model_name.lower()
        if "gpt" in model_lower or "openai" in model_lower:
            return "openai"
        elif "claude" in model_lower or "anthropic" in model_lower:
            return "anthropic"
        elif "llama" in model_lower or "meta" in model_lower:
            return "meta"
        elif "mistral" in model_lower:
            return "mistral"
        elif "gemini" in model_lower or "google" in model_lower:
            return "google"
        else:
            return model_lower.split("-")[0] if "-" in model_lower else model_lower[:5]
    
    evaluator_family = get_model_family(evaluator_model_family)
    
    # Find cases where one answer is from evaluator's model family
    relevant_cases = []
    self_enhanced = 0
    
    for result in results:
        a1_family = get_model_family(result.answer1_model or "")
        a2_family = get_model_family(result.answer2_model or "")
        
        # Check if one answer is from evaluator's family
        if a1_family == evaluator_family and a2_family != evaluator_family:
            relevant_cases.append(result)
            # Self-enhancement: model prefers own family when human doesn't
            if result.model_label == 0 and result.human_label == 1:
                self_enhanced += 1
        elif a2_family == evaluator_family and a1_family != evaluator_family:
            relevant_cases.append(result)
            # Self-enhancement: model prefers own family when human doesn't
            if result.model_label == 1 and result.human_label == 0:
                self_enhanced += 1
    
    rate = self_enhanced / len(relevant_cases) if relevant_cases else 0.0
    
    details = {
        "evaluator_family": evaluator_family,
        "relevant_cases": len(relevant_cases),
        "self_enhanced_count": self_enhanced,
        "rate": rate
    }
    
    return rate, details


def run_full_evaluation(
    human_labels: List[int],
    model_labels: List[int],
    model_scores: Optional[List[Tuple[float, float]]] = None
) -> Dict:
    """
    Run complete evaluation metrics suite.
    
    Args:
        human_labels: List of human preference labels
        model_labels: List of model preference labels
        model_scores: Optional list of score tuples
        
    Returns:
        Dictionary with all metrics
    """
    accuracy = calculate_accuracy(human_labels, model_labels)
    kappa = calculate_cohens_kappa(human_labels, model_labels)
    
    # Calculate additional statistics
    n = len(human_labels)
    human_rate_1 = sum(human_labels) / n if n > 0 else 0
    model_rate_1 = sum(model_labels) / n if n > 0 else 0
    
    results = {
        "accuracy": round(accuracy, 4),
        "cohens_kappa": round(kappa, 4),
        "sample_size": n,
        "human_preference_rate": round(human_rate_1, 4),
        "model_preference_rate": round(model_rate_1, 4),
    }
    
    # Score analysis if available
    if model_scores:
        score_gaps = [abs(s[0] - s[1]) for s in model_scores]
        results["avg_score_gap"] = round(statistics.mean(score_gaps), 2)
        results["score_gap_std"] = round(statistics.stdev(score_gaps) if len(score_gaps) > 1 else 0.0, 2)
    
    return results


def interpret_kappa(kappa: float) -> str:
    """
    Interpret Cohen's Kappa value.
    
    Standard interpretation scale:
    - < 0: Less than chance agreement
    - 0.00-0.20: Slight agreement
    - 0.21-0.40: Fair agreement
    - 0.41-0.60: Moderate agreement
    - 0.61-0.80: Substantial agreement
    - 0.81-1.00: Almost perfect agreement
    """
    if kappa < 0:
        return "Less than chance agreement"
    elif kappa < 0.20:
        return "Slight agreement"
    elif kappa < 0.40:
        return "Fair agreement"
    elif kappa < 0.60:
        return "Moderate agreement"
    elif kappa < 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"


# Example usage and tests
if __name__ == "__main__":
    # Test data
    human = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1]
    model = [1, 1, 0, 0, 0, 1, 1, 1, 0, 1]
    
    print("=== Metrics Test ===")
    accuracy = calculate_accuracy(human, model)
    print(f"Accuracy: {accuracy:.2%}")
    
    kappa = calculate_cohens_kappa(human, model)
    print(f"Cohen's Kappa: {kappa:.4f} ({interpret_kappa(kappa)})")
    
    results = run_full_evaluation(human, model)
    print(f"Full Results: {results}")