"""
Architecture Builder for D3 Framework.

Provides high-level wrappers for SAMRE and MORE protocols with D3 enhancements:
- Budgeted stopping with convergence detection
- Anonymized transcripts for jury
- Token tracking and cost analysis
"""

from typing import List, Tuple, Dict, Any
from MORE_architecture import more_scores
from SAMRE_architecture import samre_scores 
from util_adv import initiate_model

models = {"opus": "claude-3-opus-20240229", "haiku": "claude-3-haiku-20240307", "sonnet": "claude-3-sonnet-20240229", "llama3_8": "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
          "llama3_70": 'meta-llama/Llama-3-70b-chat-hf', 'llama3.1_8': "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
          "mistral": "mistralai/Mixtral-8x22B-Instruct-v0.1", "Qwen": "Qwen/Qwen2-72B-Instruct",
          "Yi": "zero-one-ai/Yi-34B-Chat", "gemma": "google/gemma-7b-it", "cohere": "command-r-plus", "gemini": 'gemini-pro',
          "gpt-4-turbo": "gpt-4-turbo-preview", "gpt-4o": "gpt-4o", "gpt-3.5-turbo":"gpt-3.5-turbo"}

models_dict = {"llama3_8": "together", "llama3_70": "together", "llama3.1_8": "together",  "mistral": "together", "Qwen": "together", "Yi": "together", "gemma": 'together', "gpt-4-turbo": "openai",
               "gpt-3.5-turbo": "openai", "gpt-4o": "openai", "opus": "claude", "haiku": "claude", "sonnet": "claude"}


def samre_arch(
    model: str, 
    temp: float, 
    question: str, 
    answer1: str, 
    answer2: str, 
    investment: float, 
    n_rounds: int, 
    n_juries: int,
    max_tokens: int = 10000,
    convergence_epsilon: float = 5.0
) -> Tuple[List[str], Tuple[int, int], Dict[str, Any]]:
    """
    Run SAMRE (Single-Advocate Multi-Round Evaluation) with D3 enhancements.
    
    Args:
        model: Model identifier
        temp: Temperature setting
        question: The question being evaluated
        answer1: First answer
        answer2: Second answer
        investment: Cost budget
        n_rounds: Maximum number of debate rounds
        n_juries: Number of jury members
        max_tokens: Token budget for budgeted stopping
        convergence_epsilon: Convergence threshold for budgeted stopping
        
    Returns:
        Tuple of (scores_list, jury_votes, stats_dict)
    """
    initiate_model(model, temp, models_dict[model])
    scores, juries, stats = samre_scores(
        question, answer1, answer2, 
        investment=investment, 
        max_rounds=n_rounds, 
        n_juries=n_juries,
        max_tokens=max_tokens,
        convergence_epsilon=convergence_epsilon
    )
    print("Returned Scores:", scores)
    if scores:
        print("Latest score:", scores[-1])
    print("Stats:", stats)
    return scores, juries, stats


def more_arch(
    model: str, 
    temperature: float, 
    question: str, 
    answer1: str, 
    answer2: str, 
    n_advocates: int, 
    investment: float, 
    n_rounds: int,
    n_juries: int = 5
) -> Tuple[List[str], Tuple[int, int], Dict[str, Any]]:
    """
    Run MORE (Multi-Advocate One-Round Evaluation) with D3 enhancements.
    
    Args:
        model: Model identifier
        temperature: Temperature setting
        question: The question being evaluated
        answer1: First answer
        answer2: Second answer
        n_advocates: Number of advocates per side
        investment: Cost budget
        n_rounds: Number of rounds (typically 1 for MORE)
        n_juries: Number of jury members
        
    Returns:
        Tuple of (scores_list, jury_votes, stats_dict)
    """
    initiate_model(model, temperature, models_dict[model])
    scores, juries, stats = more_scores(
        question, answer1, answer2, 
        investment=investment, 
        n_round=n_rounds, 
        n_advocates=n_advocates,
        n_juries=n_juries
    )
    print("Returned Scores:", scores)
    print("Jury Votes:", juries)
    print("Stats:", stats)
    return scores, juries, stats
