"""
Token Tracking for D3 Framework.

Implements cost-aware token tracking as described in Section 5.3 of the D3 paper:
"average tokens per evaluation" is a core metric for cost-accuracy analysis.

Supports tracking for multiple LLM providers (OpenAI, Together, Anthropic, etc.)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import time


@dataclass
class TokenUsage:
    """Token usage for a single API call."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    timestamp: float = field(default_factory=time.time)
    
    @property
    def cost_usd(self) -> float:
        """Estimate cost in USD based on model pricing."""
        return calculate_cost(self.model, self.prompt_tokens, self.completion_tokens)


# Approximate pricing per 1M tokens (as of 2024)
MODEL_PRICING = {
    # OpenAI
    "gpt-4-turbo": {"prompt": 10.0, "completion": 30.0},
    "gpt-4-turbo-preview": {"prompt": 10.0, "completion": 30.0},
    "gpt-4o": {"prompt": 5.0, "completion": 15.0},
    "gpt-3.5-turbo": {"prompt": 0.5, "completion": 1.5},
    
    # Anthropic
    "claude-3-opus-20240229": {"prompt": 15.0, "completion": 75.0},
    "claude-3-sonnet-20240229": {"prompt": 3.0, "completion": 15.0},
    "claude-3-haiku-20240307": {"prompt": 0.25, "completion": 1.25},
    
    # Together AI (approximate)
    "meta-llama/Meta-Llama-3-8B-Instruct-Turbo": {"prompt": 0.2, "completion": 0.2},
    "meta-llama/Llama-3-70b-chat-hf": {"prompt": 0.9, "completion": 0.9},
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": {"prompt": 0.2, "completion": 0.2},
    "mistralai/Mixtral-8x22B-Instruct-v0.1": {"prompt": 1.2, "completion": 1.2},
    "Qwen/Qwen2-72B-Instruct": {"prompt": 0.9, "completion": 0.9},
    "zero-one-ai/Yi-34B-Chat": {"prompt": 0.6, "completion": 0.6},
    "google/gemma-7b-it": {"prompt": 0.15, "completion": 0.15},
    
    # Google
    "gemini-pro": {"prompt": 0.5, "completion": 1.5},
    
    # Default for unknown models
    "default": {"prompt": 1.0, "completion": 1.0}
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate cost in USD for token usage.
    
    Args:
        model: Model identifier
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        
    Returns:
        Cost in USD
    """
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
    
    prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]
    
    return prompt_cost + completion_cost


class TokenTracker:
    """
    Tracks token usage across debate rounds and agents.
    
    Supports the cost-accuracy analysis from Table 2 of the paper:
    - Track total tokens per evaluation
    - Calculate tokens/accuracy ratio
    - Estimate costs
    """
    
    def __init__(self):
        """Initialize the token tracker."""
        self.usage_history: List[TokenUsage] = []
        self.round_totals: Dict[int, int] = {}
        self.agent_totals: Dict[str, int] = {}
        self.current_round: int = 0
        
    def add_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        agent_name: Optional[str] = None,
        round_num: Optional[int] = None
    ):
        """
        Record token usage from an API call.
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            model: Model identifier
            agent_name: Optional name of the agent (Advocate1, Judge, etc.)
            round_num: Optional round number
        """
        total = prompt_tokens + completion_tokens
        
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            model=model
        )
        
        self.usage_history.append(usage)
        
        # Track by round
        round_key = round_num if round_num is not None else self.current_round
        self.round_totals[round_key] = self.round_totals.get(round_key, 0) + total
        
        # Track by agent
        if agent_name:
            self.agent_totals[agent_name] = self.agent_totals.get(agent_name, 0) + total
            
    def set_round(self, round_num: int):
        """Set the current round number."""
        self.current_round = round_num
        
    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return sum(u.total_tokens for u in self.usage_history)
    
    @property
    def total_cost_usd(self) -> float:
        """Get total cost in USD."""
        return sum(u.cost_usd for u in self.usage_history)
    
    @property
    def prompt_tokens(self) -> int:
        """Get total prompt tokens."""
        return sum(u.prompt_tokens for u in self.usage_history)
    
    @property
    def completion_tokens(self) -> int:
        """Get total completion tokens."""
        return sum(u.completion_tokens for u in self.usage_history)
    
    def get_summary(self) -> dict:
        """
        Get a summary of token usage.
        
        Returns:
            Dictionary with usage statistics
        """
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "num_api_calls": len(self.usage_history),
            "tokens_per_round": dict(self.round_totals),
            "tokens_per_agent": dict(self.agent_totals),
            "avg_tokens_per_call": (
                self.total_tokens / len(self.usage_history) 
                if self.usage_history else 0
            )
        }
    
    def reset(self):
        """Reset all tracking data."""
        self.usage_history.clear()
        self.round_totals.clear()
        self.agent_totals.clear()
        self.current_round = 0


# Global tracker instance for convenience
_global_tracker: Optional[TokenTracker] = None


def get_global_tracker() -> TokenTracker:
    """Get or create the global token tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TokenTracker()
    return _global_tracker


def reset_global_tracker():
    """Reset the global token tracker."""
    global _global_tracker
    if _global_tracker:
        _global_tracker.reset()
    else:
        _global_tracker = TokenTracker()


def track_tokens(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    agent_name: Optional[str] = None
):
    """
    Convenience function to track tokens using global tracker.
    
    Args:
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        model: Model identifier
        agent_name: Optional agent name
    """
    get_global_tracker().add_usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model,
        agent_name=agent_name
    )