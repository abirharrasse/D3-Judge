"""
Convergence Detection and Budgeted Stopping for D3 Framework.

Implements the budgeted stopping rule from Section 2.2 of the D3 paper:
- Convergence detection: stops when score differences stabilize
- Token budget tracking: stops when cost exceeds threshold
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BudgetConfig:
    """Configuration for budgeted stopping."""
    max_rounds: int = 5
    max_tokens: int = 10000
    convergence_epsilon: float = 3.0  # Score gap stability threshold (tightened)
    convergence_window: int = 3  # Number of rounds to check for stability
    min_convergence_rounds: int = 3  # Minimum rounds before convergence can trigger


def check_convergence(
    scores: List[Tuple[int, int]], 
    epsilon: float = 3.0,
    window: int = 3,
    min_rounds: int = 3
) -> bool:
    """
    Check if the debate has converged based on score gap stability.
    
    Per the paper: "The iterative debate terminates automatically if the 
    debate has converged (e.g., the score difference remains stable)"
    
    Convergence requires:
    1. Minimum number of rounds completed
    2. Gap variance within window is below epsilon
    3. Gap is NOT monotonically increasing/decreasing (trend check)
    
    Args:
        scores: List of (score1, score2) tuples from each round
        epsilon: Maximum allowed variance in score gaps for convergence
        window: Number of recent rounds to consider (default 3)
        min_rounds: Minimum rounds before convergence can trigger
        
    Returns:
        True if converged, False otherwise
    """
    # Require minimum rounds before checking convergence
    if len(scores) < max(window, min_rounds):
        return False
    
    # Calculate score gaps for recent rounds
    recent_gaps = [abs(s[0] - s[1]) for s in scores[-window:]]
    
    # Check if gaps have stabilized (variance below epsilon)
    gap_variance = max(recent_gaps) - min(recent_gaps)
    if gap_variance >= epsilon:
        return False
    
    # Trend detection: reject if gap is STRICTLY monotonically increasing or decreasing
    # This prevents stopping when the gap is still actively growing
    # Using strict comparison (< >) to allow plateaus with small oscillations
    is_monotonic_increasing = all(
        recent_gaps[i] < recent_gaps[i + 1] 
        for i in range(len(recent_gaps) - 1)
    )
    is_monotonic_decreasing = all(
        recent_gaps[i] > recent_gaps[i + 1] 
        for i in range(len(recent_gaps) - 1)
    )
    
    # If all gaps are identical, that's true convergence (not monotonic)
    all_identical = len(set(recent_gaps)) == 1
    
    # Converged only if variance is low AND not showing a clear trend
    # (unless the gaps are all identical, which is perfect convergence)
    if all_identical:
        return True
    
    if is_monotonic_increasing or is_monotonic_decreasing:
        return False
    
    return True


def check_budget_exceeded(
    current_tokens: int,
    max_tokens: int
) -> bool:
    """
    Check if the token budget has been exceeded.
    
    Per the paper: "if a user-defined token or round budget is exceeded"
    
    Args:
        current_tokens: Current total token consumption
        max_tokens: Maximum allowed tokens
        
    Returns:
        True if budget exceeded, False otherwise
    """
    return current_tokens > max_tokens


class BudgetedStoppingController:
    """
    Controller for managing budgeted stopping in SAMRE debates.
    
    Implements Algorithm 2 from the paper:
    ```
    If CheckConvergence(S, ε) or TokenCost(Tr) > B:
        break
    ```
    """
    
    def __init__(self, config: Optional[BudgetConfig] = None):
        """Initialize with budget configuration."""
        self.config = config or BudgetConfig()
        self.scores: List[Tuple[int, int]] = []
        self.total_tokens: int = 0
        self.rounds_completed: int = 0
        
    def record_round(self, score: Tuple[int, int], tokens_used: int = 0):
        """
        Record the results of a debate round.
        
        Args:
            score: (score1, score2) tuple from this round
            tokens_used: Number of tokens used in this round
        """
        self.scores.append(score)
        self.total_tokens += tokens_used
        self.rounds_completed += 1
        
    def should_stop(self) -> Tuple[bool, str]:
        """
        Determine if the debate should stop.
        
        Returns:
            Tuple of (should_stop, reason)
        """
        # Check max rounds
        if self.rounds_completed >= self.config.max_rounds:
            return True, "max_rounds_reached"
        
        # Check token budget
        if check_budget_exceeded(self.total_tokens, self.config.max_tokens):
            return True, "token_budget_exceeded"
        
        # Check convergence
        if check_convergence(
            self.scores, 
            self.config.convergence_epsilon,
            self.config.convergence_window
        ):
            return True, "converged"
        
        return False, ""
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            "rounds_completed": self.rounds_completed,
            "total_tokens": self.total_tokens,
            "final_scores": self.scores[-1] if self.scores else None,
            "all_scores": self.scores.copy()
        }


# Convenience functions for backwards compatibility
def should_continue_debate(
    scores: List[Tuple[int, int]],
    current_round: int,
    max_rounds: int,
    current_tokens: int = 0,
    max_tokens: int = 10000,
    epsilon: float = 5.0
) -> Tuple[bool, str]:
    """
    Convenience function to check if debate should continue.
    
    Args:
        scores: List of score tuples from previous rounds
        current_round: Current round number (1-indexed)
        max_rounds: Maximum allowed rounds
        current_tokens: Current token consumption
        max_tokens: Maximum allowed tokens
        epsilon: Convergence threshold
        
    Returns:
        Tuple of (should_continue, stop_reason if stopping)
    """
    # Max rounds check
    if current_round >= max_rounds:
        return False, "max_rounds_reached"
    
    # Token budget check
    if check_budget_exceeded(current_tokens, max_tokens):
        return False, "token_budget_exceeded"
    
    # Convergence check
    if check_convergence(scores, epsilon):
        return False, "converged"
    
    return True, ""