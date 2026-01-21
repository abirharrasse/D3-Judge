"""
Transcript Anonymization for D3 Framework.

Implements anonymization as described in Section 2 of the D3 paper:
"To prevent the judge and jurors from being influenced by the source of the arguments, 
the advocates' outputs are anonymized before being entered into the debate record."

"Transcript Compilation: Upon conclusion of the debate, a complete, anonymized 
transcript is compiled"
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AnonymizationMapping:
    """Stores the mapping between original and anonymized identifiers."""
    original_to_anon: Dict[str, str]
    anon_to_original: Dict[str, str]
    
    @classmethod
    def create_default(cls) -> "AnonymizationMapping":
        """Create default advocate anonymization mapping."""
        mapping = {
            "Advocate1": "Defense A",
            "Advocate2": "Defense B",
            "AdvocateGroup1": "Defense A",
            "AdvocateGroup2": "Defense B",
            "Opponent of Advocate1": "Opposing Defense A",
            "Opponent of Advocate2": "Opposing Defense B",
        }
        reverse = {v: k for k, v in mapping.items()}
        return cls(original_to_anon=mapping, anon_to_original=reverse)


def anonymize_transcript(
    transcript: str, 
    mapping: Optional[AnonymizationMapping] = None
) -> str:
    """
    Anonymize a debate transcript by replacing advocate identifiers.
    
    Per the paper: "the advocates' outputs are anonymized before being 
    entered into the debate record"
    
    Args:
        transcript: The raw transcript text
        mapping: Optional custom anonymization mapping
        
    Returns:
        Anonymized transcript
    """
    if mapping is None:
        mapping = AnonymizationMapping.create_default()
    
    result = transcript
    
    # Sort by length (longest first) to avoid partial replacements
    sorted_mappings = sorted(
        mapping.original_to_anon.items(), 
        key=lambda x: len(x[0]), 
        reverse=True
    )
    
    for original, anon in sorted_mappings:
        # Use word boundaries for cleaner replacement
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        result = pattern.sub(anon, result)
    
    return result


def anonymize_message_list(
    messages: List[dict],
    mapping: Optional[AnonymizationMapping] = None
) -> List[dict]:
    """
    Anonymize a list of message dictionaries.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        mapping: Optional custom anonymization mapping
        
    Returns:
        List of anonymized message dicts
    """
    if mapping is None:
        mapping = AnonymizationMapping.create_default()
    
    anonymized = []
    for msg in messages:
        anon_msg = msg.copy()
        
        # Anonymize role
        if msg.get("role") in mapping.original_to_anon:
            anon_msg["role"] = mapping.original_to_anon[msg["role"]]
        
        # Anonymize content
        if "content" in msg:
            anon_msg["content"] = anonymize_transcript(msg["content"], mapping)
            
        anonymized.append(anon_msg)
    
    return anonymized


def compile_anonymized_transcript(
    question: str,
    answer1: str,
    answer2: str,
    arguments: List[Tuple[str, str]],  # List of (role, content) tuples
    judge_feedback: List[str],
    judge_scores: List[Tuple[int, int]],
    mapping: Optional[AnonymizationMapping] = None
) -> str:
    """
    Compile a complete anonymized transcript for jury deliberation.
    
    Per the paper (Section 2.3):
    "Transcript Compilation: Upon conclusion of the debate, a complete, 
    anonymized transcript is compiled, including the original question, 
    candidate answers, all arguments, and all feedback and scores from the Judge."
    
    Args:
        question: The original question
        answer1: First answer
        answer2: Second answer
        arguments: List of (role, argument) tuples
        judge_feedback: List of judge feedback strings
        judge_scores: List of (score1, score2) tuples
        mapping: Optional anonymization mapping
        
    Returns:
        Complete anonymized transcript string
    """
    if mapping is None:
        mapping = AnonymizationMapping.create_default()
    
    lines = []
    
    # Header
    lines.append("=" * 60)
    lines.append("DEBATE TRANSCRIPT (ANONYMIZED)")
    lines.append("=" * 60)
    lines.append("")
    
    # Question
    lines.append("QUESTION:")
    lines.append(question)
    lines.append("")
    
    # Answers (anonymized)
    lines.append("CANDIDATE ANSWERS:")
    lines.append("-" * 40)
    lines.append("Defense A's Position:")
    lines.append(answer1)
    lines.append("")
    lines.append("Defense B's Position:")
    lines.append(answer2)
    lines.append("")
    
    # Arguments by round
    lines.append("DEBATE ARGUMENTS:")
    lines.append("-" * 40)
    
    current_round = 0
    for i, (role, content) in enumerate(arguments):
        # Detect round changes (every 2 arguments is a new round)
        round_num = (i // 2) + 1
        if round_num != current_round:
            current_round = round_num
            lines.append(f"\n--- Round {current_round} ---")
        
        # Anonymize role
        anon_role = mapping.original_to_anon.get(role, role)
        anon_content = anonymize_transcript(content, mapping)
        
        lines.append(f"\n{anon_role}:")
        lines.append(anon_content)
        
        # Add judge feedback after each complete round
        if i % 2 == 1 and (i // 2) < len(judge_feedback):
            feedback_idx = i // 2
            lines.append(f"\nJudge's Round {round_num} Feedback:")
            lines.append(anonymize_transcript(judge_feedback[feedback_idx], mapping))
            if feedback_idx < len(judge_scores):
                lines.append(f"Scores: Defense A: {judge_scores[feedback_idx][0]}, " +
                           f"Defense B: {judge_scores[feedback_idx][1]}")
    
    # Summary
    lines.append("")
    lines.append("=" * 60)
    lines.append("END OF TRANSCRIPT")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def deanonymize_transcript(
    anonymized_transcript: str,
    mapping: Optional[AnonymizationMapping] = None
) -> str:
    """
    Reverse anonymization for debugging/logging purposes.
    
    Args:
        anonymized_transcript: The anonymized transcript
        mapping: Optional anonymization mapping
        
    Returns:
        De-anonymized transcript
    """
    if mapping is None:
        mapping = AnonymizationMapping.create_default()
    
    result = anonymized_transcript
    
    # Sort by length (longest first)
    sorted_mappings = sorted(
        mapping.anon_to_original.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )
    
    for anon, original in sorted_mappings:
        pattern = re.compile(re.escape(anon), re.IGNORECASE)
        result = pattern.sub(original, result)
    
    return result


class TranscriptBuilder:
    """
    Builder class for constructing debate transcripts progressively.
    
    Use this during debate execution to build up the transcript,
    then call build_anonymized() to get the final anonymized version.
    """
    
    def __init__(
        self,
        question: str,
        answer1: str,
        answer2: str,
        mapping: Optional[AnonymizationMapping] = None
    ):
        """Initialize the transcript builder."""
        self.question = question
        self.answer1 = answer1
        self.answer2 = answer2
        self.mapping = mapping or AnonymizationMapping.create_default()
        
        self.arguments: List[Tuple[str, str]] = []
        self.judge_feedback: List[str] = []
        self.judge_scores: List[Tuple[int, int]] = []
        
    def add_argument(self, role: str, content: str):
        """Add an argument to the transcript."""
        self.arguments.append((role, content))
        
    def add_judge_round(self, feedback: str, scores: Tuple[int, int]):
        """Add a judge's round evaluation."""
        self.judge_feedback.append(feedback)
        self.judge_scores.append(scores)
        
    def build_anonymized(self) -> str:
        """Build the complete anonymized transcript."""
        return compile_anonymized_transcript(
            question=self.question,
            answer1=self.answer1,
            answer2=self.answer2,
            arguments=self.arguments,
            judge_feedback=self.judge_feedback,
            judge_scores=self.judge_scores,
            mapping=self.mapping
        )
    
    def build_raw(self) -> str:
        """Build the raw (non-anonymized) transcript."""
        raw = compile_anonymized_transcript(
            question=self.question,
            answer1=self.answer1,
            answer2=self.answer2,
            arguments=self.arguments,
            judge_feedback=self.judge_feedback,
            judge_scores=self.judge_scores,
            mapping=AnonymizationMapping(
                original_to_anon={},  # No anonymization
                anon_to_original={}
            )
        )
        return raw