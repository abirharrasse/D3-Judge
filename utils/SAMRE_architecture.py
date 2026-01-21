import asyncio
import nest_asyncio
import re
from typing import List, Tuple, Optional
from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
import argparse

# D3 Framework enhancements
from convergence import BudgetedStoppingController, BudgetConfig, check_convergence
from anonymizer import TranscriptBuilder, anonymize_transcript
from token_tracker import TokenTracker, get_global_tracker

print("Starting improved debate script with D3 enhancements...")

def print_memory(role):
    print(f"\n--- Memory for {role.name} ---")
    for msg in role.rc.memory.get():
        print(f"{msg.role}: {msg.content[:100]}...")  # Print first 100 characters of each message
    print("-------------------------\n")

def get_last_tuple(response):
    all_matches = re.findall(r'[\(\[](\d+),\s*(\d+)[\)\]]', response)
    if all_matches:
        last_match = all_matches[-1]
        return (int(last_match[0]), int(last_match[1]))
    return None

class Summarizer(Action):
    name: str = "Summarizer"

    async def run(self, content: str) -> str:
        prompt = f"""Summarize the following content in 50 words or less, if there are any scores tuples, return them, it's important! Start summarization directly,
         no introductory sentences like here's your summary. In your summarization, only focus on the last scores, no partial ones. This is important: return the tuple of scores. These are the key points
         to summarize:\n\n{content}"""
        summary = await self._aask(prompt)
        return summary

class DefendAnswer(Action):
    PROMPT_TEMPLATE: str = """
    You're a fierce advocate defending this answer: {answer} to the question: {question}. Your opponent
    defends: {opponent_answer}. Convince the audience your argument is superior by addressing these criteria:

    1. Relevance: Show how your answer directly addresses the question and why your opponent's doesn't.
    2. Accuracy: Highlight the precision of your answer and point out any inaccuracies in your opponent's.
    3. Depth: Demonstrate the profound insights in your answer and expose the superficiality of your opponent's.
    4. Clarity: Emphasize the clarity of your answer and criticize any vagueness in your opponent's.
    5. Logic and Factuality: Stress the solid logic and facts in your answer while identifying logical fallacies or factual errors in your opponent's.

    Be assertive and don't hold back! Use rhetoric and persuasion to win over the audience but be respectful.
    Latest feedback: {feedback}
    Opponent's last argument: {opponent_argument}

    Your fierce defense (100 words max):
    """
    name: str = "DefendAnswer"

    async def run(self, question: str, answer: str, opponent_answer: str, opponent_argument: str = "", feedback: str = ""):
        prompt = self.PROMPT_TEMPLATE.format(question=question, answer=answer, opponent_answer=opponent_answer,
                                             opponent_argument=opponent_argument, feedback=feedback)
        return await self._aask(prompt)

class EnhancedJudgeAnswer(Action):


    PROMPT_TEMPLATE: str = """
    You're a critical, impartial judge in a high-stakes debate on: "{question}".
    Answer 1: "{answer1}". Answer 2: "{answer2}".
    Your goal is to provide detailed, constructive feedback that will push advocates to significantly improve their arguments.
    Current round: {current_round}
    Max rounds: {max_rounds}
    Previous scores: {previous_scores}

    Defense for 1st answer: {defense1}
    Defense for 2nd answer: {defense2}

    Analyze each argument meticulously. Be thorough and unbiased in your assessment of:
    1. Relevance to the question
    2. Accuracy of information and use of credible sources
    3. Depth of analysis and completeness of argument
    4. Clarity of expression and logical flow
    5. Strength of reasoning and factual support
    6. Effectiveness in addressing opponent's points

    For each criterion, provide a score on a scale of 1-20 and detailed justification.
    Scores should be given as [Answer1_score, Answer2_score] for each criterion.

    Your comprehensive feedback for each advocate (50 words each):
    Feedback for Advocate 1:
    Feedback for Advocate 2:

    Sum up the scores and return the final score tuple (score1, score2). Example: (95, 87)
    Your detailed scores and final tally:
    """
    name: str = "EnhancedJudgeAnswer"

    async def run(self, question: str, answer1: str, answer2: str, defense1: str, defense2: str, current_round: int, max_rounds: int, previous_scores: list):
        prompt = self.PROMPT_TEMPLATE.format(
            question=question, answer1=answer1, answer2=answer2, defense1=defense1, defense2=defense2,
            current_round=current_round, max_rounds=max_rounds, previous_scores=previous_scores
        )
        response = await self._aask(prompt)

        tuple_match = get_last_tuple(response)
        if tuple_match:
            scores = f"({tuple_match[0]}, {tuple_match[1]})"
        else:
            scores = "(0, 0)"  # Default scores if no valid tuple is found

        return response + "\n\nScores: " + scores

class VoteAction(Action):
    name: str = "VoteAction"

    async def run(self, question: str, context: str, identity: str) -> str:
        prompt = f"""As a jury member with the identity of {identity}, you are observing a debate on the question: {question}
        This is the debate:
        {context}
        
        Based on the arguments presented and taking into account the judge's evaluation (but not being overly influenced by it),
        which advocate do you think presented a more convincing case? Explain your reasoning briefly and conclude with your vote.
        
        Your response should be in the format:
        Reasoning: [Your brief explanation]
        Vote: (1, 0) if you vote for Advocate1, or (0, 1) if you vote for Advocate2
        """

        

        return await self._aask(prompt)

class EnhancedAdvocate(Role):
    def __init__(self, name: str, question: str, answer: str, opponent_answer: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.answer = answer
        self.question = question
        self.opponent_answer = opponent_answer
        self.defend_action = DefendAnswer()
        self.summarizer = Summarizer()
        self.set_actions([self.defend_action, self.summarizer])
        self._watch([self.defend_action])

    async def _act(self) -> Message:
        logger.info(f"{self.name}: Preparing argument")
        memories = self.rc.memory.get_by_role(role=self.name)
        opponent_memories = self.rc.memory.get_by_role(role=f"Opponent of {self.name}")

        opponent_argument = opponent_memories[-1].content if opponent_memories else ""
        feedback = self.rc.memory.get_by_role(role="Judge")[-1].content if self.rc.memory.get_by_role(role="Judge") else ""

        prompt = f"""
        You are {self.name} in a high-stakes debate. Your task is to present a compelling argument for your position
        while directly addressing and refuting your opponent's points.

        Question: {self.question}
        Your position: {self.answer}
        Opponent's position: {self.opponent_answer}
        Opponent's last argument: {opponent_argument}
        Judge's feedback: {feedback}

        Provide a strong, well-structured argument that:
        1. Reaffirms and expands on your position
        2. Directly addresses and refutes key points from your opponent's argument
        3. Uses specific examples, data, or expert opinions to support your claims
        4. Anticipates potential counter-arguments and preemptively addresses them
        5. Concludes with a powerful summary of why your position is superior

        Your response should be clear, logical, and persuasive. Aim for 100 words.
        """

        new_defense = await self.defend_action.run(question=self.question, answer=self.answer,
                                                   opponent_answer=self.opponent_answer,
                                                   opponent_argument=opponent_argument,
                                                   feedback=feedback)
        summary = await self.summarizer.run(new_defense)

        msg = Message(content=summary, role=self.name, cause_by=DefendAnswer)
        self.rc.memory.add(msg)
        return msg

class EnhancedJudge(Role):
    def __init__(self, question:str, answer1:str, answer2:str, **kwargs):
        super().__init__(**kwargs)
        self.name = "Judge"
        self.question = question
        self.answer1 = answer1
        self.answer2 = answer2
        self.judge_action = EnhancedJudgeAnswer()
        self.summarizer = Summarizer()
        self.set_actions([self.judge_action, self.summarizer])
        self._watch([DefendAnswer])

    async def _act(self, current_round: int, max_rounds: int, previous_scores: list) -> Message:
        logger.info("Judge: Evaluating arguments")
        memories = self.rc.memory.get(k=2)
        if len(memories) < 2:
            return Message(content="Waiting for more arguments.", role=self.name)

        advocate1_arg = memories[-2].content
        advocate2_arg = memories[-1].content

        evaluation = await self.judge_action.run(question=self.question, answer1=self.answer1, answer2=self.answer2,
                                                 defense1=advocate1_arg, defense2=advocate2_arg,
                                                 current_round=current_round, max_rounds=max_rounds,
                                                 previous_scores=previous_scores)

        summary = await self.summarizer.run(evaluation)

        msg_memory = Message(content=summary, role=self.name)
        msg = Message(content=evaluation, role=self.name, cause_by=EnhancedJudgeAnswer)
        self.rc.memory.add(msg_memory)
        return msg

class Jury(Role):
    def __init__(self, name: str, identity: str, question: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.identity = identity
        self.question = question
        self.vote_action = VoteAction()
        self.summarizer = Summarizer()
        self.set_actions([self.vote_action, self.summarizer])
        self._watch([DefendAnswer, EnhancedJudgeAnswer])

    async def _act(self) -> Message:
        logger.info(f"{self.name} ({self.identity}): Observing and voting")
        memories = self.rc.memory.get()
        context = "\n".join([f"{m.role}: {m.content}" for m in memories])

        vote = await self.vote_action.run(
            question=self.question,
            context=context,
            identity=self.identity
        )

        summary = await self.summarizer.run(vote)

        msg_memory = Message(content=summary, role=self.name)
        msg = Message(content=vote, role=self.name, cause_by=EnhancedJudgeAnswer)
        self.rc.memory.add(msg_memory)
        return msg

async def enhanced_debate(
    question: str, 
    answer1: str, 
    answer2: str, 
    investment: float = 3.0, 
    max_rounds: int = 7, 
    n_juries: int = 5,
    max_tokens: int = 10000,
    convergence_epsilon: float = 5.0
) -> Tuple[List[str], Tuple[int, int], dict]:
    """
    Enhanced debate with D3 framework features:
    - Budgeted stopping (convergence + token limits)
    - Anonymized transcripts for jury
    - Token tracking
    
    Returns:
        Tuple of (scores_list, jury_votes, stats_dict)
    """
    print("Initializing enhanced debate with D3 framework...")
    
    # Initialize D3 components
    budget_controller = BudgetedStoppingController(
        BudgetConfig(
            max_rounds=max_rounds,
            max_tokens=max_tokens,
            convergence_epsilon=convergence_epsilon
        )
    )
    transcript_builder = TranscriptBuilder(question, answer1, answer2)
    token_tracker = TokenTracker()
    
    advocate1 = EnhancedAdvocate(name="Advocate1", question=question, answer=answer1, opponent_answer=answer2)
    advocate2 = EnhancedAdvocate(name="Advocate2", question=question, answer=answer2, opponent_answer=answer1)
    judge = EnhancedJudge(question=question, answer1=answer1, answer2=answer2)

    # Create juries
    juries = []
    jury_votes = [0, 0]
    jury_identities = [
        "A retired professor of ethics",
        "A young environmental activist",
        "A middle-aged business owner",
        "A social worker specializing in community development",
        "A technology entrepreneur with a background in AI"
    ]
    for i in range(n_juries):
        identity = jury_identities[i % len(jury_identities)]
        jury = Jury(name=f"Jury{i+1}", identity=identity, question=question)
        juries.append(jury)

    print(f"Debate Question: {question}")
    print(f"Advocate1 defends: {answer1}")
    print(f"Advocate2 defends: {answer2}")
    print(f"Number of juries: {n_juries}")
    print(f"Budgeted stopping enabled: max_rounds={max_rounds}, max_tokens={max_tokens}\n")

    initial_msg = Message(content=question, role="Human", cause_by=DefendAnswer)
    advocate1.rc.memory.add(initial_msg)

    previous_scores = []
    scores = []
    stop_reason = ""

    round_count = 0
    # Budgeted stopping loop - check BEFORE running each round
    should_stop, stop_reason = budget_controller.should_stop()
    while not should_stop:
        round_count += 1
        token_tracker.set_round(round_count)
        print(f"\nStarting Round {round_count}...")

        print("Advocate1 preparing argument...")
        msg1 = await advocate1._act()
        # Track in transcript builder
        transcript_builder.add_argument("Advocate1", msg1.content)
        advocate2.rc.memory.add(Message(content=msg1.content, role="Opponent of Advocate2", cause_by=DefendAnswer))
        judge.rc.memory.add(Message(content=msg1.content, role="Advocate1", cause_by=DefendAnswer))
        # Note: Jury gets anonymized transcript at the end, not during debate

        print("Advocate2 preparing argument...")
        msg2 = await advocate2._act()
        # Track in transcript builder
        transcript_builder.add_argument("Advocate2", msg2.content)
        advocate1.rc.memory.add(Message(content=msg2.content, role="Opponent of Advocate1", cause_by=DefendAnswer))
        judge.rc.memory.add(Message(content=msg2.content, role="Advocate2", cause_by=DefendAnswer))

        print("Judge evaluating...")
        judge_msg = await judge._act(current_round=round_count, max_rounds=max_rounds, previous_scores=previous_scores)
        advocate1.rc.memory.add(Message(content=judge_msg.content, role="Judge", cause_by=EnhancedJudgeAnswer))
        advocate2.rc.memory.add(Message(content=judge_msg.content, role="Judge", cause_by=EnhancedJudgeAnswer))

        score_match = get_last_tuple(judge_msg.content)
        print("______________________last_tuples", score_match)
        if score_match:
            new_scores = score_match
            previous_scores.append(new_scores)
            scores.append(str(new_scores))
            # Track in transcript builder and budget controller
            transcript_builder.add_judge_round(judge_msg.content, new_scores)
            budget_controller.record_round(new_scores, tokens_used=0)  # TODO: integrate actual token count
            print(f"Parsed Scores: {new_scores}")
        else:
            print("Error parsing scores from judge's message")
            previous_scores.append((0, 0))
            scores.append("(0, 0)")
            budget_controller.record_round((0, 0), tokens_used=0)
        
        # Check budgeted stopping after each round
        should_stop, stop_reason = budget_controller.should_stop()
        if should_stop:
            print(f"\n*** Budgeted stopping triggered: {stop_reason} ***")
            break

    # Jury voting with ANONYMIZED transcript
    print("\nCompiling anonymized transcript for jury deliberation...")
    anonymized_transcript = transcript_builder.build_anonymized()
    print(f"Transcript compiled ({len(anonymized_transcript)} chars)")
    
    print("Jury voting...")
    jury_votes_list = []
    for jury in juries:
        # Provide anonymized transcript to jury
        jury.rc.memory.add(Message(
            content=f"ANONYMIZED DEBATE TRANSCRIPT:\n{anonymized_transcript}",
            role="Transcript",
            cause_by=EnhancedJudgeAnswer
        ))
        jury_vote = await jury._act()
        score_jury = get_last_tuple(jury_vote.content)
        parsed_jury = score_jury if score_jury else (0, 0)
        jury_votes_list.append(parsed_jury)
        print(f"{jury.name} ({jury.identity}) vote: {parsed_jury}")

    # Calculate jury votes
    for vote in jury_votes_list:
        jury_votes[0] += int(vote[0])
        jury_votes[1] += int(vote[1])

    print("\nFinal Scores:")
    for round_num, score in enumerate(scores, 1):
        print(f"Round {round_num}: {score}")

    jury_winner = "Defense A" if jury_votes[0] > jury_votes[1] else "Defense B"
    print(f"\nDebate completed.")
    if previous_scores:
        print(f"Judge's Final Scores: {previous_scores[-1]}")
    print(f"Jury Votes: Defense A: {jury_votes[0]}, Defense B: {jury_votes[1]}")
    print(f"Jury's Winner: {jury_winner}")
    if stop_reason:
        print(f"Stop reason: {stop_reason}")
    
    # Compile statistics
    stats = {
        "rounds_completed": round_count,
        "stop_reason": stop_reason,
        "token_usage": token_tracker.get_summary(),
        "budget_stats": budget_controller.get_stats()
    }

    return scores, tuple(jury_votes), stats

async def run_debate(
    question: str, 
    answer1: str, 
    answer2: str, 
    investment: float = 0.1, 
    max_rounds: int = 7, 
    n_juries: int = 5,
    max_tokens: int = 10000,
    convergence_epsilon: float = 5.0
) -> Tuple[List[str], Tuple[int, int], dict]:
    """Run a SAMRE debate with D3 framework enhancements."""
    try:
        print("Starting run_debate function with D3 enhancements...")
        scores, juries, stats = await enhanced_debate(
            question=question, 
            answer1=answer1, 
            answer2=answer2, 
            investment=investment, 
            max_rounds=max_rounds, 
            n_juries=n_juries,
            max_tokens=max_tokens,
            convergence_epsilon=convergence_epsilon
        )
        print("Debate completed successfully.")
        print(f"Stats: {stats}")
        return scores, juries, stats
    except Exception as e:
        print(f"An error occurred during the debate: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], (0, 0), {"error": str(e)}


def samre_scores(
    question: str, 
    answer1: str, 
    answer2: str, 
    investment: float = 0.1, 
    max_rounds: int = 7, 
    n_juries: int = 5,
    max_tokens: int = 10000,
    convergence_epsilon: float = 5.0
) -> Tuple[List[str], Tuple[int, int], dict]:
    """Synchronous wrapper for SAMRE debate with D3 enhancements."""
    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(run_debate(
        question, answer1, answer2, investment, max_rounds, n_juries,
        max_tokens, convergence_epsilon
    ))