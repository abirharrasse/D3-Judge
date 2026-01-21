import asyncio
import nest_asyncio
import platform
from typing import List, Tuple, Optional
import re
from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.team import Team
import argparse

# D3 Framework enhancements
from convergence import BudgetedStoppingController, BudgetConfig
from anonymizer import TranscriptBuilder, anonymize_transcript
from token_tracker import TokenTracker

print("Starting debate script with D3 enhancements...")

# Global variables for debate content


class DefendAnswer(Action):
    PROMPT_TEMPLATE: str = """
    You're an eloquent advocate (Advocate {advocate_id}) in a group defending this answer: {answer} to the question: {question}. Your opponent
    group defends: {opponent_answer}. Collaborate with your fellow advocates to convince the audience your argument is better.
    Use the latest feedback, your opponent's last argument, and your team's previous arguments to improve your case.
    Latest feedback: {feedback}
    Opponent's last argument: {opponent_argument}
    Your team's previous arguments: {team_arguments}
    Respond in under 80 words.
    Your defense:
    """
    name: str = "DefendAnswer"

    async def run(self, question: str, answer: str, opponent_answer: str, advocate_id: int, team_arguments: List[str], opponent_argument: str = "", feedback: str = ""):
        prompt = self.PROMPT_TEMPLATE.format(question=question, answer=answer, opponent_answer=opponent_answer,
                                             opponent_argument=opponent_argument, feedback=feedback,
                                             advocate_id=advocate_id, team_arguments=team_arguments)
        return await self._aask(prompt)

class AggregateDefense(Action):
    PROMPT_TEMPLATE: str = """
    You are an expert debate strategist. Your task is to aggregate and improve upon the following defenses for the answer: {answer} to the question: {question}.
    The opponent's answer is: {opponent_answer}

    Individual defenses:
    {defenses}

    Latest feedback from the judge: {feedback}

    Analyze each defense critically. Identify the strongest points, address any weaknesses, and combine the best arguments into a cohesive, powerful defense.
    Aim to create a defense that is stronger and more comprehensive than any individual argument.

    Provide your aggregated and improved defense in under 150 words:
    """
    name: str = "AggregateDefense"

    async def run(self, question: str, answer: str, opponent_answer: str, defenses: List[str], feedback: str = ""):
        prompt = self.PROMPT_TEMPLATE.format(question=question, answer=answer, opponent_answer=opponent_answer,
                                             defenses="\n".join(defenses), feedback=feedback)
        return await self._aask(prompt)

class JudgeAnswer(Action):
    PROMPT_TEMPLATE: str = """
    You're a fair, impartial judge in a debate on: "{question}". Answer 1: "{answer1}". Answer 2: "{answer2}".
    Your goal is to provide feedback that will help advocate groups improve and differentiate their arguments more clearly.
    Current round: {current_round}
    Total rounds: {total_rounds}
    Previous scores: {previous_scores}

    Defense for 1st answer: {defense1}
    Defense for 2nd answer: {defense2}

    Provide specific, constructive feedback to help each advocate group strengthen their unique position.
    Encourage them to address weaknesses and highlight distinctions.
    Aim for your feedback to lead to more divergent scores in future rounds.

    Give your feedback in under 50 words:
    """


    name: str = "JudgeAnswer"

    async def run(self, question: str, answer1: str, answer2: str, defense1: str, defense2: str, current_round: int, total_rounds: int, previous_scores: list):
        prompt = self.PROMPT_TEMPLATE.format(
            question=question, answer1=answer1, answer2=answer2, defense1=defense1, defense2=defense2,
            current_round=current_round, total_rounds=total_rounds, previous_scores=previous_scores
        )
        return await self._aask(prompt)

class ScoreAnswer(Action):
    PROMPT_TEMPLATE: str = """
    You're a critical, impartial judge in a high-stakes debate on: "{question}".
    Answer 1: "{answer1}". Answer 2: "{answer2}".
    Your goal is to provide detailed, constructive feedback that will push advocates to significantly improve their arguments.
    Total rounds: {total_rounds}
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
    name: str = "ScoreAnswer"

    async def run(self, question: str, answer1: str, answer2: str, defense1: str, defense2: str, current_round: int, total_rounds: int, previous_scores: list):
        prompt = self.PROMPT_TEMPLATE.format(
            question=question, answer1=answer1, answer2=answer2, defense1=defense1, defense2=defense2,
            current_round=current_round, total_rounds=total_rounds, previous_scores=previous_scores
        )
        response = await self._aask(prompt)

        # Extract the tuple from the response
        tuple_match = re.search(r'\((\d+),\s*(\d+)\)', response)
        if tuple_match:
            return f"({tuple_match.group(1)}, {tuple_match.group(2)})"
        else:
            return "(0, 0)"  # Default scores if no valid tuple is found

class Advocate(Role):
    def __init__(self, name: str, question:str, answer: str, opponent_answer: str, advocate_id: int, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.question = question
        self.answer = answer
        self.opponent_answer = opponent_answer
        self.advocate_id = advocate_id
        self.defend_action = DefendAnswer()
        self.set_actions([self.defend_action])
        self._watch([DefendAnswer])

    async def _act(self, team_arguments: List[str]) -> Message:
        logger.info(f"{self.name} (Advocate {self.advocate_id}): Preparing argument")
        memories = self.rc.memory.get_by_role(role=self.name)
        opponent_memories = self.rc.memory.get_by_role(role=f"Opponent of {self.name}")

        opponent_argument = opponent_memories[-1].content if opponent_memories else ""
        feedback = self.rc.memory.get_by_role(role="Judge")[-1].content if self.rc.memory.get_by_role(role="Judge") else ""

        new_defense = await self.defend_action.run(question=self.question, answer=self.answer, opponent_answer=self.opponent_answer,
                                     opponent_argument=opponent_argument, feedback=feedback,
                                     advocate_id=self.advocate_id, team_arguments=team_arguments)

        msg = Message(content=new_defense, role=self.name, cause_by=DefendAnswer)
        self.rc.memory.add(msg)
        return msg

class Aggregator(Role):
    def __init__(self, name: str, question:str, answer: str, opponent_answer: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.answer = answer
        self.question = question
        self.opponent_answer = opponent_answer
        self.aggregate_action = AggregateDefense()
        self.set_actions([self.aggregate_action])
        self._watch([AggregateDefense])

    async def _act(self, defenses: List[str]) -> Message:
        logger.info(f"{self.name}: Aggregating and improving defenses")
        feedback = self.rc.memory.get_by_role(role="Judge")[-1].content if self.rc.memory.get_by_role(role="Judge") else ""

        aggregated_defense = await self.aggregate_action.run(question=self.question, answer=self.answer,
                                                             opponent_answer=self.opponent_answer,
                                                             defenses=defenses, feedback=feedback)

        msg = Message(content=aggregated_defense, role=self.name, cause_by=AggregateDefense)
        self.rc.memory.add(msg)
        return msg

class AdvocateGroup:
    def __init__(self, name: str, question:str, answer: str, opponent_answer: str, n_advocates: int):
        self.name = name
        self.advocates = [Advocate(f"{name}_Advocate{i+1}", question, answer, opponent_answer, i+1) for i in range(n_advocates)]
        self.aggregator = Aggregator(f"{name}_Aggregator", question, answer, opponent_answer)
        self.answer = answer
        self.opponent_answer = opponent_answer

    async def act(self) -> str:
        team_arguments = [adv.rc.memory.get_by_role(role=adv.name)[-1].content if adv.rc.memory.get_by_role(role=adv.name) else "" for adv in self.advocates]

        defenses = await asyncio.gather(*[adv._act(team_arguments) for adv in self.advocates])
        individual_defenses = [d.content for d in defenses]

        aggregated_defense = await self.aggregator._act(individual_defenses)
        return aggregated_defense.content

class Judge(Role):
    def __init__(self, question, answer1, answer2, **kwargs):
        super().__init__(**kwargs)
        self.name = "Judge"
        self.question = question
        self.answer1 = answer1
        self.answer2 = answer2
        self.judge_action = JudgeAnswer()
        self.set_actions([self.judge_action])
        self._watch([DefendAnswer, AggregateDefense])

    async def _act(self, current_round: int, total_rounds: int, previous_scores: list) -> Message:
        logger.info("Judge: Evaluating arguments")
        memories = self.rc.memory.get(k=2)
        if len(memories) < 2:
            return Message(content="Waiting for more arguments.", role=self.name)

        advocate1_arg = memories[-2].content
        advocate2_arg = memories[-1].content

        evaluation = await self.judge_action.run(question=self.question, answer1=self.answer1, answer2=self.answer2,
                                                 defense1=advocate1_arg, defense2=advocate2_arg,
                                                 current_round=current_round, total_rounds=total_rounds,
                                                 previous_scores=previous_scores)

        msg = Message(content=evaluation, role=self.name)
        self.rc.memory.add(msg)
        return msg

class Scorer(Role):
    def __init__(self, question, answer1, answer2, **kwargs):
        super().__init__(**kwargs)
        self.name = "Scorer"
        self.question = question
        self.answer1 = answer1
        self.answer2 = answer2
        self.score_action = ScoreAnswer()
        self.set_actions([self.score_action])
        self._watch([DefendAnswer, AggregateDefense])

    async def _act(self, current_round: int, total_rounds: int, previous_scores: list) -> Message:
        logger.info("Scorer: Scoring arguments")
        memories = self.rc.memory.get(k=2)
        if len(memories) < 2:
            return Message(content="Waiting for more arguments.", role=self.name)

        advocate1_arg = memories[-2].content
        advocate2_arg = memories[-1].content

        scores = await self.score_action.run(question=self.question, answer1=self.answer1, answer2=self.answer2,
                                             defense1=advocate1_arg, defense2=advocate2_arg,
                                             current_round=current_round, total_rounds=total_rounds,
                                             previous_scores=previous_scores)

        msg = Message(content=scores, role=self.name)
        self.rc.memory.add(msg)
        return msg


# ==================== JURY SUPPORT FOR MORE ====================
# Per D3 paper Figure 1: Jury is used in both protocols

class VoteAction(Action):
    """Jury voting action - adapted from SAMRE for MORE protocol."""
    name: str = "VoteAction"

    async def run(self, question: str, context: str, identity: str) -> str:
        prompt = f"""As a jury member with the identity of {identity}, you are observing a debate on the question: {question}
        This is the debate:
        {context}
        
        Based on the arguments presented and taking into account the judge's evaluation (but not being overly influenced by it),
        which defense do you think presented a more convincing case? Explain your reasoning briefly and conclude with your vote.
        
        Your response should be in the format:
        Reasoning: [Your brief explanation]
        Vote: (1, 0) if you vote for Defense A, or (0, 1) if you vote for Defense B
        """
        return await self._aask(prompt)


class Jury(Role):
    """Jury role for MORE protocol with persona-based evaluation."""
    
    def __init__(self, name: str, identity: str, question: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.identity = identity
        self.question = question
        self.vote_action = VoteAction()
        self.set_actions([self.vote_action])
        self._watch([AggregateDefense, ScoreAnswer])

    async def _act(self, transcript: str) -> Message:
        logger.info(f"{self.name} ({self.identity}): Evaluating and voting")
        
        vote = await self.vote_action.run(
            question=self.question,
            context=transcript,
            identity=self.identity
        )

        msg = Message(content=vote, role=self.name, cause_by=VoteAction)
        self.rc.memory.add(msg)
        return msg

async def debate(
    question: str, 
    answer1: str, 
    answer2: str, 
    investment: float = 3.0, 
    n_round: int = 5, 
    n_advocates: int = 3,
    n_juries: int = 5
) -> Tuple[List[str], Tuple[int, int], dict]:
    """
    MORE debate with D3 framework features:
    - Multi-advocate groups with aggregation
    - Jury deliberation with anonymized transcripts
    - Token tracking
    
    Returns:
        Tuple of (scores_list, jury_votes, stats_dict)
    """
    print("Initializing MORE debate with D3 framework...")
    
    # Initialize D3 components
    transcript_builder = TranscriptBuilder(question, answer1, answer2)
    token_tracker = TokenTracker()
    
    advocate_group1 = AdvocateGroup(name="AdvocateGroup1", question=question, answer=answer1, opponent_answer=answer2, n_advocates=n_advocates)
    advocate_group2 = AdvocateGroup(name="AdvocateGroup2", question=question, answer=answer2, opponent_answer=answer1, n_advocates=n_advocates)
    judge = Judge(question=question, answer1=answer1, answer2=answer2)
    scorer = Scorer(question=question, answer1=answer1, answer2=answer2)
    
    # Initialize juries with diverse personas (per D3 paper)
    jury_identities = [
        "A retired professor of ethics",
        "A young environmental activist",
        "A middle-aged business owner",
        "A social worker specializing in community development",
        "A technology entrepreneur with a background in AI"
    ]
    juries = [
        Jury(name=f"Jury{i+1}", identity=jury_identities[i % len(jury_identities)], question=question)
        for i in range(n_juries)
    ]

    print(f"Debate Question: {question}")
    print(f"AdvocateGroup1 ({n_advocates} advocates) defends: {answer1}")
    print(f"AdvocateGroup2 ({n_advocates} advocates) defends: {answer2}")
    print(f"Number of juries: {n_juries}\n")

    initial_msg = Message(content=question, role="Human", cause_by=DefendAnswer)
    for adv in advocate_group1.advocates + advocate_group2.advocates:
        adv.rc.memory.add(initial_msg)

    previous_scores = []
    scores = []

    for i in range(n_round):
        token_tracker.set_round(i + 1)
        print(f"Starting Round {i+1}...")

        print("AdvocateGroup1 preparing argument...")
        msg1 = await advocate_group1.act()
        print(f"AdvocateGroup1 aggregated argument: {msg1[:100]}...")
        transcript_builder.add_argument("AdvocateGroup1", msg1)
        for adv in advocate_group2.advocates:
            adv.rc.memory.add(Message(content=msg1, role=f"Opponent of {adv.name}", cause_by=AggregateDefense))
        judge.rc.memory.add(Message(content=msg1, role="AdvocateGroup1"))
        scorer.rc.memory.add(Message(content=msg1, role="AdvocateGroup1"))

        print("AdvocateGroup2 preparing argument...")
        msg2 = await advocate_group2.act()
        print(f"AdvocateGroup2 aggregated argument: {msg2[:100]}...")
        transcript_builder.add_argument("AdvocateGroup2", msg2)
        for adv in advocate_group1.advocates:
            adv.rc.memory.add(Message(content=msg2, role=f"Opponent of {adv.name}", cause_by=AggregateDefense))
        judge.rc.memory.add(Message(content=msg2, role="AdvocateGroup2"))
        scorer.rc.memory.add(Message(content=msg2, role="AdvocateGroup2"))

        print("Judge evaluating...")
        judge_msg = await judge._act(current_round=i+1, total_rounds=n_round, previous_scores=previous_scores)
        print(f"Judge evaluation: {judge_msg.content[:100]}...")
        for adv in advocate_group1.advocates + advocate_group2.advocates:
            adv.rc.memory.add(judge_msg)
        advocate_group1.aggregator.rc.memory.add(judge_msg)
        advocate_group2.aggregator.rc.memory.add(judge_msg)

        print("Scorer scoring...")
        score_msg = await scorer._act(current_round=i+1, total_rounds=n_round, previous_scores=previous_scores)
        print(f"Raw Scores: {score_msg.content}")
        scores.append(score_msg.content)

        # Parse and store the new scores
        try:
            new_scores = eval(score_msg.content)
            if not isinstance(new_scores, tuple) or len(new_scores) != 2:
                raise ValueError("Invalid score format")
            previous_scores.append(new_scores)
            transcript_builder.add_judge_round(judge_msg.content, new_scores)
            print(f"Parsed Scores: {new_scores}")
        except Exception as e:
            print(f"Error parsing scores: {e}")
            previous_scores.append((0, 0))

        print()  # Add a blank line between rounds

    # Compile ANONYMIZED transcript for jury deliberation
    print("\nCompiling anonymized transcript for jury deliberation...")
    anonymized_transcript = transcript_builder.build_anonymized()
    print(f"Transcript compiled ({len(anonymized_transcript)} chars)")
    
    # Jury voting (NEW for MORE protocol - per D3 paper Figure 1)
    print("\nJury voting...")
    jury_votes = [0, 0]
    jury_votes_list = []
    
    # Extract tuple from vote response
    def get_vote_tuple(response):
        all_matches = re.findall(r'[\(\[]([01]),\s*([01])[\)\]]', response)
        if all_matches:
            last_match = all_matches[-1]
            return (int(last_match[0]), int(last_match[1]))
        return None
    
    for jury in juries:
        jury_vote = await jury._act(anonymized_transcript)
        vote_tuple = get_vote_tuple(jury_vote.content)
        parsed_vote = vote_tuple if vote_tuple else (0, 0)
        jury_votes_list.append(parsed_vote)
        print(f"{jury.name} ({jury.identity}) vote: {parsed_vote}")
        jury_votes[0] += parsed_vote[0]
        jury_votes[1] += parsed_vote[1]

    # Print final scores
    print("\nFinal Judge Scores:") 
    for round_num, (score1, score2) in enumerate(previous_scores, 1):
        print(f"Round {round_num}: Defense A - {score1}, Defense B - {score2}")
    
    # Determine winners
    final_scores = previous_scores[-1] if previous_scores else (0, 0)
    judge_winner = "Defense A" if final_scores[0] > final_scores[1] else "Defense B"
    jury_winner = "Defense A" if jury_votes[0] > jury_votes[1] else "Defense B"
    
    print(f"\nJury Votes: Defense A: {jury_votes[0]}, Defense B: {jury_votes[1]}")
    print(f"Judge's Winner: {judge_winner}")
    print(f"Jury's Winner: {jury_winner}")

    print("\nDebate completed.")
    
    # Compile statistics
    stats = {
        "rounds_completed": n_round,
        "n_advocates": n_advocates,
        "n_juries": n_juries,
        "judge_final_scores": final_scores,
        "jury_votes": tuple(jury_votes),
        "jury_winner": jury_winner,
        "judge_winner": judge_winner,
        "token_usage": token_tracker.get_summary()
    }
    
    return scores, tuple(jury_votes), stats

async def run_debate(
    question: str, 
    answer1: str, 
    answer2: str, 
    investment: float = 0.1, 
    n_round: int = 3, 
    n_advocates: int = 3,
    n_juries: int = 5
) -> Tuple[List[str], Tuple[int, int], dict]:
    """Run a MORE debate with D3 framework enhancements."""
    try:
        print("Starting run_debate function with D3 enhancements...")
        scores, jury_votes, stats = await debate(
            question=question, 
            answer1=answer1, 
            answer2=answer2, 
            investment=investment, 
            n_round=n_round, 
            n_advocates=n_advocates,
            n_juries=n_juries
        )
        print("Debate completed successfully.")
        print(f"Stats: {stats}")
        return scores, jury_votes, stats
    except Exception as e:
        print(f"An error occurred during the debate: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], (0, 0), {"error": str(e)}

nest_asyncio.apply()

def more_scores(
    question: str, 
    answer1: str, 
    answer2: str, 
    investment: float = 0.1, 
    n_round: int = 3, 
    n_advocates: int = 3,
    n_juries: int = 5
) -> Tuple[List[str], Tuple[int, int], dict]:
    """Synchronous wrapper for MORE debate with D3 enhancements."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(run_debate(
        question, answer1, answer2, investment, n_round, n_advocates, n_juries
    ))