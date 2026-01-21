import re
from openai import OpenAI
import os
from anthropic import Anthropic
from together import Together
import google.generativeai as genai



claude_api = os.getenv("CLAUDE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

client_anth = Anthropic(api_key = claude_api)
client_togeth = Together(api_key =  together_api_key)
client_openai = OpenAI(api_key = openai_api_key)
genai.configure(api_key = gemini_api_key)


models = {"opus": "claude-3-opus-20240229", "haiku": "claude-3-haiku-20240307", "sonnet": "claude-3-sonnet-20240229", "llama3_8": "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
          "llama3_70": 'meta-llama/Llama-3-70b-chat-hf', 'llama3.1_8': "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
          "mistral": "mistralai/Mixtral-8x22B-Instruct-v0.1", "Qwen": "Qwen/Qwen2-72B-Instruct",
          "Yi": "zero-one-ai/Yi-34B-Chat", "gemma": "google/gemma-7b-it", "cohere": "command-r-plus", "gemini": 'gemini-pro',
          "gpt-4-turbo": "gpt-4-turbo-preview", "gpt-4o": "gpt-4o", "gpt-3.5-turbo":"gpt-3.5-turbo"}


def judge_answers_other(judge_model, temp, question, answer1, answer2):

  prompt = f"""
  You are a highly accurate judge. You will be given two defenses of two answers.
  Judge which one of the answers is the best based only on the defenses provided.
   Do not use any knowledge you have. Choose the answer based on the following criteria: Relevance, Accuracy, Depth and Completeness, Logic and Factuality,
   Clarity and Readability. Return only a tuple of the scores: (score1, score2). Scores should be between 20 and 1, close to 20 for the best answer
   and close to 1 for the worst.
   Don't overthink and do not use your world knowledge. Just compare
  \n
  Question : {question}
  First defense: {answer1}
  Second defense: {answer2}
  Scores:
  """



  if judge_model in  ["gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"]:
    response = client_openai.chat.completions.create(
      model=models[judge_model],
      messages=[
          {"role": "user", "content": prompt}
      ],
      temperature=temp
      )
    scores = response.choices[0].message.content

  elif judge_model in ["opus", "haiku", "sonnet"]:
    message = client_anth.messages.create(
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        model = models[judge_model],
        )
    scores = message.content[0].text
  elif judge_model == 'cohere':
    response = co.chat(
    model="command-r-plus",
    message= prompt
      )
    scores = response.text
  elif judge_model == 'gemini':
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    scores = response.text
  else:
    message = client_togeth.chat.completions.create(
        model=models[judge_model],
        temperature=temp,
        messages=[{"role": "user", "content": prompt}],
    )
    scores = message.choices[0].message.content
  return scores


def judge_answers(judge_model, temp, question, answer1, answer2):

  prompt = f"""
  You are a fair, impartial judge scoring a debate on the following question: {question}.

  Answer 1: {answer1}
  Answer 2: {answer2}

  Score each answer on a scale of 1-20 for each of the following criteria:
    1. Relevance to the question
    2. Accuracy of information and use of credible sources
    3. Depth of analysis and completeness of argument
    4. Clarity of expression and logical flow
    5. Strength of reasoning and factual support
    6. Effectiveness in addressing opponent's points
  Provide scores as [Answer1_score, Answer2_score] for each criterion in a list format, then sum for final scores. Please keep an eye on the slightest difference that should make a difference in the scoring. Don't overthink!

  Relevance:
  Accuracy:
  Depth:
  Clarity:
  Logic and Factuality:
  Addressing opponent's points:
  Final Scores (sum of above) as a tuple (example: (18, 9)):

  Explain your scoring, focusing on why one answer is better than the other based on the criteria above. Keep your explanation concise but informative.

  Finally, return  the final score tuple (score1, score2) as a tuple (in parentheses). Example: (18, 9)

  Your scores and explanation:
  """



  if judge_model in  ["gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"]:
    response = client_openai.chat.completions.create(
      model=models[judge_model],
      messages=[
          {"role": "user", "content": prompt}
      ],
      temperature=temp
      )
    scores = response.choices[0].message.content

  elif judge_model in ["opus", "haiku", "sonnet"]:
    message = client_anth.messages.create(
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        model = models[judge_model],
        )
    scores = message.content[0].text
  elif judge_model == 'cohere':
    response = co.chat(
    model="command-r-plus",
    message= prompt
      )
    scores = response.text
  elif judge_model == 'gemini':
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    scores = response.text
  else:
    message = client_togeth.chat.completions.create(
        model=models[judge_model],
        temperature=temp,
        messages=[{"role": "user", "content": prompt}],
    )
    scores = message.choices[0].message.content

  return scores