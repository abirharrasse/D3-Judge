from openai import OpenAI
import os
import pandas as pd
import re
import ast
import random
from together import Together
from tqdm import tqdm
import json
from util_adv import initiate_model, average_scores
from basemodel import judge_answers_other, judge_answers
from arch_builder import samre_arch, more_arch




os.environ['OPENAI_API_KEY'] = "sk-aa"
os.environ['TOGETHER_API_KEY'] = ''
os.environ['CLAUDE_API_KEY'] = ''
os.environ['GEMINI_API_KEY'] = ''
os.environ['COHERE_API_KEY'] = ''


models = {"opus": "claude-3-opus-20240229", "haiku": "claude-3-haiku-20240307", "sonnet": "claude-3-sonnet-20240229", "llama3_8": "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
          "llama3_70": 'meta-llama/Llama-3-70b-chat-hf', 'llama3.1_8': "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
          "mistral": "mistralai/Mixtral-8x22B-Instruct-v0.1", "Qwen": "Qwen/Qwen2-72B-Instruct",
          "Yi": "zero-one-ai/Yi-34B-Chat", "gemma": "google/gemma-7b-it", "cohere": "command-r-plus", "gemini": 'gemini-pro',
          "gpt-4-turbo": "gpt-4-turbo-preview", "gpt-4o": "gpt-4o", "gpt-3.5-turbo":"gpt-3.5-turbo"}

models_dict = {"llama3_8": "together", "llama3_70": "together", "llama3.1_8": "together",  "mistral": "together", "Qwen": "together", "Yi": "together", "gemma": 'together', "gpt-4-turbo": "openai",
               "gpt-3.5-turbo": "openai", "gpt-4o": "openai", "opus": "claude", "haiku": "claude", "sonnet": "claude"}

initiate_model("llama3_8", 0, models_dict["llama3_8"])
mt = pd.read_excel("data/mt_bench_human_judgments.xlsx")



my_models = ["llama3_8", "mistral", "Qwen", "gemma", "llama3.1_8", "llama3_70", "Yi"]
human_scores = []
llm_scores = []
llm_scores_many = []
llm_scores_sim = []
scores_juries = []
llm_scores_sim_other= []
sample_size = len(mt)
all_indices = list(range(len(mt)))
results = {}
# Initialize a progress bar
pbar = tqdm(total=len(mt)*len(my_models), desc="Processing", unit="sample")

for model_pl in my_models:
  print(f"Model played: {model_pl}")
  i = 0
  while len(llm_scores) < sample_size and i < len(all_indices):
      idx = all_indices[i]
      try:
          question = mt.iloc[idx].iloc[0]
          answer1 = mt.iloc[idx].iloc[1]
          answer2 = mt.iloc[idx].iloc[2]

          human_scores.append(mt.iloc[idx].iloc[3])

          # Judge answers
          text = judge_answers(model_pl, 0, question, answer1, answer2)
          print(f"Judge answers text: {text}")

          scores_match = re.search(r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)', text)
          if scores_match:
              scores_sim = scores_match.group()
              print(f"Matched scores: {scores_sim}")
              sc_sim = ast.literal_eval(scores_sim)
              llm_scores_sim.append(1 if sc_sim[0] > sc_sim[1] else 0)
          else:
              print("No scores found in the text")
              raise ValueError("No scores found in judge_answers output")

          print(f"LLM scores sim: {llm_scores_sim}")





          text_other = judge_answers_other(model_pl, 0, question, answer1, answer2)
          print(f"Judge answers text other: {text_other}")

          scores_match_other = re.search(r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)', text_other)
          if scores_match_other:
              scores_sim_other = scores_match_other.group()
              print(f"Matched scores other: {scores_sim_other}")
              sc_sim_other = ast.literal_eval(scores_sim_other)
              llm_scores_sim_other.append(1 if sc_sim_other[0] > sc_sim_other[1] else 0)
          else:
              print("No scores found in the text other")
              raise ValueError("No scores found in judge_answers output other")

          print(f"LLM scores sim other: {llm_scores_sim_other}")

  #         # Judge 2 advocates
          scores_lit, juries = samre_arch(model_pl, 0, question, answer1, answer2, investment=0.1, n_rounds=4, n_juries=3)
          scores_list = [ast.literal_eval(item) for item in scores_lit]
          if scores_list:
              sc = average_scores(scores_list)
              print(f"Average scores: {sc}")
              llm_scores.append(1 if sc[0] > sc[1] else 0)
          else:
              print("No valid scores returned from judge2advocates_modified")
              raise ValueError("No valid scores from judge2advocates_modified")

          print(f"LLM scores: {llm_scores}")

          if juries:
            scores_juries.append(1 if juries[0] > juries[1] else 0)

          print(f"LLM juries: {scores_juries}")

          #many advocates
          scores_lit_many = more_arch(model_pl, 0, question, answer1, answer2, n_advocates=3, investment=0.1, n_rounds=1)
          scores_list_many = [ast.literal_eval(item) for item in scores_lit_many]

          if scores_list_many:
              sc_many = scores_list_many[-1]
              print(f"Scores: {sc}")
              llm_scores_many.append(1 if sc[0] > sc[1] else 0)
          else:
              print("No valid scores returned from judge_multi_advocates")
              raise ValueError("No valid scores from judge_multi_advocates")

          print(f"LLM scores many: {llm_scores_many}")

          # Update progress bar
          pbar.update(1)
      except Exception as e:
          print(f"Error processing row {idx}: {e}")
          # Remove the appended scores if an error occurred
          if len(human_scores) > len(llm_scores):
              human_scores.pop()
          if len(llm_scores_sim) > len(llm_scores):
              llm_scores_sim.pop()
              llm_scores_sim_other.pop()

      i += 1
      # # Calculate accuracies
      count = sum(llm_score == human_score for llm_score, human_score in zip(llm_scores, human_scores))
      accuracy = count / len(llm_scores)
      print(f"Accuracy (judge2advocates): {accuracy:.2f}")


      count_many = sum(llm_score == human_score for llm_score, human_score in zip(llm_scores_many, human_scores))
      accuracy_many = count_many / len(llm_scores_many)
      print(f"Accuracy (many advocates): {accuracy_many:.2f}")

      count_sim = sum(llm_score == human_score for llm_score, human_score in zip(llm_scores_sim, human_scores))
      accuracy_sim = count_sim / len(llm_scores_sim)
      print(f"Accuracy (judge_answers): {accuracy_sim:.2f}")


      count_sim_other = sum(llm_score == human_score for llm_score, human_score in zip(llm_scores_sim_other, human_scores))
      accuracy_sim_other = count_sim_other / len(llm_scores_sim_other)
      print(f"Accuracy (judge_answers_other): {accuracy_sim_other:.2f}")


      count_juries = sum(llm_score == human_score for llm_score, human_score in zip(scores_juries, human_scores))
      accuracy_juries = count_juries / len(scores_juries)
      print(f"Accuracy (juries): {accuracy_juries:.2f}")

  results[model_pl] = {"accuracy_basemodel1":accuracy_sim,  "accuracy_basemodel2": accuracy_sim_other, "accuracy_judge": accuracy , "accuracy_judge&juries": accuracy_juries,  "accuracy_many_advocates": accuracy_many}
# Close the progress bar
pbar.close()


print("Experiment Finished")
print(results)

# File path for the JSON file
json_file_path = '/content/MetaGPT/experiment_results.json'

try:
    # Read existing data
    with open(json_file_path, 'r') as file:
        existing_data = json.load(file)
except FileNotFoundError:
    existing_data = {}

existing_data.update(results)

# Write updated data back to file
with open(json_file_path, 'w') as file:
    json.dump(existing_data, file, indent=4)

print(f"Results appended to {json_file_path}")