import yaml
import os

openai_api_key = os.getenv("OPENAI_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")

models = {"opus": "claude-3-opus-20240229", "haiku": "claude-3-haiku-20240307", "sonnet": "claude-3-sonnet-20240229", "llama3_8": "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
          "llama3_70": 'meta-llama/Llama-3-70b-chat-hf', 'llama3.1_8': "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
          "mistral": "mistralai/Mixtral-8x22B-Instruct-v0.1", "Qwen": "Qwen/Qwen2-72B-Instruct",
          "Yi": "zero-one-ai/Yi-34B-Chat", "gemma": "google/gemma-7b-it", "cohere": "command-r-plus", "gemini": 'gemini-pro',
          "gpt-4-turbo": "gpt-4-turbo-preview", "gpt-4o": "gpt-4o", "gpt-3.5-turbo":"gpt-3.5-turbo"}



def initiate_model(model_name, temp, provider):
    config = {}
    if provider == "together":
        config = {
            "llm": {
                "api_type": "together",
                "model": models[model_name],
                "base_url": "https://api.together.xyz",
                "api_key": together_api_key,
                "temperature": temp
            }
        }
    elif provider == "openai":
        config = {
            "llm": {
                "api_type": "openai",
                "model": models[model_name],
                "base_url": "https://api.openai.com/v1",
                "api_key": openai_api_key
            }
        }

    # Ensure the directory exists
    os.makedirs(os.path.expanduser("~/.metagpt"), exist_ok=True)

    # Write the configuration to the file
    with open(os.path.expanduser("~/.metagpt/config2.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    
    
def average_scores(tuples_list):
    if not tuples_list:
        return ()

    # Initialize sums for each position in the tuples
    num_elements = len(tuples_list[0])
    sums = [0] * num_elements

    # Sum up all elements for each position
    for tpl in tuples_list:
        for i in range(num_elements):
            sums[i] += tpl[i]

    # Calculate averages
    num_tuples = len(tuples_list)
    averages = tuple(s / num_tuples for s in sums)

    return averages