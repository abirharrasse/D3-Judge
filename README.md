# LLM-Based-Judging-Architectures

<p align="center">
  <img src="images/main_figure.jpg" alt="Architectures Diagram" width="90%" />
</p>

## Background
As large language models (LLMs) evolve, evaluating their outputs becomes increasingly complex. Traditional methods, such as human assessments, are often costly and inconsistent, while automated metrics may fail to capture the nuances of LLM performance.

To address these challenges, we propose a novel framework that uses LLMs as **advocates** in a dynamic, **courtroom-inspired** system. This approach involves LLMs acting as advocates, judges, and juries to provide a comprehensive evaluation of model outputs.

This framework integrates decision theory, bounded rationality, and economic incentives to offer a robust evaluation method. Subsequent sections will detail the architecture and experiments validating its effectiveness.



## Data

### Dataset

This project uses the `lmsys/mt_bench_human_judgments` dataset from Hugging Face.

### Preprocessing

The script `src/preprocess_mt_bench.py` processes the raw data into an Excel file (`data/mt_bench_human_judgments.xlsx`) with the following columns:

- **Question**: Aggregated user questions.
- **Response_A**: Responses from Model A.
- **Response_B**: Responses from Model B.
- **Model_A_Score**: Binary score for Model A (1 for a win, 0 for a loss).
- **Model_B_Score**: Binary score for Model B (1 for a win, 0 for a loss).

Run the preprocessing with:
```bash
python data/preprocess_mt_bench.py
```
## Installation and Setup

Our agentic architecture is built on **MetaGPT**, a framework designed to efficiently manage interactions between agents using shared memory.

To use the MetaGPT framework, follow these steps:

1. **Clone the Repository**
   
First, clone the repository and navigate to the relevant directory:
```bash
git clone https://github.com/abirharrasse/LLM-Judging-Architectures  && cd LLM-Judging-Architectures/MetaGPT_LLM_advocates
```
3. **Install Dependencies**
   
Install the necessary packages:
```bash
pip install --upgrade -e .
pip install together -q
```
3. Initialize Configuration

Set up the configuration file for the MetaGPT framework:
```bash
metagpt --init-config
```
Before running this command, ensure you're in the correct directory:
```bash
import os
os.chdir('/content/LLM-Judging-Architectures/MetaGPT_LLM_advocates')
print(os.getcwd())
```
4. Set API Keys
   
Set the required API keys to run your experiments:
```bash
os.environ['OPENAI_API_KEY'] = 'sk-'
os.environ['TOGETHER_API_KEY'] = '' 
os.environ['CLAUDE_API_KEY'] = ''
os.environ['GEMINI_API_KEY'] = ''
os.environ['COHERE_API_KEY'] = ''
```
## Running the evaluation architectures

#### Single Advocate Multi-Round Evaluation (SAMRE): 
To run the **SAMRE** evaluation framework:
Begin by accessing the directory `LLM_Judging_Architectures`:
```bash
import os
os.chdir('/content/LLM_Judging_Architectures')
print(os.getcwd())
```
then call the `samre_experiment` function with the appropriate parameters: 
```bash
! python utils/experiments.py --experiment samre --model_pl "mistral" --temperature 0.7 --question "What is the impact of AI on healthcare?" --answer1 "AI can improve diagnostic accuracy." --answer2 "AI might introduce bias in diagnosis." --investment 0.1 --n_rounds 4 --n_juries 3

```
Where:

- **model**: Select one of the models supported by our framework, accessible via Together or OpenAI.
- **question**: The question to which answer1 and answer2 respond.
- **answer1** and **answer2**: The answers to be evaluated.
- **investment**: The maximum cost allocated for the experiment.
- **n_rounds**: The number of evaluation rounds to conduct.
- **n_juries**: The number of juries involved in the evaluation.



#### Multi-Advocate One-Round Evaluation (MORE) 

To run the **MORE** evaluation framework:
Begin by accessing the directory `LLM_Judging_Architectures`:
```bash
import os
os.chdir('/content/LLM_Judging_Architectures')
print(os.getcwd())
```
then call the `more_experiment` function with the appropriate parameters: 
```bash
! python utils/experiments.py --experiment more --model_pl "mistral" --temperature 0.7 --question "How does AI influence education?" --answer1 "AI personalizes learning." --answer2 "AI could limit creativity." --n_advocates 2 --investment 3  --n_rounds 1


```
Where:
- **n_advocates**: The number of advocates involved in the evaluation.
- **n_rounds**: The number of evaluation rounds to conduct.


## Full Dataset Evaluation

To execute our architectures on the entire `mt-bench` dataset and ompare it to the `basemodel.py`, use the following code:
```bash
!python utils/main.py
```
