import argparse
import ast
from arch_builder import samre_arch, more_arch

# Function to average scores
def average_scores(scores_list):
    return [sum(scores) / len(scores) for scores in zip(*scores_list)]

# Judge 2 advocates with SAMRE architecture
def samre_experiment(model_pl, temperature, question, answer1, answer2, investment=0.1, n_rounds=4, n_juries=3):
    scores_lit, juries = samre_arch(model_pl, temperature, question, answer1, answer2, investment, n_rounds, n_juries)
    scores_list = [ast.literal_eval(item) for item in scores_lit]

    if scores_list:
        sc = average_scores(scores_list)
        print(f"Average scores: {sc}")
        result = "Answer 1 wins" if sc[0] > sc[1] else "Answer 2 wins"
    else:
        print("No valid scores returned from samre_arch")
        raise ValueError("No valid scores from samre_arch")
    
    print(f"The court result: {result}")
    
    if juries:
        scores_juries = "Answer 1 wins" if juries[0] > juries[1] else "Answer 2 wins"
    
    print(f"The court result based on the juries: {scores_juries}")

# Judge multiple advocates with MORE architecture
def more_experiment(model_pl, temperature, question, answer1, answer2, n_advocates=3, investment=0.1, n_rounds=1):
    scores_lit_many = more_arch(model_pl, temperature, question, answer1, answer2, n_advocates, investment, n_rounds)
    scores_list_many = [ast.literal_eval(item) for item in scores_lit_many]

    if scores_list_many:
        sc_many = scores_list_many[-1]
        print(f"Scores: {sc_many}")
        result_more = "Answer 1 wins" if sc_many[0] > sc_many[1] else "Answer 2 wins"
    else:
        print("No valid scores returned from more_arch")
        raise ValueError("No valid scores from more_arch")

    print(f"The court result: {result_more}")

# Main function to handle command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Run SAMRE or MORE experiments")
    
    parser.add_argument("--experiment", choices=['samre', 'more'], required=True, help="Choose the experiment to run: 'samre' or 'more'")
    parser.add_argument("--model_pl", required=True, help="The model platform to use")
    parser.add_argument("--temperature", type=float, required=True, help="The temperature setting for the model")
    parser.add_argument("--question", required=True, help="The question being debated")
    parser.add_argument("--answer1", required=True, help="The first answer to evaluate")
    parser.add_argument("--answer2", required=True, help="The second answer to evaluate")
    parser.add_argument("--investment", type=float, default=0.1, help="The maximum cost dedicated to the experiment")
    parser.add_argument("--n_rounds", type=int, default=4, help="The number of rounds for the SAMRE experiment")
    parser.add_argument("--n_juries", type=int, default=3, help="The number of juries for the SAMRE experiment")
    parser.add_argument("--n_advocates", type=int, default=3, help="The number of advocates for the MORE experiment")

    args = parser.parse_args()

    if args.experiment == 'samre':
        samre_experiment(args.model_pl, args.temperature, args.question, args.answer1, args.answer2, args.investment, args.n_rounds, args.n_juries)
    elif args.experiment == 'more':
        more_experiment(args.model_pl, args.temperature, args.question, args.answer1, args.answer2, args.n_advocates, args.investment, args.n_rounds)

if __name__ == "__main__":
    main()