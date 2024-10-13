import pickle
from evaluate import load

BASE_DIR = "..."
def evaluate_perplexity(model_id):
    data_path = f"{BASE_DIR}/eval_ppl_data_7_domains.pkl"
    
    with open(data_path, 'rb') as file:
        data = pickle.load(file)

    domain_sizes = len(data) // 7  
    results = {}

    perplexity = load("perplexity", module_type="metric")

    for i, domain in enumerate(['amazon', 'arxiv', 'github', 'pubmed', 'owtc', 'books', 'wiki']):
        domain_data_start = i * domain_sizes
        domain_data_end = (i + 1) * domain_sizes
        domain_data = []
        for index in range(domain_data_start, domain_data_end):
            domain_data.append(data[index])

        eval_result = perplexity.compute(predictions=domain_data, model_id=model_id)
        results[domain] = eval_result['mean_perplexity']

    return results

import os
import pandas as pd
def evaluate_all_models():
    model_directory = f"{BASE_DIR}/training_gpt2_large"
    results_table = pd.DataFrame()

    for model_folder in os.listdir(model_directory)[::-1]:
        model_id = os.path.join(model_directory, model_folder)
        print(model_id)
        try:
            model_results = evaluate_perplexity(model_id)
            model_results['model_id'] = model_folder
            results_table = pd.concat([results_table, pd.DataFrame([model_results])], ignore_index=True)
        except Exception as e:
            print(f"Error processing model {model_folder}: {e}")
        print(results_table)
    return results_table

# Run evaluation
results_df = evaluate_all_models()
print(results_df)
results_df.to_csv(".../ppl.csv")