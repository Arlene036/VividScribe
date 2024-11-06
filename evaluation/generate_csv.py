import os
import json
import pandas as pd

# Define the folder path where the JSON files are stored
folder_path = '../results'  # Replace with the actual folder path

# Initialize an empty list to store the results
results = []

# Initialize an empty list to store the results
results = []

# Iterate through each JSON file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        
        # Load the JSON content
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract model name (from filename without extension) and metrics for valor_nonverbal and vast
        model_name = filename.replace('.json', '')
        if 'valor_nonverbal' in data and 'vast' in data:
            results.append({
                'Model': model_name,
                'Bleu_1_valor_nonverbal': data['valor_nonverbal'].get('Bleu_1', None),
                'Bleu_2_valor_nonverbal': data['valor_nonverbal'].get('Bleu_2', None),
                'Bleu_3_valor_nonverbal': data['valor_nonverbal'].get('Bleu_3', None),
                'Bleu_4_valor_nonverbal': data['valor_nonverbal'].get('Bleu_4', None),
                'METEOR_valor_nonverbal': data['valor_nonverbal'].get('METEOR', None),
                'ROUGE_L_valor_nonverbal': data['valor_nonverbal'].get('ROUGE_L', None),
                'CIDEr_valor_nonverbal': data['valor_nonverbal'].get('CIDEr', None),
                'Bleu_1_vast': data['vast'].get('Bleu_1', None),
                'Bleu_2_vast': data['vast'].get('Bleu_2', None),
                'Bleu_3_vast': data['vast'].get('Bleu_3', None),
                'Bleu_4_vast': data['vast'].get('Bleu_4', None),
                'METEOR_vast': data['vast'].get('METEOR', None),
                'ROUGE_L_vast': data['vast'].get('ROUGE_L', None),
                'CIDEr_vast': data['vast'].get('CIDEr', None)
            })

# Create a DataFrame from the collected results
df = pd.DataFrame(results)

# Sort the models by unimodal -> multimodal -> competitive
def custom_sort(model_name):
    if model_name.startswith("unimodal"):
        return 1
    elif model_name.startswith("multimodal"):
        return 2
    elif model_name.startswith("competitive"):
        return 3
    else:
        return 4  # Default case if the naming doesn't match

# Apply the custom sort and sort the dataframe
df['sort_key'] = df['Model'].apply(custom_sort)
df_sorted = df.sort_values(by=['sort_key', 'Model']).drop(columns='sort_key')


# Save the DataFrame to a CSV file
csv_file_path = '../results/model_performance.csv'
df_sorted.to_csv(csv_file_path, index=False)