import json
import pandas as pd

# File paths
HUMAN_JSON_FILE = "pan24-generative-authorship-news-train/human.jsonl"
MACHINE_JSON_FILE = "pan24-generative-authorship-news-train/machines/vicgalle-gpt2-open-instruct-v1.jsonl"
#MACHINE_JSON_FILE2 = "pan24-generative-authorship-news-train/machines/meta-llama-llama-2-70b-chat-hf.jsonl"
TRAINING_DATA_CSV = "training_data.csv"

# Step 1: Load the JSON files
def load_json(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error decoding line in {file_path}: {e}")
    return data


# Step 2: Combine data and assign labels
def prepare_training_data(human_file, machine_file, output_csv):
    # Load human and machine datasets
    human_data = load_json(human_file)
    machine_data = load_json(machine_file)
   
    # Convert to DataFrame
    human_df = pd.DataFrame(human_data)
    machine_df = pd.DataFrame(machine_data)

    # Add labels: 0 for human, 1 for machine
    human_df['label'] = 0
    machine_df['label'] = 1

    # Combine datasets
    combined_df = pd.concat([human_df, machine_df], ignore_index=True)
    
    # Shuffle the data
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    combined_df.to_csv(output_csv, index=False)
    print(f"Training data saved to {output_csv}")

# Main workflow
prepare_training_data(HUMAN_JSON_FILE, MACHINE_JSON_FILE, TRAINING_DATA_CSV)
