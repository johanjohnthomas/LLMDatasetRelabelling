import os
import pandas as pd
import requests

# Filepaths for your CSV files
input_csv = 'Suicide_Detection.csv'
output_csv = 'DeepSeekR1DistillQwen7b_Suicide_Detection.csv'

# How often to save intermediate progress (e.g., every 1,000 rows)
CHECKPOINT_INTERVAL = 1000

# LMStudio/QWEN API endpoint (based on the curl example)
api_url = "http://localhost:1234/v1/chat/completions"


def classify_text(text):
    """Classify text using DeepSeek-R1-Distill-Qwen-7B in chat format."""
    payload = {
        "model": "deepseek-r1-distill-qwen-7b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a text classification assistant. The user will provide some text, "
                    "and you must classify it strictly as 'suicide ideation' or 'not suicide ideation'."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Text to classify:\n{text}\n\n"
                    "Answer strictly with either 'suicide ideation' or 'not suicide ideation'."
                )
            }
        ],
        "temperature": 0,
        "max_tokens": 5,
        "stream": False
    }

    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()

        # The response structure should have 'choices' -> [
        #   { 'message': { 'role': ..., 'content': ... } }
        # ]
        json_data = response.json()
        classification = json_data['choices'][0]['message']['content'].strip().lower()

        return classification
    except requests.exceptions.RequestException as e:
        print(f"Error with the API: {e}")
        return None


def process_csv(input_csv, output_csv):
    """Process the CSV, classify new entries, and track changes with checkpointing."""
    
    # --- Step 1: Load or initialize the DataFrame ---
    if os.path.exists(output_csv):
        # If a checkpoint file already exists, load it
        df = pd.read_csv(output_csv)
    else:
        # No checkpoint file - read the original data
        df = pd.read_csv(input_csv)

    # --- Step 2: Ensure required columns exist ---
    if 'New_Class' not in df.columns:
        df['New_Class'] = None

    if 'Changed' not in df.columns:
        df['Changed'] = False

    # Reclassified = 1 means this row is done; 0 means not yet classified
    if 'Reclassified' not in df.columns:
        df['Reclassified'] = 0

    # --- Step 3: Classify only rows that haven't been classified yet (Reclassified == 0) ---
    reclass_count = 0  # tracks how many rows we've classified this run

    for index, row in df.iterrows():
        # Skip rows that have already been classified
        if row['Reclassified'] == 1:
            continue

        original_class = str(row['class']).lower()  # handle possible NaN, so convert to str
        text = row['text']
        new_class = classify_text(text)

        if new_class:
            df.at[index, 'New_Class'] = new_class
            df.at[index, 'Changed'] = (original_class != new_class)
            df.at[index, 'Reclassified'] = 1
            reclass_count += 1

            # --- Step 4: Check if we should save a checkpoint ---
            if reclass_count % CHECKPOINT_INTERVAL == 0:
                df.to_csv(output_csv, index=False)
                print(f"Checkpoint saved after {reclass_count} classifications.")

    # --- Step 5: Save final updated CSV after all (or remaining) rows ---
    df.to_csv(output_csv, index=False)
    print(f"Updated dataset saved to {output_csv}. Total newly classified rows: {reclass_count}")


if __name__ == "__main__":
    process_csv(input_csv, output_csv)
