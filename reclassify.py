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

        # Optionally enforce strictly 'suicide ideation' or 'not suicide ideation'
        if classification not in ["suicide ideation", "not suicide ideation"]:
            classification = None

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
        print(f"Loaded existing progress from '{output_csv}'.")
    else:
        # No checkpoint file - read the original data
        df = pd.read_csv(input_csv)
        print(f"Starting new classification from '{input_csv}'.")

    # --- Step 2: Ensure required columns exist ---
    if 'New_Class' not in df.columns:
        df['New_Class'] = None

    if 'Changed' not in df.columns:
        df['Changed'] = False

    if 'Reclassified' not in df.columns:
        df['Reclassified'] = 0

    # --- Step 3: Find how many rows still need classification ---
    rows_to_classify = df[df['Reclassified'] == 0].shape[0]
    print(f"Rows needing classification: {rows_to_classify}")

    if rows_to_classify == 0:
        print("All rows have already been classified. Exiting.")
        return

    reclass_count = 0  # Number of rows classified this run
    total_classified_now = 0  # Will track how many we've done in this run, to show progress

    # --- Step 4: Classify only rows that haven't been classified yet (Reclassified == 0) ---
    for index, row in df.iterrows():
        if row['Reclassified'] == 1:
            continue  # skip rows already done

        original_class = str(row['class']).lower()  # handle possible NaN
        text = row['text']
        new_class = classify_text(text)

        if new_class:
            df.at[index, 'New_Class'] = new_class
            df.at[index, 'Changed'] = (original_class != new_class)
            df.at[index, 'Reclassified'] = 1
            reclass_count += 1
            total_classified_now += 1

            # Print progress after each classification (you could change this to every 100, for less verbosity)
            rows_remaining = rows_to_classify - total_classified_now
            print(f"Classified row index {index}. "
                  f"New_Class: {new_class} | Changed: {df.at[index, 'Changed']} | "
                  f"Remaining: {rows_remaining}")

            # --- Step 5: Check if we should save a checkpoint ---
            if reclass_count % CHECKPOINT_INTERVAL == 0:
                df.to_csv(output_csv, index=False)
                print(f"[Checkpoint] Saved after {reclass_count} classifications in this run.")

    # --- Step 6: Save final updated CSV after all (or remaining) rows ---
    df.to_csv(output_csv, index=False)
    print(f"Updated dataset saved to {output_csv}.")
    print(f"Total newly classified rows in this run: {reclass_count}")


if __name__ == "__main__":
    process_csv(input_csv, output_csv)
