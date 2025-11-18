# 01_run_openai.ipynb

import openai
import pandas as pd
import time
import os


openai.api_key = os.getenv("OPENAI_API_KEY")
# Load test data
file_path = "../data/data_for_processing.xlsx"
df = pd.read_excel(file_path)


# Define a function to generate prompt
PROMPT_TEMPLATE = """
Rate the grammatical acceptability of the following sentence on a scale from 1 (completely unacceptable) to 5 (fully acceptable):

"{sentence}"

Just respond with a number from 1 to 5.
"""

def query_openai(sentence, model="gpt-4"):
    prompt = PROMPT_TEMPLATE.format(sentence=sentence)
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a linguist evaluating grammaticality."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0
        )
        reply = response["choices"][0]["message"]["content"].strip()
        return reply
    except Exception as e:
        print(f"Error: {e}")
        return None

# Run evaluation (limit to N examples for now)
results = []
N = 10  # Change this to len(df) when ready

for i, row in df.iloc[:N].iterrows():
    sent = row["sentence"]
    score = query_openai(sent)
    print(f"{i}: {sent} -> {score}")
    results.append(score)
    time.sleep(1.2)  # to avoid hitting rate limits

# Store results
out_df = df.iloc[:N].copy()
out_df["gpt-4_score"] = results
out_df.to_csv("../results/scores_openai.csv", index=False)
print("Done.")
