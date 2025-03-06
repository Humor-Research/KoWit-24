import os
import json

import pandas as pd
from transformers import pipeline


# Configs

PROJECT_PATH = "/home/ambaranov/paper-2025-anonymous-submission"
TARGET_FILE = "wordplay_interpretation_mistral_nemo_predictions_extended_ria.json"
TARGET_FILE_PATH = os.path.join(PROJECT_PATH, f"Data/predictions/{TARGET_FILE}")
MODEL_PATH = "/home/ambaranov/paper-2024/LLM predict/Mistral/Mistral-Nemo-Instruct-2407"

# Load model

chatbot = pipeline(
    "text-generation",
    model=MODEL_PATH,
    max_new_tokens=2048,
    temperature=0.3,
    device="cuda"
)

# Load data

df = pd.read_json(
    os.path.join(PROJECT_PATH, "Data/processed_data/ria_interpretation_extended.json"),
    orient="index"
)

all_preds = list()

for i, row in df.iterrows():
        messages = [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["user_prompt"]},
        ]

        all_preds.append(
              chatbot(messages)
        )

df["mistral_nemo_preds"] = all_preds

df.to_json(
    TARGET_FILE_PATH, orient="index", force_ascii=False
)