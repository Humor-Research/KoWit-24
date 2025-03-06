import os
import json

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


# Configs

PROJECT_PATH = "/home/ambaranov/paper-2025-anonymous-submission"
TARGET_FILE = "wordplay_interpretation_gigachat_lite_predictions.json"
TARGET_FILE_PATH = os.path.join(PROJECT_PATH, f"Data/predictions/{TARGET_FILE}")
MODEL_PATH = "/home/ambaranov/GigaChat-20B-A3B-instruct"

# Load model

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained("/home/ambaranov/paper-2025-anonymous-submission/Notebooks/wordplay_interpretation/GigaChat-Lite")

# Load data

df = pd.read_json(
    os.path.join(PROJECT_PATH, "Data/processed_data/dataset_wordplay_interpretation_propmts.json"),
    orient="index"
)

all_preds = list()

for i, row in df.iterrows():
        messages = [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["user_prompt"]},
        ]

        input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        outputs = model.generate(input_tensor.to(model.device))

        all_preds.append(
              tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=False)
        )

df["gigachat_lite_preds"] = all_preds

df.to_json(
    TARGET_FILE_PATH, orient="index", force_ascii=False
)