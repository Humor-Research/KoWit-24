import os
import json

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat


def predict_yes_or_not_gigachat(system_prompt, user_prompt):

    messages = [
        SystemMessage(
            content=system_prompt
        ),
        HumanMessage(
            content=user_prompt
        )
    ]

    result = llm.invoke(messages).content

    return str(result).lower()


def predict(row):
    result = None
    if row["gigachat_max_pred"] is None:
        try:
            result = predict_yes_or_not_gigachat(row["system_prompt"], row["user_prompt"])
        except:
            result = None
    else:
        return row["gigachat_max_pred"]
    
    return result


# Configs

PROJECT_PATH = "/home/alex/paper-2025-anonymous-submission/"
TARGET_FILE = "wordplay_interpretation_gigachat_max_predictions_extended.json"
TARGET_FILE_PATH = os.path.join(PROJECT_PATH, f"Data/predictions/{TARGET_FILE}")

# Load token

with open(os.path.join(PROJECT_PATH, "api_credentials.json")) as f:
    configs = json.load(f)
token = configs["gigachat_token"]

# Load data

df = pd.read_json(
    os.path.join(PROJECT_PATH, "Data/processed_data/dataset_wordplay_interpretation_propmts_extended.json"),
    orient="index"
)

if os.path.exists(TARGET_FILE_PATH):
    df_result = pd.read_json(
        TARGET_FILE_PATH,
        orient="index"
    )
else:
    df_result = df.copy()
    df_result["gigachat_max_pred"] = None

# Load token

with open(os.path.join(PROJECT_PATH, "api_credentials.json")) as f:
    configs = json.load(f)
token = configs["gigachat_token"]


print("Rows without predictions:", df_result["gigachat_max_pred"].isna().sum())

llm = GigaChat(
    credentials=token,
    verify_ssl_certs=False,
    model="GigaChat-Max",
    profanity_check=False,
    max_tokens=2048,
    temperature=0.1
)

df_result["gigachat_max_pred"] = df_result.progress_apply(predict, axis=1)

# save results

df_result.to_json(
    TARGET_FILE_PATH,
    orient="index",
    force_ascii=False
)