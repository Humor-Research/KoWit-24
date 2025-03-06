# Imports 
import json
import time
import os

import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()

# Configs

PROJECT_PATH = "/home/alex/paper-2025-anonymous-submission/"
YANDEX_CLOUD_FOLDER = "b1gvds71t2fp1r39mft9"
YANDEX_CLOUD_MODEL = "yandexgpt/latest"
YANDEX_CLOUD_API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
TARGET_FILE = "wordplay_detection_yandex_gpt4_predictions_ria.json"
TARGET_FILE_PATH = os.path.join(PROJECT_PATH, f"Data/predictions/{TARGET_FILE}")

# Load data

df = pd.read_json(
    os.path.join(PROJECT_PATH, "Data/processed_data/ria_detection.json"),
    orient="index"
)

if os.path.exists(TARGET_FILE_PATH):
    df_result = pd.read_json(
        TARGET_FILE_PATH,
        orient="index"
    )
else:
    df_result = df.copy()
    df_result["yagpt_pred"] = None

# Load token

with open(os.path.join(PROJECT_PATH, "api_credentials.json")) as f:
    configs = json.load(f)
token = configs["yagpt_token"]


def predict_yes_or_not_yandex(user_prompt, system_prompt, token):

    prompt = {
            "modelUri": f"gpt://{YANDEX_CLOUD_FOLDER}/{YANDEX_CLOUD_MODEL}",
            "completionOptions": {
                "stream": False,
                "temperature": 0.1,
                "maxTokens": "128"
            },
            "messages": [
                {
                    "role": "system",
                    "text": system_prompt
                },
                {
                    "role": "user",
                    "text": user_prompt
                }
            ]
    }
    
    url = YANDEX_CLOUD_API_URL
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {token}"
    }
    
    try:
        response = requests.post(url, headers=headers, json=prompt)
        result = response.text
        result = json.loads(result)
    except:
        result = dict()
    
    if "result" in result:
        result = str.lower(result["result"]["alternatives"][0]["message"]["text"])
    else:
        result = None

    return result

def predict(row):
    if row["yagpt_pred"] is None:
        return predict_yes_or_not_yandex(row["user_prompt"], row["system_prompt"], token)
    else:
        return row["yagpt_pred"]
    
print("Rows without predictions:", df_result["yagpt_pred"].isna().sum())

df_result["yagpt_pred"] = df_result.progress_apply(predict, axis=1)

# save results

df_result.to_json(
    TARGET_FILE_PATH,
    orient="index"
)