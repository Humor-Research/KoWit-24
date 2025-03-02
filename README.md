# KoWit-24 

[ Paper ](#)|[ Slides ](#)|[ Dataset ](https://huggingface.co/datasets/Humor-Research/KoWit-24)|[ Prompts ](https://smith.langchain.com/hub/humor-research)

## Overview

We present KoWit-24, a dataset with fine-grained annotation of wordplay in 2,700 Russian news headlines. KoWit-24 annotations include the presence of wordplay, its type, wordplay anchors, and words/phrases the wordplay refers to.

## Content

- [Overview](#Overview)
- [Content](#Content)
- [Dataset](#Dataset)
  - [Description](#Description)
  - [Key features](#Key-features)
  - [How to load and use](#How-to-load-and-use)


## Dataset

### Description

### Key features

### How to load and use

```python
from datasets import load_dataset
data_files = {"test": "dataset.csv", "dev": "dev_dataset.csv"}
dataset = load_dataset("Humor-Research/KoWit-24", data_files=data_files)

```

## Repository structure

## Experiments

### Wordplay detection


### Wordplay interpretation


#### Automatic interpretation evaluation


### How to run an experiment with another LLM

To facilitate the evaluation of alternative large language models (LLMs) for detection and interpretation tasks, the prompts utilized in the experiments have been made available on the LangChain Hub, while the corresponding data have been deposited on the HuggingFace Hub.

Example:
```python
# Imports
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain import hub

# Load model
model_path = hf_hub_download(repo_id="Vikhrmodels/Vikhr-Llama-3.2-1B-instruct-GGUF",
                             filename="Vikhr-Llama-3.2-1B-Q4_K_M.gguf",
                             local_dir=".")

llm = LlamaCpp(
        model_path=model_path,
        n_ctx=2048,
        temperature=0.1,
        top_p=0.9,
        max_tokens=256
)

# Load prompt
prompt = hub.pull("humor-research/wordplay_detection")

# Load dataset
data_files = {"test": "dataset.csv", "dev": "dev_dataset.csv"}
dataset = load_dataset("Humor-Research/KoWit-24", data_files=data_files)

# Invoke LLM
predicted = list()
for example in dataset["test"]:
    task = prompt.format(
        headline=example["headline"],
        lead=example["lead"]
    )
    predicted.append(
        llm.invoke(task)
    )
    break
```

## Citation