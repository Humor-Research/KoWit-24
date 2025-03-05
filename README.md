# KoWit-24 

[ Paper ](https://arxiv.org/abs/2503.01510)|[ Slides ](#)|[ Dataset ](https://huggingface.co/datasets/Humor-Research/KoWit-24)|[ Prompts ](https://smith.langchain.com/hub/humor-research)

## Overview

We present KoWit-24, a dataset with fine-grained annotation of wordplay in 2,700 Russian news headlines. KoWit-24 annotations include the presence of wordplay, its type, wordplay anchors, and words/phrases the wordplay refers to.

## Content

- [Overview](#Overview)
- [Content](#Content)
- [Dataset](#Dataset)
  - [Description](#Description)
  - [Key features](#Key-features)
  - [How to load and use](#How-to-load-and-use)
- [Repository structure](#Repository-structure)
- [Experiments](#Experiments)
  - [Wordplay detection](#Wordplay-detection)
  - [Wordplay interpretation](#Wordplay-interpretation)
    - [Automatic interpretation evaluation](#Automatic-interpretation-evaluation)
  - [Table of results](#Table-of-results)
  - [How to run an experiment with another LLM](#How-to-run-an-experiment-with-another-LLM)
- [Citation](#Citation)


## Dataset

### Description

Dataset contains manual annotated 2,700 headlines, of which 1,340 contained wordplay, so the dataset is almost perfectly balanced. For all headlines identified as containing wordplay, annotations were generated, including the original substring, a reference string, and a link to Wikipedia or Wiktionary. Distribution of headlines by wordplay type can be seen in Table 1. The most frequent wordplay mechanism in our dataset appeared to be the modification of existing well-known phrases – collocations, idiomatic expressions, or named entities.

|                 | Wordplay type       | #   | AAL  | Links |
|-----------------|---------------------|-----|------|-------|
| Puns            | Polysemy            | 190 | 1.51 |       |
| Puns            | Homonymy            | 26  | 1.57 |       |
| Puns            | Phonetic similarity | 98  | 1.80 |       |
| Transformations | Collocation         | 423 | 2.64 | 126   |
| Transformations | Idiom               | 177 | 3.43 | 118   |
| Transformations | Reference           | 353 | 3.73 | 214   |
|                 | Nonce word          | 185 |      |       |
|                 | Oxymoron            | 48  |      |       |

*Table 1. Wordplay types, average anchor length in words (AAL), and wiki links in KOWIT-24*

### Key features

Unlike the majority of existing humor collections of canned jokes, KoWit-24 provides wordplay contexts – each headline is accompanied by the news lead and summary. The most common type of wordplay in the dataset is the transformation of collocations, idioms, and named entities – the mechanism that has been underrepresented in previous humor datasets. Moreover the dataset contains manually created annotations that provide information about what the wordplay refers to. Incorporating this annotation into the dataset **enables automated evaluation** of the large language model’s wordplay interpretations.

Dataset entry example:
```
{'article_url': 'https://www.kommersant.ru/doc/5051268',
 'date': '2021-10-27',
 'headline': 'Диалектический пиломатериализм',
 'is_wordplay': True,
 'lead': 'Цены на фанеру и доски начали снижаться вслед за спросом',
 'summary': 'Пиломатериалы и лесопромышленная продукция начинают дешеветь по '
            'мере завершения строительного сезона. По мнению аналитиков и '
            'некоторых участников рынка, этому способствует сокращение спроса '
            'на фоне летнего всплеска цен. И хотя на некоторые продукты, '
            'например OSB, цена упала уже на треть, она все еще вдвое выше '
            'уровня конца прошлого года. До конца года можно ожидать '
            'стабилизации цен, полагают участники рынка, но едва ли '
            'возвращения к средним многолетним значениям.'},
'annotations': [{'end_index': 30,
                  'headline_substring': 'Диалектический пиломатериализм',
                  'reference_string': 'Диалектический материализм',
                  'reference_url': 'https://ru.wikipedia.org/wiki/Диалектический_материализм',
                  'start_index': 0,
                  'wordplay_type': 'Reference'},
                 {'end_index': 30,
                  'headline_substring': 'пиломатериализм',
                  'reference_string': ['материализм', 'пиломатериалы'],
                  'reference_url': ['', ''],
                  'start_index': 15,
                  'wordplay_type': 'Nonce word'}]
```

### How to load and use

```python
from datasets import load_dataset
data_files = {"test": "dataset.csv", "dev": "dev_dataset.csv"}
dataset = load_dataset("Humor-Research/KoWit-24", data_files=data_files)

```

## Repository structure

TODO

## Experiments

For the experiments, we allocated 200 records (100 from each class) for the development set, making sure that all wordplay types were represented. Thus, the test set contains 2,500 headlines (1,290 with and 1,310 without wordplay). We experimented with two tasks – wordplay detection and wordplay interpretation. We employed five LLMs: GPT-4o, Mistral NeMo 12B, YandexGPT4, GigaChat Lite, and GigaChat Max. 

### Wordplay detection

### Wordplay interpretation

#### Automatic interpretation evaluation

### Table of results

| Model         | Detection with simple prompt, P/R | Detection with extended prompt, P/R | Interpretation manual, R | Interpretation auto, R |
|---------------|-----------------------------------|-------------------------------------|--------------------------|------------------------|
| GigaChat Lite | 0.50 / 0.50                       | 0.53 / 0.72                         | 0.11                     | 0.19                   |
| GigaChat Max  | 0.62 / 0.48                       | 0.68 / 0.59                         | 0.28                     | 0.28                   |
| YandexGPT4    | 0.83 / 0.10                       | 0.76 / 0.24                         | 0.20                     | 0.22                   |
| Mistral Nemo  | 0.00 / 0.00                       | 0.00 / 0.00                         | 0.24                     | 0.30                   |
| GPT-4o        | 0.62 / 0.81                       | 0.65 / 0.88                         | 0.48                     | 0.43                   |

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

```
@misc{baranov2025kowit24richlyannotateddataset,
      title={KoWit-24: A Richly Annotated Dataset of Wordplay in News Headlines}, 
      author={Alexander Baranov and Anna Palatkina and Yulia Makovka and Pavel Braslavski},
      year={2025},
      eprint={2503.01510},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.01510}, 
}

```