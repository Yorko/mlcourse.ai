---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(assignment04_extra)=


# Assignment 4 (demo). Sarcasm detection with logistic regression


```{figure} /_static/img/ods_stickers.jpg
```

Author: [Yury Kashnitsky](https://www.linkedin.com/in/festline/). All content is distributed under the [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.


**Same assignment as a [Kaggle Notebook](https://www.kaggle.com/kashnitsky/a4-demo-sarcasm-detection-with-logit) + [solution](https://www.kaggle.com/kashnitsky/a4-demo-sarcasm-detection-with-logit-solution).**


We'll be using the dataset from the [paper](https://arxiv.org/abs/1704.05579) "A Large Self-Annotated Corpus for Sarcasm" with >1mln comments from Reddit, labeled as either sarcastic or not. A processed version can be found on Kaggle in a form of a [Kaggle Dataset](https://www.kaggle.com/danofer/sarcasm).

Sarcasm detection is easy.


```{figure} /_static/img/yeah.jpeg
:width: 400px
```


```{code-cell} ipython3
# some necessary imports
import os
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
```

The dataset is a bit too large to be stored on GitHub. So, we download it from Google Drive. Alternatively, you can download the "train-balanced-sarcasm.csv.zip" file from [Kaggle](https://www.kaggle.com/danofer/sarcasm?select=train-balanced-sarcasm.csv.zip) ([alternative link](https://drive.google.com/file/d/1KbBdJaEY8RF4GXzoihWgH0RdqoVBZ_oi/view?usp=sharing)) and place the file in a convenient place, then modify the `DATA_PATH` below accordingly.

```{code-cell} ipython3
def download_file_from_gdrive(file_url, filename, out_path='../../_static', overwrite=False):
    """
    Downloads a file from GDrive given an URL
    :param file_url: a string formated as https://drive.google.com/uc?id=<file_id>
    :param: the desired file name
    :param: the desired folder where the file will be downloaded to
    :param overwrite: whether to overwrite the file if it already exists
    """
    file_exists = os.path.exists(f'{out_path}/{filename}')

    if (file_exists and overwrite) or (not file_exists):
    	os.system(f'gdown {file_url} -O {out_path}/{filename}')
```

```{code-cell} ipython3
FILE_URL = 'https://drive.google.com/uc?id=1KbBdJaEY8RF4GXzoihWgH0RdqoVBZ_oi'
FILE_NAME = 'train-balanced-sarcasm.csv.zip'
DATA_PATH = '../../_static/data/'

download_file_from_gdrive(file_url=FILE_URL, filename= FILE_NAME, out_path=DATA_PATH)

train_df = pd.read_csv(DATA_PATH + "train-balanced-sarcasm.csv.zip")
```

```{code-cell} ipython3
train_df.head()
```


```{code-cell} ipython3
train_df.info()
```

Some comments are missing, so we drop the corresponding rows.


```{code-cell} ipython3
train_df.dropna(subset=["comment"], inplace=True)
```

We notice that the dataset is indeed balanced


```{code-cell} ipython3
train_df["label"].value_counts()
```

We split data into training and validation parts.


```{code-cell} ipython3
train_texts, valid_texts, y_train, y_valid = train_test_split(
    train_df["comment"], train_df["label"], random_state=17
)
```

## Tasks:
1. Analyze the dataset, make some plots. This [Kernel](https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc) might serve as an example
2. Build a Tf-Idf + logistic regression pipeline to predict sarcasm (`label`) based on the text of a comment on Reddit (`comment`).
3. Plot the words/bigrams which a most predictive of sarcasm (you can use [eli5](https://github.com/TeamHG-Memex/eli5) for that)
4. (optionally) add subreddits as new features to improve model performance. Apply here the Bag of Words approach, i.e. treat each subreddit as a new feature.

## Links:
  - Machine learning library [Scikit-learn](https://scikit-learn.org/stable/index.html) (a.k.a. sklearn)
  - Kernels on [logistic regression](https://www.kaggle.com/kashnitsky/topic-4-linear-models-part-2-classification) and its applications to [text classification](https://www.kaggle.com/kashnitsky/topic-4-linear-models-part-4-more-of-logit), also a [Kernel](https://www.kaggle.com/kashnitsky/topic-6-feature-engineering-and-feature-selection) on feature engineering and feature selection
  - [Kaggle Kernel](https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle) "Approaching (Almost) Any NLP Problem on Kaggle"
  - [ELI5](https://github.com/TeamHG-Memex/eli5) to explain model predictions
