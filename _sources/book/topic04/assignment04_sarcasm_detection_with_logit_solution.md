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

(assignment04_solution)=


# Assignment 4 (demo). Solution. Sarcasm detection with logistic regression


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

### Part 1. Exploratory data analysis

Distribution of lengths for sarcastic and normal comments is almost the same.


```{code-cell} ipython3
train_df.loc[train_df["label"] == 1, "comment"].str.len().apply(np.log1p).hist(
    label="sarcastic", alpha=0.5
)
train_df.loc[train_df["label"] == 0, "comment"].str.len().apply(np.log1p).hist(
    label="normal", alpha=0.5
)
plt.legend();
```


```{code-cell} ipython3
from wordcloud import STOPWORDS, WordCloud
```


```{code-cell} ipython3
wordcloud = WordCloud(
    background_color="black",
    stopwords=STOPWORDS,
    max_words=200,
    max_font_size=100,
    random_state=17,
    width=800,
    height=400,
)
```

Word cloud are nice, but not very useful


```{code-cell} ipython3
plt.figure(figsize=(16, 12))
wordcloud.generate(str(train_df.loc[train_df["label"] == 1, "comment"]))
plt.imshow(wordcloud);
```


```{code-cell} ipython3
plt.figure(figsize=(16, 12))
wordcloud.generate(str(train_df.loc[train_df["label"] == 0, "comment"]))
plt.imshow(wordcloud);
```

Let's analyze whether some subreddits are more "sarcastic" on average than others


```{code-cell} ipython3
sub_df = train_df.groupby("subreddit")["label"].agg([np.size, np.mean, np.sum])
sub_df.sort_values(by="sum", ascending=False).head(10)
```


```{code-cell} ipython3
sub_df[sub_df["size"] > 1000].sort_values(by="mean", ascending=False).head(10)
```

The same for authors doesn't yield much insight. Except for the fact that somebody's comments were sampled - we can see the same amounts of sarcastic and non-sarcastic comments.


```{code-cell} ipython3
sub_df = train_df.groupby("author")["label"].agg([np.size, np.mean, np.sum])
sub_df[sub_df["size"] > 300].sort_values(by="mean", ascending=False).head(10)
```


```{code-cell} ipython3
sub_df = (
    train_df[train_df["score"] >= 0]
    .groupby("score")["label"]
    .agg([np.size, np.mean, np.sum])
)
sub_df[sub_df["size"] > 300].sort_values(by="mean", ascending=False).head(10)
```


```{code-cell} ipython3
sub_df = (
    train_df[train_df["score"] < 0]
    .groupby("score")["label"]
    .agg([np.size, np.mean, np.sum])
)
sub_df[sub_df["size"] > 300].sort_values(by="mean", ascending=False).head(10)
```

### Part 2. Training the model


```{code-cell} ipython3
# build bigrams, put a limit on maximal number of features
# and minimal word frequency
tf_idf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)
# multinomial logistic regression a.k.a softmax classifier
logit = LogisticRegression(C=1, n_jobs=4, solver="lbfgs", random_state=17, verbose=1)
# sklearn's pipeline
tfidf_logit_pipeline = Pipeline([("tf_idf", tf_idf), ("logit", logit)])
```


```{code-cell} ipython3
%%time
tfidf_logit_pipeline.fit(train_texts, y_train)
```


```{code-cell} ipython3
%%time
valid_pred = tfidf_logit_pipeline.predict(valid_texts)
```


```{code-cell} ipython3
accuracy_score(y_valid, valid_pred)
```

### Part 3. Explaining the model


```{code-cell} ipython3
def plot_confusion_matrix(
    actual,
    predicted,
    classes,
    normalize=False,
    title="Confusion matrix",
    figsize=(7, 7),
    cmap=plt.cm.Blues,
    path_to_save_fig=None,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(actual, predicted).T
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("Predicted label")
    plt.xlabel("True label")

    if path_to_save_fig:
        plt.savefig(path_to_save_fig, dpi=300, bbox_inches="tight")
```

Confusion matrix is quite balanced.


```{code-cell} ipython3
plot_confusion_matrix(
    y_valid,
    valid_pred,
    tfidf_logit_pipeline.named_steps["logit"].classes_,
    figsize=(8, 8),
)
```

Indeed, we can recognize some phrases indicative of sarcasm. Like "yes sure".


```{code-cell} ipython3
import eli5

eli5.show_weights(
    estimator=tfidf_logit_pipeline.named_steps["logit"],
    vec=tfidf_logit_pipeline.named_steps["tf_idf"],
)
```

So sarcasm detection is easy.
<img src="https://habrastorage.org/webt/1f/0d/ta/1f0dtavsd14ncf17gbsy1cvoga4.jpeg" />

### Part 4. Improving the model


```{code-cell} ipython3
subreddits = train_df["subreddit"]
train_subreddits, valid_subreddits = train_test_split(subreddits, random_state=17)
```

We'll have separate Tf-Idf vectorizers for comments and for subreddits. It's possible to stick to a pipeline as well, but in that case it becomes a bit less straightforward. [Example](https://stackoverflow.com/questions/36731813/computing-separate-tfidf-scores-for-two-different-columns-using-sklearn)


```{code-cell} ipython3
tf_idf_texts = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)
tf_idf_subreddits = TfidfVectorizer(ngram_range=(1, 1))
```

Do transformations separately for comments and subreddits.


```{code-cell} ipython3
%%time
X_train_texts = tf_idf_texts.fit_transform(train_texts)
X_valid_texts = tf_idf_texts.transform(valid_texts)
```


```{code-cell} ipython3
X_train_texts.shape, X_valid_texts.shape
```


```{code-cell} ipython3
%%time
X_train_subreddits = tf_idf_subreddits.fit_transform(train_subreddits)
X_valid_subreddits = tf_idf_subreddits.transform(valid_subreddits)
```


```{code-cell} ipython3
X_train_subreddits.shape, X_valid_subreddits.shape
```

Then, stack all features together.


```{code-cell} ipython3
from scipy.sparse import hstack

X_train = hstack([X_train_texts, X_train_subreddits])
X_valid = hstack([X_valid_texts, X_valid_subreddits])
```


```{code-cell} ipython3
X_train.shape, X_valid.shape
```

Train the same logistic regression.


```{code-cell} ipython3
logit.fit(X_train, y_train)
```


```{code-cell} ipython3
%%time
valid_pred = logit.predict(X_valid)
```


```{code-cell} ipython3
accuracy_score(y_valid, valid_pred)
```

As we can see, accuracy slightly increased.

## Links:
  - Machine learning library [Scikit-learn](https://scikit-learn.org/stable/index.html) (a.k.a. sklearn)
  - Kernels on [logistic regression](https://www.kaggle.com/kashnitsky/topic-4-linear-models-part-2-classification) and its applications to [text classification](https://www.kaggle.com/kashnitsky/topic-4-linear-models-part-4-more-of-logit), also a [Kernel](https://www.kaggle.com/kashnitsky/topic-6-feature-engineering-and-feature-selection) on feature engineering and feature selection
  - [Kaggle Kernel](https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle) "Approaching (Almost) Any NLP Problem on Kaggle"
  - [ELI5](https://github.com/TeamHG-Memex/eli5) to explain model predictions
