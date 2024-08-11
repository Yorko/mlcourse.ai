#!/usr/bin/env python
# coding: utf-8

# <img src="https://habrastorage.org/webt/ia/m9/zk/iam9zkyzqebnf_okxipihkgjwnw.jpeg" />
#     
# **<center>[mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course** </center><br>
# Author: [Yury Kashnitskiy](https://yorko.github.io) (@yorko). [mlcourse.ai](https://mlcourse.ai) is powered by [OpenDataScience (ods.ai)](https://ods.ai/) © 2017—2022

# # <center>Assignment #6. Solution</center><a class="tocSkip">
# ## <center> Beating benchmarks in "How good is your Medium article?"</center><a class="tocSkip">
#     
# [Competition](https://www.kaggle.com/c/how-good-is-your-medium-article). The task is to beat "Assignment 6 baseline". You can refer to [this simple Ridge baseline](https://www.kaggle.com/kashnitsky/ridge-countvectorizer-baseline?rvi=1).
# 
# -----

# In[1]:


import json
import os
import pickle
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Lasso, Ridge, RidgeCV, SGDRegressor
from sklearn.metrics import mean_absolute_error
from tqdm.notebook import tqdm


# The following code will help to throw away all HTML tags from an article content.

# In[2]:


from html.parser import HTMLParser


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return "".join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


# Supplementary function to read a JSON line without crashing on escape characters.

# In[3]:


def read_json_line(line=None):
    result = None
    try:
        result = json.loads(line)
    except Exception as e:
        # Find the offending character index:
        idx_to_replace = int(str(e).split(" ")[-1].replace(")", ""))
        # Remove the offending character:
        new_line = list(line)
        new_line[idx_to_replace] = " "
        new_line = "".join(new_line)
        return read_json_line(line=new_line)
    return result


# Parse JSON and extract some features.

# In[4]:


def extract_features_and_write(path_to_data, inp_filename, is_train=True):

    features = ["content", "published", "title", "author"]
    prefix = "train" if is_train else "test"
    feature_files = [
        open(
            os.path.join(path_to_data, "{}_{}.txt".format(prefix, feat)),
            "w",
            encoding="utf-8",
        )
        for feat in features
    ]

    with open(
        os.path.join(path_to_data, inp_filename), encoding="utf-8"
    ) as inp_json_file:

        for line in tqdm(inp_json_file):
            json_data = read_json_line(line)
            for i, feat in enumerate(features):
                if feat == "published":
                    info = json_data[feat]["$date"]
                elif feat == "author":
                    info = json_data[feat]["twitter"]
                    if info:
                        info = info.replace("\n", " ").replace("@", " ")
                    else:
                        info = ""
                elif feat == "content" or feat == "title":
                    info = json_data[feat].replace("\n", " ").replace("\r", " ")
                    info = strip_tags(info)
                feature_files[i].write(info + "\n")


# Download the [competition data](https://www.kaggle.com/c/how-good-is-your-medium-article/data) and place it where it's convenient for you. You can modify the path to data below.

# In[5]:


PATH_TO_DATA = "../../_static/data/assignment6/"  # modify this if you need to


# In[6]:


get_ipython().run_cell_magic('time', '', 'extract_features_and_write(PATH_TO_DATA, "train.json", is_train=True)\n')


# In[7]:


get_ipython().run_cell_magic('time', '', 'extract_features_and_write(PATH_TO_DATA, "test.json", is_train=False)\n')


# **Tf-Idf with article content.**

# In[8]:


tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=100000)


# In[9]:


get_ipython().run_cell_magic('time', '', 'with open(\n    os.path.join(PATH_TO_DATA, "train_content.txt"), encoding="utf-8"\n) as input_train_file:\n    X_train_content_sparse = tfidf_vectorizer.fit_transform(input_train_file)\n\nwith open(\n    os.path.join(PATH_TO_DATA, "test_content.txt"), encoding="utf-8"\n) as input_test_file:\n    X_test_content_sparse = tfidf_vectorizer.transform(input_test_file)\n')


# In[10]:


X_train_content_sparse.shape, X_test_content_sparse.shape


# **Tf-Idf with titles.**

# In[11]:


get_ipython().run_cell_magic('time', '', 'tfidf_vectorizer_title = TfidfVectorizer(ngram_range=(1, 3), max_features=100000)\n\nwith open(\n    os.path.join(PATH_TO_DATA, "train_title.txt"), encoding="utf-8"\n) as input_train_file:\n    X_train_title_sparse = tfidf_vectorizer_title.fit_transform(input_train_file)\n\nwith open(\n    os.path.join(PATH_TO_DATA, "test_title.txt"), encoding="utf-8"\n) as input_test_file:\n    X_test_title_sparse = tfidf_vectorizer_title.transform(input_test_file)\n')


# In[12]:


X_train_title_sparse.shape, X_test_title_sparse.shape


# **Add time features.**

# In[13]:


def add_time_features(path_to_publication_time_file):

    df = pd.read_csv(path_to_publication_time_file, names=["time"])
    df["time"] = df["time"].apply(
        lambda t: pd.to_datetime(t.replace("T", " ").replace("Z", ""))
    )
    df["hour"] = df["time"].apply(lambda ts: ts.hour)
    df["month"] = df["time"].apply(lambda ts: ts.month)

    df["weekend"] = (
        df["time"]
        .apply(lambda ts: ts.weekday() == 5 or ts.weekday() == 6)
        .astype("int")
    )

    df["day"] = ((df["hour"] >= 12) & (df["hour"] <= 18)).astype("int")
    df["morning"] = ((df["hour"] >= 7) & (df["hour"] <= 11)).astype("int")
    df["night"] = ((df["hour"] >= 0) & (df["hour"] <= 5)).astype("int")

    cols = ["day", "morning", "night", "month", "weekend"]
    X_time_features_sparse = csr_matrix(df[cols].values)

    return X_time_features_sparse


# In[14]:


get_ipython().run_cell_magic('time', '', 'X_train_time_features_sparse = add_time_features(\n    os.path.join(PATH_TO_DATA, "train_published.txt")\n)\nX_test_time_features_sparse = add_time_features(\n    os.path.join(PATH_TO_DATA, "test_published.txt")\n)\n')


# In[15]:


X_train_time_features_sparse.shape, X_test_time_features_sparse.shape


# **Add authors.**

# In[16]:


get_ipython().run_cell_magic('time', '', 'author_train = pd.read_csv(\n    os.path.join(PATH_TO_DATA, "train_author.txt"),\n    names=["author"],\n    skip_blank_lines=False,\n)\nauthor_train = pd.get_dummies(author_train)\n\nauthor_test = pd.read_csv(\n    os.path.join(PATH_TO_DATA, "test_author.txt"),\n    names=["author"],\n    skip_blank_lines=False,\n)\nauthor_test = pd.get_dummies(author_test)\n\nunique_authors_train = list(set(author_train.columns) - set(author_test.columns))\nunique_authors_test = list(set(author_test.columns) - set(author_train.columns))\n\nauthor_test = author_test.drop(unique_authors_test, axis=1)\nauthor_train = author_train.drop(unique_authors_train, axis=1)\n\nX_train_author_sparse = csr_matrix(author_train.values)\nX_test_author_sparse = csr_matrix(author_test.values)\n')


# In[17]:


X_train_author_sparse.shape, X_test_author_sparse.shape


# **Join all sparse matrices.**

# In[18]:


X_train_sparse = hstack(
    [
        X_train_content_sparse,
        X_train_title_sparse,
        X_train_author_sparse,
        X_train_time_features_sparse,
    ]
).tocsr()


# In[19]:


X_test_sparse = hstack(
    [
        X_test_content_sparse,
        X_test_title_sparse,
        X_test_author_sparse,
        X_test_time_features_sparse,
    ]
).tocsr()


# In[20]:


X_train_sparse.shape, X_test_sparse.shape


# **Read train target and split data for validation.**

# In[23]:


train_target = pd.read_csv(
    os.path.join(PATH_TO_DATA, "train_log1p_recommends.csv"), index_col="id"
)
y_train = train_target["log_recommends"].values


# In[24]:


train_part_size = int(0.7 * train_target.shape[0])
X_train_part_sparse = X_train_sparse[:train_part_size, :]
y_train_part = y_train[:train_part_size]
X_valid_sparse = X_train_sparse[train_part_size:, :]
y_valid = y_train[train_part_size:]


# **Train a simple Ridge model and check MAE on the validation set.**

# In[25]:


get_ipython().run_cell_magic('time', '', 'ridge_reg = Ridge(random_state=17)\nridge_reg.fit(X_train_part_sparse, y_train_part)\nridge_valid_pred = ridge_reg.predict(X_valid_sparse)\nprint(mean_absolute_error(y_valid, ridge_valid_pred))\n')


# Plot distributions of tagets and predictions for the validation set.

# In[26]:


plt.hist(y_valid, bins=30, alpha=0.5, color="red", label="true", range=(0, 10))
plt.hist(
    ridge_valid_pred, bins=30, alpha=0.5, color="green", label="pred", range=(0, 10)
)
plt.legend();


# **Train the same Ridge with all available data, make predictions for the test set and form a submission file.**

# In[27]:


get_ipython().run_cell_magic('time', '', 'ridge_reg.fit(X_train_sparse, y_train)\nridge_test_pred = ridge_reg.predict(X_test_sparse)\n')


# In[28]:


def write_submission_file(
    prediction,
    filename,
    path_to_sample=os.path.join(PATH_TO_DATA, "sample_submission.csv"),
):
    submission = pd.read_csv(path_to_sample, index_col="id")

    submission["log_recommends"] = prediction
    submission.to_csv(filename)


# In[29]:


write_submission_file(
    ridge_test_pred, os.path.join(PATH_TO_DATA, "assignment6_medium_submission.csv")
)


# **With this you get ~ 1.73877 on public leaderboard.**
# 
# **Now's the time for dirty Kaggle hacks. Form a submission file with all zeroes. Make a submission. What do you get if you think about? How is it going to help you with modifying your predictions?**

# In[ ]:


write_submission_file(
    np.zeros_like(ridge_test_pred),
    os.path.join(PATH_TO_DATA, "medium_all_zeros_submission.csv"),
)


# In[ ]:


mean_test_target = 4.33328


# **Calculate mean target for the test set.**

# In[ ]:


y_train.mean()


# **Now we now that we need to add the difference between test and train mean targets.**

# In[ ]:


ridge_test_pred_modif = ridge_test_pred + mean_test_target - y_train.mean()


# In[ ]:


write_submission_file(
    ridge_test_pred_modif,
    os.path.join(PATH_TO_DATA, "assignment6_medium_submission_with_hack.csv"),
)


# That's it for the assignment. In case you'd like to try some more ideas for improvement:
# 
# - Engineer good features, this is the key to success. Some simple features will be based on publication time, authors, content length and so on
# - You may not ignore HTML and extract some features from there
# - You'd better experiment with your validation scheme. You should see a correlation between your local improvements and LB score
# - Try TF-IDF, ngrams, Word2Vec and GloVe embeddings
# - Try various NLP techniques like stemming and lemmatization
# - Tune hyperparameters. In our example, we've left only 50k features and used C=1 as a regularization parameter, this can be changed
# - SGD and Vowpal Wabbit will train much faster
# - Play around with blending and/or stacking. An intro is given in [this Kernel](https://www.kaggle.com/kashnitsky/ridge-and-lightgbm-simple-blending) by @yorko 
# - And neural nets of course. We don't cover them in this course byt still transformer-based architectures will likely perform well in such types of tasks
