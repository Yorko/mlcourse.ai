#!/usr/bin/env python
# coding: utf-8

# <img src="https://habrastorage.org/webt/ia/m9/zk/iam9zkyzqebnf_okxipihkgjwnw.jpeg" />
#     
# **<center>[mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course** </center><br>
# Author: [Yury Kashnitskiy](https://yorko.github.io) (@yorko). [mlcourse.ai](https://mlcourse.ai) is powered by [OpenDataScience (ods.ai)](https://ods.ai/) © 2017—2022

# # <center>Assignment #6. Task</center><a class="tocSkip">
# ## <center> Beating benchmarks in "How good is your Medium article?"</center><a class="tocSkip">
#     
# [Competition](https://www.kaggle.com/c/how-good-is-your-medium-article). The task is to beat "Assignment 6 baseline" (~1.45 Public LB score). You can refer to [this simple Ridge baseline](https://www.kaggle.com/kashnitsky/ridge-countvectorizer-baseline?rvi=1).
# 
# -----

# In[1]:


import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
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


# Extract features `content`, `published`, `title` and `author`, write them to separate files for train and test sets.

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


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# Download the [competition data](https://www.kaggle.com/c/how-good-is-your-medium-article/data) and place it where it's convenient for you. You can modify the path to data below.

# In[5]:


PATH_TO_DATA = "../../_static/data/assignment6/"  # modify this if you need to


# In[6]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# **Add the following groups of features:**
#     - Tf-Idf with article content (ngram_range=(1, 2), max_features=100000 but you can try adding more)
#     - Tf-Idf with article titles (ngram_range=(1, 2), max_features=100000 but you can try adding more)
#     - Time features: publication hour, whether it's morning, day, night, whether it's a weekend
#     - Bag of authors (i.e. One-Hot-Encoded author names)

# In[7]:


X_train_content_sparse = csr_matrix(np.empty([10, 100000]))  # change this
X_train_title_sparse = csr_matrix(np.empty([10, 100000]))  # change this
X_train_author_sparse = csr_matrix(np.empty([10, 100000]))  # change this
X_train_time_features_sparse = csr_matrix(np.empty([10, 5]))  # change this

X_test_content_sparse = csr_matrix(np.empty([5, 100000]))  # change this
X_test_title_sparse = csr_matrix(np.empty([5, 100000]))  # change this
X_test_author_sparse = csr_matrix(np.empty([5, 100000]))  # change this
X_test_time_features_sparse = csr_matrix(np.empty([5, 5]))  # change this


# In[8]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# **Join all sparse matrices.**

# In[9]:


X_train_sparse = hstack(
    [
        X_train_content_sparse,
        X_train_title_sparse,
        X_train_author_sparse,
        X_train_time_features_sparse,
    ]
).tocsr()


# In[10]:


X_test_sparse = hstack(
    [
        X_test_content_sparse,
        X_test_title_sparse,
        X_test_author_sparse,
        X_test_time_features_sparse,
    ]
).tocsr()


# **Read train target and split data for validation.**

# In[11]:


train_target = pd.read_csv(
    os.path.join(PATH_TO_DATA, "train_log1p_recommends.csv"), index_col="id"
)
y_train = train_target["log_recommends"].values


# In[12]:


train_part_size = int(0.7 * train_target.shape[0])
X_train_part_sparse = X_train_sparse[:train_part_size, :]
y_train_part = y_train[:train_part_size]
X_valid_sparse = X_train_sparse[train_part_size:, :]
y_valid = y_train[train_part_size:]


# **Train a simple Ridge model and check MAE on the validation set.**

# In[13]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# **Train the same Ridge with all available data, make predictions for the test set and form a submission file.**

# In[14]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# In[15]:


ridge_test_pred = np.empty([34645, 1])  # change this


# In[16]:


def write_submission_file(
    prediction,
    filename,
    path_to_sample=os.path.join(PATH_TO_DATA, "sample_submission.csv"),
):
    submission = pd.read_csv(path_to_sample, index_col="id")

    submission["log_recommends"] = prediction
    submission.to_csv(filename)


# In[17]:


write_submission_file(
    ridge_test_pred, os.path.join(PATH_TO_DATA, "assignment6_medium_submission.csv")
)


# **Now's the time for dirty Kaggle hacks. Form a submission file with all zeros. Make a submission. What do you get if you think about it? How is it going to help you with modifying your predictions?**

# In[18]:


write_submission_file(
    np.zeros_like(ridge_test_pred),
    os.path.join(PATH_TO_DATA, "medium_all_zeros_submission.csv"),
)


# **Modify predictions in an appropriate way (based on your all-zero submission) and make a new submission.**

# In[19]:


ridge_test_pred_modif = ridge_test_pred
# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# In[20]:


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
