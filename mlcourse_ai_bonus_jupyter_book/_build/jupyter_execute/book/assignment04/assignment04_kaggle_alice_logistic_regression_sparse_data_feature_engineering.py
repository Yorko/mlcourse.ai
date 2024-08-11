#!/usr/bin/env python
# coding: utf-8

# <img src="https://habrastorage.org/webt/ia/m9/zk/iam9zkyzqebnf_okxipihkgjwnw.jpeg" />
#     
# **[mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course**
# Authors: [Yury Kashnitsky](https://yorko.github.io) (@yorko), and Yury Isakov. Edited by Anna Tarelina (@feuerengel), and Kolchenko Sergey (@KolchenkoSergey). [mlcourse.ai](https://mlcourse.ai) is powered by [OpenDataScience (ods.ai)](https://ods.ai/) © 2017—2022

# # <center> Assignment #4. Task </center> <a class="tocSkip">
# ## <center>  Kaggle Competition. User Identification with Logistic Regression <br>(beating baselines in the "Alice" competition) </center>  <a class="tocSkip">
# 
#     
# Today we will reproduce a couple of baselines in the  Kaggle Inclass competition ["Catch Me If You Can: Intruder Detection through Webpage Session Tracking"](https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2) (a.k.a. "Alice") and discuss ideas for further improvement. You'll sense the spirit of Kaggle competitions: brainstorming model or validation scheme improvements, adding new features, implementing new ideas in code, climbing the leaderboard – that's exciting! And that's very useful as well, especially when you're only starting your ML journey. In this assignment, we'll also work with sparse matrices (which is often the only way to work with some types of data, from the computation perspective), train Logistic Regression models, and practice feature engineering. 
# 
# Prior to working on the assignment, you'd better check out the corresponding course material:
#  1. [Classification, Decision Trees and k Nearest Neighbors](https://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/jupyter_english/topic03_decision_trees_kNN/topic3_decision_trees_kNN.ipynb?flush_cache=true), the same as an interactive web-based [Kaggle Notebook](https://www.kaggle.com/kashnitsky/topic-3-decision-trees-and-knn) (basics of machine learning are covered here)
#  1. Linear classification and regression in 5 parts: 
#     - [ordinary least squares](https://www.kaggle.com/kashnitsky/topic-4-linear-models-part-1-ols)
#     - [linear classification](https://www.kaggle.com/kashnitsky/topic-4-linear-models-part-2-classification)
#     - [regularization](https://www.kaggle.com/kashnitsky/topic-4-linear-models-part-3-regularization)
#     - [logistic regression: pros and cons](https://www.kaggle.com/kashnitsky/topic-4-linear-models-part-4-more-of-logit)
#     - [validation](https://www.kaggle.com/kashnitsky/topic-4-linear-models-part-5-validation)
#  1. You can also practice with demo assignments, which are simpler and already shared with solutions: 
#     - "Sarcasm detection with logistic regression": [assignment](https://www.kaggle.com/kashnitsky/a4-demo-sarcasm-detection-with-logit) + [solution](https://www.kaggle.com/kashnitsky/a4-demo-sarcasm-detection-with-logit-solution)
#     - "Linear regression as optimization": [assignment](https://www.kaggle.com/kashnitsky/a4-demo-linear-regression-as-optimization/edit) (solution cannot be officially shared as the assignment, also by @yorko, is a part of a Coursera specialization)
#     - "Exploring OLS, Lasso and Random Forest in a regression task": [assignment](https://www.kaggle.com/kashnitsky/a6-demo-linear-models-and-rf-for-regression) + [solution](https://www.kaggle.com/kashnitsky/a6-demo-regression-solution)
#  1. Alice baseline with logistic regression and "bag of sites", [Notebook](https://www.kaggle.com/kashnitsky/alice-logistic-regression-baseline)
#  1. Correct time-aware cross-validation scheme, more features, and hyperparameter optimization, [Notebook](https://www.kaggle.com/kashnitsky/correct-time-aware-cross-validation-scheme) 
#  1. Model validation in a competition, [Notebook](https://www.kaggle.com/kashnitsky/model-validation-in-a-competition) – this one reproduces a solution with **0.95055** Public LB ROC AUC and gives a lot of hints how to proceed with this competition.
#  1. Other [Notebooks](https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/code?competitionId=7173&sortBy=voteCount) in this competition. You can share yours as well, but not high-performing ones (Public LB ROC AUC shall be < 0.95055). Please do not spoil the competitive spirit. 
#  1. Two videos on logistic regression: [mlcourse.ai/video](https://mlcourse.ai/video)
#  
# **Your task is to:**
#   1. "Follow me". Complete the missing code and submit your answers via [the google form](https://forms.gle/SmawpchSerqhK4C9A). 
#   1. "Freeride". Come up with good features to beat the baselines "A4 baseline (10 credits)" (**0.95343** Public LB ROC-AUC, press "Load more" in the bottom of the [Leaderboard](https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/leaderboard) to actually see it) and "A4 strong baseline (20 credits)" (**0.95965** Public LB ROC-AUC). 
#      
# 
# *If you are sure that something is not 100% correct with the assignment/solution, please leave your feedback via the mentioned webform ↑*
# 
# -----

# ### Part 1. Follow me <a class="tocSkip">

# <img src='../../_static/img/followme_alice.png' width=50%>
# 
# *image credit [@muradosmann](https://www.instagram.com/muradosmann/?hl=en)*

# In[1]:


# Import libraries and set desired options
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sns.set()
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# **Problem description**
# 
# In this competition, we'll analyze the sequence of websites consequently visited by a particular person and try to predict whether this person is Alice or someone else. As a metric we will use [ROC AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic).

# **1. Data Downloading and Transformation**
# Register on [Kaggle](https://www.kaggle.com/), if you have not done it before.
# Go to the competition [page](https://inclass.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2) and download the data.
# 
# First, read the training and test sets. Then we'll explore the data in hand and do a couple of simple exercises. 

# In[2]:


PATH_TO_DATA = Path("../../_static/data/assignment4")


# In[3]:


# Read the training and test data sets, change paths if needed
times = ["time%s" % i for i in range(1, 11)]
# customize the paths if needed
train_df = pd.read_csv(
    PATH_TO_DATA / "train_sessions.csv.zip", index_col="session_id", parse_dates=times
)
test_df = pd.read_csv(
    PATH_TO_DATA / "test_sessions.csv.zip", index_col="session_id", parse_dates=times
)

# Sort the data by time
train_df = train_df.sort_values(by="time1")

# Look at the first rows of the training set
train_df.head(2)


# The training data set contains the following features:
# 
# - **site1** – id of the first visited website in the session
# - **time1** – visiting time for the first website in the session
# - ...
# - **site10** – id of the tenth visited website in the session
# - **time10** – visiting time for the tenth website in the session
# - **target** – target variable, 1 for Alice's sessions, and 0 for the other users' sessions
#     
# User sessions are chosen in the way that they are shorter than 30 min. long and contain no more than 10 websites. I.e. a session is considered over either if a user has visited 10 websites or if a session has lasted over 30 minutes.
# 
# There are some empty values in the table, it means that some sessions contain less than ten websites. Replace empty values with 0 and change columns types to integer. Also load the websites dictionary and check how it looks like:

# In[4]:


# Change site1, ..., site10 columns type to integer and fill NA-values with zeros
sites = ["site%s" % i for i in range(1, 11)]
train_df[sites] = train_df[sites].fillna(0).astype(np.uint16)
test_df[sites] = test_df[sites].fillna(0).astype(np.uint16)

# Load websites dictionary
with open(PATH_TO_DATA / "site_dic.pkl", "rb") as input_file:
    site_dict = pickle.load(input_file)

# Create dataframe for the dictionary
sites_dict = pd.DataFrame(
    list(site_dict.keys()), index=list(site_dict.values()), columns=["site"]
)
print(u"Websites total:", sites_dict.shape[0])
sites_dict.head()


# **2. Brief Exploratory Data Analysis**

# Before we start training models, we have to perform Exploratory Data Analysis ([EDA](https://en.wikipedia.org/wiki/Exploratory_data_analysis)). Today, we are going to perform a shorter version, but we will use other techniques as we move forward. Let's check which websites in the training data set are the most visited. As you can see, they are Google services and a bioinformatics website (a website with 'zero'-index is our missed values, just ignore it):

# In[5]:


# Top websites in the training data set
top_sites = (
    pd.Series(train_df[sites].values.flatten())
    .value_counts()
    .sort_values(ascending=False)
    .head(5)
)
print(top_sites)
sites_dict.loc[top_sites.drop(0).index]


# __<font color = 'red'> **Question 1:** </font>  What kind of websites does Alice visit the most?__
# - videohostings
# - social networks
# - torrent trackers
# - news

# In[6]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# Now let us look at the timestamps and try to characterize sessions as timeframes:

# In[7]:


# Create a separate dataframe where we will work with timestamps
time_df = pd.DataFrame(index=train_df.index)
time_df["target"] = train_df["target"]

# Find sessions' starting and ending
time_df["min"] = train_df[times].min(axis=1)
time_df["max"] = train_df[times].max(axis=1)

# Calculate sessions' duration in seconds
time_df["seconds"] = (time_df["max"] - time_df["min"]) / np.timedelta64(1, "s")

time_df.head()


# In order to perform the next task, generate descriptive statistics as you did in the first assignment.
# 
# *In the next question, we are using the notion of "approximately the same". To be strict, let's define it: $a$ is approximately the same as $b$ ($a \approx b $) if their difference is less than or equal to 5% of the maximum between $a$ and $b$, i.e. $a \approx b \leftrightarrow \frac{|a-b|}{max(a,b)} \leq 0.05$.*
# 
# __<font color = 'red'> **Question 2:** </font>  Select all correct statements:__
# 
# 
# - on average, Alice's session is shorter than that of other users
# - more than 1% of all sessions in the dataset belong to Alice
# - minimum and maximum duration of Alice's and other users' sessions are approximately the same
# - standard deviation of Alice's sessions duration is approximately the same as for non-Alice's sessions
# - less than a quarter of Alice's sessions are greater than or equal to 40 seconds

# In[8]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# In order to train our first model, we need to prepare the data. First of all, exclude the target variable from the training set. Now both training and test sets have the same number of columns, therefore aggregate them into one dataframe.  Thus, all transformations will be performed simultaneously on both training and test data sets. 
# 
# On the one hand, it leads to the fact that both data sets have one feature space (you don't have to worry that you forgot to transform a feature in some data sets). On the other hand, processing time will increase. 
# For the enormously large sets it might turn out that it is impossible to transform both data sets simultaneously (and sometimes you have to split your transformations into several stages only for train/test data set).
# In our case, with this particular data set, we are going to perform all the transformations for the whole united dataframe at once, and before training the model or making predictions we will just take its appropriate part.

# In[9]:


# Our target variable
y_train = train_df["target"]

# United dataframe of the initial data
full_df = pd.concat([train_df.drop("target", axis=1), test_df])

# Index to split the training and test data sets
idx_split = train_df.shape[0]


# For the very basic model, we will use only the visited websites in the session (but we will not take into account timestamp features). The point behind this data selection is: *Alice has her favorite sites, and the more often you see these sites in the session, the higher probability that this is Alice's session, and vice versa.*
# 
# Let us prepare the data, we will take only features `site1, site2, ... , site10` from the whole dataframe. Keep in mind that the missing values are replaced with zero. Here is how the first rows of the dataframe look like:

# In[10]:


# Dataframe with indices of visited websites in session
full_sites = full_df[sites]
full_sites.head()


# Sessions are sequences of website indices, and data in this representation is useless for machine learning method (just think, what happens if we switched all ids of all websites). 
# 
# According to our hypothesis (Alice has favorite websites), we need to transform this dataframe so each website has a corresponding feature (column) and its value is equal to number of this website visits in the session. It can be done in two lines:

# In[11]:


# sequence of indices
sites_flatten = full_sites.values.flatten()

# and the matrix we are looking for
# (make sure you understand which of the `csr_matrix` constructors is used here)
# a further toy example will help you with it
full_sites_sparse = csr_matrix(
    (
        [1] * sites_flatten.shape[0],
        sites_flatten,
        range(0, sites_flatten.shape[0] + 10, 10),
    )
)[:, 1:]


# In[12]:


full_sites_sparse.shape


# If you understand what just happened here, then you can skip the next passage (perhaps, you can handle logistic regression too?), If not, then let us figure it out.
# 
# **Important detour #1: Sparse Matrices**
# 
# Let us estimate how much memory it will require to store our data in the example above. Our united dataframe contains 336 thousand samples of 48 thousand integer features in each. It's easy to calculate the required amount of memory, roughly:
# 
# $$336\ K * 48\ K * 8\ bytes \approx 16* 10^9 * 8\ bytes = 128\ GB,$$
# 
# (that's the [exact](http://www.wolframalpha.com/input/?i=336358*48371*8+bytes) value). Obviously, ordinary mortals have no such volumes (strictly speaking, Python may allow you to create such a matrix, but it will not be easy to do anything with it). The interesting fact is that most of the elements of our matrix are zeros. If we count non-zero elements, then it will be about 1.8 million, i.е. slightly more than 10% of all matrix elements. Such a matrix, where most elements are zeros, is called sparse, and the ratio between the number of zero elements and the total number of elements is called the sparseness of the matrix.
# 
# For the work with such matrices you can use `scipy.sparse` library, check [documentation](https://docs.scipy.org/doc/scipy-0.18.1/reference/sparse.html) to understand what possible types of sparse matrices are, how to work with them and in which cases their usage is most effective. You can learn how they are arranged, for example, in Wikipedia [article](https://en.wikipedia.org/wiki/Sparse_matrix).
# Note, that a sparse matrix contains only non-zero elements, and you can get the allocated memory size like this (significant memory savings are obvious):

# In[13]:


# How much memory does a sparse matrix occupy?
print(
    "{} elements * {} bytes = {} bytes".format(
        full_sites_sparse.count_nonzero(), 8, full_sites_sparse.count_nonzero() * 8
    )
)
# Or just like this:
print("sparse_matrix_size = {} bytes".format(full_sites_sparse.data.nbytes))


# Let us explore how the matrix with the websites has been formed using a mini example. Suppose we have the following table with user sessions:
# 
# | id | site1 | site2 | site3 |
# |---|---|---|---|
# | 1 | 1 | 0 | 0 |
# | 2 | 1 | 3 | 1 |
# | 3 | 2 | 3 | 4 |
# 
# There are 3 sessions, and no more than 3 websites in each. Users visited four different sites in total (there are numbers from 1 to 4 in the table cells). And let us assume that the mapping is:
# 
#  1. vk.com
#  2. habrahabr.ru 
#  3. yandex.ru
#  4. ods.ai
# 
# If the user has visited less than 3 websites during the session, the last few values will be zero. We want to convert the original dataframe in a way that each session has a corresponding row which shows the number of visits to each particular site. I.e. we want to transform the previous table into the following form:
# 
# | id | vk.com | habrahabr.ru | yandex.ru | ods.ai |
# |---|---|---|---|---|
# | 1 | 1 | 0 | 0 | 0 |
# | 2 | 2 | 0 | 1 | 0 |
# | 3 | 0 | 1 | 1 | 1 |
# 
# 
# To do this, use the constructor: `csr_matrix ((data, indices, indptr))` and create a frequency table (see examples, code and comments on the links above to see how it works). Here we set all the parameters explicitly for greater clarity:

# In[14]:


# data, create the list of ones, length of which equal to the number of elements in the initial dataframe (9)
# By summing the number of ones in the cell, we get the frequency,
# number of visits to a particular site per session
data = [1] * 9

# To do this, you need to correctly distribute the ones in cells
# Indices - website ids, i.e. columns of a new matrix. We will sum ones up grouping them by sessions (ids)
indices = [1, 0, 0, 1, 3, 1, 2, 3, 4]

# Indices for the division into rows (sessions)
# For example, line 0 is the elements between the indices [0; 3) - the rightmost value is not included
# Line 1 is the elements between the indices [3; 6)
# Line 2 is the elements between the indices [6; 9)
indptr = [0, 3, 6, 9]

# Aggregate these three variables into a tuple and compose a matrix
# To display this matrix on the screen transform it into the usual "dense" matrix
csr_matrix((data, indices, indptr)).todense()


# As you might have noticed, there are not four columns in the resulting matrix (corresponding to number of different websites) but five. A zero column has been added, which indicates if the session was shorter (in our mini example we took sessions of three). This column is excessive and should be removed from the dataframe (do that yourself).
# 
# __<font color = 'red'> **Question 3:** </font> What is the sparseness of the matrix in our toy example?__
# 
# 
# 
# - 42%
# - 47%
# - 50%
# - 53%
# 
# 
# 

# In[15]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# Another benefit of using sparse matrices is that there are special implementations of both matrix operations and machine learning algorithms for them, which sometimes allows to significantly accelerate operations due to the data structure peculiarities. This applies to logistic regression as well. Now everything is ready to build our first model.
# 
# **3. Training the first model**
# 
# So, we have an algorithm and data for it. Let us build our first model, using [logistic regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) implementation from ` Sklearn` with default parameters. We will use the first 90% of the data for training (the training data set is sorted by time), and the remaining 10% for validation. Let's write a simple function that returns the quality of the model and then train our first classifier:

# In[16]:


def get_auc_lr_valid(X, y, C=1.0, seed=17, ratio=0.9):
    # Split the data into the training and validation sets
    idx = int(round(X.shape[0] * ratio))
    # Classifier training
    lr = LogisticRegression(C=C, random_state=seed, solver="liblinear").fit(
        X[:idx, :], y[:idx]
    )
    # Prediction for validation set
    y_pred = lr.predict_proba(X[idx:, :])[:, 1]
    # Calculate the quality
    score = roc_auc_score(y[idx:], y_pred)

    return score


# In[17]:


get_ipython().run_cell_magic('time', '', '# Select the training set from the united dataframe (where we have the answers)\nX_train = full_sites_sparse[:idx_split, :]\n\n# Calculate metric on the validation set\nprint(get_auc_lr_valid(X_train, y_train))\n')


# The first model demonstrated the quality  of 0.92 on the validation set. Let's take it as the first baseline and starting point. To make a prediction on the test data set **we need to train the model again on the entire training data set** (until this moment, our model used only part of the data for training), which will increase its generalizing ability:

# In[18]:


# Function for writing predictions to a file
def write_to_submission_file(
    predicted_labels, out_file, target="target", index_label="session_id"
):
    predicted_df = pd.DataFrame(
        predicted_labels,
        index=np.arange(1, predicted_labels.shape[0] + 1),
        columns=[target],
    )
    predicted_df.to_csv(out_file, index_label=index_label)


# In[19]:


# Train the model on the whole training data set
# Use random_state=17 for repeatability
# Parameter C=1 by default, but here we set it explicitly
lr = LogisticRegression(C=1.0, random_state=17, solver="liblinear").fit(
    X_train, y_train
)

# Make a prediction for test data set
X_test = full_sites_sparse[idx_split:, :]
y_test = lr.predict_proba(X_test)[:, 1]

# Write it to the file which could be submitted
write_to_submission_file(y_test, "baseline_1.csv")


# If you follow these steps and upload the answer to the competition [page](https://inclass.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2), you will get `ROC AUC = 0.90812` on the public leaderboard ("A4 baseline 1").
# 
# **4. Model Improvement: Feature Engineering**
# 
# Now we are going to improve the quality of our model by engineering new features. 

# Create a feature that will be a number in YYYYMM format from the date when the session was held, for example 201407 -- year 2014 and 7th month. Thus, we will take into account the monthly [linear trend](http://people.duke.edu/~rnau/411trend.htm) for the entire period of the data provided.

# In[20]:


# Dataframe for new features
full_new_feat = pd.DataFrame(index=full_df.index)

# Add start_month feature
full_new_feat["start_month"] = (
    full_df["time1"].apply(lambda ts: 100 * ts.year + ts.month).astype("float64")
)


# __<font color = 'red'> **Question 4:** </font>  Plot the graph of the number of Alice sessions versus the new feature, start_month. Choose the correct statement:__
# 
# 
# - Alice wasn't online at all for the entire period
# - From the beginning of 2013 to mid-2014, the number of Alice's sessions per month decreased
# - The number of Alice's sessions per month is generally constant for the entire period
# - From the beginning of 2013 to mid-2014, the number of Alice's sessions per month increased
# 
# *Hint: the graph will be more explicit if you treat `start_month` as a categorical ordinal variable*.

# In[21]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# In this way, we have an illustration and thoughts about the usefulness of the new feature, add it to the training sample and check the quality of the new model:

# In[22]:


# Add the new feature to the sparse matrix
tmp = full_new_feat[["start_month"]].values
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split, :], tmp[:idx_split, :]]))

# Compute the metric on the validation set
print(get_auc_lr_valid(X_train, y_train))


# The quality of the model has decreased significantly. We added a feature that definitely seemed useful to us, but its usage only worsened the model. Why did it happen?
# 
# **Important detour #2: is it necessary to scale features?**
# 
# Here we give an intuitive reasoning (a rigorous mathematical justification for one or another aspect in linear models you can easily find on the internet). Consider the features more closely: those of them that correspond to the number of visits to a particular web-site per session vary from 0 to 10. The feature `start_month` has a completely different range: from 201301 to 201412, this means the contribution of this variable is significantly greater than the others. It would seem that problem can be avoided if we put less weight in a linear combination of attributes in this case, but in our case logistic regression with regularization is used (by default, this parameter is `C = 1`), which penalizes the model the stronger the greater its weights are. Therefore, for linear methods with regularization, it is recommended to convert features to the same scale (you can read more about the regularization, for example, [here](https://habrahabr.ru/company/ods/blog/322076/)).
# 
# One way to do this is standardization: for each observation you need to subtract the average value of the feature and divide this difference by the standard deviation:
# 
# $$ x^{*}_{i} = \dfrac{x_{i} - \mu_x}{\sigma_x}$$
# 
# The following practical tips can be given:
# - It is recommended to scale features if they have essentially different ranges or different units of measurement (for example, the country's population is indicated in units, and the country's GNP in trillions)
# - Scale features if you do not have a reason/expert opinion to give a greater weight to any of them
# - Scaling can be excessive if the ranges of some of your features differ from each other, but they are in the same system of units (for example, the proportion of middle-aged people and people over 80 among the entire population)
# - If you want to get an interpreted model, then build a model without regularization and scaling (most likely, its quality will be worse)
# - Binary features (which take only values of 0 or 1) are usually left without conversion, (but)
# - If the quality of the model is crucial, try different options and select one where the quality is better
# 
# Getting back to `start_month`, let us rescale the new feature and train the model again. This time the quality has increased:

# In[23]:


# Add the new standardized feature to the sparse matrix
tmp = StandardScaler().fit_transform(full_new_feat[["start_month"]])
X_train = csr_matrix(hstack([full_sites_sparse[:idx_split, :], tmp[:idx_split, :]]))

# Compute metric on the validation set
print(get_auc_lr_valid(X_train, y_train))


# __<font color = 'red'> **Question 5:** </font>  Add to the training set a new feature "n_unique_sites" – the number of the unique web-sites in a session. Calculate how the quality on the validation set has changed__
# 
# 
# - It has decreased. It is better not to add a new feature.
# - It has not changed
# - It has decreased. The new feature should be scaled.
# - I am confused, and I do not know if it's necessary to scale a new feature.
# 
# *Tips: use the nunique() function from `pandas`. Do not forget to include the start_month in the set. Will you scale a new feature? Why?*

# In[24]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# __<font color = 'red'> **Question 6:** </font>  Add two new features: start_hour and morning. Calculate the metric. Which of these features gives an improvement?__
# 
# The `start_hour` feature is the hour at which the session started (from 0 to 23), and the binary feature `morning` is equal to 1 if the session started in the morning and 0 if the session started later (we assume that morning means `start_hour` is equal to 11 or less).
# 
# Will you scale the new features? Make your assumptions and test them in practice.
# 
# 
# - None of the features gave an improvement :(
# - `start_hour` feature gave an improvement, and `morning` did not
# - `morning` feature gave an improvement, and `start_hour` did not
# - Both features gave an improvement
# 
# *Tip: find suitable functions for working with time series data in [documentation](http://pandas.pydata.org/pandas-docs/stable/api.html). Do not forget to include the `start_month` feature.*

# In[25]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)
full_new_feat["morning"] = 0  # change this
full_new_feat["start_hour"] = 0  # change this


# **5. Regularization and Parameter Tuning**
# 
# We have introduced features that improve the quality of our model in comparison with the first baseline. Can we do even better? After we have changed the training and test sets, it almost always makes sense to search for the optimal hyperparameters - the parameters of the model that do not change during training.
# 
# For example, in week 3, you learned that, in decision trees, the depth of the tree is a hyperparameter, but the feature by which splitting occurs and its threshold is not. 
# 
# In the logistic regression that we use, the weights of each feature are changing, and we find their optimal values during training; meanwhile, the regularization parameter remains constant. This is the hyperparameter that we are going to optimize now.
# 
# Calculate the quality on a validation set with a regularization parameter, which is equal to 1 by default:

# In[26]:


# Compose the training set
tmp_scaled = StandardScaler().fit_transform(
    full_new_feat[["start_month", "start_hour", "morning"]]
)
X_train = csr_matrix(
    hstack([full_sites_sparse[:idx_split, :], tmp_scaled[:idx_split, :]])
)

# Capture the quality with default parameters
score_C_1 = get_auc_lr_valid(X_train, y_train)
print(score_C_1)


# We will try to beat this result by optimizing the regularization parameter. We will take a list of possible values of C and calculate the quality metric on the validation set for each of C-values:

# In[27]:


# List of possible C-values
Cs = np.logspace(-3, 1, 10)

# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# Plot the graph of the quality metric (AUC-ROC) versus the value of the regularization parameter. The value of quality metric corresponding to the default value of C=1 is represented by a horizontal dotted line:

# In[28]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# __<font color = 'red'> **Question 7:** </font>  What is the value of parameter C (if rounded to 2 decimals) that corresponds to the highest model quality?__
# 
# 
# - 0.17
# - 0.46
# - 1.29
# - 3.14

# In[29]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)
C = 1  # change this


# For the last task in this assignment: train the model using the optimal regularization parameter you found (do not round up to two digits like in the last question). If you do everything correctly and submit your solution, you should see `ROC AUC = 0.92784` on the public leaderboard ("A4 baseline 2"):

# In[30]:


# Prepare the training and test data
tmp_scaled = StandardScaler().fit_transform(
    full_new_feat[["start_month", "start_hour", "morning"]]
)
X_train = csr_matrix(
    hstack([full_sites_sparse[:idx_split, :], tmp_scaled[:idx_split, :]])
)
X_test = csr_matrix(
    hstack([full_sites_sparse[idx_split:, :], tmp_scaled[idx_split:, :]])
)

# Train the model on the whole training data set using optimal regularization parameter
lr = LogisticRegression(C=C, random_state=17, solver="liblinear").fit(X_train, y_train)

# Make a prediction for the test set
y_test = lr.predict_proba(X_test)[:, 1]

# Write it to the submission file
write_to_submission_file(y_test, "baseline_2.csv")


# In this part of the assignment, you have learned how to use sparse matrices, train logistic regression models, create new features and selected the best ones, learned why you need to scale features, and how to select hyperparameters. That's a lot!

# ### Part 2. Freeride <a class="tocSkip">

# <img src='../../_static/img/snowboard.jpg' width=70%>
# 
# *Yorko in Sheregesh, arguably, the best place in Russia for snowboarding and skiing.*

# In this part, you'll need to beat the 2 more baselines mentioned in the beginning of this assignment. No more step-by-step instructions. But it'll be very helpful for you to study the Notebook "[Correct time-aware cross-validation scheme](https://www.kaggle.com/kashnitsky/correct-time-aware-cross-validation-scheme)".
# 
# Here are a few tips for finding new features: think about what you can come up with using existing features, try multiplying or dividing two of them, justify or decline your hypotheses with plots, extract useful information from time series data (time1 ... time10), do not hesitate to convert an existing feature (for example, take a logarithm), etc. Checkout other [Notebooks](https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/kernels). We encourage you to try new ideas and models - it's fun!
# 
# When you get into Kaggle and Xgboost, you'll feel like that, and it's fine :)
# 
# <img src='../../_static/img/xgboost_meme.jpg' width=30%>
