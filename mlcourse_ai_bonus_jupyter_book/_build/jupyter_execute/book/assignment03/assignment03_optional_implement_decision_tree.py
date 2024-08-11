#!/usr/bin/env python
# coding: utf-8

# <img src="https://habrastorage.org/webt/ia/m9/zk/iam9zkyzqebnf_okxipihkgjwnw.jpeg" />
#     
# **<center>[mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course** </center><br>
# Author: [Yury Kashnitsky](https://yorko.github.io) (@yorko). Edited by Anna Tarelina (@feuerengel).[mlcourse.ai](https://mlcourse.ai) is powered by [OpenDataScience (ods.ai)](https://ods.ai/) © 2017—2021

# # <center>Assignment #3. Optional part. Task </center> <a class="tocSkip">
# ## <center>Implementation of the decision tree algorithm </center><a class="tocSkip">
# 
# If you want to truly understand the algorithm that you are using, it's good to implement it from scratch. Have to say, it's not necessary if you just want to apply an algorithm in practice. However, if you love math, algorithms and programming, you'll enjoy this task.  
# <img src='../../_static/img/quote_feynman.jpg' width=50%>
# 
# ### Your task is to:
#  1. write code and perform computations in the cells below;
#  2. choose answers in the [webform](https://docs.google.com/forms/d/1SYwUD0Yx_bcykq6EqFQ4Ug0KaQWG4L7bAlL5yGedZnw).
#     
# 
# *If you are sure that something is not 100% correct with the assignment/solution, please leave your feedback via the mentioned webform ↑*
# 
# -----

# In[1]:


import numpy as np

# if seaborn is not yet installed, run `pip install seaborn` in terminal
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.datasets import (
    load_boston,
    load_digits,
    make_classification,
    make_regression,
)
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

sns.set()
import matplotlib.pyplot as plt

# sharper plots
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# Let's fix `random_state` (a.k.a. random seed) beforehand.

# In[2]:


RANDOM_STATE = 17


# **Implement the class `DecisionTree`**
# **Specification:**
# - the class is inherited from `sklearn.BaseEstimator`;
# - class constructor has the following parameters: 
#     - `max_depth` - maximum depth of the tree (`numpy.inf` by default); 
#     - `min_samples_split` - the minimum number of instances in a node for a splitting to be done (2 by default); 
#     - `criterion` - split criterion ('gini' or 'entropy' for classification, 'variance' or 'mad_median' for regression; 'gini' by default);
# - the class has several methods: `fit`, `predict` and `predict_proba`;
# - the`fit` method takes the matrix of instances `X` and a target vector `y` (`numpy.ndarray` objects) and returns an instance of the class `DecisionTree` representing the decision tree trained on the dataset `(X, y)` according to parameters set in the constructor; 
# - the `predict_proba` method takes the matrix of instances `X` and returns the matrix `P` of a size `X.shape[0] x K`, where `K` is the number of classes and $p_{ij}$ is the probability of an instance in $i$-th row of `X` to belong to class $j \in \{1, \dots, K\}$.
# - the `predict` method takes the matrix of instances `X` and returns a prediction vector; in case of classification, prediction for an instance $x_i$ falling into leaf $L$ will be the class, mostly represented among instances in $L$. In case of regression, it'll be the mean value of targets for all instances in leaf $L$.
# 
# A note on `criterion`: this is the functional to be maximized to find an optimal partition at a given node has the form $Q(X, j, t) = F(X) - \dfrac{|X_l|}{|X|} F(X_l) - \dfrac{|X_r|}{|X|} F(X_r),$ where $X$ are samples at a given node, $X_l$ and $X_r$ are partitions of samples $X$ into two parts with the following condition $[x_j < t]$, and $F(X)$ is a partition criterion.
#     
# For classification: let $p_i$ be the fraction of the instances of the $i$-th class in the dataset $X$.
# - 'gini': Gini impurity $F(X) = 1 -\sum_{i = 1}^K p_i^2$.
# - 'entropy': Entropy $F(X) = -\sum_{i = 1}^K p_i \log_2(p_i)$.
#     
# For regression: $y_j = y(x_j)$ - is a target for an instance $x_j$, $y = (y_1, \dots, y_{|X|})$ - is a target vector.
# - 'variance': Variance (mean quadratic deviation from average) $F(X) = \dfrac{1}{|X|} \sum_{x_j \in X}(y_j - \dfrac{1}{|X|}\sum_{x_i \in X}y_i)^2$
# - 'mad_median': Mean deviation from the median $F(X) = \dfrac{1}{|X|} \sum_{x_j \in X}|y_j - \mathrm{med}(y)|$
#     

# In[3]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


def entropy(y):
    pass


def gini(y):
    pass


def variance(y):
    pass


def mad_median(y):
    pass


# The `Node` class implements a node in the decision tree.

# In[4]:


class Node:
    def __init__(self, feature_idx=0, threshold=0, labels=None, left=None, right=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.labels = labels
        self.left = left
        self.right = right


# Let's determine the function for calculating a prediction in a leaf. For regression, let's take the mean for all values in a leaf, for classification - the most popular class in leaf.

# In[5]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)

def regression_leaf(y):
    pass


def classification_leaf(y):
    pass


class DecisionTree(BaseEstimator):
    def __init__(
        self, max_depth=np.inf, min_samples_split=2, criterion="gini", debug=False
    ):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass


# ## Testing the implemented algorithm

# ### Classification

# Download the dataset `digits` using the method `load_digits`. Split the data into train and test with the `train_test_split` method, use parameter values `test_size=0.2`, and `random_state=17`. Try to train shallow decision trees and make sure that gini and entropy criteria return different results.

# In[6]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# Using 5-folds cross-validation (`GridSearchCV`) pick up the optimal values of the `max_depth` and `criterion` parameters. For the parameter `max_depth` use range(3, 11), for criterion use {'gini', 'entropy'}. Quality measure is `scoring`='accuracy'.

# In[7]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# Draw the plot of the mean quality measure `accuracy` for criteria `gini` and `entropy` depending on `max_depth`.

# In[8]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# **<font color='red'>Question 1.</font> Choose all correct statements:**
# 1. Optimal value of the `max_depth` parameter is on the interval [4, 9] for both criteria.
# 2. Created plots have no intersection on the interval [3, 10]
# 3. Created plots intersect each other only once on the interval [3, 10].
# 4. The best quality for `max_depth` on the interval [3, 10] is reached using `gini` criterion .
# 5. Accuracy is strictly increasing at least for one of the criteria, when `max_depth` is also increasing on the interval [3, 10]

# **<font color='red'>Question 2.</font> What are the optimal values for max_depth and criterion parameters?**
# 1. max_depth = 7, criterion = 'gini';
# 2. max_depth = 7, criterion = 'entropy';
# 3. max_depth = 10, criterion = 'entropy';
# 4. max_depth = 10, criterion = 'gini';
# 5. max_depth = 9, criterion = 'entropy';
# 6. max_depth = 9, criterion = 'gini';

# Train decision tree on `(X_train, y_train)` using the optimal values of `max_depth` and `criterion`. Compute class probabilities for `X_test`.

# In[9]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# Using the given matrix, compute the mean class probabilities for all instances in `X_test`.

# In[10]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# **<font color='red'>Question 3.</font> What is the maximum probability in a resulted vector?**
# 1. 0.127
# 2. 0.118
# 3. 1.0
# 4. 0.09

# ## Regression

# Download the dataset `boston` using the method `load_boston`. Split the data into train and test with the `train_test_split` method, use parameter values `test_size=0.2`, `random_state=17`. Try to train shallow regression decision trees and make sure that `variance` and `mad_median` criteria return different results.

# In[11]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# Using 5-folds cross-validation (`GridSearchCV`) pick up the optimal values of the `max_depth` and `criterion` parameters. For the parameter `max_depth` use `range(2, 9)`, for `criterion` use {'variance', 'mad_median'}. Quality measure is `scoring`='neg_mean_squared_error'.

# In[12]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# Draw the plot of the mean quality measure `neg_mean_squared_error` for criteria `variance` and `mad_median` depending on `max_depth`.

# In[13]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# **<font color='red'>Question 4.</font> Choose all correct statements:**
# 1. Created plots have no intersection on the interval [2, 8].
# 2. Created plots intersect each other only once on the interval [2, 8].
# 3. Optimal value of the `max_depth` for each of the criteria is on the border of the interval [2, 8].
# 4. The best quality at `max_depth` on the interval [2, 8] is reached using `mad_median` criterion.

# **<font color='red'>Question 5.</font> What are the optimal values for `max_depth` and `criterion` parameters?**
# 1. max_depth = 9, criterion = 'variance';
# 2. max_depth = 5, criterion = 'mad_median';
# 3. max_depth = 4, criterion = 'variance';
# 4. max_depth = 2, criterion = 'mad_median';
# 5. max_depth = 4, criterion = 'mad_median';
# 6. max_depth = 5, criterion = 'variance'.
