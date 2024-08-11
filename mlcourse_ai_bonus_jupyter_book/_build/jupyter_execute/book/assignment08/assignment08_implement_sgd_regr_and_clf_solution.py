#!/usr/bin/env python
# coding: utf-8

# <img src="https://habrastorage.org/webt/ia/m9/zk/iam9zkyzqebnf_okxipihkgjwnw.jpeg" />
#     
# **<center>[mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course** </center><br>
# Author: [Yury Kashnitsky](https://yorko.github.io) (@yorko). Translated by Sergey Oreshkov.  [mlcourse.ai](https://mlcourse.ai) is powered by [OpenDataScience (ods.ai)](https://ods.ai/) © 2017—2022

# # <center>Assignment #8. Solution </center><a class="tocSkip">
# 
# ## <center>Implementing Stochastic Gradient Descent for regression and classification  </center><a class="tocSkip">
# ## <center> 

# Here we implement two algorithms  – a regressor and a classifier – driven by stochastic gradient descent (SGD). 
# 
# 
# ### Your task is to:
#  1. write code and perform computations in the cells below;
#  2. choose answers in the [webform](https://forms.gle/gC4PN9ntDru4sbZU7).
# 
# *If you are sure that something is not 100% correct with the assignment/solution, please leave your feedback via the mentioned webform ↑*
# 
# -----
# 

# ## Plan
# 1. [Linear regression and SGD](#1.-Linear-regression-and-Stochastic-Gradient-Descent-)
# 1. [Logistic regression and SGD](#2.-Logistic-Regression-and-SGD)
# 1. [Logistic regression and SGDClassifier for movie review sentiment analysis](#3.-Logistic-regression-and-SGDClassifier-and-movie-review-classification-task)

# ## 1. Linear regression and Stochastic Gradient Descent <a class="tocSkip">
# 
# In [this article](https://mlcourse.ai/articles/topic8-sgd-vw/) we described how to train an online regressor, while minimizing squared error function. Let's implement this algorithm. 
#     
# **Note:** the implementation closely follows the mentioned article. It's vanilla MSE minimization with no regularization (just to make it easier a bit). We'll add $L_2$-regularization later when we implement the SGD version of logistic regression. 

# In[1]:


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# sharper plots
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import seaborn as sns
from matplotlib import pyplot as plt

sns.set()


# Implement class `SGDRegressor`. Specs for the class:
# - class is inherited from `sklearn.base.BaseEstimator`
# - constructor takes parameters `eta` – gradient step ($10^{-3}$ by default) and `n_iter` – dataset pass count (10 by default)
# - constructor also must create `mse_` and `weights_` lists in order to track mean squared error and weight vector during gradient descent iterations
# - Class has `fit` and `predict` methods
# - The `fit` method takes matrix `X` and vector `y` (`numpy.array` objects) as parameters, appends column of ones to  `X` on the left side, initializes the weight vector `w` with **zeros** and then does `n_iter` iterations of weight updates (check [the article](https://mlcourse.ai/articles/topic8-sgd-vw/), and for every iteration logs mean squared error and the weight vector `w` in corresponding lists that were created in the constructor. 
# - Additionally the `fit` method will create `w_` variable to store weights which produce the best mean squared error
# - The `fit` method must return current object of `SGDRegressor` class, i.e. `self`
# - The `predict` method takes `X` matrix, adds column of ones to the left of this matrix and returns a prediction vector, using weight vector `w_`, created by the `fit` method.

# In[2]:


class SGDRegressor(BaseEstimator):
    def __init__(self, eta=1e-3, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.mse_ = []
        self.weights_ = []

    def fit(self, X, y):
        X = np.hstack([np.ones([X.shape[0], 1]), X])

        w = np.zeros(X.shape[1])

        for it in tqdm(range(self.n_iter)):
            for i in range(X.shape[0]):

                new_w = w.copy()
                new_w[0] += self.eta * (y[i] - w.dot(X[i, :]))
                for j in range(1, X.shape[1]):
                    new_w[j] += self.eta * (y[i] - w.dot(X[i, :])) * X[i, j]
                w = new_w.copy()

                self.weights_.append(w)
                self.mse_.append(mean_squared_error(y, X.dot(w)))

        self.w_ = self.weights_[np.argmin(self.mse_)]

        return self

    def predict(self, X):
        X = np.hstack([np.ones([X.shape[0], 1]), X])

        return X.dot(self.w_)


# Let us test out the algorithm on basic example of height/weight data. We will predict height(in inches) having a weight (lbs).

# In[3]:


data_demo = pd.read_csv("../../_static/data/assignment8/weights_heights.csv")


# In[4]:


plt.scatter(data_demo["Weight"], data_demo["Height"])
plt.xlabel("Weight (lbs)")
plt.ylabel("Height (Inch)");


# In[5]:


X, y = data_demo["Weight"].values, data_demo["Height"].values


# We leave 70% of the data as a training set, and 30% as a test set. Also we scale the data.

# In[6]:


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=17
)


# In[7]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape([X_train.shape[0], 1]))
X_valid_scaled = scaler.transform(X_valid.reshape([X_valid.shape[0], 1]))


# Train the created `SGDRegressor` with `(X_train_scaled, y_train)` data. Leave default parameter values for now.

# In[8]:


sgd_reg = SGDRegressor(n_iter=1)
sgd_reg.fit(X_train_scaled, y_train)


# Draw a chart with the training process – the dependency of mean squared error on the i-th SGD iteration number.

# In[9]:


plt.plot(range(len(sgd_reg.mse_)), sgd_reg.mse_);


# Print the minimal value of mean squared error and the best weights vector.

# In[10]:


np.min(sgd_reg.mse_), sgd_reg.w_


# Draw the chart of model weights ($w_0$ and $w_1$) as they changed during training.

# In[11]:


plt.plot(range(len(sgd_reg.weights_)), [w[0] for w in sgd_reg.weights_])
plt.plot(range(len(sgd_reg.weights_)), [w[1] for w in sgd_reg.weights_]);


# Make a prediction for the hold-out test set `(X_valid_scaled, y_valid)` and check the corresponding MSE value.

# In[12]:


mean_squared_error(y_valid, sgd_reg.predict(X_valid_scaled))


# Now do the same thing for the `LinearRegression` class from `sklearn.linear_model`. Evaluate MSE for the same hold-out set.

# In[13]:


from sklearn.linear_model import LinearRegression

lm = LinearRegression().fit(X_train_scaled, y_train)

lm.coef_, lm.intercept_

mean_squared_error(y_valid, lm.predict(X_valid_scaled))


# **<font color='red'>Question 1.</font> In which decimal do we see the difference between MSE for linear regressor and `SGDRegressor`?**
#  - 2
#  - 3
#  - 4
#  - 5 **<font color='red'>[+]</font>**

# ## 2. Logistic Regression and SGD
# Let us sort out now, how the very same stochastic approach can help to train logistic regression.

# Let's consider a classification task, where $X$ – is a training dataset with size $\ell \times (d+1)$ (first column is a vector of ones), and $y$ – is the target vector, $y_i \in \{-1, 1\}$. In [topic 4, part 2](https://mlcourse.ai/book/topic04/topic4_linear_models_part2_logit_likelihood_learning.html) of this course we described how logistic regression with $L_2$-regularization yields the following minimization problem:
# 
# $$\large C\sum_{i=1}^\ell \log{(1 + e^{-y_iw^Tx_i})} + \frac{1}{2}\sum_{j=1}^d w_j^2 \rightarrow min_w$$

# **<font color='red'>Question 2.</font> Which formula will be used for update of logistic regression weights during stochastic gradient descent training?**
#  - $w_j^{(t+1)} = w_j^{(t)} + \eta (Cy_i x_{ij} \sigma(y_iw^Tx_i) +  \delta_{j\neq0} w_j)$
#  - $w_j^{(t+1)} = w_j^{(t)} - \eta (Cy_i x_{ij} \sigma(-y_iw^Tx_i) +  \delta_{j\neq0}w_j)$
#  - $w_j^{(t+1)} = w_j^{(t)} - \eta (Cy_i x_{ij} \sigma(y_iw^Tx_i) -  \delta_{j\neq0}w_j )$
#  - $w_j^{(t+1)} = w_j^{(t)} + \eta (Cy_i x_{ij} \sigma(-y_iw^Tx_i) -  \delta_{j\neq0}w_j)$  **<font color='red'>[+]</font>**
#  
# Here:
# - $i \in {0,\ldots, \ell-1}, j \in {0,\ldots, d}$
# - C – regularization coefficient
# - $x_{ij} $ – element of the X matrix at row $i$ and column $j$ (indexing starts from 0), 
# - $x_i$ – $i$-th row of $X$ matrix (indexing from 0), 
# - $w_j^{(t)}$ – value of $j$-th element of weights vector $w$ during step $t$ of stochastic gradient descent
# - $\eta$ – small constant value, step of gradient descent
# - $\delta_{j\neq0}$ – Kronecker symbol, i.e. 1, if $j\neq0$ and $0$ otherwise 

# **<font color='red'>Solution:</font>**
# $$ J(w) = C\sum_{i=1}^\ell \log{(1 + e^{-y_iw^Tx_i})} + \frac{1}{2}\sum_{j=1}^d w_j^2$$
# 
# First, differentiate $f(z) = \log{(1 + e^{-z})}$:
# 
# $$\frac{df}{dz} = \frac{1}{1 + e^{-z}}\frac{d(1 + e^{-z})}{dz} =  -\frac{1}{1 + e^{-z}}e^{-z} = -\frac{1}{e^z+1} = -\sigma(-z)$$
# 
# Next,
# $$\frac{\partial{J}}{\partial{w_0}} = -C\sum_{i=1}^\ell \sigma(-y_iw^Tx_i) \frac{d(y_iw^Tx_i)}{dw_0} = -C\sum_{i=1}^\ell \sigma(-y_iw^Tx_i)~y_i$$
# 
# For $j \neq 0$:
# 
# $$\frac{\partial{J}}{\partial{w_j}} = -C\sum_{i=1}^\ell \sigma(-y_iw^Tx_i) \frac{d(y_iw^Tx_i)}{dw_j} + \frac{d(\frac{1}{2}\sum_{j=1}^d w_j^2)}{dw_j} = -C\sum_{i=1}^\ell \sigma(-y_iw^Tx_i)~y_ix_{ij} + w_j$$
# 
# Weights update for gradient descent (not stochastic this time):
# 
# $$w_j^{(t+1)} = w_j^{(t)} -\eta \frac{\partial{J}}{\partial{w_j}}$$
# 
# or
# 
# $$w_0^{(t+1)} = w_0^{(t)} +\eta C\sum_{i=1}^\ell \sigma(-y_iw^Tx_i)~y_i$$
# $$w_j^{(t+1)} = w_j^{(t)} +\eta (C\sum_{i=1}^\ell \sigma(-y_iw^Tx_i)~y_ix_{ij} - w_j), j\in 1 \ldots d$$
# 
# With stochastic approach we remove summation:
# 
# $$w_j^{(t+1)} = w_j^{(t)} + \eta (Cy_i x_{ij} \sigma(-y_iw^Tx_i) -  \delta_{j\neq0}w_j)$$
# 

# Let's implement the `SGDClassifier` class. Specs for this class:
# - we inherit our class from `sklearn.base.BaseEstimator`
# - constructor has the following parameters: `eta` – gradient descent step (default value is $10^{-3}$), `n_iter` – iterations count (default is 10) and `C` – regularization coefficient
# - additionally, let's create  `loss_` and `weights_` lists in order to track values of logistic loss and weight vector over gradient descent iterations
# - class has `fit`, `predict` and `predict_proba` methods
# - The `fit` method has parameters `X`(matrix) and `y`(vector) (`numpy.array` objects, we assume we have binary classification, and values in vector `y` can only be either -1 or 1). We add a column of ones to `X` from the left, initialize `w` vector with **zeros** and iteratively (`n_iter` times) update weights with expression we got earlier, also logging log_loss and weight `w` values into corresponding lists. 
# - In the end `fit` must create `w_` and store weight vector with the smallest loss value
# - The `fit` method must return object of `SGDClassifier` type, i.e. `self`
# - The `predict_proba` method gets `X` matrix, adds column of ones from the left and returns matrix with predictions (you get same matrices from `predict_proba` methods in `sklearn`), using vector `w_` we created by `fit` method
# - `predict` method calls `predict_proba` and returns answer vector: -1, if predicted probability of 1 is less than 0.5 and 1 otherwise
# - And **important**: in order to avoid problems with computation of big and small values with exponent (overflow & underflow) use the following `sigma` function.

# In[14]:


def sigma(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


# In[15]:


class SGDClassifier(BaseEstimator):
    def __init__(self, C=1, eta=1e-3, n_iter=10):
        self.eta = eta
        self.C = C
        self.n_iter = n_iter
        self.loss_ = []
        self.weights_ = []

    def fit(self, X, y):
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        best_loss = np.inf
        w = np.zeros(X.shape[1])

        for it in tqdm(range(self.n_iter)):
            for i in range(X.shape[0]):

                new_w = w.copy()

                new_w[0] += self.eta * self.C * y[i] * sigma(-y[i] * w.dot(X[i, :]))
                for j in range(1, X.shape[1]):
                    new_w[j] += self.eta * (
                        self.C * y[i] * X[i, j] * sigma(-y[i] * w.dot(X[i, :])) - w[j]
                    )

                w = new_w.copy()

                self.loss_.append(log_loss(y, sigma(X.dot(w))))
                self.weights_.append(w)

        self.w_ = self.weights_[np.argmin(self.loss_)]
        return self

    def predict_proba(self, X):
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        p_vec = sigma(X.dot(self.w_)).reshape([X.shape[0], 1])
        return np.hstack([1 - p_vec, p_vec])

    def predict(self, X):
        pred_probs = self.predict_proba(X)[:, 1]
        signs = np.sign(pred_probs - 0.5)
        # zeros can remain if pred_probs = 0.5 exactly
        signs[np.where(signs == 0)] = 1
        return signs


# Let's test `SGDClassifier` with breast cancer UCI dataset.

# In[16]:


from sklearn.datasets import load_breast_cancer


# In[17]:


cancer = load_breast_cancer()
# change labels in y from 0 to -1
X, y = cancer.data, [-1 if i == 0 else 1 for i in cancer.target]


# Let's split dataset - 70% for training and 30% – as a holdout set. Let's also scale the data.

# In[18]:


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=17
)


# In[19]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)


# Train `SGDClassifier` with the scaled training set with the following parameters: `C`=1, `eta`=$10^{-3}$ and `n_iter`=3.

# In[20]:


sgd_clf = SGDClassifier(C=1, n_iter=3, eta=1e-3)
sgd_clf.fit(X_train_scaled, y_train)


# Make a plot of `log_loss` as it's changing during training.

# In[21]:


plt.plot(range(len(sgd_clf.loss_)), sgd_clf.loss_);


# Now train `SGDClassifier` with `C`=1000 increasing the number of iterations over the training set to 10.

# In[22]:


sgd_clf = SGDClassifier(C=1000, n_iter=10)
sgd_clf.fit(X_train_scaled, y_train)
plt.plot(range(len(sgd_clf.loss_)), sgd_clf.loss_);


# Now check out the model weights vector with minimal loss on the training set.

# **<font color='red'>Question 3.</font> Which feature has the highest impact on probability of benign tumour, according to  `SGDClassifier` model? (be careful to check the length of the weight vector you get after training,  and compare with number of features in our task)**
#  - worst compactness
#  - worst smoothness
#  - worst concavity **<font color='red'>[+]</font>**
#  - concave points error
#  - concavity error
#  - compactness error
#  - worst fractal dimension

# In[23]:


best_w = sgd_clf.weights_[np.argmin(sgd_clf.loss_)]


# In[24]:


np.min(best_w), np.max(best_w)


# In[25]:


pd.DataFrame(
    {"coef": best_w, "feat": ["intercept"] + list(cancer.feature_names)}
).sort_values(by="coef")


# In[26]:


cancer.feature_names[np.argmin(best_w) - 1]


# Compute log_loss and ROC AUC for hold-out validation set, and do all the same with  `sklearn.linear_model.LogisticRegression` (leave default parameters for this object, only set random_state=17) and compare results.

# In[27]:


log_loss(y_valid, sgd_clf.predict_proba(X_valid_scaled)[:, 1])


# In[28]:


roc_auc_score(y_valid, sgd_clf.predict_proba(X_valid_scaled)[:, 1])


# In[29]:


from sklearn.linear_model import LogisticRegression

logit = LogisticRegression(random_state=17).fit(X_train_scaled, y_train)

cancer.feature_names[np.argmin(logit.coef_.flatten())]

log_loss(y_valid, logit.predict_proba(X_valid_scaled)[:, 1])

roc_auc_score(y_valid, logit.predict_proba(X_valid_scaled)[:, 1])


# ## 3. Logistic regression and SGDClassifier and movie review classification task

# Let's look at logistic regression and its SGD variation for classification of reviews from IMDB. We know this task from the 4-th and the 8-th topics of the course, and we also used it here in the 5-th bonus assignment. 
# 
# We will import the data and train `CountVectorizer` with the available data.

# In[30]:


from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier


# In[31]:


reviews = pd.read_csv("../../_static/data/assignment5/movie_reviews_train.csv.zip")


# In[32]:


reviews.head()


# Perform train/validation split.

# In[33]:


reviews_train, reviews_valid, y_train, y_valid = train_test_split(
    reviews["text"], reviews["label"], test_size=0.3, random_state=17
)


# We are going to train `CountVectorizer` with data we have, while counting bigrams. By doing this we come to sparse data representation, where we have a feature for each unique word and pair of consecutive words. We get over 1.5 billion of features therefore.

# In[34]:


get_ipython().run_cell_magic('time', '', 'cv = CountVectorizer(ngram_range=(1, 2))\nX_train = cv.fit_transform(reviews_train)\nX_valid = cv.transform(reviews_valid)\n')


# In[35]:


X_train.shape, X_valid.shape


# Train logistic regression with data `(X_train, y_train)` and default parameters (except for `random_state`= 17 to get reproducible result) and calculate ROC AUC with the validation set. Measure the time of the model's training. We don't need to scale the data because our features are counters and they are already spread across roughly the same ranges.

# In[36]:


get_ipython().run_cell_magic('time', '', 'logit = LogisticRegression(solver="lbfgs", random_state=17, max_iter=100)\nlogit.fit(X_train, y_train)\nroc_auc_score(y_valid, logit.predict_proba(X_valid)[:, 1])\n')


# --------
# Now we move to the online algorithm. We implemented our `SGDClassifier` and understood how it works, but additional efforts are required to make it effective, for example, to support sparse features. Let's now switch to the `sklearn`-implementation of the SGD-algorithm. Check out [SGDClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) docs, and make a note in which aspects `SGDClassifier` from `Sklearn` is more advanced than our own SGD. 

# **<font color='red'>Question 4.</font> In which aspects is the `Sklearn` implementation more advanced than the `SGDClassifier` that we implemented? Select all correct options.**
#  - Gradient descent step can be varied **<font color='red'>[+]</font>**
#  - Linear SVM is implemented **<font color='red'>[+]</font>**
#  - Early stopping is implemented to avoid overfitting
#  - Can be run on multiple CPUs **<font color='red'>[+]</font>**
#  - LASSO is supported **<font color='red'>[+]</font>**
#  - Online learning of decision trees is supported
#  - Mini-batch training is supported (i.e. weight updates weights with several training examples at a time) **<font color='red'>[+]</font>**

# Run 100 iterations of SGD-logistic regression (again, `random_state`=17) with the same data. Measure the training time and note how faster the SGD version is.

# In[37]:


sgd_logit = SGDClassifier(loss="log", random_state=17, max_iter=100)


# In[38]:


get_ipython().run_cell_magic('time', '', 'sgd_logit.fit(X_train, y_train)\nroc_auc_score(y_valid, sgd_logit.predict_proba(X_valid)[:, 1])\n')


# **<font color='red'>Question 5.</font> In which decimal do we see the difference between validation ROC AUC-s for logistic regression and `Sklearn` SGD classifier with logistic loss function?**
#  - 2
#  - 3 **<font color='red'>[+]</font>**
#  - 4
#  - 5
