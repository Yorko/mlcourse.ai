#!/usr/bin/env python
# coding: utf-8

# <img src="https://habrastorage.org/webt/ia/m9/zk/iam9zkyzqebnf_okxipihkgjwnw.jpeg" />
#     
# **<center>[mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course** </center><br>
# Author: [Yury Kashnitsky](https://yorko.github.io) (@yorko). Edited by Anna Tarelina (@feuerengel), and Mikhail Korshchikov (@MS4). [mlcourse.ai](https://mlcourse.ai) is powered by [OpenDataScience (ods.ai)](https://ods.ai/) © 2017—2022

# # <center> Assignment #3. Solution </center> <a class="tocSkip">
# ## <center> Decision trees for classification and regression </center> <a class="tocSkip">

# In this assignment, we will find out how a decision tree works in a regression task, then will build and tune classification decision trees for identifying heart diseases.
# 
# ### Your task is to:
#  1. write code and perform computations in the cells below;
#  2. choose answers in the [webform](https://docs.google.com/forms/d/1ZUYREiTJjg8IiZAIIMdWRODtPe1qfC0UBYq_i2WFe1c/). 
#  
# 
# *f you are sure that something is not 100% correct with the assignment/solution, please leave your feedback via the mentioned webform ↑*
# 
# -----

# In[1]:


import numpy as np
import pandas as pd

# if seaborn is not yet installed, run `pip install seaborn` in terminal
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

sns.set()
import matplotlib.pyplot as plt

# sharper plots
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# ## 1. Decision trees for regression: a toy example

# Let's consider the following one-dimensional regression problem. We need to build a function $\large a(x)$ to approximate the dependency $\large y = f(x)$ using the mean-squared error criterion: $\large \min \sum_i {(a(x_i) - f(x_i))}^2$.

# In[2]:


X = np.linspace(-2, 2, 7)
y = X ** 3  # original dependecy

plt.scatter(X, y)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$");


# Let's make several steps to build a decision tree. In the case of a **regression** task, at prediction time, the leaf returns the average value for all observations in this leaf.
# 
# Let's start with a tree of depth 0, i.e. all observations placed in a single leaf. 
# 
# <br>You'll need to build a tree with only one node (also called **root**) that contains all train observations (instances). 
# <br>How will predictions of this tree look like for $x \in [-2, 2]$? <br> Create an appropriate plot using a pen, paper and Python if needed (but no `sklearn` is needed yet).

# In[3]:


# Solution code
xx = np.linspace(-2, 2, 100)
predictions = [np.mean(y) for x in xx]

X = np.linspace(-2, 2, 7)
y = X ** 3

plt.scatter(X, y)
plt.plot(xx, predictions, c="red");


# **Making first splits.**
# <br>Let's split the data according to the following condition $[x < 0]$. It gives us the tree of depth 1 with two leaves. To clarify, for all instances with $x \geqslant 0$ the tree will return some value, for all instances with $x < 0$ it will return another value. Let's create a similar plot for predictions of this tree.

# In[4]:


# Solution code
xx = np.linspace(-2, 2, 200)
predictions = [np.mean(y[X < 0]) if x < 0 else np.mean(y[X >= 0]) for x in xx]

X = np.linspace(-2, 2, 8)
y = X ** 3

plt.scatter(X, y)
plt.plot(xx, predictions, c="red");


# In the decision tree algorithm, the feature and the threshold for splitting are chosen according to some criterion. The commonly used criterion for regression is based on variance: 
# 
# $$\large Q(X, y, j, t) = D(X, y) - \dfrac{|X_l|}{|X|} D(X_l, y_l) - \dfrac{|X_r|}{|X|} D(X_r, y_r),$$
# 
# where $\large X$ and $\large y$ are a feature matrix and a target vector (correspondingly) for training instances in a current node, $\large X_l, y_l$ and $\large X_r, y_r$ are splits of samples $\large X, y$ into two parts w.r.t. $\large [x_j < t]$ (by $\large j$-th feature and threshold $\large t$), $\large |X|$, $\large |X_l|$, $\large |X_r|$ (or, the same, $\large |y|$, $\large |y_l|$, $\large |y_r|$) are sizes of appropriate samples, and $\large D(X, y)$ is variance of answers $\large y$ for all instances in $\large X$:
# 
# $$\large D(X, y) = \dfrac{1}{|X|} \sum_{j=1}^{|X|}(y_j – \dfrac{1}{|X|}\sum_{i = 1}^{|X|}y_i)^2$$
# 
# Here $\large y_i = y(x_i)$ is the answer for the $\large x_i$ instance. Feature index $\large j$ and threshold $\large t$ are chosen to maximize the value of criterion  $\large Q(X, y, j, t)$ for each split.
# 
# In our 1D case,  there's only one feature so $\large Q$ depends only on threshold $\large t$ and training data $\large X$ and $\large y$. Let's designate it $\large Q_{1d}(X, y, t)$ meaning that the criterion no longer depends on feature index $\large j$, i.e. in 1D case $\large j = 1$.

# In[5]:


def regression_var_criterion(X, y, t):
    # Solution code
    X_left, X_right = X[X < t], X[X >= t]
    y_left, y_right = y[X < t], y[X >= t]
    return (
        np.var(y)
        - X_left.shape[0] / X.shape[0] * np.var(y_left)
        - X_right.shape[0] / X.shape[0] * np.var(y_right)
    )


# Create the plot of criterion $\large Q_{1d}(X, y, t)$  as a function of threshold value $t$ on the interval $\large [-1.9, 1.9]$.

# In[6]:


# Solution code
thresholds = np.linspace(-1.9, 1.9, 100)
crit_by_thres = [regression_var_criterion(X, y, thres) for thres in thresholds]

plt.plot(thresholds, crit_by_thres)
plt.xlabel("threshold")
plt.ylabel("Regression criterion");


# **<font color='red'>Question 1.</font> What is the worst threshold value (to perform a split) according to the variance criterion?**
# - -1.9
# - -1.3
# - 0 **<font color='red'>[+]</font>**
# - 1.3
# - 1.9 

# Then let's make splitting in each of the leaves nodes. 
# <br> Take your tree with first threshold [$x<0$].
# <br> Now add a split in the left branch (where previous split was $x < 0$) using the criterion $[x < -1.5]$, in the right branch (where previous split was $x \geqslant 0$) with the following criterion $[x < 1.5]$. 
# <br>It gives us a tree of depth 2 with 7 nodes and 4 leaves. Create a plot of this tree predictions for $x \in [-2, 2]$.

# In[7]:


# Solution code
xx = np.linspace(-2, 2, 200)


def prediction(x, X, y):
    if x >= 1.5:
        return np.mean(y[X >= 1.5])
    elif x < 1.5 and x >= 0:
        return np.mean(y[(X >= 0) & (X < 1.5)])
    elif x >= -1.5 and x < 0:
        return np.mean(y[(X < 0) & (X >= -1.5)])
    else:
        return np.mean(y[X < -1.5])


predictions = [prediction(x, X, y) for x in xx]

X = np.linspace(-2, 2, 7)
y = X ** 3

plt.scatter(X, y)
plt.plot(xx, predictions, c="red");


# **<font color='red'>Question 2.</font> Tree predictions is a piecewise-constant function, right? How many "pieces" (horizontal segments in the plot that you've just built) are there in the interval [-2, 2]?**
# - 2
# - 4 **<font color='red'>[+]</font>**
# - 6
# - 8

# ## 2. Building a decision tree for predicting heart diseases
# Let's read the data on heart diseases. The dataset can be downloaded from the course repo from [here](https://github.com/Yorko/mlcourse.ai/blob/master/data/mlbootcamp5_train.csv) by clicking on `Download` and then selecting `Save As` option. If you work with Git, then the dataset is already there in `data/mlbootcamp5_train.csv`.
# 
# **Problem**
# 
# Predict presence or absence of cardiovascular disease (CVD) using the patient examination results.
# 
# **Data description**
# 
# There are 3 types of input features:
# 
# - *Objective*: factual information;
# - *Examination*: results of medical examination;
# - *Subjective*: information given by the patient.
# 
# | Feature | Variable Type | Variable      | Value Type |
# |---------|--------------|---------------|------------|
# | Age | Objective Feature | age | int (days) |
# | Height | Objective Feature | height | int (cm) |
# | Weight | Objective Feature | weight | float (kg) |
# | Gender | Objective Feature | gender | categorical code |
# | Systolic blood pressure | Examination Feature | ap_hi | int |
# | Diastolic blood pressure | Examination Feature | ap_lo | int |
# | Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal |
# | Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal |
# | Smoking | Subjective Feature | smoke | binary |
# | Alcohol intake | Subjective Feature | alco | binary |
# | Physical activity | Subjective Feature | active | binary |
# | Presence or absence of cardiovascular disease | Target Variable | cardio | binary |
# 
# All of the dataset values were collected at the moment of medical examination.

# In[8]:


df = pd.read_csv(
    "../../_static/data/assignment3/mlbootcamp5_train.csv", index_col="id", sep=";"
)


# In[9]:


df.head()


# Transform the features: 
# - create "age in years" dividing age by 365.25 and taking floor ($\lfloor{x}\rfloor$ is the largest integer that is less than or equal to $x$) 
# - create 3 binary features based on `cholesterol`.
# - create 3 binary features based on `gluc`.
# <br> Binary features equal to 1, 2 or 3. This method is called dummy-encoding or One Hot Encoding (OHE). It is more convenient to use `pandas.get_dummies`. There is no need to use the original features `cholesterol` and `gluc` after encoding.

# In[10]:


# Solution code
df["age_years"] = (df.age / 365.25).astype("int")


# In[11]:


# Solution code
train_df = pd.get_dummies(df, columns=["cholesterol", "gluc"]).drop(["cardio"], axis=1)
target = df.cardio


# In[12]:


# Solution code
train_df.head()


# Split data into train and holdout parts in the proportion of 7/3 using `sklearn.model_selection.train_test_split` with `random_state=17`.

# In[13]:


# Solution code
X_train, X_valid, y_train, y_valid = train_test_split(
    train_df.values, target.values, test_size=0.3, random_state=17
)


# Train a decision tree on the dataset `(X_train, y_train)` with **max depth equal to 3** and `random_state=17`. Plot this tree with `sklearn.tree.export_graphviz` and Graphviz. Here we need to mention that `sklearn` doesn't draw decision trees on its own, but is able to output a tree in the `.dot` format that can be used by Graphviz for visualization. 
# 
# How to plot a decision tree, alternatives:
#  1. Install Graphviz and pydotplus yourself (see below)
#  2. Use our docker image with all needed packages already installed
#  3. Easy way: execute `print(dot_data.getvalue())` with `dot_data` defined below (this can be done without pydotplus and Graphviz), go to http://www.webgraphviz.com, paste the graph code string (digraph Tree {...) and generate a nice picture

# Take a look how trees are visualized in the [3rd part](https://mlcourse.ai/articles/topic3-dt-knn/) of course materials.

# There are may be some troubles with graphviz for Windows users.
# The error is 'GraphViz's executables not found'.
# <br>To fix that – install Graphviz from [here](https://graphviz.gitlab.io/_pages/Download/Download_windows.html).
# <br>Then add graphviz path to your system PATH variable. You can do this manually, but don't forget to restart kernel.
# <br>Or just run this code:

# In[14]:


# import os
# path_to_graphviz = '' # your path to graphviz (C:\\Program Files (x86)\\Graphviz2.38\\bin\\ for example)
# os.environ["PATH"] += os.pathsep + path_to_graphviz


# In[15]:


# Solution code
tree = DecisionTreeClassifier(max_depth=3, random_state=17).fit(X_train, y_train)


# In[16]:


from io import StringIO

# pip install pydotplus
import pydotplus

# Solution code
from IPython.display import Image


# In[17]:


# Solution code
dot_data = StringIO()
export_graphviz(tree, feature_names=train_df.columns, out_file=dot_data, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# **<font color='red'>Question 3.</font> Which 3 features are used to make predictions in the created decision tree?**
# - age, ap_lo, chol=1
# - age, ap_hi, chol=3 **<font color='red'>[+]</font>**
# - smoke, age, gender
# - alco, weight, gluc=3

# Make predictions for holdout data `(X_valid, y_valid)` with the trained decision tree. Calculate accuracy.

# In[18]:


# Solution code
tree_pred_valid = tree.predict(X_valid)


# In[19]:


# Solution code
tree_acc_valid = accuracy_score(y_valid, tree_pred_valid)
tree_acc_valid


# Set up the depth of the tree using cross-validation on the dataset `(X_train, y_train)` in order to increase quality of the model. Use `GridSearchCV` with 5 folds. Fix `random_state=17` and change  `max_depth` from 2 to 10.

# In[20]:


get_ipython().run_cell_magic('time', '', '# Solution code\ntree_params = {"max_depth": list(range(2, 11))}\n\ntree_grid = GridSearchCV(\n    DecisionTreeClassifier(random_state=17), tree_params, cv=5, scoring="accuracy"\n)\n\ntree_grid.fit(X_train, y_train)\n')


# Draw the plot to show how mean accuracy is changing in regards to `max_depth` value on cross-validation.

# In[21]:


# Solution code
plt.plot(tree_params["max_depth"], tree_grid.cv_results_["mean_test_score"])
plt.xlabel("Max depth")
plt.ylabel("Mean CV accuracy");


# Print the best value of `max_depth` where the mean value of cross-validation quality metric reaches maximum. Also compute accuracy on holdout data. This can be done with the trained instance of the class `GridSearchCV`.

# In[22]:


# Solution code
print("Best params:", tree_grid.best_params_)
print("Best cross validaton score", tree_grid.best_score_)


# In[23]:


# Solution code
tuned_tree_acc_valid = accuracy_score(y_valid, tree_grid.predict(X_valid))
tuned_tree_acc_valid


# Сalculate the effect of `GridSearchCV`: check out the expression (acc2 - acc1) / acc1 * 100%, where acc1 and acc2 are accuracies on holdout data before and after tuning max_depth with GridSearchCV respectively.

# In[24]:


# Solution code
(tuned_tree_acc_valid - tree_acc_valid) / tree_acc_valid * 100


# **<font color='red'>Question 4.</font> Choose all correct statements:**
# - There exists a local maximum of accuracy on the built validation curve **<font color='red'>[+]</font>**
# - `GridSearchCV` increased holdout accuracy by **more** than 1% 
# - There is **no** local maximum of accuracy on the built validation curve
# - `GridSearchCV` increased holdout accuracy by **less** than 1% **<font color='red'>[+]</font>**

# Take a look at the SCORE table to estimate ten-year risk of fatal cardiovascular disease in Europe. [Source paper](https://academic.oup.com/eurheartj/article/24/11/987/427645).
# 
# <img src='../../_static/img/SCORE2007-eng.png' width=70%>
# 
# Let's create new features according to this picture:
# - $age \in [40,50), age \in [50,55), age \in [55,60), age \in [60,65) $ (4 features)
# - systolic blood pressure: $ap\_hi \in [120,140), ap\_hi \in [140,160), ap\_hi \in [160,180),$ (3 features)
# 
# If the values of age or blood pressure don't fall into any of the intervals then all binary features will be equal to zero. 
# 
# <br>Add a ``smoke`` feature.
# <br>Build the ``cholesterol``  and ``gender`` features. Transform the ``cholesterol`` to 3 binary features according to it's 3 unique values ( ``cholesterol``=1,  ``cholesterol``=2 and  ``cholesterol``=3). Transform the ``gender`` from 1 and 2 into 0 and 1. It is better to rename it to ``male`` (0 – woman, 1 – man). In general, this is typically done with ``sklearn.preprocessing.LabelEncoder`` but here in case of only 2 unique values it's not necessary.
# 
# Finally, the decision tree is built using these 12 binary features (excluding all original features that we had before this feature engineering part).
# 
# Create a decision tree with the limitation `max_depth=3` and train it on the whole train data. Use the `DecisionTreeClassifier` class with fixed `random_state=17`, but all other arguments (except for `max_depth` and `random_state`) should be left with their default values.
# 
# **<font color='red'>Question 5.</font> Which binary feature is the most important for heart disease detection (i.e., it is placed in the root of the tree)?**
# - Systolic blood pressure from 160 to 180 (mmHg)
# - Cholesterol level == 3
# - Systolic blood pressure from 140 to 160 (mmHg) **<font color='red'>[+]</font>**
# - Age from 50 to 55 (years)
# - Smokes / doesn't smoke
# - Age from 60 to 65 (years)

# In[25]:


# Solution code
sub_df = pd.DataFrame(df.smoke.copy())
sub_df["male"] = df.gender - 1

sub_df["age_40_50"] = ((df.age_years >= 40) & (df.age_years < 50)).astype("int")
sub_df["age_50_55"] = ((df.age_years >= 50) & (df.age_years < 55)).astype("int")
sub_df["age_55_60"] = ((df.age_years >= 55) & (df.age_years < 60)).astype("int")
sub_df["age_60_65"] = ((df.age_years >= 60) & (df.age_years < 65)).astype("int")

sub_df["ap_hi_120_140"] = ((df.ap_hi >= 120) & (df.ap_hi < 140)).astype("int")
sub_df["ap_hi_140_160"] = ((df.ap_hi >= 140) & (df.ap_hi < 160)).astype("int")
sub_df["ap_hi_160_180"] = ((df.ap_hi >= 160) & (df.ap_hi < 180)).astype("int")

sub_df["chol=1"] = (df.cholesterol == 1).astype("int")
sub_df["chol=2"] = (df.cholesterol == 2).astype("int")
sub_df["chol=3"] = (df.cholesterol == 3).astype("int")


# In[26]:


# Solution code
sub_df.head()


# In[27]:


# Solution code
tree = DecisionTreeClassifier(max_depth=3, random_state=17).fit(sub_df, target)


# In[28]:


# Solution code
dot_data = StringIO()
export_graphviz(tree, feature_names=sub_df.columns, out_file=dot_data, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# Although it is possible to improve such plots, still it is clear that the decision tree first checks whether the systolic blood pressure is in range from 140 to 160 mmHg.
