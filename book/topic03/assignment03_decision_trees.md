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

(assignment03)=

# Assignment #3 (demo). Decision trees with a toy task and the UCI Adult dataset

```{figure} /_static/img/ods_stickers.jpg
```

**<center>[mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course** </center><br>

Authors: [Maria Sumarokova](https://www.linkedin.com/in/mariya-sumarokova-230b4054/), and [Yury Kashnitsky](https://www.linkedin.com/in/festline/). Translated and edited by Gleb Filatov, Aleksey Kiselev, [Anastasia Manokhina](https://www.linkedin.com/in/anastasiamanokhina/), [Egor Polusmak](https://www.linkedin.com/in/egor-polusmak/), and [Yuanyuan Pao](https://www.linkedin.com/in/yuanyuanpao/). All content is distributed under the [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

Same assignment as a [Kaggle Kernel](https://www.kaggle.com/kashnitsky/a3-demo-decision-trees) + [solution](https://www.kaggle.com/kashnitsky/a3-demo-decision-trees-solution). Fill in the answers in the [web-form](https://docs.google.com/forms/d/1wfWYYoqXTkZNOPy1wpewACXaj2MZjBdLOL58htGWYBA/edit).

Let's start by loading all necessary libraries:


```{code-cell} ipython3
%matplotlib inline
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (10, 8)
import collections
from io import StringIO

import numpy as np
import pandas as pd
import pydotplus  # pip install pydotplus
import seaborn as sns
from ipywidgets import Image
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
```

## Part 1. Toy dataset "Will They? Won't They?"

Your goal is to figure out how decision trees work by walking through a toy problem. While a single decision tree does not yield outstanding results, other performant algorithms like gradient boosting and random forests are based on the same idea. That is why knowing how decision trees work might be useful.

We'll go through a toy example of binary classification - Person A is deciding whether they will go on a second date with Person B. It will depend on their looks, eloquence, alcohol consumption (only for example), and how much money was spent on the first date.

### Creating the dataset


```{code-cell} ipython3
# Create dataframe with dummy variables
def create_df(dic, feature_list):
    out = pd.DataFrame(dic)
    out = pd.concat([out, pd.get_dummies(out[feature_list])], axis=1)
    out.drop(feature_list, axis=1, inplace=True)
    return out


# Some feature values are present in train and absent in test and vice-versa.
def intersect_features(train, test):
    common_feat = list(set(train.keys()) & set(test.keys()))
    return train[common_feat], test[common_feat]
```


```{code-cell} ipython3
features = ["Looks", "Alcoholic_beverage", "Eloquence", "Money_spent"]
```

### Training data


```{code-cell} ipython3
df_train = {}
df_train["Looks"] = [
    "handsome",
    "handsome",
    "handsome",
    "repulsive",
    "repulsive",
    "repulsive",
    "handsome",
]
df_train["Alcoholic_beverage"] = ["yes", "yes", "no", "no", "yes", "yes", "yes"]
df_train["Eloquence"] = ["high", "low", "average", "average", "low", "high", "average"]
df_train["Money_spent"] = ["lots", "little", "lots", "little", "lots", "lots", "lots"]
df_train["Will_go"] = LabelEncoder().fit_transform(["+", "-", "+", "-", "-", "+", "+"])

df_train = create_df(df_train, features)
df_train
```

### Test data


```{code-cell} ipython3
df_test = {}
df_test["Looks"] = ["handsome", "handsome", "repulsive"]
df_test["Alcoholic_beverage"] = ["no", "yes", "yes"]
df_test["Eloquence"] = ["average", "high", "average"]
df_test["Money_spent"] = ["lots", "little", "lots"]
df_test = create_df(df_test, features)
df_test
```


```{code-cell} ipython3
# Some feature values are present in train and absent in test and vice-versa.
y = df_train["Will_go"]
df_train, df_test = intersect_features(train=df_train, test=df_test)
df_train
```


```{code-cell} ipython3
df_test
```

### Draw a decision tree (by hand or in any graphics editor) for this dataset. Optionally you can also implement tree construction and draw it here.

1\. What is the entropy $S_0$ of the initial system? By system states, we mean values of the binary feature "Will_go" - 0 or 1 - two states in total.


```{code-cell} ipython3
# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)
```

2\. Let's split the data by the feature "Looks_handsome". What is the entropy $S_1$ of the left group - the one with "Looks_handsome". What is the entropy $S_2$ in the opposite group? What is the information gain (IG) if we consider such a split?


```{code-cell} ipython3
# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)
```

### Train a decision tree using sklearn on the training data. You may choose any depth for the tree.


```{code-cell} ipython3
# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)
```

### Additional: display the resulting tree using graphviz. You can use pydot or a web-service, e.g. [this one](https://onlineconvertfree.com/convert-format/dot-to-png/).


```{code-cell} ipython3
# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)
```

## Part 2. Functions for calculating entropy and information gain.

Consider the following warm-up example: we have 9 blue balls and 11 yellow balls. Let ball have label **1** if it is blue, **0** otherwise.


```{code-cell} ipython3
balls = [1 for i in range(9)] + [0 for i in range(11)]
```

<img src = '../../_static/img/decision_tree3.png'>

Next split the balls into two groups:

<img src = '../../_static/img/decision_tree4.png'>


```{code-cell} ipython3
# two groups
balls_left = [1 for i in range(8)] + [0 for i in range(5)]  # 8 blue and 5 yellow
balls_right = [1 for i in range(1)] + [0 for i in range(6)]  # 1 blue and 6 yellow
```

### Implement a function to calculate the Shannon Entropy


```{code-cell} ipython3
def entropy(a_list):
    # You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)
    pass
```

Tests


```{code-cell} ipython3
print(entropy(balls))  # 9 blue и 11 yellow
print(entropy(balls_left))  # 8 blue и 5 yellow
print(entropy(balls_right))  # 1 blue и 6 yellow
print(entropy([1, 2, 3, 4, 5, 6]))  # entropy of a fair 6-sided die
```

3\. What is the entropy of the state given by the list **balls_left**?

4\. What is the entropy of a fair dice? (where we look at a dice as a system with 6 equally probable states)?


```{code-cell} ipython3
# information gain calculation
def information_gain(root, left, right):
    """ root - initial data, left and right - two partitions of initial data"""

    # You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)
    pass
```

5\. What is the information gain from splitting the initial dataset into **balls_left** and **balls_right** ?


```{code-cell} ipython3
def information_gains(X, y):
    """Outputs information gain when splitting with each feature"""

    # You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)
    pass
```

### Optional:
- Implement a decision tree building algorithm by calling `information_gains` recursively
- Plot the resulting tree

## Part 3. The "Adult" dataset

**Dataset description:**

[Dataset](http://archive.ics.uci.edu/ml/machine-learning-databases/adult) UCI Adult (no need to download it, we have a copy in the course repository): classify people using demographic data - whether they earn more than \$50,000 per year or not.

Feature descriptions:

- **Age** – continuous feature
- **Workclass** –  continuous feature
- **fnlwgt** – final weight of object, continuous feature
- **Education** –  categorical feature
- **Education_Num** – number of years of education, continuous feature
- **Martial_Status** –  categorical feature
- **Occupation** –  categorical feature
- **Relationship** – categorical feature
- **Race** – categorical feature
- **Sex** – categorical feature
- **Capital_Gain** – continuous feature
- **Capital_Loss** – continuous feature
- **Hours_per_week** – continuous feature
- **Country** – categorical feature

**Target** – earnings level, categorical (binary) feature.

**Reading train and test data**


```{code-cell} ipython3
# for Jupyter-book, we copy data from GitHub, locally, to save Internet traffic,
# you can specify the data/ folder from the root of your cloned
# https://github.com/Yorko/mlcourse.ai repo, to save Internet traffic
DATA_PATH = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/master/data/"
```


```{code-cell} ipython3
data_train = pd.read_csv(DATA_PATH + "adult_train.csv", sep=";")
```


```{code-cell} ipython3
data_train.tail()
```


```{code-cell} ipython3
data_test = pd.read_csv(DATA_PATH + "adult_test.csv", sep=";")
```


```{code-cell} ipython3
data_test.tail()
```


```{code-cell} ipython3
# necessary to remove rows with incorrect labels in test dataset
data_test = data_test[
    (data_test["Target"] == " >50K.") | (data_test["Target"] == " <=50K.")
]

# encode target variable as integer
data_train.loc[data_train["Target"] == " <=50K", "Target"] = 0
data_train.loc[data_train["Target"] == " >50K", "Target"] = 1

data_test.loc[data_test["Target"] == " <=50K.", "Target"] = 0
data_test.loc[data_test["Target"] == " >50K.", "Target"] = 1
```

**Primary data analysis**


```{code-cell} ipython3
data_test.describe(include="all").T
```


```{code-cell} ipython3
data_train["Target"].value_counts()
```


```{code-cell} ipython3
fig = plt.figure(figsize=(25, 15))
cols = 5
rows = int(data_train.shape[1] / cols)
for i, column in enumerate(data_train.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if data_train.dtypes[column] == object:
        data_train[column].value_counts().plot(kind="bar", axes=ax)
    else:
        data_train[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2);
```

**Checking data types**


```{code-cell} ipython3
data_train.dtypes
```


```{code-cell} ipython3
data_test.dtypes
```

As we see, in the test data, age is treated as type **object**. We need to fix this.


```{code-cell} ipython3
data_test["Age"] = data_test["Age"].astype(int)
```

Also we'll cast all **float** features to **int** type to keep types consistent between our train and test data.


```{code-cell} ipython3
data_test["fnlwgt"] = data_test["fnlwgt"].astype(int)
data_test["Education_Num"] = data_test["Education_Num"].astype(int)
data_test["Capital_Gain"] = data_test["Capital_Gain"].astype(int)
data_test["Capital_Loss"] = data_test["Capital_Loss"].astype(int)
data_test["Hours_per_week"] = data_test["Hours_per_week"].astype(int)
```

**Fill in missing data for continuous features with their median values, for categorical features with their mode.**


```{code-cell} ipython3
# choose categorical and continuous features from data

categorical_columns = [
    c for c in data_train.columns if data_train[c].dtype.name == "object"
]
numerical_columns = [
    c for c in data_train.columns if data_train[c].dtype.name != "object"
]

print("categorical_columns:", categorical_columns)
print("numerical_columns:", numerical_columns)
```


```{code-cell} ipython3
# we see some missing values
data_train.info()
```


```{code-cell} ipython3
# fill missing data

for c in categorical_columns:
    data_train[c].fillna(data_train[c].mode()[0], inplace=True)
    data_test[c].fillna(data_train[c].mode()[0], inplace=True)

for c in numerical_columns:
    data_train[c].fillna(data_train[c].median(), inplace=True)
    data_test[c].fillna(data_train[c].median(), inplace=True)
```


```{code-cell} ipython3
# no more missing values
data_train.info()
```

We'll dummy code some categorical features: **Workclass**, **Education**, **Martial_Status**, **Occupation**, **Relationship**, **Race**, **Sex**, **Country**. It can be done via pandas method **get_dummies**


```{code-cell} ipython3
data_train = pd.concat(
    [data_train[numerical_columns], pd.get_dummies(data_train[categorical_columns])],
    axis=1,
)

data_test = pd.concat(
    [data_test[numerical_columns], pd.get_dummies(data_test[categorical_columns])],
    axis=1,
)
```


```{code-cell} ipython3
set(data_train.columns) - set(data_test.columns)
```


```{code-cell} ipython3
data_train.shape, data_test.shape
```

**There is no Holland in the test data. Create new zero-valued feature.**


```{code-cell} ipython3
data_test["Country_ Holand-Netherlands"] = 0
```


```{code-cell} ipython3
set(data_train.columns) - set(data_test.columns)
```


```{code-cell} ipython3
data_train.head(2)
```


```{code-cell} ipython3
data_test.head(2)
```


```{code-cell} ipython3
X_train = data_train.drop(["Target"], axis=1)
y_train = data_train["Target"]

X_test = data_test.drop(["Target"], axis=1)
y_test = data_test["Target"]
```

### 3.1 Decision tree without parameter tuning

Train a decision tree **(DecisionTreeClassifier)** with a maximum depth of 3, and evaluate the accuracy metric on the test data. Use parameter **random_state = 17** for results reproducibility.


```{code-cell} ipython3
# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit) (read-only in a JupyterBook, pls run jupyter-notebook to edit)
# tree =
# tree.fit
```

Make a prediction with the trained model on the test data.


```{code-cell} ipython3
# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)
# tree_predictions = tree.predict
```


```{code-cell} ipython3
# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)
# accuracy_score
```

6\. What is the test set accuracy of a decision tree with maximum tree depth of 3 and **random_state = 17**?

### 3.2 Decision tree with parameter tuning

Train a decision tree **(DecisionTreeClassifier, random_state = 17).** Find the optimal maximum depth using 5-fold cross-validation **(GridSearchCV)**.


```{code-cell} ipython3
tree_params = {"max_depth": range(2, 11)}

locally_best_tree = GridSearchCV  # You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)

locally_best_tree.fit
# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)
```

Train a decision tree with maximum depth of 9 (it is the best **max_depth** in my case), and compute the test set accuracy. Use parameter **random_state = 17** for reproducibility.


```{code-cell} ipython3
# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)
# tuned_tree =
# tuned_tree.fit
# tuned_tree_predictions = tuned_tree.predict
# accuracy_score
```

7\. What is the test set accuracy of a decision tree with maximum depth of 9 and **random_state = 17**?

### 3.3 (Optional) Random forest without parameter tuning

Let's take a sneak peek of upcoming lectures and try to use a random forest for our task. For now, you can imagine a random forest as a bunch of decision trees, trained on slightly different subsets of the training data.

Train a random forest **(RandomForestClassifier)**. Set the number of trees to 100 and use **random_state = 17**.


```{code-cell} ipython3
# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)
# rf =
# rf.fit # You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)
```

Make predictions for the test data and assess accuracy.


```{code-cell} ipython3
# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)
```

### 3.4 (Optional) Random forest with parameter tuning

Train a random forest **(RandomForestClassifier)**. Tune the maximum depth and maximum number of features for each tree using **GridSearchCV**.


```{code-cell} ipython3
# forest_params = {'max_depth': range(10, 21),
#                 'max_features': range(5, 105, 20)}

# locally_best_forest = GridSearchCV # You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)

# locally_best_forest.fit # You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)
```

Make predictions for the test data and assess accuracy.


```{code-cell} ipython3
# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)
```
