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

(topic05_part2)=

# Topic 5. Ensembles and random forest. Part 2. Random Forest

<img src="https://habrastorage.org/webt/ia/m9/zk/iam9zkyzqebnf_okxipihkgjwnw.jpeg" />

**<center>[mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course** </center><br>


Authors: [Vitaliy Radchenko](https://www.linkedin.com/in/vitaliyradchenk0/), and [Yury Kashnitsky](https://yorko.github.io). Translated and edited by [Christina Butsko](https://www.linkedin.com/in/christinabutsko/), [Egor Polusmak](https://www.linkedin.com/in/egor-polusmak/), [Anastasia Manokhina](https://www.linkedin.com/in/anastasiamanokhina/), [Anna Shirshova](http://linkedin.com/in/anna-shirshova-b908458b), and [Yuanyuan Pao](https://www.linkedin.com/in/yuanyuanpao/). This material is subject to the terms and conditions of the [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. Free use is permitted for any non-commercial purpose.


## Article outline

1. [Algorithm](#algorithm)
2. [Comparison with Decision Trees and Bagging](#comparison-with-decision-trees-and-bagging)
3. [Parameters](#parameters)
4. [Variance and Decorrelation](#variance-and-decorrelation)
5. [Bias](#bias)
6. [Extremely Randomized Trees](#extremely-randomized-trees)
7. [Similarities between Random Forest and k-Nearest Neighbors](#similarities-between-random-forest-and-k-nearest-neighbors)
8. [Transformation of a dataset into a high-dimensional representation](#transformation-of-a-dataset-into-a-high-dimensional-representation)
9. [Pros and cons of random forests](#pros-and-cons-of-random-forests)
10. [Useful resources](#useful-resources)

$\DeclareMathOperator{\Var}{Var}$
$\DeclareMathOperator{\Cov}{Cov}$
$\DeclareMathOperator{\Corr}{Corr}$
$\DeclareMathOperator{\Err}{Err}$
$\DeclareMathOperator{\Bias}{Bias}$
$\DeclareMathOperator{\E}{\mathbb{E}}$

Leo Breiman managed to apply bootstrapping not only in statistics but also in machine learning. He, along with Adel Cutler, extended and improved the random forest algorithm [proposed by Tin Kam Ho](https://ieeexplore.ieee.org/document/598994). They combined the construction of uncorrelated trees using [CART](https://en.wikipedia.org/wiki/Decision_tree_learning), bagging, and the [random subspace method](https://en.wikipedia.org/wiki/Random_subspace_method).  

Decision trees are a good choice for the base classifier in bagging because they are quite sophisticated and can achieve zero classification error on any sample. The random subspace method reduces the correlation between the trees and thus prevents overfitting. With bagging, the base algorithms are trained on different random subsets of the original feature set.

The following algorithm constructs an ensemble of models using the random subspace method:

1. Let the number of instances be equal to $\large \ell$, and the number of features be equal to $\large d$.
2. Choose $\large L$ as the number of individual models in the ensemble.
3. For each model $\large l$, choose the number of features $\large dl < d$. As a rule, the same value of $\large dl$ is used for all the models.
4. For each model $\large l$, create a training set by selecting $\large dl$ features at random from the whole set of $\large d$ features.
5. Train each model.
6. Apply the resulting ensemble model to a new instance by combining the results from all the models in $\large L$. You can use either majority voting or aggregation of the posterior probabilities.

## 1. Algorithm

The algorithm for constructing a random forest of $\large N$ trees goes as follows:

* For each $\large k = 1, \dots, N$:

    * Generate a bootstrap sample $\large X_k$.
    * Build a decision tree $\large b_k$ on the sample $\large X_k$:

        * Pick the best feature according to the given criteria. Split the sample by this feature to create a new tree level. Repeat this procedure until the sample is exhausted.
        * Building the tree until any of its leaves contains no more than $\large n_\text{min}$ instances or until a certain depth is reached.
        * For each split, we first randomly pick $\large m$ features from the $\large d$ original ones and then search for the next best split only among the subset.

The final classifier is defined by:
$$\large a(x) = \frac{1}{N}\sum_{k = 1}^N b_k(x)$$

We use the majority voting for classification and the mean for regression.

For classification problems, it is advisable to set $\large m = \sqrt{d}$. For regression problems, we usually take $\large m = \frac{d}{3}$, where $\large d$ is the number of features. It is recommended to build each tree until all of its leaves contain only $\large n_\text{min} = 1$ examples for classification and $\large n_\text{min} = 5$ examples for regression.

You can see random forest as bagging of decision trees with the modification of selecting a random subset of features at each split.

## 2. Comparison with Decision Trees and Bagging


```{code-cell} ipython3
# Disable warnings in Anaconda
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from matplotlib import pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 10, 6
%config InlineBackend.figure_format = 'retina'

import seaborn as sns
from sklearn.datasets import make_circles
from sklearn.ensemble import (BaggingClassifier, BaggingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
```


```{code-cell} ipython3
n_train = 150  
n_test = 1000  
noise = 0.1

# Generate data
def f(x):
    x = x.ravel()
    return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)

def generate(n_samples, noise):
    X = np.random.rand(n_samples) * 10 - 5
    X = np.sort(X).ravel()
    y = np.exp(-X ** 2) + 1.5 * np.exp(-(X - 2) ** 2)\
        + np.random.normal(0.0, noise, n_samples)
    X = X.reshape((n_samples, 1))

    return X, y

X_train, y_train = generate(n_samples=n_train, noise=noise)
X_test, y_test = generate(n_samples=n_test, noise=noise)

# One decision tree regressor
dtree = DecisionTreeRegressor().fit(X_train, y_train)
d_predict = dtree.predict(X_test)

plt.figure(figsize=(10, 6))
plt.plot(X_test, f(X_test), "b")
plt.scatter(X_train, y_train, c="b", s=20)
plt.plot(X_test, d_predict, "g", lw=2)
plt.xlim([-5, 5])
plt.title("Decision tree, MSE = %.2f"
          % np.sum((y_test - d_predict) ** 2))

# Bagging with a decision tree regressor
bdt = BaggingRegressor(DecisionTreeRegressor()).fit(X_train, y_train)
bdt_predict = bdt.predict(X_test)

plt.figure(figsize=(10, 6))
plt.plot(X_test, f(X_test), "b")
plt.scatter(X_train, y_train, c="b", s=20)
plt.plot(X_test, bdt_predict, "y", lw=2)
plt.xlim([-5, 5])
plt.title("Bagging for decision trees, MSE = %.2f" % np.sum((y_test - bdt_predict) ** 2));

# Random Forest
rf = RandomForestRegressor(n_estimators=10).fit(X_train, y_train)
rf_predict = rf.predict(X_test)

plt.figure(figsize=(10, 6))
plt.plot(X_test, f(X_test), "b")
plt.scatter(X_train, y_train, c="b", s=20)
plt.plot(X_test, rf_predict, "r", lw=2)
plt.xlim([-5, 5])
plt.title("Random forest, MSE = %.2f" % np.sum((y_test - rf_predict) ** 2));
```

As we can see from our graphs and the MSE values above, a random forest of 10 trees achieves a better result than a single decision tree and is comparable to bagging with 10 trees. The main difference between random forests and bagging is that, in a random forest, the best feature for a split is selected from a random subset of the available features while, in bagging, all features are considered for the next best split.

We can also look at the advantages of random forests and bagging in classification problems:


```{code-cell} ipython3
np.random.seed(42)
X, y = make_circles(n_samples=500, factor=0.1, noise=0.35, random_state=42)
X_train_circles, X_test_circles, y_train_circles, y_test_circles = \
    train_test_split(X, y, test_size=0.2)

dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train_circles, y_train_circles)

x_range = np.linspace(X.min(), X.max(), 100)
xx1, xx2 = np.meshgrid(x_range, x_range)
y_hat = dtree.predict(np.c_[xx1.ravel(), xx2.ravel()])
y_hat = y_hat.reshape(xx1.shape)
plt.contourf(xx1, xx2, y_hat, alpha=0.2)
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', alpha=.7)
plt.title("Decision tree")
plt.show()

b_dtree = BaggingClassifier(DecisionTreeClassifier(),
                            n_estimators=300, random_state=42)
b_dtree.fit(X_train_circles, y_train_circles)

x_range = np.linspace(X.min(), X.max(), 100)
xx1, xx2 = np.meshgrid(x_range, x_range)
y_hat = b_dtree.predict(np.c_[xx1.ravel(), xx2.ravel()])
y_hat = y_hat.reshape(xx1.shape)
plt.contourf(xx1, xx2, y_hat, alpha=0.2)
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', alpha=.7)
plt.title("Bagging (decision trees)")
plt.show()

rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train_circles, y_train_circles)

x_range = np.linspace(X.min(), X.max(), 100)
xx1, xx2 = np.meshgrid(x_range, x_range)
y_hat = rf.predict(np.c_[xx1.ravel(), xx2.ravel()])
y_hat = y_hat.reshape(xx1.shape)
plt.contourf(xx1, xx2, y_hat, alpha=0.2)
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', alpha=.7)
plt.title("Random forest")
plt.show()
```

The figures above show that the decision boundary of the decision tree is quite jagged and has a lot of acute angles that suggest overfitting and a weak ability to generalize. We would have trouble making reliable predictions on new test data. In contrast, the bagging algorithm has a rather smooth boundary and has no obvious signs of overfitting.

Now, let's investigate some parameters which can help us increase the model accuracy.

## 3. Parameters

The [scikit-learn library](http://scikit-learn.org/stable/) implements random forests by providing two estimators: `RandomForestClassifier` and `RandomForestRegressor`.

The full list of random forest parameters for regression is shown below:


   - `n_estimators` — the number of trees in the forest (default = 10)
   - `criterion` — the function used to measure the quality of a split. Supported criteria are “mse” for the mean squared error, which is equal to variance reduction as feature selection criterion, and “mae” for the mean absolute error (default = "mse")
   - `max_features` — the number of features to consider when looking for the best split. You can specify the number or percentage of features, or choose from the available values: "auto" (all features), "sqrt", "log2". (default = "auto")
   -  `max_depth` — the maximum depth of the tree (default means that nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples)
   -  `min_samples_split` — the minimum number of samples required to split an internal node. Can be specified as the number or as a percentage of a total number of samples (default = 2)
   -  `min_samples_leaf` — the minimum number of samples required at a leaf node(default = 1)
   -  `min_weight_fraction_leaf` — the minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided (default = 0)
   - `max_leaf_nodes` — the maximum number of leaves (default = no restrictions)
   - `min_impurity_split` — threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf (default = 1е-7)
   - `bootstrap` — whether bootstrap samples are used when building trees(default = True)
   - `oob_score` — whether to use out-of-bag samples to estimate the R^2 on unseen data (default = False)
   - `n_jobs` — the number of jobs to run in parallel for both fit and predict. If -1, then the number of jobs is set to the number of cores (default = 1)
   - `random_state` — if int, random_state is the seed used by the random number generator; if RandomState instance, random_state is the random number generator; if None, the random number generator is the RandomState instance used by np.random (default = None)
   - `verbose` — controls the verbosity of the tree building process (default = 0)
   - `warm_start` — when set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest (default = False)



In case of classification, parameters are mostly the same. Only the following differ for `RandomForestClassifier` as compared to `RandomForestRegressor`:

   - `criterion` — the function used to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. Note: this parameter is tree-specific (default = "gini")
   - `class_weight` — the weight of each class (by default all weights equal to 1, but you can create a dictionary with weights or specify it as "balanced" - uses the values of classes to automatically adjust weights inversely proportional to class frequencies in the input data or as "balanced_subsample" - the same as “balanced” except that weights are computed based on the bootstrap sample for every tree grown)

Below are the parameters which we need to pay attention to when we are building a new model:

- `n_estimators` — the number of trees in the forest;
- `criterion` — the function used to measure the quality of a split;
- `max_features` — the number of features to consider when looking for the best split;
- `min_samples_leaf` — the minimum number of samples required to be at a leaf node;
- `max_depth` — the maximum depth of the tree.

### Practice with random forests in a real problem

In this example we will look at predicting customer churn. This is a classification problem, so we will use accuracy for model evaluation.

First, let's build a simple classifier which we will use as a baseline. For the sake of simplicity, we will use only numeric features.


```{code-cell} ipython3
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score)

# for Jupyter-book, we copy data from GitHub, locally, to save Internet traffic,
# you can specify the data/ folder from the root of your cloned
# https://github.com/Yorko/mlcourse.ai repo, to save Internet traffic
DATA_PATH = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/master/data/"

# Load data
df = pd.read_csv(DATA_PATH + "telecom_churn.csv")

# Choose the numeric features
cols = []
for i in df.columns:
    if (df[i].dtype == "float64") or (df[i].dtype == 'int64'):
        cols.append(i)

# Divide the dataset into the input and target
X, y = df[cols].copy(), np.asarray(df["Churn"],dtype='int8')

# Initialize a stratified split of our dataset for the validation process
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the classifier with the default parameters
rfc = RandomForestClassifier(random_state=42, n_jobs=-1)

# Train it on the training set
results = cross_val_score(rfc, X, y, cv=skf)

# Evaluate the accuracy on the test set
print("CV accuracy score: {:.2f}%".format(results.mean() * 100))
```

We have accuracy equal to 91.18%. Now, let's try to improve this result, and take a look at the behavior of the learning curves when we change the basic parameters.

Let's start with the number of trees:


```{code-cell} ipython3
# Initialize the validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create lists to save the values of accuracy on training and test sets
train_acc = []
test_acc = []
temp_train_acc = []
temp_test_acc = []
trees_grid = [5, 10, 15, 20, 30, 50, 75, 100]

for ntrees in trees_grid:
    rfc = RandomForestClassifier(n_estimators=ntrees, random_state=42, n_jobs=-1)
    temp_train_acc = []
    temp_test_acc = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rfc.fit(X_train, y_train)
        temp_train_acc.append(rfc.score(X_train, y_train))
        temp_test_acc.append(rfc.score(X_test, y_test))
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)

train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
print("Best CV accuracy is {:.2f}% with {} trees".format(max(test_acc.mean(axis=1))*100,
                                                        trees_grid[np.argmax(test_acc.mean(axis=1))]))
```


```{code-cell} ipython3
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(trees_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')
ax.plot(trees_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')
ax.fill_between(trees_grid, test_acc.mean(axis=1) - test_acc.std(axis=1),
                test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)
ax.fill_between(trees_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1),
                test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)
ax.legend(loc='best')
ax.set_ylim([0.88,1.02])
ax.set_ylabel("Accuracy")
ax.set_xlabel("N_estimators");
```

As you can see, when a certain number of trees is reached, our accuracy on the test set is very close to the asymptote. You can decide by yourself which value would be the optimal number of trees for your problem.

The figures also show that we achieved 100% accuracy on the training set, which tells us that we overfit. In order to avoid overfitting, we need to add regularization parameters to our model.

We will start with the maximum depth of trees `max_depth` and fix the number of trees at 100:


```{code-cell} ipython3
# Create lists to save accuracy values on the training and test sets
train_acc = []
test_acc = []
temp_train_acc = []
temp_test_acc = []
max_depth_grid = [3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24]

for max_depth in max_depth_grid:
    rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=max_depth)
    temp_train_acc = []
    temp_test_acc = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rfc.fit(X_train, y_train)
        temp_train_acc.append(rfc.score(X_train, y_train))
        temp_test_acc.append(rfc.score(X_test, y_test))
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)

train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
print("Best CV accuracy is {:.2f}% with {} max_depth".format(max(test_acc.mean(axis=1))*100,
                                                        max_depth_grid[np.argmax(test_acc.mean(axis=1))]))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(max_depth_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')
ax.plot(max_depth_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')
ax.fill_between(max_depth_grid, test_acc.mean(axis=1) - test_acc.std(axis=1),
                test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)
ax.fill_between(max_depth_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1),
                test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)
ax.legend(loc='best')
ax.set_ylim([0.88,1.02])
ax.set_ylabel("Accuracy")
ax.set_xlabel("Max_depth");
```

Parameter `max_depth` copes well with the regularization of our model and it does not overfit as badly as before. The model accuracy has increased slightly.

Another important parameter worth tuning is `min_samples_leaf`. It also contributes to regularization.


```{code-cell} ipython3
# Create lists to save accuracy values on the training and test sets
train_acc = []
test_acc = []
temp_train_acc = []
temp_test_acc = []
min_samples_leaf_grid = [1, 3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24]

for min_samples_leaf in min_samples_leaf_grid:
    rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                                 min_samples_leaf=min_samples_leaf)
    temp_train_acc = []
    temp_test_acc = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rfc.fit(X_train, y_train)
        temp_train_acc.append(rfc.score(X_train, y_train))
        temp_test_acc.append(rfc.score(X_test, y_test))
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)

train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
print("Best CV accuracy is {:.2f}% with {} min_samples_leaf".format(max(test_acc.mean(axis=1))*100,
                                                        min_samples_leaf_grid[np.argmax(test_acc.mean(axis=1))]))
```


```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(min_samples_leaf_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')
ax.plot(min_samples_leaf_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')
ax.fill_between(min_samples_leaf_grid, test_acc.mean(axis=1) - test_acc.std(axis=1),
                test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)
ax.fill_between(min_samples_leaf_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1),
                test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)
ax.legend(loc='best')
ax.set_ylim([0.88,1.02])
ax.set_ylabel("Accuracy")
ax.set_xlabel("Min_samples_leaf");
```

In this case, we do not see an improvement in accuracy on the validation set, but we significantly reduce the overfitting down to 2% while keeping the accuracy at about 92%.

Let's consider the parameter `max_features`. For classification, the value $\large \sqrt{d}$ (the total number of features) is typically used as the default choice. Let's check whether it would be optimal to use 4 features in our case:


```{code-cell} ipython3
# Create lists to save accuracy values on the training and test sets
train_acc = []
test_acc = []
temp_train_acc = []
temp_test_acc = []
max_features_grid = [2, 4, 6, 8, 10, 12, 14, 16]

for max_features in max_features_grid:
    rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                                 max_features=max_features)
    temp_train_acc = []
    temp_test_acc = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rfc.fit(X_train, y_train)
        temp_train_acc.append(rfc.score(X_train, y_train))
        temp_test_acc.append(rfc.score(X_test, y_test))
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)

train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
print("Best CV accuracy is {:.2f}% with {} max_features".format(max(test_acc.mean(axis=1))*100,
                                                        max_features_grid[np.argmax(test_acc.mean(axis=1))]))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(max_features_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')
ax.plot(max_features_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')
ax.fill_between(max_features_grid, test_acc.mean(axis=1) - test_acc.std(axis=1),
                test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)
ax.fill_between(max_features_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1),
                test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)
ax.legend(loc='best')
ax.set_ylim([0.88,1.02])
ax.set_ylabel("Accuracy")
ax.set_xlabel("Max_features");
```

In our case, the optimal number of features is equal to 10. This is the value at which the best result is achieved.

We have seen how the learning curves change with different values of the basic parameters. Now, let's use `GridSearch` to find the optimal parameters for our example:


```{code-cell} ipython3
# Initialize the set of parameters for exhaustive search and fit
parameters = {'max_features': [4, 7, 10, 13],
              'min_samples_leaf': [1, 3, 5, 7],
              'max_depth': [5, 10, 15, 20]}
rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
gcv = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
gcv.fit(X, y)
```


```{code-cell} ipython3
gcv.best_params_, gcv.best_score_
```

## 4. Variance and Decorrelation

Let's write the variance of a random forest as

$$\large \Var f(x) = \rho(x)\sigma^2(x)$$

$$\large \rho(x) = \Corr\left[T(x_1,\Theta_1(Z)),T(x_2,\Theta_2(Z))\right],$$

where

* $\large \rho(x)$ is the sample correlation coefficient between any two trees used in averaging:
* $\large \Theta_1(Z)$ and $\large \Theta_2(Z)$ are a randomly selected pair of trees on randomly selected elements of the sample $Z$;
* $\large T(x,\Theta_i(Z))$ is the output of the $\large i$-th tree classifier on an input vector $\large x$;
* $\large \sigma^2(x)$ is the sample variance of any randomly selected tree:

$$\large \sigma^2(x) = \Var\left[T(x,\Theta(X))\right]$$

It is easy to confuse $\large \rho(x)$ with the average correlation between the trained trees in a given random forest when we consider trees as N-vectors and calculate the average pairwise correlation between them. But this is not the case.

In fact, this conditional correlation is not directly related to the averaging process, and the dependence of $\large \rho(x)$ on $\large x$ warns us of this difference. $\large \rho(x)$ is the theoretical correlation between a pair of random trees estimated on the input $\large x$. Its value comes from the repeated sampling of the training set from the population $\large Z$ and the subsequent random choice of a pair of trees. In statistics jargon, this is the correlation caused by the sampling distribution of $\large Z$ and $\large \Theta$.

The conditional covariance of any pair of trees is equal to 0 because bootstrapping and feature selection are independent and identically distributed.


If we consider the variance of a single tree, it barely depends on the parameters of the splitting ($\large m$). But they are crucial for ensembles. The variance of a tree is much higher than the one of an ensemble. The book *The Elements of Statistical Learning (Trevor Hastie, Robert Tibshirani and Jerome Friedman)* has a great example that demonstrates this fact:

![image](../../_static/img/topic5_variance_rf.png)

## 5. Bias

Just as in bagging, the bias of a random forest is the same as the bias of a single tree $\large T(x,\Theta(Z))$:

$$ \large \begin{array}{rcl} \Bias &=& \mu(x) - \E_Z \, f_{rf}(x) \\
&=& \mu(x) - \E_Z \, \E_{\Theta | Z} \, T(x,\Theta(Z))\end{array}$$  

In absolute value, the bias is usually higher than that of an unpruned tree because randomization and sample space reduction impose their own restrictions on the model. Therefore, the improvements in prediction accuracy obtained by bagging and random forests are solely the result of variance reduction.

##  6. Extremely Randomized Trees

Extremely Randomized Trees employ a greater degree of randomization at the cut-point choice when splitting a tree node. As in random forests, a random subset of features is used. But, instead of the search for the optimal thresholds, their values are selected at random for each possible feature, and the best one among these randomly generated thresholds is used as the best rule to split the node. This usually trades off a slight reduction in the model variance with a small increase of the bias.

In the `scikit-learn` library, there are 2 implementations of Extremely Randomized Trees: [ExtraTreesClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier) and [ExtraTreesRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html#sklearn.ensemble.ExtraTreesRegressor).

This method should be used if you have greatly overfit with random forests or gradient boosting.

## 7. Similarities between Random Forest and k-Nearest Neighbors
The random forest method is similar to the nearest neighbors technique. Random forests predictions are based on labels of alike examples from the training set. The more often these examples appear in the same leaf of a tree, the higher their similarity. Let's prove this formally.

Let's consider a regression problem with the quadratic loss function. Let $\large T_n(x)$ be the number of the leaf of the $\large n$-th tree in a random forest with input $\large x$.
The algorithm response for the input vector $\large x$ equals the averaged response over all the examples of the training sample that fall into the leaf $\large T_n(x)$. This can be written as

$$\large b_n(x) = \sum_{i=1}^{l}w_n(x,x_i)y_i,$$

where

$$ \large w_n(x, x_i) = \frac{\left[T_n(x) = T_n(x_i)\right]}{\sum_{j=1}^{l}\left[T_n(x) = T_n(x_j)\right]}$$  

Then, the response of the composition is

$$ \large \begin{array}{rcl} a_n(x) &=& \frac{1}{N}\sum_{n=1}^{N}\sum_{i=1}^{l}w_n(x,x_i)y_i \\
&=& \sum_{i=1}^{l}\left(\frac{1}{N}\sum_{j=1}^{N}w_n(x,x_j)\right)y_i \end{array}$$

You can see that the response of a random forest is a weighted sum of responses over all training examples.

It is also worth noting that the number of the leaf $ \large T_n(x)$, where the instance $\large x$ ended up, is a valuable feature by itself. For example, the following approach works well: 1) A composition of a small number of trees is trained on a sample using a random forest or gradient boosting. 2) The categorical features $\large T_1(x), \dots, T_n(x)$ are added to the sample.

These new features are the result of the non-linear space splitting, and they provide information about similarity between examples. In the book *The Elements of Statistical Learning*, there is a good illustrative example that demonstrates this similarity between random forests and the k-nearest neighbors technique:

![image](../../_static/img/topic5_knn_vs_rf.png)

## 8. Transformation of a dataset into a high-dimensional representation

Random forests are mostly used in supervised learning, but there is a way to apply them in the unsupervised setting.

Using the `scikit-learn` method [RandomTreesEmbedding](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomTreesEmbedding.html#sklearn.ensemble.RandomTreesEmbedding), we can transform our dataset into a high-dimensional, [sparse](https://en.wikipedia.org/wiki/Sparse_matrix) representation. We first build extremely randomized trees and then use the index of the leaf containing the example as a new feature.

For example, if the input appears in the first leaf, we assign $1$ as the feature value; if not, we assign $0$. This is a so-called *binary coding*. We can control the number of features and the sparseness of data by increasing or decreasing the number of trees and their depth. Because nearby data points are likely to fall into the same leaf, this transformation provides an implicit nonparametric estimate of their density.

## 9. Pros and cons of random forests

### Pros:

- High prediction accuracy; will perform better than linear algorithms in most problems; the accuracy is comparable with that of boosting.
- Robust to outliers, thanks to random sampling.
- Insensitive to the scaling of features as well as any other monotonic transformations due to the random subspace selection.
- Doesn't require fine-grained parameter tuning, works quite well out-of-the-box. With tuning, it is possible to achieve a 0.5–3% gain in accuracy, depending on the problem setting and data.
- Efficient for datasets with a large number of features and classes.
- Handles both continuous and discrete variables equally well.
- Rarely overfits. In practice, an increase in the tree number almost always improves the composition. But, after reaching a certain number of trees, the learning curve is very close to the asymptote.
- There are developed methods to estimate feature importance.
- Works well with missing data and maintains good accuracy even when a large part of data is missing.
- Provides means to weight classes on the whole dataset as well as for each tree sample.
- Under the hood, calculates proximities between pairs of examples that can subsequently be used in clustering, outlier detection, or interesting data representations.
- The above functionality and properties may be extended to unlabeled data to enable unsupervised clustering, data visualization, and outlier detection.
- Easily parallelized and highly scalable.

### Cons:

 - In comparison with a single decision tree, Random Forest's output is more difficult to interpret.
 - There are no formal p-values for feature significance estimation.
 - Performs worse than linear methods in the case of sparse data: text inputs, bag of words, etc.
 - Unlike linear regression, Random Forest is unable to extrapolate. But, this can be also regarded as an advantage because outliers do not cause extreme values in Random Forests.
 - Prone to overfitting in some problems, especially, when dealing with noisy data.
 - In the case of categorical variables with varying level numbers, random forests favor variables with a greater number of levels. The tree will fit more towards a feature with many levels because this gains greater accuracy.
 - If a dataset contains groups of correlated features, preference might be given to groups of smaller size ("correlation bias"). See [this work](https://academic.oup.com/bioinformatics/article/27/14/1986/194387)
 - The resulting model is large and requires a lot of RAM.

## 10. Useful resources
- Main course [site](https://mlcourse.ai), [course repo](https://github.com/Yorko/mlcourse.ai), and YouTube [channel](https://www.youtube.com/watch?v=QKTuw4PNOsU&list=PLVlY_7IJCMJeRfZ68eVfEcu-UcN9BbwiX)
- mlcourse.ai [lecture](https://www.youtube.com/watch?v=neXJL-AqI_c) on Random Forest
- Medium ["story"](https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-5-ensembles-of-algorithms-and-random-forest-8e05246cbba7) based on this notebook
- Course materials as a [Kaggle Dataset](https://www.kaggle.com/kashnitsky/mlcourse)
- If you read Russian: an [article](https://habr.com/ru/company/ods/blog/324402/) on Habr.com with ~ the same material. And a [lecture](https://youtu.be/G0DmuuFeC30) on YouTube
- Chapter 15 of the book “[Elements of Statistical Learning](https://statweb.stanford.edu/~tibs/ElemStatLearn/)” by Jerome H. Friedman, Robert Tibshirani, and Trevor Hastie.
- More about practical applications of random forests and other algorithms can be found in the [official documentation](http://scikit-learn.org/stable/modules/ensemble.html) of `scikit-learn`.
- For a more in-depth discussion of variance and decorrelation of random forests, see the [original paper](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf).
http://ect.bell-labs.com/who/tkh/publications/papers/odt.pdf
