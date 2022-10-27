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

(assignment05_solution)=

# Assignment #5 (demo). Logistic Regression and Random Forest in the credit scoring problem. Solution

<img src="https://habrastorage.org/webt/ia/m9/zk/iam9zkyzqebnf_okxipihkgjwnw.jpeg" />

**<center>[mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course** </center><br>
Author: Vitaly Radchenko. All content is distributed under the [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.


**Same assignment as a [Kaggle Kernel](https://www.kaggle.com/kashnitsky/a5-demo-logit-and-rf-for-credit-scoring) + [solution](https://www.kaggle.com/kashnitsky/a5-demo-logit-and-rf-for-credit-scoring-sol).**

In this assignment, you will build models and answer questions using data on credit scoring.

Please write your code in the cells with the "Your code here" placeholder. Then, answer the questions in the [form](https://docs.google.com/forms/d/1gKt0DA4So8ohKAHZNCk58ezvg7K_tik26d9QND7WC6M/edit).

Let's start with a warm-up exercise.

**Question 1.** There are 5 jurors in a courtroom. Each of them can correctly identify the guilt of the defendant with 70% probability, independent of one another. What is the probability that the jurors will jointly reach the correct verdict if the final decision is by majority vote?

1. 70.00%
2. 83.20%
3. 83.70%
4. 87.50%

**Answer:** 3.

**Solution:**

We will use the formula for $\mu$ from the article. Since the majority of votes begin with $3$, then $m = 3, ~N = 5, ~p = 0.7$. Substitute these values into the formula to get:

$$\large \mu = \sum_{i=3}^{5}{5 \choose i}0.7^i(1-0.7)^{5-i} = 83.70\%$$

Great! Let's move on to machine learning.

## Credit scoring problem setup

### Problem

Predict whether the customer will repay their credit within 90 days. This is a binary classification problem; we will assign customers into good or bad categories based on our prediction.

### Data description

| Feature | Variable Type | Value Type | Description |
|:--------|:--------------|:-----------|:------------|
| age | Input Feature | integer | Customer age |
| DebtRatio | Input Feature | real | Total monthly loan payments (loan, alimony, etc.) / Total monthly income percentage |
| NumberOfTime30-59DaysPastDueNotWorse | Input Feature | integer | The number of cases when client has overdue 30-59 days (not worse) on other loans during the last 2 years |
| NumberOfTimes90DaysLate | Input Feature | integer | Number of cases when customer had 90+dpd overdue on other credits |
| NumberOfTime60-89DaysPastDueNotWorse | Input Feature | integer | Number of cased when customer has 60-89dpd (not worse) during the last 2 years |
| NumberOfDependents | Input Feature | integer | The number of customer dependents |
| SeriousDlqin2yrs | Target Variable | binary: <br>0 or 1 | Customer hasn't paid the loan debt within 90 days |


Let's set up our environment:


```{code-cell} ipython3
# Disable warnings in Anaconda
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
```


```{code-cell} ipython3
from matplotlib import rcParams

rcParams["figure.figsize"] = 11, 8
```

Let's write the function that will replace *NaN* values with the median for each column.


```{code-cell} ipython3
def fill_nan(table):
    for col in table.columns:
        table[col] = table[col].fillna(table[col].median())
    return table
```

Now, read the data:


```{code-cell} ipython3
# for Jupyter-book, we copy data from GitHub, locally, to save Internet traffic,
# you can specify the data/ folder from the root of your cloned
# https://github.com/Yorko/mlcourse.ai repo, to save Internet traffic
DATA_PATH = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/master/data/"
```


```{code-cell} ipython3
data = pd.read_csv(DATA_PATH + "credit_scoring_sample.csv", sep=";")
data.head()
```

Look at the variable types:


```{code-cell} ipython3
data.dtypes
```

Check the class balance:


```{code-cell} ipython3
ax = data["SeriousDlqin2yrs"].hist(orientation="horizontal", color="red")
ax.set_xlabel("number_of_observations")
ax.set_ylabel("unique_value")
ax.set_title("Target distribution")

print("Distribution of the target:")
data["SeriousDlqin2yrs"].value_counts() / data.shape[0]
```

Separate the input variable names by excluding the target:


```{code-cell} ipython3
independent_columns_names = [x for x in data if x != "SeriousDlqin2yrs"]
independent_columns_names
```

Apply the function to replace *NaN* values:


```{code-cell} ipython3
table = fill_nan(data)
```

Separate the target variable and input features:


```{code-cell} ipython3
X = table[independent_columns_names]
y = table["SeriousDlqin2yrs"]
```

## Bootstrapping

**Question 2.** Make an interval estimate of the average age for the customers who delayed the repayment with the confidence level equal 90%. Use the example from the article for reference. Also, use `np.random.seed(0)` as it was done in the article. What is the resulting interval estimate?

1. 52.59 – 52.86
2. 45.71 – 46.13
3. 45.68 – 46.17
4. 52.56 – 52.88

**Answer:** 2.

**Solution:**


```{code-cell} ipython3
def get_bootstrap_samples(data, n_samples):
    """Generate samples using bootstrapping."""
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples


def stat_intervals(stat, alpha):
    """Make an interval estimate."""
    boundaries = np.percentile(stat, [100 * alpha / 2.0, 100 * (1 - alpha / 2.0)])
    return boundaries


# Save the ages of those who let a delay
churn = data[data["SeriousDlqin2yrs"] == 1]["age"].values

# Set the random seed for reproducibility
np.random.seed(0)

# Generate bootstrap samples and calculate the mean for each sample
churn_mean_scores = [np.mean(sample) for sample in get_bootstrap_samples(churn, 1000)]

# Print the interval estimate for the sample means
print("Mean interval", stat_intervals(churn_mean_scores, 0.1))
```

## Logistic regression

Let's set up to use logistic regression:


```{code-cell} ipython3
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
```

Now, we will create a LogisticRegression model and use class_weight='balanced' to make up for our unbalanced classes.


```{code-cell} ipython3
lr = LogisticRegression(random_state=5, class_weight="balanced")
```

Let's try to find the best regularization coefficient, which is the coefficient `C` for logistic regression. Then, we will have an optimal model that is not overfit and is a good predictor of the target variable.


```{code-cell} ipython3
parameters = {"C": (0.0001, 0.001, 0.01, 0.1, 1, 10)}
```

In order to find the optimal value of `C`, let's apply stratified 5-fold validation and look at the *ROC AUC* against different values of the parameter `C`. Use the `StratifiedKFold` function for this:


```{code-cell} ipython3
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
```

One of the important metrics of model quality is the *Area Under the Curve (AUC)*. *ROC AUC* varies from 0 to 1. The closer ROC AUC to 1, the better the quality of the classification model.

**Question 3.** Perform a *Grid Search* with the scoring metric "roc_auc" for the parameter `C`. Which value of the parameter `C` is optimal?

1. 0.0001
2. 0.001
3. 0.01
4. 0.1
5. 1
6. 10

**Answer:** 2.

**Solution:**


```{code-cell} ipython3
grid_search = GridSearchCV(lr, parameters, n_jobs=-1, scoring="roc_auc", cv=skf)
grid_search = grid_search.fit(X, y)
grid_search.best_estimator_
```

**Question 4.** Can we consider the best model stable? The model is *stable* if the standard deviation on validation is less than 0.5%. Save the *ROC AUC* value of the best model, it will be useful for the following tasks.

1. Yes
2. No

**Answer:** 2.

**Solution:**


```{code-cell} ipython3
grid_search.cv_results_["std_test_score"][1]
```

The *ROC AUC* value of the best model:


```{code-cell} ipython3
grid_search.best_score_
```

## Feature importance

**Question 5.** *Feature importance* is defined by the absolute value of its corresponding coefficient. First you need to normalize all the feature values so that it will be correct to compare them. What is the most important feature for the best logistic regression model?

1. age
2. NumberOfTime30-59DaysPastDueNotWorse
3. DebtRatio
4. NumberOfTimes90DaysLate
5. NumberOfTime60-89DaysPastDueNotWorse
6. MonthlyIncome
7. NumberOfDependents

**Answer:** 2.

**Solution:**


```{code-cell} ipython3
from sklearn.preprocessing import StandardScaler

lr = LogisticRegression(C=0.001, random_state=5, class_weight="balanced")
scal = StandardScaler()
lr.fit(scal.fit_transform(X), y)

pd.DataFrame(
    {"feat": independent_columns_names, "coef": lr.coef_.flatten().tolist()}
).sort_values(by="coef", ascending=False)
```

**Question 6.** Calculate how much `DebtRatio` affects the prediction using the [softmax function](https://en.wikipedia.org/wiki/Softmax_function). What is its value?

1. 0.38
2. -0.02
3. 0.11
4. 0.24

**Answer:** 3.

**Solution:**


```{code-cell} ipython3
print((np.exp(lr.coef_[0]) / np.sum(np.exp(lr.coef_[0])))[2])
```

**Question 7.** Let's see how we can interpret the impact of our features. For this, recalculate the logistic regression with absolute values, that is without scaling. Next, modify the customer's age by adding 20 years, keeping the other features unchanged. How many times will the chance that the customer will not repay their debt increase? You can find an example of the theoretical calculation [here](https://www.unm.edu/~schrader/biostat/bio2/Spr06/lec11.pdf).

1. -0.01
2. 0.70
3. 8.32
4. 0.66

**Answer:** 2.

**Solution:**


```{code-cell} ipython3
lr = LogisticRegression(C=0.001, random_state=5, class_weight="balanced")
lr.fit(X, y)

pd.DataFrame(
    {"feat": independent_columns_names, "coef": lr.coef_.flatten().tolist()}
).sort_values(by="coef", ascending=False)
```


```{code-cell} ipython3
np.exp(lr.coef_[0][0] * 20)
```

It is $\exp^{\beta\delta}$ times more likely that the customer won't repay the debt, where $\delta$ is the feature value increment. That means that if we increased the age by 20 years, the odds that the customer won't repay would increase by 0.69 times.

## Random Forest

Import the Random Forest classifier:


```{code-cell} ipython3
from sklearn.ensemble import RandomForestClassifier
```

Initialize Random Forest with 100 trees and balance target classes:


```{code-cell} ipython3
rf = RandomForestClassifier(
    n_estimators=100, n_jobs=-1, random_state=42, class_weight="balanced"
)
```

We will search for the best parameters among the following values:


```{code-cell} ipython3
parameters = {
    "max_features": [1, 2, 4],
    "min_samples_leaf": [3, 5, 7, 9],
    "max_depth": [5, 10, 15],
}
```

Also, we will use the stratified k-fold validation again. You should still have the `skf` variable.

**Question 8.** How much higher is the *ROC AUC* of the best random forest model than that of the best logistic regression on validation? Select the closest answer.

1. 0.04
2. 0.03
3. 0.02
4. 0.01

**Answer:** 2.

**Solution:**


```{code-cell} ipython3
%%time
rf_grid_search = GridSearchCV(
    rf, parameters, n_jobs=-1, scoring="roc_auc", cv=skf, verbose=True
)
rf_grid_search = rf_grid_search.fit(X, y)
print(rf_grid_search.best_score_ - grid_search.best_score_)
```

**Question 9.** What feature has the weakest impact in Random Forest model?

1. age
2. NumberOfTime30-59DaysPastDueNotWorse
3. DebtRatio
4. NumberOfTimes90DaysLate
5. NumberOfTime60-89DaysPastDueNotWorse
6. MonthlyIncome
7. NumberOfDependents

**Answer:** 7.

**Solution:**


```{code-cell} ipython3
independent_columns_names[
    np.argmin(rf_grid_search.best_estimator_.feature_importances_)
]
```

Rating of the feature importance:


```{code-cell} ipython3
pd.DataFrame(
    {
        "feat": independent_columns_names,
        "coef": rf_grid_search.best_estimator_.feature_importances_,
    }
).sort_values(by="coef", ascending=False)
```

**Question 10.** What is the most significant advantage of using *Logistic Regression* versus *Random Forest* for this problem?

1. Spent less time for model fitting;
2. Fewer variables to iterate;
3. Feature interpretability;
4. Linear properties of the algorithm.

**Answer:** 3.

**Solution:**

On the one hand, the Random Forest model works better for our credit scoring problem. Its performance is 4% higher. The reason for such a result is a small number of features and the compositional property of random forests.

On the other hand, the main advantage of Logistic Regression is that we can interpret the feature impact on the model outcome.

## Bagging

Import modules and set up the parameters for bagging:


```{code-cell} ipython3
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

parameters = {
    "max_features": [2, 3, 4],
    "max_samples": [0.5, 0.7, 0.9],
    "base_estimator__C": [0.0001, 0.001, 0.01, 1, 10, 100],
}
```

**Question 11.** Fit a bagging classifier with `random_state=42`. For the base classifiers, use 100 logistic regressors and use `RandomizedSearchCV` instead of `GridSearchCV`. It will take a lot of time to iterate over all 54 variants, so set the maximum number of iterations for `RandomizedSearchCV` to 20. Don't forget to set the parameters `cv` and `random_state=1`. What is the best *ROC AUC* you achieve?

1. 80.75%
2. 80.12%
3. 79.62%
4. 76.50%

**Answer:** 1.

**Solution:**

_(the following code is commented out for the Jupyter-book version as it takes ~16 min. to run, a bit too long for CI/CD)_


```{code-cell} ipython3
# bg = BaggingClassifier(
#     LogisticRegression(class_weight="balanced"),
#     n_estimators=100,
#     n_jobs=-1,
#     random_state=42,
# )
# r_grid_search = RandomizedSearchCV(
#     bg,
#     parameters,
#     n_jobs=-1,
#     scoring="roc_auc",
#     cv=skf,
#     n_iter=20,
#     random_state=1,
#     verbose=True,
# )
# r_grid_search = r_grid_search.fit(X, y)
```


```{code-cell} ipython3
# r_grid_search.best_score_
# 0.8076172570918905
```

**Question 12.** Give an interpretation of the best parameters for bagging. Why are these values of `max_features` and `max_samples` the best?

1. For bagging it's important to use as few features as possible;
2. Bagging works better on small samples;
3. Less correlation between single models;
4. The higher the number of features, the lower the loss of information.

**Answer:** 3.

**Solution:**

The advantage of *Random Forest* is that the trees in the composition are not highly correlated. Similarly, for bagging with logistic regression, the weaker correlation between single models, the higher the accuracy. Since in logistic regression there is almost no randomness, we have to change the set of features to minimize the correlation between our single models.
