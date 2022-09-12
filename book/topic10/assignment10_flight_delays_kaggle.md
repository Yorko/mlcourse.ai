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

(assignment10)=

# Assignment #10 (demo). Gradient boosting

<img src="https://habrastorage.org/webt/ia/m9/zk/iam9zkyzqebnf_okxipihkgjwnw.jpeg" />

**<center>[mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course** </center><br>

Author: [Yury Kashnitsky](https://yorko.github.io). All content is distributed under the [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.


Your task is to beat at least 2 benchmarks in this [Kaggle Inclass competition](https://www.kaggle.com/c/flight-delays-spring-2018). Here you won’t be provided with detailed instructions. We only give you a brief description of how the second benchmark was achieved using Xgboost. Hopefully, at this stage of the course, it's enough for you to take a quick look at the data in order to understand that this is the type of task where gradient boosting will perform well. Most likely it will be Xgboost, however, we’ve got plenty of categorical features here.

```{figure} /_static/img/xgboost_meme.jpg
:name: ods_stickers
:width: 444px
```


```{code-cell} ipython3
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
```


```{code-cell} ipython3
# for Jupyter-book, we copy data from GitHub, locally, to save Internet traffic,
# you can specify the data/ folder from the root of your cloned
# https://github.com/Yorko/mlcourse.ai repo, to save Internet traffic
DATA_PATH = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/master/data/"
```


```{code-cell} ipython3
train = pd.read_csv(DATA_PATH + "flight_delays_train.csv")
test = pd.read_csv(DATA_PATH + "flight_delays_test.csv")
```


```{code-cell} ipython3
train.head()
```


```{code-cell} ipython3
test.head()
```

Given flight departure time, carrier's code, departure airport, destination location, and flight distance, you have to predict departure delay for more than 15 minutes. As the simplest benchmark, let's take Xgboost classifier and two features that are easiest to take: DepTime and Distance. Such model results in 0.68202 on the LB.


```{code-cell} ipython3
X_train = train[["Distance", "DepTime"]].values
y_train = train["dep_delayed_15min"].map({"Y": 1, "N": 0}).values
X_test = test[["Distance", "DepTime"]].values

X_train_part, X_valid, y_train_part, y_valid = train_test_split(
    X_train, y_train, test_size=0.3, random_state=17
)
```

We'll train Xgboost with default parameters on part of data and estimate holdout ROC AUC.


```{code-cell} ipython3
xgb_model = XGBClassifier(seed=17)

xgb_model.fit(X_train_part, y_train_part)
xgb_valid_pred = xgb_model.predict_proba(X_valid)[:, 1]

roc_auc_score(y_valid, xgb_valid_pred)
```

Now we do the same with the whole training set, make predictions to test set and form a submission file. This is how you beat the first benchmark.


```{code-cell} ipython3
xgb_model.fit(X_train, y_train)
xgb_test_pred = xgb_model.predict_proba(X_test)[:, 1]

pd.Series(xgb_test_pred, name="dep_delayed_15min").to_csv(
    "xgb_2feat.csv", index_label="id", header=True
)
```

The second benchmark in the leaderboard was achieved as follows:

- Features `Distance` and `DepTime` were taken unchanged
- A feature `Flight` was created from features `Origin` and `Dest`
- Features `Month`, `DayofMonth`, `DayOfWeek`, `UniqueCarrier` and `Flight` were transformed with OHE (`LabelBinarizer`)
- Logistic regression and gradient boosting (xgboost) were trained. Xgboost hyperparameters were tuned via cross-validation. First, the hyperparameters responsible for model complexity were optimized, then the number of trees was fixed at 500 and learning step was tuned.
- Predicted probabilities were made via cross-validation using `cross_val_predict`. A linear mixture of logistic regression and gradient boosting predictions was set in the form $w_1 * p_{logit} + (1 - w_1) * p_{xgb}$, where $p_{logit}$ is a probability of class 1, predicted by logistic regression, and $p_{xgb}$ – the same for xgboost. $w_1$ weight was selected manually.
- A similar combination of predictions was made for test set.

Following the same steps is not mandatory. That’s just a description of how the result was achieved by the author of this assignment. Perhaps you might not want to follow the same steps, and instead, let’s say, add a couple of good features and train a random forest of a thousand trees.

Good luck!
