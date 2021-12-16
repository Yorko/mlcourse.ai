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

(assignment01_solution)=

# Assignment #1 (demo). Exploratory data analysis with Pandas. Solution

```{figure} /_static/img/ods_stickers.jpg
:name: ods_stickers
```

**<center>[mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course** </center><br>

Author: [Yury Kashnitsky](https://www.linkedin.com/in/festline/). Translated and edited by [Sergey Isaev](https://www.linkedin.com/in/isvforall/), [Artem Trunov](https://www.linkedin.com/in/datamove/), [Anastasia Manokhina](https://www.linkedin.com/in/anastasiamanokhina/), and [Yuanyuan Pao](https://www.linkedin.com/in/yuanyuanpao/). All content is distributed under the [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.


**Same assignment as a [Kaggle Kernel](https://www.kaggle.com/kashnitsky/a1-demo-pandas-and-uci-adult-dataset) + [solution](https://www.kaggle.com/kashnitsky/a1-demo-pandas-and-uci-adult-dataset-solution).**

**In this task you should use Pandas to answer a few questions about the [Adult](https://archive.ics.uci.edu/ml/datasets/Adult) dataset. (You don't have to download the data – it's already  in the repository). Choose the answers in the [web-form](https://docs.google.com/forms/d/1uY7MpI2trKx6FLWZte0uVh3ULV4Cm_tDud0VDFGCOKg).**

Unique values of features (for more information please see the link above):
- `age`: continuous;
- `workclass`: `Private`, `Self-emp-not-inc`, `Self-emp-inc`, `Federal-gov`, `Local-gov`, `State-gov`, `Without-pay`, `Never-worked`;
- `fnlwgt`: continuous;
- `education`: `Bachelors`, `Some-college`, `11th`, `HS-grad`, `Prof-school`, `Assoc-acdm`, `Assoc-voc`, `9th`, `7th-8th`, `12th`, `Masters`, `1st-4th`, `10th`, `Doctorate`, `5th-6th`, `Preschool`;
- `education-num`: continuous;
- `marital-status`: `Married-civ-spouse`, `Divorced`, `Never-married`, `Separated`, `Widowed`, `Married-spouse-absent`, `Married-AF-spouse`,
- `occupation`: `Tech-support`, `Craft-repair`, `Other-service`, `Sales`, `Exec-managerial`, `Prof-specialty`, `Handlers-cleaners`, `Machine-op-inspct`, `Adm-clerical`, `Farming-fishing`, `Transport-moving`, `Priv-house-serv`, `Protective-serv`, `Armed-Forces`;
- `relationship`: `Wife`, `Own-child`, `Husband`, `Not-in-family`, `Other-relative`, `Unmarried`;
- `race`: `White`, `Asian-Pac-Islander`, `Amer-Indian-Eskimo`, `Other`, `Black`;
- `sex`: `Female`, `Male`;
- `capital-gain`: continuous.
- `capital-loss`: continuous.
- `hours-per-week`: continuous.
- `native-country`: `United-States`, `Cambodia`, `England`, `Puerto-Rico`, `Canada`, `Germany`, `Outlying-US(Guam-USVI-etc)`, `India`, `Japan`, `Greece`, `South`, `China`, `Cuba`, `Iran`, `Honduras`, `Philippines`, `Italy`, `Poland`, `Jamaica`, `Vietnam`, `Mexico`, `Portugal`, `Ireland`, `France`, `Dominican-Republic`, `Laos`, `Ecuador`, `Taiwan`, `Haiti`, `Columbia`, `Hungary`, `Guatemala`, `Nicaragua`, `Scotland`, `Thailand`, `Yugoslavia`, `El-Salvador`, `Trinadad&Tobago`, `Peru`, `Hong`, `Holand-Netherlands`;
- `salary`: `>50K`, `<=50K`.


```{code-cell} ipython3
import numpy as np
import pandas as pd

pd.set_option("display.max.columns", 100)
# to draw pictures in jupyter notebook
%matplotlib inline
# we don't like warnings
# you can comment the following 2 lines if you'd like to
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
```


```{code-cell} ipython3
# for Jupyter-book, we copy data from GitHub, locally, to save Internet traffic,
# you can specify the data/ folder from the root of your cloned
# https://github.com/Yorko/mlcourse.ai repo, to save Internet traffic
DATA_URL = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/master/data/"
```


```{code-cell} ipython3
data = pd.read_csv(DATA_URL + "adult.data.csv")
data.head()
```

**1. How many men and women (*sex* feature) are represented in this dataset?**


```{code-cell} ipython3
data["sex"].value_counts()
```

**2. What is the average age (*age* feature) of women?**


```{code-cell} ipython3
data[data["sex"] == "Female"]["age"].mean()
```

**3. What is the proportion of German citizens (*native-country* feature)?**


```{code-cell} ipython3
float((data["native-country"] == "Germany").sum()) / data.shape[0]
```

**4-5. What are mean value and standard deviation of the age of those who receive more than 50K per year (*salary* feature) and those who receive less than 50K per year? **


```{code-cell} ipython3
ages1 = data[data["salary"] == ">50K"]["age"]
ages2 = data[data["salary"] == "<=50K"]["age"]
print(
    "The average age of the rich: {0} +- {1} years, poor - {2} +- {3} years.".format(
        round(ages1.mean()),
        round(ages1.std(), 1),
        round(ages2.mean()),
        round(ages2.std(), 1),
    )
)
```

**6. Is it true that people who earn more than 50K have at least high school education? (*education* – `Bachelors`, `Prof-school`, `Assoc-acdm`, `Assoc-voc`, `Masters` or `Doctorate` feature)**


```{code-cell} ipython3
data[data["salary"] == ">50K"]["education"].unique()  # No
```

**7. Display age statistics for each race (*race* feature) and each gender (*sex* feature). Use *groupby()* and *describe()*. Find the maximum age of men of `Amer-Indian-Eskimo` race.**


```{code-cell} ipython3
for (race, sex), sub_df in data.groupby(["race", "sex"]):
    print("Race: {0}, sex: {1}".format(race, sex))
    print(sub_df["age"].describe())
```

**8. Among whom is the proportion of those who earn a lot (`>50K`) greater: married or single men (*marital-status* feature)? Consider as married those who have a *marital-status* starting with *Married* (`Married-civ-spouse`, `Married-spouse-absent` or `Married-AF-spouse`), the rest are considered bachelors.**


```{code-cell} ipython3
data[
    (data["sex"] == "Male")
    & (data["marital-status"].isin(["Never-married", "Separated", "Divorced"]))
]["salary"].value_counts()
```


```{code-cell} ipython3
data[(data["sex"] == "Male") & (data["marital-status"].str.startswith("Married"))][
    "salary"
].value_counts()
```


```{code-cell} ipython3
data["marital-status"].value_counts()
```

It's good to be married :)

**9. What is the maximum number of hours a person works per week (*hours-per-week* feature)? How many people work such a number of hours and what is the percentage of those who earn a lot among them?**


```{code-cell} ipython3
max_load = data["hours-per-week"].max()
print("Max time - {0} hours./week.".format(max_load))

num_workaholics = data[data["hours-per-week"] == max_load].shape[0]
print("Total number of such hard workers {0}".format(num_workaholics))

rich_share = (
    float(
        data[(data["hours-per-week"] == max_load) & (data["salary"] == ">50K")].shape[0]
    )
    / num_workaholics
)
print("Percentage of rich among them {0}%".format(int(100 * rich_share)))
```

**10. Count the average time of work (*hours-per-week*) those who earning a little and a lot (*salary*) for each country (*native-country*).**

Simple method:


```{code-cell} ipython3
for (country, salary), sub_df in data.groupby(["native-country", "salary"]):
    print(country, salary, round(sub_df["hours-per-week"].mean(), 2))
```

Elegant method:


```{code-cell} ipython3
pd.crosstab(
    data["native-country"],
    data["salary"],
    values=data["hours-per-week"],
    aggfunc=np.mean,
).T
```
