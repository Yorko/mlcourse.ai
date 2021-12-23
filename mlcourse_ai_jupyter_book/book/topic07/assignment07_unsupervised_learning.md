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

(assignment07)=

# Assignment #7 (demo). Unupervised learning

<img src="https://habrastorage.org/webt/ia/m9/zk/iam9zkyzqebnf_okxipihkgjwnw.jpeg" />

**<center>[mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course** </center><br>

Authors: [Olga Daykhovskaya](https://www.linkedin.com/in/odaykhovskaya/), [Yury Kashnitsky](https://yorko.github.io). This material is subject to the terms and conditions of the [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. Free use is permitted for any non-commercial purpose.

**Same assignment as a [Kaggle Notebook](https://www.kaggle.com/kashnitsky/a7-demo-unsupervised-learning) + [solution](https://www.kaggle.com/kashnitsky/a7-demo-unsupervised-learning-solution).**

In this task, we will look at how data dimensionality reduction and clustering methods work. At the same time, we'll practice solving classification task again.

We will work with the [Samsung Human Activity Recognition](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) dataset. The data comes from accelerometers and gyros of Samsung Galaxy S3 mobile phones ( you can find more info about the features using the link above), the type of activity of a person with a phone in his/her pocket is also known – whether he/she walked, stood, lay, sat or walked up or down the stairs.

First, we pretend that the type of activity is unknown to us, and we will try to cluster people purely on the basis of available features. Then we solve the problem of determining the type of physical activity as a classification problem.

Fill the code where needed ("Your code is here") and answer the questions in the [web form](https://docs.google.com/forms/d/1wBf5UoRndv6PpzIwYnM9f0ysoGa4Yqcqle-HBlBP5QQ/edit).


```{code-cell} ipython3
import os
from zipfile import ZipFile
from pathlib import Path
import requests

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm_notebook

%matplotlib inline
from matplotlib import pyplot as plt

plt.style.use(["seaborn-darkgrid"])
plt.rcParams["figure.figsize"] = (12, 9)
plt.rcParams["font.family"] = "DejaVu Sans"

from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

RANDOM_STATE = 17
```


```{code-cell} ipython3
def load_har_dataset(url, extract_path: Path, filename: str, overwrite=False):
    # check if existed already
    filepath = extract_path / filename
    if filepath.exists() and not overwrite:
        print("The dataset is already in place.")
        return

    print("Downloading the dataset from:  ", url)
    response = requests.get(url)

    with open(filepath, 'wb') as f:
        f.write(response.content)

    with ZipFile(filepath, 'r') as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall(extract_path)
```


```{code-cell} ipython3
FILE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
FILE_NAME = "UCI HAR Dataset.zip"
DATA_PATH = Path("../../_static/data/")

load_har_dataset(url=FILE_URL, extract_path=DATA_PATH, filename=FILE_NAME)

PATH_TO_SAMSUNG_DATA = DATA_PATH / FILE_NAME.strip('.zip')
```


```{code-cell} ipython3
X_train = np.loadtxt(PATH_TO_SAMSUNG_DATA / "train" / "X_train.txt")
y_train = np.loadtxt(PATH_TO_SAMSUNG_DATA / "train" / "y_train.txt").astype(int)

X_test = np.loadtxt(PATH_TO_SAMSUNG_DATA / "test" / "X_test.txt")
y_test = np.loadtxt(PATH_TO_SAMSUNG_DATA / "test" / "y_test.txt").astype(int)
```


```{code-cell} ipython3
# Checking dimensions
assert X_train.shape == (7352, 561) and y_train.shape == (7352,)
assert X_test.shape == (2947, 561) and y_test.shape == (2947,)
```

For clustering, we do not need a target vector, so we'll work with the combination of training and test samples. Merge `X_train` with `X_test`, and `y_train` with `y_test`.


```{code-cell} ipython3
# You code here (read-only in a JupyterBook, pls download as .ipynb and run jupyter-notebook to edit)
```

Define the number of unique values of the labels of the target class.


```{code-cell} ipython3
# np.unique(y)
```


```{code-cell} ipython3
# n_classes = np.unique(y).size
```

[These labels correspond to:](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.names)
- 1 – walking
- 2 – walking upstairs
- 3 – walking downstairs
- 4 – sitting
- 5 – standing
- 6 – laying down

Scale the sample using `StandardScaler` with default parameters.


```{code-cell} ipython3
# You code here (read-only in a JupyterBook, pls download as .ipynb and run jupyter-notebook to edit)
```

Reduce the number of dimensions using PCA, leaving as many components as necessary to explain at least 90% of the variance of the original (scaled) data. Use the scaled dataset and fix `random_state` (RANDOM_STATE constant).


```{code-cell} ipython3
# You code here (read-only in a JupyterBook, pls download as .ipynb and run jupyter-notebook to edit)
# pca =
# X_pca =
```

**Question 1:** <br>
What is the minimum number of principal components required to cover the 90% of the variance of the original (scaled) data?


```{code-cell} ipython3
# You code here (read-only in a JupyterBook, pls download as .ipynb and run jupyter-notebook to edit)
```

**Answer options:**
- 56
- 65
- 66
- 193

**Вопрос 2:**<br>
What percentage of the variance is covered by the first principal component? Round to the nearest percent.

**Answer options:**
- 45
- 51
- 56
- 61


```{code-cell} ipython3
# You code here (read-only in a JupyterBook, pls download as .ipynb and run jupyter-notebook to edit)
```

Visualize data in projection on the first two principal components.


```{code-cell} ipython3
# You code here (read-only in a JupyterBook, pls download as .ipynb and run jupyter-notebook to edit)
# plt.scatter(, , c=y, s=20, cmap='viridis');
```

**Question 3:**<br>
If everything worked out correctly, you will see a number of clusters, almost perfectly separated from each other. What types of activity are included in these clusters? <br>

**Answer options:**
- 1 cluster: all 6 activities
- 2 clusters: (walking, walking upstairs, walking downstairs ) and (sitting, standing, laying)
- 3 clusters: (walking), (walking upstairs, walking downstairs) and (sitting, standing, laying)
- 6 clusters

------------------------------

Perform clustering with the `KMeans` method, training the model on data with reduced dimensionality (by PCA). In this case, we will give a clue to look for exactly 6 clusters, but in general case we will not know how many clusters we should be looking for.

Options:

- `n_clusters` = n_classes (number of unique labels of the target class)
- `n_init` = 100
- `random_state` = RANDOM_STATE (for reproducibility of the result)

Other parameters should have default values.


```{code-cell} ipython3
# You code here (read-only in a JupyterBook, pls download as .ipynb and run jupyter-notebook to edit)
```

Visualize data in projection on the first two principal components. Color the dots according to the clusters obtained.


```{code-cell} ipython3
# You code here (read-only in a JupyterBook, pls download as .ipynb and run jupyter-notebook to edit)
# plt.scatter(, , c=cluster_labels, s=20, cmap='viridis');
```

Look at the correspondence between the cluster marks and the original class labels and what kinds of activities the `KMeans` algorithm is confused at.


```{code-cell} ipython3
# tab = pd.crosstab(y, cluster_labels, margins=True)
# tab.index = ['walking', 'going up the stairs',
#             'going down the stairs', 'sitting', 'standing', 'laying', 'all']
# tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['all']
# tab
```

We see that for each class (i.e., each activity) there are several clusters. Let's look at the maximum percentage of objects in a class that are assigned to a single cluster. This will be a simple metric that characterizes how easily the class is separated from others when clustering.

Example: if for class "walking downstairs" (with 1406 instances belonging to it), the distribution of clusters is:
 - cluster 1 - 900
 - cluster 3 - 500
 - cluster 6 - 6,

then such a share will be 900/1406 $ \approx $ 0.64.


**Question 4:** <br>
Which activity is separated from the rest better than others based on the simple metric described above? <br>

**Answer options:**
- walking
- standing
- walking downstairs
- all three options are incorrect

It can be seen that kMeans does not distinguish activities very well. Use the elbow method to select the optimal number of clusters. Parameters of the algorithm and the data we use are the same as before, we change only `n_clusters`.


```{code-cell} ipython3
# # You code here (read-only in a JupyterBook, pls download as .ipynb and run jupyter-notebook to edit)
# inertia = []
# for k in tqdm_notebook(range(1, n_classes + 1)):
#     pass
```

**Question 5:** <br>
How many clusters can we choose according to the elbow method? <br>

**Answer options:**
- 1
- 2
- 3
- 4

------------------------

Let's try another clustering algorithm, described in the article – agglomerative clustering.


```{code-cell} ipython3
# ag = AgglomerativeClustering(n_clusters=n_classes,
#                              linkage='ward').fit(X_pca)
```

Calculate the Adjusted Rand Index (`sklearn.metrics`) for the resulting clustering and for ` KMeans` with the parameters from the 4th question.


```{code-cell} ipython3
# You code here (read-only in a JupyterBook, pls download as .ipynb and run jupyter-notebook to edit)
```

**Question 6:** <br>
Select all the correct statements. <br>

** Answer options:**
- According to ARI, KMeans handled clustering worse than Agglomerative Clustering
- For ARI, it does not matter which tags are assigned to the cluster, only the partitioning of instances into clusters matters
- In case of random partitioning into clusters, ARI will be close to zero

-------------------------------

You can notice that the task is not very well solved when we try to detect several clusters (> 2). Now, let's solve the classification problem, given that the data is labeled.

For classification, use the support vector machine – class `sklearn.svm.LinearSVC`. In this course, we didn't study this algorithm separately, but it is well-known and you can read about it, for example [here](http://cs231n.github.io/linear-classify/#svmvssoftmax).

Choose the `C` hyperparameter for` LinearSVC` using `GridSearchCV`.

- Train the new `StandardScaler` on the training set (with all original features), apply scaling to the test set
- In `GridSearchCV`, specify `cv` = 3.


```{code-cell} ipython3
# # You code here (read-only in a JupyterBook, pls download as .ipynb and run jupyter-notebook to edit)
# scaler = StandardScaler()
# X_train_scaled =
# X_test_scaled =
```


```{code-cell} ipython3
svc = LinearSVC(random_state=RANDOM_STATE)
svc_params = {"C": [0.001, 0.01, 0.1, 1, 10]}
```


```{code-cell} ipython3
# %%time
# # You code here (read-only in a JupyterBook, pls download as .ipynb and run jupyter-notebook to edit)
# best_svc = None
```


```{code-cell} ipython3
# best_svc.best_params_, best_svc.best_score_
```

**Question 7**<br>
Which value of the hyperparameter `C` was chosen the best on the basis of cross-validation? <br>

**Answer options:**
- 0.001
- 0.01
- 0.1
- 1
- 10


```{code-cell} ipython3
# y_predicted = best_svc.predict(X_test_scaled)
```


```{code-cell} ipython3
# tab = pd.crosstab(y_test, y_predicted, margins=True)
# tab.index = ['walking', 'climbing up the stairs',
#              'going down the stairs', 'sitting', 'standing', 'laying', 'all']
# tab.columns = ['walking', 'climbing up the stairs',
#              'going down the stairs', 'sitting', 'standing', 'laying', 'all']
# tab
```

**Question 8:** <br>
Which activity type is worst detected by SVM in terms of precision? Recall?<br>

**Answer options:**
- precision – going up the stairs, recall – laying
- precision – laying, recall – sitting
- precision – walking, recall – walking
- precision – standing, recall – sitting

Finally, do the same thing as in Question 7, but add PCA.

- Use `X_train_scaled` and` X_test_scaled`
- Train the same PCA as before, on the scaled training set, apply scaling to the test set
- Choose the hyperparameter `C` via cross-validation on the training set with PCA-transformation. You will notice how much faster it works now.

**Question 9:** <br>
What is the difference between the best quality (accuracy) for cross-validation in the case of all 561 initial characteristics and in the second case, when the principal component method was applied? Round to the nearest percent. <br>

**Answer options:**
- quality is the same
- 2%
- 4%
- 10%
- 20%


```{code-cell} ipython3
# You code here (read-only in a JupyterBook, pls download as .ipynb and run jupyter-notebook to edit)
```

**Question 10:** <br>
Select all the correct statements:

**Answer options:**
- Principal component analysis in this case allowed to reduce the model training time, while the quality (mean cross-validation accuracy) suffered greatly, by more than 10%
- PCA can be used to visualize data, but there are better methods for this task, for example, tSNE. However, PCA has lower computational complexity
- PCA builds linear combinations of initial features, and in some applications they might be poorly interpreted by humans
