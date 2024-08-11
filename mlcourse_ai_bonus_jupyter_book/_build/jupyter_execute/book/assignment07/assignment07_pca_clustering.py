#!/usr/bin/env python
# coding: utf-8

# <img src="https://habrastorage.org/webt/ia/m9/zk/iam9zkyzqebnf_okxipihkgjwnw.jpeg" />
#     
# **<center>[mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course** </center><br>
# Author: [Yury Kashnitskiy](https://yorko.github.io) (@yorko). Translated and edited by [Egor Polusmak](https://www.linkedin.com/in/egor-polusmak/), [Anastasia Manokhina](https://www.linkedin.com/in/anastasiamanokhina/), [Eugene Mashkin](https://www.linkedin.com/in/eugene-mashkin-88490883/), and [Yuanyuan Pao](https://www.linkedin.com/in/yuanyuanpao/). [mlcourse.ai](https://mlcourse.ai) is powered by [OpenDataScience (ods.ai)](https://ods.ai/) © 2017—2022

# # <center>Assignment #7. Task </center> <a class="tocSkip">
#     
# ## <center> Principal Component Analysis and Clustering </center><a class="tocSkip">

# In this assignment, we are going to walk through `sklearn` built-in implementations of dimensionality reduction and clustering methods and apply these techniques to the popular "faces" dataset.
# 
# 
# **Your task is to:**
#  1. write code and perform computations in the cells below;
#  2. choose answers in the [webform](https://forms.gle/W1dtNVMSVpSQBt7s5).
#  
# *If you are sure that something is not 100% correct with the assignment/solution, please leave your feedback via the mentioned webform ↑*
# 
# 
# ## 1. Principal Component Analysis
# 
# First import all required modules:

# In[1]:


import seaborn as sns
from matplotlib import pyplot as plt

sns.set(style="white")
import numpy as np
import pandas as pd
from sklearn import datasets, metrics
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Use the given toy data set:

# In[2]:


X = np.array(
    [
        [2.0, 13.0],
        [1.0, 3.0],
        [6.0, 19.0],
        [7.0, 18.0],
        [5.0, 17.0],
        [4.0, 9.0],
        [5.0, 22.0],
        [6.0, 11.0],
        [8.0, 25.0],
    ]
)


# In[3]:


plt.scatter(X[:, 0], X[:, 1])
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$");


# **<font color='red'>Question 1.</font> What is the angle between the $x_1$ axis and the vector corresponding to the first principal component for this data (don't forget to scale data using `StandardScaler`)?**
# 
# - 30 degrees
# - 45 degrees
# - 60 degrees
# - 75 degrees

# In[4]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# **<font color='red'>Question 2.</font> What are the eigenvalues of the $X^{\text{T}}X$ matrix, given $X$, a scaled matrix from the previous question?**
# 
#  - 4 and 1.42
#  - 16.2 and 2702.8
#  - 4.02 and 51.99
#  - 15.97 and 2.03

# In[5]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# **<font color='red'>Question 3.</font> What is the meaning of the two numbers from the previous question?**
# 
# - their squares tell what part of the initial data's variance is explained by principal components
# - they define a rotation angle between the first principal component and the initial axis
# - those numbers tell what part of the initial data's variance is explained by principal components
# - the square roots of those numbers define a rotation angle between the first principal component and the initial axis

# Let's load a dataset of peoples' faces and output their names. (This step requires stable, fast internet connection.)

# In[6]:


lfw_people = datasets.fetch_lfw_people(
    min_faces_per_person=50, resize=0.4, data_home="../../_static/data/faces"
)

print(
    "%d objects, %d features, %d classes"
    % (lfw_people.data.shape[0], lfw_people.data.shape[1], len(lfw_people.target_names))
)
print("\nPersons:")
for name in lfw_people.target_names:
    print(name)


# Let's look at some faces. All images are stored in a handy `lfw_people.images` array.

# In[7]:


fig = plt.figure(figsize=(8, 6))

for i in range(15):
    ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(lfw_people.images[i], cmap="gray")


# **<font color='red'>Question 4.</font> What's the _minimal_ number of principal components is needed to explain 90% variance in the data (scaled using  `StandardScaler`)?**
# 
# - 75
# - 76
# - 77
# - 78
# 
# For this task, you should be using the [`svd_solver='randomized'`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) parameter, which is a PCA approximation, but it significantly increases performance on large data sets. Use fixed `random_state=1` for comparable results.

# In[8]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# Print a picture showing the first 30 principal components (don't be scared when you see the results). In order to create it, use 30 vectors from `pca.components_`, reshape them to their initial size (50 x 37), and display. Specify `cmap`='binary'.

# In[9]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# **<font color='red'>Question 5.</font> Within the first 30 principal components, which one brightens the left side of the face? More specifically, which principal component corresponds to a linear combination of the initial features (pixels' intensity), which, when shown as an image, looks like a photo highlighted from the right side (the same as a face highlighted from its left side)?**
# 
# - 1
# - 2
# - 4
# - 5

# Now let's create a projection of faces onto the space of the first two principal components. 
# 
# **<font color='red'>Question 6.</font> Who looks the least similar to the other people in the dataset if we only consider the two first principal components?**
# 
# To answer this question, take the first two principal components from the scaled data, evaluate two mean principal components' values for each person over all their images in the dataset (again, use both svd_solver='randomized' and random_state=1). Then, with 12 two-dimensional points, find the one which has the largest distance from the others (by Euclidean distance). You can do this either precisely or approximately using `sklearn.metrics.euclidean_distances` and `seaborn.heatmap`.
# 
# - Colin Powell
# - George W Bush
# - Jacques Chirac
# - Serena Williams

# In[10]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# ## 2. Clustering

# For the next question, load the housing prices dataset:

# In[11]:


boston = datasets.load_boston()
X = boston.data


# Using the elbow-method (reference [article 7](https://medium.com/@libfun/db7879568417) of the course), find the optimal number of clusters to set as a hyperparameter for the k-means algorithm.

# **<font color='red'>Question 7.</font> What is the optimal number of clusters to use on housing prices data set according to the elbow method? Use `random_state=1` in the k-means method, and don't scale the data.**
# 
# - 2
# - 3
# - 4
# - 5
# 
# In this case, we are looking for the most significant curve fracture on the `Cluster number vs Centroid distances` graph. Consider the number of clusters from 2 to 10. Use `random_state=1` for the k-means algorithm initialization.

# In[12]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# Go back to the faces dataset (that is already scaled). Imagine that we did not know the names for who was each photo but that we knew that there were 12 different people. Let's compare clustering results from 4 algorithms - k-means, Agglomerative clustering, Affinity Propagation, and Spectral clustering. Use the same respective parameters as in the end of [this article](https://medium.com/@libfun/db7879568417), only change the number of clusters to 12.

# In[13]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# **<font color='red'>Question 8.</font> Question 8. Select all of the correct statements:**
# 
# - Agglomerative clustering worked better than others according to all metrics
# - Clustering results are disappointing with no metric exceeding 0.35
# - Considering only 2 clusters (whether it is Serena Williams or not) and comparing clustering results with a binary vector, we can see that clustering algorithms work better, with some metrics exceeding 66%

# Use the coordinates of the 12 "average" people's images you got before. Draw a dendrogram for them. Use `scipy.cluster.hierarchy` and `scipy.spatial.distance.pdist`, take parameters values from the appropriate example in the [article](https://medium.com/@libfun/db7879568417).

# In[14]:


from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# **<font color='red'>Question 9.</font> Look at the dendrogram and consider a step when just two clusters are left: Serena Williams vs. all.  Who was the last person added to the "big" cluster?**
# 
# - Gerhard Schroeder
# - Jean Chretien 
# - John Ashcroft
# - Junichiro Koizumi

# In[15]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)

