#!/usr/bin/env python
# coding: utf-8

# <img src="https://habrastorage.org/webt/ia/m9/zk/iam9zkyzqebnf_okxipihkgjwnw.jpeg" />
#     
# **<center>[mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course** </center><br>
# Authors: Arina Lopukhova (@erynn), and [Yury Kashnitsky](https://yorko.github.io) (@yorko). Edited by Vadim Shestopalov (@vchulski). [mlcourse.ai](https://mlcourse.ai) is powered by [OpenDataScience (ods.ai)](https://ods.ai/) © 2017—2022

# # <center>Assignment #1. Task </center> <a class="tocSkip">
# ## <center>Exploratory Data Analysis (EDA) of Olympic games with Pandas </center> <a class="tocSkip">
#     
# <img src="https://habrastorage.org/webt/my/70/d9/my70d97xhwfj8krp2q2qmn_smww.png" width=50%>

# There are ten questions about 120 years of Olympic history in this assignment.
# 
# ### Your task is to:
#  1. write code and perform computations in the cells below;
#  2. choose answers in the [webform](https://docs.google.com/forms/d/1SJcNJxnY5Wwb_sDSxmb-FyghraeOt-ZmQMA0XzOLu4I).
#  
# *If you are sure that something is not 100% correct with the assignment/solution, please leave your feedback via the mentioned webform ↑*
# 
# -----

# In[1]:


from pathlib import Path

import pandas as pd


# In[2]:


PATH_TO_DATA = Path("../../_static/data/assignment1")


# In[3]:


df = pd.read_csv(PATH_TO_DATA / "athlete_events.csv.zip", index_col="ID")
df.head(2)


#  The dataset has the following features:
# 
# - __ID__ – Unique number for each athlete
# - __Name__ – Athlete's name
# - __Sex__ – M or F
# - __Age__ – Integer
# - __Height__ – In centimeters
# - __Weight__ – In kilograms
# - __Team__ – Team name
# - __NOC__ – National Olympic Committee 3-letter code
# - __Games__ – Year and season
# - __Year__ – Integer
# - __Season__ – Summer or Winter
# - __City__ – Host city
# - __Sport__ – Sport
# - __Event__ – Event
# - __Medal__ – Gold, Silver, Bronze, or NA

# **<font color='red'>Question 1.</font> How old were the youngest male and female participants of the 1992 Olympics?**
# 
# - 16 and 15
# - 14 and 13 
# - 13 and 11
# - 11 and 12

# In[4]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# **<font color='red'>Question 2.</font> What was the percentage of male basketball players among all the male participants of the 2012 Olympics? Round the answer to the first decimal.**
# 
# *Hint:* drop duplicate athletes where necessary to count each athlete just once. This applies to other questions as well.
# 
# - 0.2
# - 1.5 
# - 2.5
# - 7.7

# In[5]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# **<font color='red'>Question 3.</font> What are the mean and standard deviation of height for female tennis players who participated in the 2000 Olympics? Round the answer to the first decimal.**
# 
# - 171.8 and 6.5
# - 179.4 and 10
# - 180.7 and 6.7
# - 182.4 and 9.1 

# In[6]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# **<font color='red'>Question 4.</font> Find the heaviest athlete among 2006 Olympics participants. What sport did he or she do?**
# 
# - Judo
# - Bobsleigh 
# - Skeleton
# - Boxing

# In[7]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# **<font color='red'>Question 5.</font> How many times did John Aalberg participate in the Olympics held in different years?**
# 
# - 0
# - 1 
# - 2
# - 3 

# In[8]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# **<font color='red'>Question 6.</font> How many gold medals in tennis did the Switzerland team win at the 2008 Olympics?**
# 
# - 0
# - 1 
# - 2
# - 3 

# In[9]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# **<font color='red'>Question 7.</font>  Is it true that Spain won fewer medals than Italy at the 2016 Olympics? Do not consider NaN values in the _Medal_ column.**
# 
# - Yes
# - No

# In[10]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# **<font color='red'>Question 8.</font> What are the least and most common age groups among the participants of the 2008 Olympics?**
# - [45-55] and [25-35) correspondingly
# - [45-55] and [15-25) correspondingly
# - [35-45) and [25-35) correspondingly
# - [45-55] and [35-45) correspondingly

# In[11]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# **<font color='red'>Question 9.</font> Is it true that there were Summer Olympics held in Atlanta? Is it true that there were Winter Olympics held in Squaw Valley?**
# 
# - Yes, Yes
# - Yes, No
# - No, Yes 
# - No, No 

# In[12]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# **<font color='red'>Question 10.</font> What is the absolute difference between the number of unique sports at the 1994 Olympics and 2002 Olympics?**
# 
# - 3
# - 10
# - 15 
# - 27 

# In[13]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)

