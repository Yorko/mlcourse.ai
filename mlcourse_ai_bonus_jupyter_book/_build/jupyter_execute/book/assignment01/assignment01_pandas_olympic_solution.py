#!/usr/bin/env python
# coding: utf-8

# <img src="https://habrastorage.org/webt/ia/m9/zk/iam9zkyzqebnf_okxipihkgjwnw.jpeg" />
#     
# **<center>[mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course** </center><br>
# Authors: Arina Lopukhova (@erynn), and [Yury Kashnitsky](https://yorko.github.io) (@yorko). Edited by Vadim Shestopalov (@vchulski). [mlcourse.ai](https://mlcourse.ai) is powered by [OpenDataScience (ods.ai)](https://ods.ai/) © 2017—2021 

# # <center>Assignment #1. Solution </center> <a class="tocSkip">
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
# - 11 and 12 **<font color='red'>[+]</font>**

# Straightforward way:

# In[4]:


# solution code
# male
df.loc[(df.Year == 1992) & (df["Sex"] == "M"), "Age"].min()


# In[5]:


# female
df.loc[(df.Year == 1992) & (df["Sex"] == "F"), "Age"].min()


# **<font color='red'>Question 2.</font> What was the percentage of male basketball players among all the male participants of the 2012 Olympics? Round the answer to the first decimal.**
# 
# *Hint:* drop duplicate athletes where necessary to count each athlete just once. This applies to other questions as well.
# 
# - 0.2
# - 1.5 
# - 2.5 **<font color='red'>[+]</font>**
# - 7.7

# In[6]:


# solution code
q = '(Year == 2012) & (Sex == "M")'
# dfquery(q) equals to data[(dfYear == 2012) & (dfSex == 'M')]
sportsmen_count = len(df.query(q).drop_duplicates(subset="Name"))

basketball_count = len(
    df.query(q).query('Sport == "Basketball"').drop_duplicates(subset="Name")
)

print("%.1f" % (basketball_count / sportsmen_count * 100))


# **<font color='red'>Question 3.</font> What are the mean and standard deviation of height for female tennis players who participated in the 2000 Olympics? Round the answer to the first decimal.**
# 
# - 171.8 and 6.5 **<font color='red'>[+]</font>**
# - 179.4 and 10
# - 180.7 and 6.7
# - 182.4 and 9.1 

# In[7]:


# solution code
q = '(Year == 2000) & (Sex == "F") & (Sport == "Tennis")'
(df.query(q).describe().Height)


# **<font color='red'>Question 4.</font> Find the heaviest athlete among 2006 Olympics participants. What sport did he or she do?**
# 
# - Judo
# - Bobsleigh 
# - Skeleton **<font color='red'>[+]</font>**
# - Boxing

# In[8]:


# solution code
max_weight_idx = df[df.Year == 2006].Weight.idxmax()

(df[df.Year == 2006].loc[max_weight_idx].Sport)


# In[9]:


df[df.Year == 2006].Season.unique()


# **<font color='red'>Question 5.</font> How many times did John Aalberg participate in the Olympics held in different years?**
# 
# - 0
# - 1 
# - 2 **<font color='red'>[+]</font>**
# - 3 

# Straightforward way:

# In[10]:


# solution code
len(df[df.Name == "John Aalberg"].drop_duplicates(subset="Year").Year)


# "Fancier" way:

# In[11]:


len(df[df.Name == "John Aalberg"].groupby("Year"))


# **<font color='red'>Question 6.</font> How many gold medals in tennis did the Switzerland team win at the 2008 Olympics?**
# 
# - 0
# - 1 
# - 2 **<font color='red'>[+]</font>**
# - 3 

# Straightforward way:

# In[12]:


# solution code
q = '(Year == 2008) & (Team == "Switzerland") & (Medal == "Gold") & (Sport == "Tennis")'
df.query(q).shape[0]


# "Fancier" way:

# In[13]:


q = '(Year == 2008) & (Team == "Switzerland")'
(df.query(q).groupby(["Medal", "Sport"]).get_group(("Gold", "Tennis")).shape[0])


# **<font color='red'>Question 7.</font> Is it true that Spain won fewer medals than Italy at the 2016 Olympics? Do not consider NaN values in the _Medal_ column.**
# 
# - Yes **<font color='red'>[+]</font>**
# - No

# Straightforward way:

# In[14]:


# solution code
spain_medals = len(
    df.query('(Year == 2016) & (Team == "Spain")').dropna(subset=["Medal"])
)

italy_medals = len(
    df.query('(Year == 2016) & (Team == "Italy")').dropna(subset=["Medal"])
)

spain_medals, italy_medals


# "Fancier" way:

# In[15]:


(
    df[df.Year == 2016]
    .dropna(subset=["Medal"])
    .groupby("Team")
    .size()[["Spain", "Italy"]]
)


# **<font color='red'>Question 8.</font> What are the least and most common age groups among the participants of the 2008 Olympics?**
# - [45-55] and [25-35) correspondingly **<font color='red'>[+]</font>**
# - [45-55] and [15-25) correspondingly
# - [35-45) and [25-35) correspondingly
# - [45-55] and [35-45) correspondingly

# Straightforward way:

# In[16]:


# solution code
participants_2008 = df[df.Year == 2008].drop_duplicates(subset="Name")

print("[15, 25): ", len(participants_2008.query("(Age >= 15) & (Age < 25)")))

print("[25, 35): ", len(participants_2008.query("(Age >= 25) & (Age < 35)")))

print("[35, 45): ", len(participants_2008.query("(Age >= 35) & (Age < 45)")))

print("[45, 55]: ", len(participants_2008.query("(Age >= 45) & (Age <= 55)")))


# "Fancier" way:

# In[17]:


def age_category(age):
    """Maps age to four categories"""

    if 15 <= age < 25:
        return "[15, 25)"
    elif 25 <= age < 35:
        return "[25, 35)"
    elif 35 <= age < 45:
        return "[35, 45)"
    elif 45 <= age <= 55:
        return "[45, 55]"


# map() applies age_category() function to every value in dfAge
df["age_category"] = df.Age.map(age_category)
(df[df.Year == 2008].drop_duplicates(subset="Name").groupby("age_category").size())


# **<font color='red'>Question 9.</font>  Is it true that there were Summer Olympics held in Atlanta? Is it true that there were Winter Olympics held in Squaw Valley?**
# - Yes, Yes **<font color='red'>[+]</font>**
# - Yes, No
# - No, Yes 
# - No, No 

# In[18]:


# solution code
pd.crosstab(df.City, df.Season).loc[["Atlanta", "Squaw Valley"]]


# **<font color='red'>Question 10.</font> What is the absolute difference between the number of unique sports at the 1994 Olympics and 2002 Olympics?**
# 
# - 3 **<font color='red'>[+]</font>**
# - 10
# - 15 
# - 27 

# In[19]:


# solution code
abs(df[df.Year == 1994].Sport.nunique() - df[df.Year == 2002].Sport.nunique())

