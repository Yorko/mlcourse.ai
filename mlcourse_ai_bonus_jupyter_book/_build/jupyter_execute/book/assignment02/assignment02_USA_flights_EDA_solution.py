#!/usr/bin/env python
# coding: utf-8

# <img src="https://habrastorage.org/webt/ia/m9/zk/iam9zkyzqebnf_okxipihkgjwnw.jpeg" />
#     
# **<center>[mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course** </center><br>
# Authors: [Yury Kashnitsky](https://yorko.github.io), and [Maxim Keremet](https://www.linkedin.com/in/maximkeremet/). Translated and edited by  [Artem Trunov](https://www.linkedin.com/in/datamove/), and [Aditya Soni](https://www.linkedin.com/in/aditya-soni-0505a9124/). [mlcourse.ai](https://mlcourse.ai) is powered by [OpenDataScience (ods.ai)](https://ods.ai/) © 2017—2022

# # <center>Assignment #2. Solution </center><a class="tocSkip">
# 
# ## <center>Exploratory Data Analysis (EDA) of US flights with Pandas, Matplotlib, and Seaborn </center><a class="tocSkip">
# 
# <img src='https://habrastorage.org/webt/z9/io/wb/z9iowbwlya0sadrr0rf_am0ffm0.jpeg' width=50%>
# 
# Prior to working on the assignment, you'd better check out the corresponding course material:
#  - [Exploratory data analysis with Pandas](https://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/jupyter_english/topic01_pandas_data_analysis/topic1_pandas_data_analysis.ipynb?flush_cache=true), the same as an interactive web-based [Kaggle Notebook](https://www.kaggle.com/kashnitsky/topic-1-exploratory-data-analysis-with-pandas)
#  - [Visualization: from Simple Distributions to Dimensionality Reduction](https://mlcourse.ai/notebooks/blob/master/jupyter_english/topic02_visual_data_analysis/topic2_visual_data_analysis.ipynb?flush_cache=true), the same as a [Kaggle Notebook](https://www.kaggle.com/kashnitsky/topic-2-visual-data-analysis-in-python)
#  - [Overview of Seaborn, Matplotlib and Plotly libraries](https://mlcourse.ai/notebooks/blob/master/jupyter_english/topic02_visual_data_analysis/topic2_additional_seaborn_matplotlib_plotly.ipynb?flush_cache=true), the same as a [Kaggle Notebook](https://www.kaggle.com/kashnitsky/topic-2-part-2-seaborn-and-plotly) 
#  - first lectures in [this](https://www.youtube.com/watch?v=QKTuw4PNOsU&list=PLVlY_7IJCMJeRfZ68eVfEcu-UcN9BbwiX) YouTube playlist 
#  - you can also practice with demo assignments, which are simpler and already shared with solutions: [A1 demo](https://www.kaggle.com/kashnitsky/a1-demo-pandas-and-uci-adult-dataset), [solution](https://www.kaggle.com/kashnitsky/a1-demo-pandas-and-uci-adult-dataset-solution), [A2 demo](https://www.kaggle.com/kashnitsky/a2-demo-analyzing-cardiovascular-data), [solution](https://www.kaggle.com/kashnitsky/a2-demo-analyzing-cardiovascular-data-solution)
# 
# ### Your task is to:
#  1. write code and perform computations in the cells below;
#  2. choose answers in the [webform](https://docs.google.com/forms/d/1GXgR4TsqoTH_nQkrgluqWBElpK0emfhNzdZWZQObtnw).
# 
# *If you are sure that something is not 100% correct with the assignment/solution, please leave your feedback via the mentioned webform ↑*
#     
# -----

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd

# if seaborn is not yet installed, run `pip install seaborn` in terminal
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt

# sharper plots
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# * Download the data [archive](https://drive.google.com/file/d/1kOEcWtcTbbrlhVf1wlYtFQUF_kY92U5O/view?usp=sharing) (Archived ~ 114 Mb, unzipped – ~ 690 Mb). No need to unzip – pandas can unzip on the fly.
# * Place it in the `"../../_static/data/assignment2"` folder, or change the path below according to your location.
# * The dataset has information about carriers and flights between US airports during the year 2008. 
# * Column description is available [here](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ). Visit this site to find ex. meaning of flight cancellation codes.

# Consider the following terms we use:
# * unique flight – a record (row) in the dataset
# * completed flight – flight that is not cancelled (Cancelled==0 in the dataset)
# * flight code – a combination of ['UniqueCarrier','FlightNum'], i.e.  UA52
# * airport code – a three letter airport alias from 'Origin or 'Dest' columns

# **Reading data into memory and creating a Pandas ``DataFrame`` object**
# 
# (This may take a while, be patient)
# 
# We are not going to read in the whole dataset. In order to reduce memory footprint, we instead load only needed columns and cast them to suitable data types.

# In[2]:


dtype = {
    "DayOfWeek": np.uint8,
    "DayofMonth": np.uint8,
    "Month": np.uint8,
    "Cancelled": np.uint8,
    "Year": np.uint16,
    "FlightNum": np.uint16,
    "Distance": np.uint16,
    "UniqueCarrier": str,
    "CancellationCode": str,
    "Origin": str,
    "Dest": str,
    "ArrDelay": np.float16,
    "DepDelay": np.float16,
    "CarrierDelay": np.float16,
    "WeatherDelay": np.float16,
    "NASDelay": np.float16,
    "SecurityDelay": np.float16,
    "LateAircraftDelay": np.float16,
    "DepTime": np.float16,
}


# In[3]:


PATH_TO_DATA = Path("../../_static/data/assignment2")


# In[4]:


# change the path if needed
flights_df = pd.read_csv(
    PATH_TO_DATA / "flights_2008.csv.bz2", usecols=dtype.keys(), dtype=dtype
)


# **Check the number of rows and columns and print column names.**

# In[5]:


print(flights_df.shape)
print(flights_df.columns)


# **Print first 5 rows of the dataset.**

# In[6]:


flights_df.head()


# **Transpose the frame to see all features at once.**

# In[7]:


flights_df.head().T


# **Examine data types of all features and total dataframe size in memory.**

# In[8]:


flights_df.info()


# **Get basic statistics of each feature.**

# In[9]:


flights_df.describe().T


# **Count unique Carriers and plot their relative share of flights:**

# In[10]:


flights_df["UniqueCarrier"].nunique()


# In[11]:


flights_df.groupby("UniqueCarrier").size().plot(kind="bar");


# **We can also _group by_ category/categories in order to calculate different aggregated statistics.**
# 
# **For example, finding top-3 flight codes, that have the largest total distance traveled in year 2008.**

# In[12]:


flights_df.groupby(["UniqueCarrier", "FlightNum"])["Distance"].sum().sort_values(
    ascending=False
).iloc[:3]


# **Another way:**

# In[13]:


flights_df.groupby(["UniqueCarrier", "FlightNum"]).agg(
    {"Distance": [np.mean, np.sum, "count"], "Cancelled": np.sum}
).sort_values(("Distance", "sum"), ascending=False).iloc[0:3]


# **Number of flights by days of week and months:**

# In[14]:


pd.crosstab(flights_df.Month, flights_df.DayOfWeek)


# **It can also be handy to color such tables in order to easily notice outliers:**

# In[15]:


plt.imshow(
    pd.crosstab(flights_df.Month, flights_df.DayOfWeek),
    cmap="seismic",
    interpolation="none",
);


# **Flight distance histogram:**

# In[16]:


flights_df.hist("Distance", bins=20);


# **Making a histogram of flight frequency by date.**

# In[17]:


flights_df["Date"] = pd.to_datetime(
    flights_df.rename(columns={"DayofMonth": "Day"})[["Year", "Month", "Day"]]
)


# In[18]:


num_flights_by_date = flights_df.groupby("Date").size()


# In[19]:


num_flights_by_date.plot();


# **Do you see a weekly pattern above? And below?**

# In[20]:


num_flights_by_date.rolling(window=7).mean().plot();


# **We'll need a new column in our dataset - departure hour, let's create it.**
# 
# As we see, `DepTime` is distributed from 1 to 2400 (it is given in the `hhmm` format, check the [column description](https://www.transtats.bts.gov/Fields.asp?Table_ID=236) again). We'll treat departure hour as `DepTime` // 100 (divide by 100 and apply the `floor` function). However, now we'll have both hour 0 and hour 24. Hour 24 sounds strange, we'll set it to be 0 instead (a typical imperfectness of real data, however, you can check that it affects only 521 rows, which is sort of not a big deal). So now values of a new column `DepHour` will be distributed from 0 to 23. There are some missing values, for now we won't fill in them, just ignore them. 

# In[21]:


flights_df["DepHour"] = flights_df["DepTime"] // 100
flights_df["DepHour"].replace(to_replace=24, value=0, inplace=True)


# In[22]:


flights_df["DepHour"].describe()


# ### Now it's your turn. Answer the questions below.

# **<font color='red'>Question 1.</font> How many unique carriers are there in our dataset?**
# 
# - 10
# - 15
# - 20 **<font color='red'>[+]</font>**
# - 25 

# In[23]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# In[24]:


# Solution code
flights_df["UniqueCarrier"].nunique()


# **<font color='red'>Question 2.</font>  We have  both cancelled and completed flights in the dataset. Check if there are more completed or cancelled flights. What is the difference?** <br>
# 
# - Cancelled overweights completed by 329 flights
# - Completed overweights cancelled by 6734860 flights **<font color='red'>[+]</font>**
# - Cancelled overweights completed by 671 flights
# - Completed overweights cancelled by 11088967 flights

# In[25]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# Straightforward way:

# In[26]:


# Solution code
cancelled_flights = flights_df[flights_df["Cancelled"] == 1].shape[0]
completed_flights = flights_df[flights_df["Cancelled"] == 0].shape[0]
print("There are {} cancelled flights.".format(cancelled_flights))
print("There are {} completed flights.".format(completed_flights))
print(
    "The difference is {} flights.".format(abs(completed_flights - cancelled_flights))
)


# Nicer:

# In[27]:


result_df = flights_df.groupby("Cancelled").size()
result_df


# In[28]:


print("The difference is {} flights.".format(abs(result_df[0] - result_df[1])))


# **<font color='red'>Question 3.</font> Find a flight with the longest departure delay and a flight with the longest arrival delay. Do they have the same destination airport, and if yes, what is its code?**
# 
# - yes, ATL
# - yes, HNL
# - yes, MSP **<font color='red'>[+]</font>**
# - no

# In[29]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# In[30]:


# Solution code
print(flights_df[flights_df["DepDelay"] == flights_df["DepDelay"].max()]["Dest"])
print(flights_df[flights_df["ArrDelay"] == flights_df["ArrDelay"].max()]["Dest"])


# **<font color='red'>Question 4.</font> Find the carrier that has the greatest number of cancelled flights.**
# 
# - AA
# - MQ **<font color='red'>[+]</font>**
# - WN
# - CO 

# In[31]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# To find the carrier with the most number of cancelled flights: 

# In[32]:


flights_df[flights_df["Cancelled"] == 1]["UniqueCarrier"].mode()


# If we want to know the number of cancelled flights as well:

# In[33]:


flights_df[flights_df["Cancelled"] == 1]["UniqueCarrier"].value_counts().head()


# **<font color='red'>Question 5.</font> Let's examine departure time and consider distribution by hour (column `DepHour` that we've created earlier). Which hour has the highest percentage of flights?**<br>
# 
# *Hint:* Check time format [here](https://www.transtats.bts.gov/Fields.asp?Table_ID=236).
# 
# - 1 am 
# - 5 am  
# - 8 am  **<font color='red'>[+]</font>**
# - 3 pm 

# In[34]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# In[35]:


result_df = (flights_df["DepHour"].value_counts(normalize=True) * 100).sort_index()
print(result_df.head(10))
result_df.idxmax()


# **<font color='red'>Question 6.</font> OK, now let's examine cancelled flight distribution by time. Which hour has the least percentage of cancelled flights?**<br>
# 
# - 2 am 
# - 9 pm  
# - 8 am  
# - 3 am **<font color='red'>[+]</font>**

# In[36]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# In[37]:


flights_df.groupby("DepHour")["Cancelled"].mean().sort_values().head()


# **<font color='red'>Question 7.</font> Is there any hour that didn't have any cancelled flights at all? Check all that apply.**
# 
# - 3 **<font color='red'>[+]</font>**
# - 19
# - 22
# - 4 

# In[38]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# Actually, no code needed. From the previous question, we see that it's 3am. But if we were to check this point specifically:
# 

# In[39]:


set(range(24)).difference(
    flights_df.loc[flights_df["Cancelled"] == 1, "DepHour"].unique()
)


# **<font color='red'>Question 8.</font> Find the busiest hour, or in other words, the hour when the number of departed flights reaches its maximum.**<br>
# 
# *Hint:* Consider only *completed* flights.
# 
# - 4
# - 7
# - 8 **<font color='red'>[+]</font>**
# - 17 

# In[40]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# In[41]:


# Solution code
flights_df[(flights_df["Cancelled"] == 0)].groupby(["DepHour"]).size().idxmax()


# **<font color='red'>Question 9.</font> Since we know the departure hour, it might be interesting to examine the average delay for corresponding hour. Are there any cases, when the planes on average departed earlier than they should have done? And if yes, at what departure hours did it happen?**<br>
# 
# *Hint:* Consider only *completed* flights.
# 
# - no, there are no such cases
# - yes, at 5-6 am **<font color='red'>[+]</font>**
# - yes, at 9-10 am
# - yes, at 2-4 pm

# In[42]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# In[43]:


# Solution code
flights_df[flights_df["Cancelled"] == 0].groupby("DepHour")["DepDelay"].mean()


# **<font color='red'>Question 10.</font> Considering only the completed flights by the carrier, that you have found in Question 4, find the distribution of these flights by hour. At what time does the greatest number of its planes depart?**<br>
# 
# - at noon
# - at 7 am 
# - at 8 am
# - at 10 am **<font color='red'>[+]</font>**

# In[44]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# In[45]:


# Solution code
flights_df[
    (flights_df["Cancelled"] == 0) & (flights_df["UniqueCarrier"] == "MQ")
].groupby(["DepHour"]).size().idxmax()


# **<font color='red'>Question 11.</font> Find top-10 carriers in terms of the number of *completed* flights (_UniqueCarrier_ column)?**
# 
# **Which of the listed below is _not_ in your top-10 list?**
# - DL
# - AA
# - OO
# - EV **<font color='red'>[+]</font>**

# In[46]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# In[47]:


# Solution code
top10_airlines = (
    flights_df.loc[flights_df["Cancelled"] == 0, "UniqueCarrier"]
    .value_counts()
    .head(10)
)
top10_airlines


# In[48]:


question_airlines = ["DL", "AA", "OO", "EV"]
set(question_airlines) - set(top10_airlines.index)


# **<font color='red'>Question 12.</font> . Plot distributions of flight cancellation reasons (CancellationCode).**
# 
# **What is the most frequent reason for flight cancellation? (Use this [link](https://www.transtats.bts.gov/Fields.asp?Table_ID=236) to translate codes into reasons)**
# - Carrier
# - Weather conditions **<font color='red'>[+]</font>**
# - National Air System
# - Security reasons

# In[49]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# In[50]:


# Solution code
sns.countplot(flights_df["CancellationCode"].sort_values());


# In[51]:


flights_df["CancellationCode"].mode()


# *From the link above:*
# ```
# A - Carrier 
# B - Weather 
# C - National Air System 
# D - Security
# ```

# **<font color='red'>Question 13.</font> Which route is the most frequent, in terms of the number of flights?**
# 
# (Take a look at _'Origin'_ and _'Dest'_ features. Consider _A->B_ and _B->A_ directions as _different_ routes) 
# 
#  - New-York – Washington (JFK-IAD)
#  - San-Francisco – Los-Angeles (SFO-LAX) **<font color='red'>[+]</font>**
#  - San-Jose – Dallas (SJC-DFW)
#  - New-York – San-Francisco (JFK-SFO)

# In[52]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# In[53]:


# Solution code
flights_df["Route"] = flights_df["Origin"] + "->" + flights_df["Dest"]
flights_df["Route"].value_counts().head()


# Or without creating a new feature

# In[54]:


flights_df[["Origin", "Dest"]].groupby(["Origin", "Dest"]).size().idxmax()


# **<font color='red'>Question 14.</font> . Find top-5 delayed routes (count how many times they were delayed on departure). From all flights on these 5 routes, count all flights with weather conditions contributing to a delay.**
# 
# _Hint_ : consider only positive delays
# 
# - 449 
# - 539 
# - 549 
# - 668 **<font color='red'>[+]</font>**

# In[55]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# In[56]:


# Solution code
# find top5 routes with most delayed flights
top5_delayed = (
    flights_df[flights_df["DepDelay"] > 0]
    .groupby("Route")
    .size()
    .sort_values(ascending=False)
    .head(5)
)
top5_delayed


# In[57]:


# reduce to only flights from top5 delayed routes
flights_df_top5_delays = flights_df[flights_df["Route"].isin(top5_delayed.index)]


# In[58]:


# now the answer
(flights_df_top5_delays["WeatherDelay"] > 0).sum()


# **<font color='red'>Question 15.</font> Examine the hourly distribution of departure times. Choose all correct statements:**
# 
#  - Flights are normally distributed within time interval [0-23] (Search for: Normal distribution, bell curve).
#  - Flights are uniformly distributed within time interval [0-23].
#  - In the period from 0 am to 4 am there are considerably less flights than from 7 pm to 8 pm. **<font color='red'>[+]</font>**

# In[59]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# In[60]:


# Solution code
flights_df["DepHour"].value_counts(sort=False).plot(
    kind="bar", title="Number of flights depending on departure hour"
);


# Checking the 3rd point separately.

# In[61]:


round((flights_df["DepHour"] == 19).sum() / (flights_df["DepHour"] < 5).sum())


# **<font color='red'>Question 16.</font> Show how the number of flights changes through time (on the daily/weekly/monthly basis) and interpret the findings.**
# 
# **Choose all correct statements:**
# - The number of flights during weekends is less than during weekdays (working days). **<font color='red'>[+]</font>**
# - The lowest number of flights is on Sunday.
# - There are less flights during winter than during summer.  **<font color='red'>[+]</font>**
# 
# _Hint_: Look for official meteorological winter months for the Northern Hemisphere.

# In[62]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# In[63]:


# Solution code
num_flights_by_day_of_week = (
    flights_df[flights_df["Cancelled"] == 0].groupby("DayOfWeek").size()
)
num_flights_by_day_of_week.plot(kind="bar");


# By the way, here is an illustrative example of "how to lie with statistics". Compare the next plot with the previous one.

# In[64]:


num_flights_by_day_of_week.head()


# In[65]:


num_flights_by_day_of_week.plot();


# In[66]:


num_flights_by_month = flights_df[flights_df["Cancelled"] == 0].groupby("Month").size()
num_flights_by_month.plot(kind="bar");


# **<font color='red'>Question 17.</font> Examine the distribution of cancellation reasons with time. Make a bar plot of cancellation reasons aggregated by months.**
# 
# **Choose all correct statements:**
# - October has the lowest number of cancellations due to weather. **<font color='red'>[+]</font>**
# - The highest number of cancellations in September is due to Security reasons.
# - April's top cancellation reason is carriers. **<font color='red'>[+]</font>**
# - Flights cancellations due to National Air System are more frequent than those due to carriers.

# In[67]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# *Reminder on Cancellatoin codes:*
# ```
# A - Carrier 
# B - Weather 
# C - National Air System 
# D - Security
# ```

# In[68]:


# create a month name list
import calendar

month_names = []

for month_idx in flights_df["Month"].unique():
    month_names.append(calendar.month_name[month_idx])

ax = flights_df.groupby(["Month", "CancellationCode"]).size().unstack().plot(kind="bar")

ax.set_xticklabels(month_names, rotation=90)
plt.show()


# **<font color='red'>Question 18.</font> Which month has the greatest number of cancellations due to Carrier?** 
# - May
# - January
# - September
# - April **<font color='red'>[+]</font>**

# In[69]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# In[70]:


# Solution code
flights_df.loc[flights_df["CancellationCode"] == "A", "Month"].value_counts().head()


# Or, as a nice plot:

# In[71]:


import calendar
import random

import matplotlib.colors as colors

ax = (
    flights_df[flights_df["CancellationCode"] == "A"]
    .groupby(["Month"])["UniqueCarrier"]
    .count()
    .plot(kind="bar", color="coral")
)
ax.set_xticklabels(month_names, rotation=90)
plt.show()


# **<font color='red'>Question 19.</font> Identify the carrier with the greatest number of cancellations due to carrier in the corresponding month from the previous question.**
# 
# - 9E
# - EV
# - HA
# - AA **<font color='red'>[+]</font>**

# In[72]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# In[73]:


# Solution code
flights_df.loc[
    (flights_df["CancellationCode"] == "A") & (flights_df["Month"] == 4),
    "UniqueCarrier",
].value_counts().head()


# Or, as a nice plot (with much more efforts of course):

# In[74]:


import random

import matplotlib.colors as colors

colors_list = list(colors._colors_full_map.values())
selected_colors = random.sample(colors_list, 20)

cancelled_Carrier_April = (
    flights_df[(flights_df["CancellationCode"] == "A") & (flights_df["Month"] == 4)]
    .groupby(["Date", "UniqueCarrier"])
    .size()
    .unstack()
)

# get the peak coordinates for annotation
max_value = cancelled_Carrier_April.max().max()
carrier_of_max_value = cancelled_Carrier_April.max().idxmax()
date_of_max_value = cancelled_Carrier_April[carrier_of_max_value].idxmax()

fig, ax = plt.subplots()

cancelled_Carrier_April.plot(color=selected_colors, figsize=(10, 8), ax=ax)
ax.annotate(carrier_of_max_value, (str(date_of_max_value), max_value))
fig.autofmt_xdate()


# **<font color='red'>Question 20.</font> Examine median arrival and departure delays (in time) by carrier. Which carrier has the lowest median delay time for both arrivals and departures? Leave only non-negative values of delay times ('ArrDelay', 'DepDelay').
# ([Boxplots](https://seaborn.pydata.org/generated/seaborn.boxplot.html) can be helpful in this exercise, as well as it might be a good idea to remove outliers in order to build nice graphs. You can exclude delay time values higher than a corresponding .95 percentile).**
# 
# - EV
# - OO
# - AA
# - AQ **<font color='red'>[+]</font>** 

# In[75]:


# You code here (read-only in a JupyterBook, pls run jupyter-notebook to edit)


# In[76]:


# Solution code
for delay_type in ["ArrDelay", "DepDelay"]:
    sub_df = flights_df[
        (flights_df[delay_type] >= 0)
        & (flights_df[delay_type] < flights_df[delay_type].quantile(0.95))
    ]
    sns.boxplot(x="UniqueCarrier", y=delay_type, data=sub_df)
    plt.show()


# That's it! Now go and do 30 push-ups! :)
