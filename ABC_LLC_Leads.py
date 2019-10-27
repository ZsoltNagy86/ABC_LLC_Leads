#!/usr/bin/env python
# coding: utf-8

# # <span style="color:darkslategray">ABC LLC - Leads for Health Insurance Services </span>

# ### <span style="color:dimgray">Importing packages</span>

# In[1]:


# Importing general packages
import pandas as pd
import dfply

from dfply import *
import numpy as np

# Importing packages for vizualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ### <span style="color:dimgray">Loading data</span>

# <b>Features of df_bids:</b><br/>
# ● created: time of the bid<br />
# ● leadID: unique ID of lead<br />
# ● max_bid: bid offered by Abc LLC for that particular lead<br />
# ● won: 1 if their bid was the highest and they won the bid, 0 otherwise

# In[2]:


# Reading Bids csv file
df_bids = pd.read_csv('C:/Users/ZsoltNagy/Desktop/github_projects/ABC_LLC_Leads_Boberdoo/ABC_LLC_Leads/Data/bids_hw.csv', index_col=0)
df_bids.head(5)


# <b>Features of df_leads:</b><br />
# ● leadID: unique ID of the lead<br />
# ● state_18: the state where the lead lives in<br />
# ● gender_18: lead's gender, which is 1 if male, and 2 if female<br />
# ● estimated_household_income_18: this is the lead's estimation of their own household income per year in dollars <br />
# ● cpa_status_18: 1 for closed leads, 0 for lost leads<br />
# ● premium_amount_18: value of the contract signed, to be paid monthly in dollars<br />

# In[3]:


# Reading Leads csv file
df_leads = pd.read_csv('C:/Users/ZsoltNagy/Desktop/github_projects/ABC_LLC_Leads_Boberdoo/ABC_LLC_Leads/Data/leads_hw.csv', index_col=0)
df_leads.head(5)


# ### <span style="color:dimgray"> Merging dataset </span>

# In[4]:


# Joining the two dataframe
df = df_leads >> left_join(df_bids, by = "leadID")
df.head(10)


# ### <span style="color:dimgray"> Exploring the dataset </span>

# In[5]:


# Checking the type of variables
df.info()


# In[6]:


# Describing continuous variables
df.describe()


# In[7]:


#Missing values computation
df.isnull().sum()


# #### <span style="color:darkgray"> Univariate analysis on closed leads </span>

# In[8]:


df_closed = df >> mask(X.cpa_status_18 == 1)


# In[9]:


df_closed.dtypes


# In[10]:


df_closed.describe()


# In[11]:


# Distribution of gender 
sns.countplot(x='gender_18', data=df_closed, palette="BuPu")
plt.title('Distribution of Gender')


# In[12]:


sns.boxplot(x="age_18", data=df_closed, orient="v", palette="BuPu")


# In[27]:


# Distribution of ages

df_closed.hist('age_18', bins=30, color = "lightsteelblue", ec="teal")
plt.title('Distribution of Age')
plt.xlabel('Age')


# In[14]:


sns.boxplot(x="estimated_household_income_18", data=df_closed, orient="v", palette="BuPu")


# In[15]:


#Excluding the outlier case 
df_closed = df_closed >> mask(X.estimated_household_income_18 < 250000)


# In[28]:


df_closed.hist('estimated_household_income_18', bins=20, color = "lightsteelblue", ec="teal")
plt.title('Distribution of HH Income')
plt.xlabel('HH Income')


# In[17]:


sns.boxplot(x="premium_amount_18", data=df_closed, orient="v", palette="BuPu")


# In[18]:


#Excluding the outlier case 
df_closed = df_closed >> mask(X.premium_amount_18 < max(df_closed['premium_amount_18'])) 


# In[29]:


df_closed = df_closed >> mask(X.premium_amount_18 > 0) 

df_closed.hist(column='premium_amount_18', bins=25, color = "lightsteelblue", ec="teal")
plt.title('Value of the contract')
plt.xlabel('Contract Value')


# In[20]:


# Distribution of closed leads by states 
df_state = df_closed >> group_by(X.state_18, X.gender_18) >> summarize(N=n(X.leadID))


plt.figure(figsize=(16, 16))
g = sns.catplot(x='state_18', y='N', hue='gender_18', data=df_state, palette="BuPu", kind='bar', legend=False)
plt.title('Distribution of leads by states')
plt.legend(title='Gender', loc='upper left', labels=['Male', 'Female'])
g.fig.set_figwidth(10)
g.fig.set_figheight(7)


# ### <span style="color:dimgray"> Abc LLC's current most typical client </span>

# In[52]:


# Creating new categorical age variable
#bins = pd.IntervalIndex.from_tuples([(0, 17), (18, 44), (45, 64), (65, 150)])
bins=[0, 18, 45, 65, np.inf]
df_closed['Age_c'] = pd.cut(round(df_closed['age_18'], 0), bins=bins, labels=['Under_18', '18-44', '45-64', '65_and_Above'])


# In[53]:


# Checking classes of age
df_closed >> group_by(X.Age_c) >> summarize(N=n(X.leadID), Min=X.age_18.min(), Max=X.age_18.max())


# In[32]:


# Creating new categorical income variable
df_closed['Income_c'] = pd.cut(df_closed['estimated_household_income_18'], bins=4, labels=['Lowest', 'Lower_middle', 'Upper_middle', 'Highest'])


# In[39]:


# Checking classes of income
df_closed >> group_by(X.Income_c) >> summarize(N=n(X.leadID), Min=X.estimated_household_income_18.min(), Max=X.estimated_household_income_18.max())


# In[60]:


#Finding the 5 most typical customer groups
df_typ_cust = df_closed >> group_by(X.state_18, X.gender_18, X.Age_c, X.Income_c) >> summarize(N = n(X.leadID))
df_typ_cust.sort_values(by='N', ascending=False).head(5)

