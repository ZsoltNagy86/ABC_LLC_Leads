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

# In[87]:


df = df_leads >> left_join(df_bids, by = "leadID")
df.head(10)


# ### <span style="color:dimgray"> Abc LLC's current most typical client </span>

# #### <span style="color:darkgray"> Exploring the dataset </span>

# In[5]:


# Checking the type of variables
df.info()


# In[6]:


# Describing continuous variables
df.describe()


# In[196]:


#Missing values computation
df.isnull().sum()


# #### <span style="color:darkgray"> Univariate analysis on closed leads </span>

# In[180]:


df_closed = df >> mask(X.cpa_status_18 == 1)


# In[181]:


df_closed.dtypes


# In[182]:


df_closed.describe()


# In[183]:


# Distribution of gender 
sns.countplot(x='gender_18', data=df_closed, palette="BuPu")
plt.title('Distribution of Gender')


# In[184]:


sns.boxplot(x="age_18", data=df_closed, orient="v", palette="BuPu")


# In[185]:


# Distribution of ages

df_closed.hist('age_18', bins=30)
plt.title('Distribution of Age')
plt.xlabel('Age')


# In[195]:


sns.boxplot(x="estimated_household_income_18", data=df_closed, orient="v", palette="BuPu")


# In[188]:


#Excluding the outlier case 
df_closed = df_closed >> mask(X.estimated_household_income_18 < 250000)


# In[189]:


df_closed.hist('estimated_household_income_18', bins=20)
plt.title('Distribution of HH Income')
plt.xlabel('HH Income')


# In[190]:


sns.boxplot(x="premium_amount_18", data=df_closed, orient="v", palette="BuPu")


# In[191]:


#Excluding the outlier case 
df_closed = df_closed >> mask(X.premium_amount_18 < max(df_closed['premium_amount_18'])) 


# In[194]:


df_closed = df_closed >> mask(X.premium_amount_18 > 0) 

df_closed.hist(column='premium_amount_18', bins=25)
plt.title('Value of the contract')
plt.xlabel('Contract Value')


# In[172]:





# In[229]:


# Distribution of closed leads by states 
df_state = df_closed >> group_by(X.state_18, X.gender_18) >> summarize(N=n(X.leadID))


plt.figure(figsize=(16, 16))
g = sns.catplot(x='state_18', y='N', hue='gender_18', data=df_state, palette="BuPu", kind='bar', legend=False)
plt.title('Distribution of leads by states')
plt.legend(title='Gender', loc='upper left', labels=['Male', 'Female'])
g.fig.set_figwidth(10)
g.fig.set_figheight(7)


# In[10]:


df >> mask(X.cpa_status_18 == 1) >> group_by(X.gender_18, X.state_18) >> summarize(age = X.age_18.mean(), income = X.estimated_household_income_18.mean(), contr_size = X.premium_amount_18.mean())

