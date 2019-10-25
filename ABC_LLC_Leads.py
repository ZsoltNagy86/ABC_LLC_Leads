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

# In[14]:


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

# In[15]:


# Reading Leads csv file
df_leads = pd.read_csv('C:/Users/ZsoltNagy/Desktop/github_projects/ABC_LLC_Leads_Boberdoo/ABC_LLC_Leads/Data/leads_hw.csv', index_col=0)
df_leads.head(5)


# ### <span style="color:dimgray">Merging dataset</span>

# In[17]:


df = df_leads >> left_join(df_bids, by = "leadID")
df.head(10)


# In[ ]:




