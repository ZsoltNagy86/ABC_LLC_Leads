#!/usr/bin/env python
# coding: utf-8

# # <span style="color:darkslategray">ABC LLC - Leads for Health Insurance Services </span>

# ### <span style="color:dimgray">Importing packages</span>

# In[3]:


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

# In[6]:


df_bids = pd.read_csv('C:/Users/ZsoltNagy/Desktop/github_projects/ABC_LLC_Leads_Boberdoo/ABC_LLC_Leads/Data/bids_hw.csv')
print(df_bids.head(5))

df_leads = pd.read_csv('C:/Users/ZsoltNagy/Desktop/github_projects/ABC_LLC_Leads_Boberdoo/ABC_LLC_Leads/Data/leads_hw.csv')
print(df_leads.head(5))

