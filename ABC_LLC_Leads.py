#!/usr/bin/env python
# coding: utf-8

# # <span style="color:darkslategray">ABC LLC - Leads for Health Insurance Services </span>

# ### <span style="color:dimgray">Importing packages</span>

# In[80]:


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

# Importing packages for encoding and standardization
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler

# Importing packages for modeling
from sklearn.cluster import KMeans


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


# #### <span style="color:steelblue"> Univariate analysis on closed leads </span>

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


# ### <span style="color:dimgray"> Customer segmentation using K-means clustering </span>

# #### <span style="color:steelblue"> Customer segmentation on closed leads </span>

# In[169]:


# One-hot-encoding categorical variables

# Selecting relevant features
df_closed_enc = df_closed >> select(X.leadID, 
                                    X.gender_18, 
                                    X.age_18, 
                                    X.estimated_household_income_18, 
                                    X.premium_amount_18, 
                                    X.state_18)

cat_columns = ["gender_18", "state_18"]
df_closed_enc = pd.get_dummies(df_closed_enc, 
                               prefix_sep="__",
                               columns=cat_columns)

df_closed_enc = df_closed_enc.set_index('leadID')

df_closed_enc.head(10)


# In[170]:


# Normalizing/Standardizing data to ensure that unit of dimension does not distort relative near-ness of observations

# Using standardization because of presence of possible outliers: 
df_columns = list(df_closed_enc.columns)
rsc = RobustScaler()
df_closed_st = rsc.fit_transform(df_closed_enc)
df_closed_st = pd.DataFrame(df_closed_st, columns=df_columns)
df_closed_st.head(10)


# In[171]:


# Determining the number of clusters for K-means
clusters_range = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in clusters_range]
score = [kmeans[i].fit(df_closed_st).score(df_closed_st) for i in range(len(kmeans))]

plt.plot(clusters_range,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Number of clusters by score')
plt.show()


# In[98]:


clusters_range = [2,3,4,5,6,7,8,9,10,11,12,13,14]
inertias =[] 
for c in clusters_range:
    kmeans = KMeans(n_clusters=c, random_state=0).fit(df_closed_st)
    inertias.append(kmeans.inertia_)
plt.figure()
plt.plot(clusters_range,inertias, marker='o')


# In[173]:


# Running K-means cluster on the encoded dataframe with 4 clusters based on elbow method
kmens = KMeans(n_clusters=4, random_state=0).fit(df_closed_st)
# Adding cluster variable to closed lead dataframe 
df_closed['Clusters'] = kmens.labels_
df_closed


# In[190]:


# Checking the clusters characteristics

#Creating cluster df
cluster_df = df_closed >> group_by(X.Clusters) >> summarize(Gender_dist = X.gender_18.mean(), 
                                                            Avg_Age = X.age_18.mean(), 
                                                            Avg_Inc = X.estimated_household_income_18.mean(), 
                                                            Avg_Contr_size = X.premium_amount_18.mean(),
                                                            Avg_MBids = X.max_bid.mean(),
                                                            SD_MBids = sd(X.max_bid),
                                                            N=n(X.leadID))

# Adding info about from what states existing customers are coming from 
df_freq_state = df_closed >> group_by(X.state_18, X.Clusters) >> summarize(State_N = n(X.state_18))
df_freq_state = df_freq_state.sort_values(by=['Clusters', 'State_N'], ascending = [True, False])
df_freq_state = df_freq_state.pivot(index='Clusters', columns='state_18', values='State_N')
cluster_df['Most_Freq_States'] = pd.Series()
for rows in range(0,len(df_freq_state)):
    string = ", "
    string = string.join(df_freq_state.loc[rows].sort_values(ascending=False).index[0:5])
    cluster_df.loc[rows, 'Most_Freq_States'] = string
cluster_df


# #### <span style="color:steelblue"> Customer segmentation on lost leads </span>

# In[180]:


# Creating dataframe for lost leads
df_lost = df >> mask(X.cpa_status_18 == 0)


# In[184]:


# Checking dataframe
df_lost.describe()


# In[ ]:





# In[ ]:


# One-hot-encoding categorical variables

# Selecting relevant features
df_closed_enc = df_closed >> select(X.leadID, 
                                    X.gender_18, 
                                    X.age_18, 
                                    X.estimated_household_income_18, 
                                    X.premium_amount_18, 
                                    X.state_18)

cat_columns = ["gender_18", "state_18"]
df_closed_enc = pd.get_dummies(df_closed_enc, 
                               prefix_sep="__",
                               columns=cat_columns)

df_closed_enc = df_closed_enc.set_index('leadID')

df_closed_enc.head(10)

