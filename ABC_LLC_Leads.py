#!/usr/bin/env python
# coding: utf-8

# # <span style="color:darkslategray">ABC LLC - Leads for Health Insurance Services </span>

# ### <span style="color:dimgray">Importing packages</span>

# In[41]:


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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# Importing packages for custering
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc


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


# In[13]:


# Distribution of ages

df_closed.hist('age_18', bins=30, color = "lightsteelblue", ec="teal")
plt.title('Distribution of Age')
plt.xlabel('Age')


# In[14]:


sns.boxplot(x="estimated_household_income_18", data=df_closed, orient="v", palette="BuPu")


# In[15]:


#Excluding the outlier cases 

def outliers_iqr(x, mplyr):
    quartile_1, quartile_3 = np.percentile(x, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * mplyr)
    upper_bound = quartile_3 + (iqr * mplyr)
    return np.where((x > upper_bound) | (x < lower_bound))
indexes_to_drop = list(outliers_iqr(df_closed['estimated_household_income_18'], 3.5))

df_closed = df_closed.drop(df_closed.index[indexes_to_drop])


# In[16]:


df_closed.hist('estimated_household_income_18', bins=20, color = "lightsteelblue", ec="teal")
plt.title('Distribution of HH Income')
plt.xlabel('HH Income')


# In[17]:


sns.boxplot(x="premium_amount_18", data=df_closed, orient="v", palette="BuPu")


# In[18]:


#Excluding the outlier case 
df_closed = df_closed >> mask(X.premium_amount_18 < max(df_closed['premium_amount_18'])) 


# In[19]:


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

# In[21]:


# Creating new categorical age variable
#bins = pd.IntervalIndex.from_tuples([(0, 17), (18, 44), (45, 64), (65, 150)])
bins=[0, 18, 45, 65, np.inf]
df_closed['Age_c'] = pd.cut(round(df_closed['age_18'], 0), bins=bins, labels=['Under_18', '18-44', '45-64', '65_and_Above'])


# In[22]:


# Checking classes of age
df_closed >> group_by(X.Age_c) >> summarize(N=n(X.leadID), Min=X.age_18.min(), Max=X.age_18.max())


# In[23]:


# Creating new categorical income variable
df_closed['Income_c'] = pd.cut(df_closed['estimated_household_income_18'], bins=4, labels=['Lowest', 'Lower_middle', 'Upper_middle', 'Highest'])


# In[24]:


# Checking classes of income
df_closed >> group_by(X.Income_c) >> summarize(N=n(X.leadID), Min=X.estimated_household_income_18.min(), Max=X.estimated_household_income_18.max())


# In[25]:


#Finding the 5 most typical customer groups
df_typ_cust = df_closed >> group_by(X.state_18, X.gender_18, X.Age_c, X.Income_c) >> summarize(N = n(X.leadID), Avg_Contr_Size = X.premium_amount_18.mean())
df_typ_cust.sort_values(by='N', ascending=False).head(5)


# #### <span style="color:darkred"> Question 1: Analysis on most typical client </span>

# #### <span style="color:sienna"> <i> The typical client of ABC LLC is a young adult male who lives in New York, whose household income belongs to the lowest segment of the sample and average contract size is around 740 dollars. </i></span>

# ### <span style="color:dimgray"> Customer segmentation using K-means and hierarchical clustering </span>

# #### <span style="color:steelblue"> Customer segmentation on closed leads </span>

# In[26]:


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

df_closed_enc.head(10)


# In[27]:


# Standardizing data to ensure that unit of dimension does not distort relative near-ness of observations

# Using MinMaxScale considering the presence of high number of binnary features: 
df_columns = ['age_18', 'estimated_household_income_18', 'premium_amount_18']
mms = MinMaxScaler()
df_closed_st_mm = mms.fit_transform(df_closed_enc[['age_18', 'estimated_household_income_18', 'premium_amount_18']])
df_closed_st_mm = pd.DataFrame(df_closed_st_mm, columns=df_columns)
df_closed_st_mm['leadID'] = list(df_closed_enc['leadID'])
df_closed_st_mm = df_closed_st_mm >> left_join(df_closed_enc, by='leadID') >> drop(['leadID', 'age_18_y', 'estimated_household_income_18_y', 'premium_amount_18_y'])
df_closed_st_mm.head(10)


# In[28]:


# Determining the number of clusters for K-means
clusters_range = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in clusters_range]
score = [kmeans[i].fit(df_closed_st_mm).score(df_closed_st_mm) for i in range(len(kmeans))]

plt.plot(clusters_range,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Number of clusters by score')
plt.show()


# In[29]:


clusters_range = [2,3,4,5,6,7,8,9,10,11,12,13,14]
inertias =[] 
for c in clusters_range:
    kmeans = KMeans(n_clusters=c, random_state=0).fit(df_closed_st_mm)
    inertias.append(kmeans.inertia_)
plt.figure()
plt.plot(clusters_range,inertias, marker='o')


# In[30]:


# Running K-means cluster on the encoded dataframe with 4 clusters based on elbow method
kmens = KMeans(n_clusters=4, random_state=0).fit(df_closed_st_mm)
# Adding cluster variable to closed lead dataframe 
df_closed['Clusters_mm'] = kmens.labels_
df_closed.head(10)


# In[33]:


# Checking the clusters characteristics

#Creating cluster df
cluster_df_mm = df_closed >> group_by(X.Clusters_mm) >> summarize(Gender_dist = X.gender_18.mean(), 
                                                            Avg_Age = X.age_18.mean(), 
                                                            Avg_Inc = X.estimated_household_income_18.mean(), 
                                                            Avg_Contr_size = X.premium_amount_18.mean(),
                                                            Avg_MBids = X.max_bid.mean(),
                                                            SD_MBids = sd(X.max_bid),
                                                            N=n(X.leadID))

# Adding info about what states existing customers are coming from 
df_freq_state = df_closed >> group_by(X.state_18, X.Clusters_mm) >> summarize(State_N = n(X.state_18))
df_freq_state = df_freq_state.sort_values(by=['Clusters_mm', 'State_N'], ascending = [True, False])
df_freq_state = df_freq_state.pivot(index='Clusters_mm', columns='state_18', values='State_N')
cluster_df_mm['Most_Freq_States'] = pd.Series()
for rows in range(0,len(df_freq_state)):
    string = ", "
    string = string.join(df_freq_state.loc[rows].sort_values(ascending=False).index[0:5])
    cluster_df_mm.loc[rows, 'Most_Freq_States'] = string
cluster_df_mm


# In[34]:


# Standardizing data to ensure that unit of dimension does not distort relative near-ness of observations

# Using RobustScaler considering the presence of possible outliers: 
df_columns = ['age_18', 'estimated_household_income_18', 'premium_amount_18']
rsc = RobustScaler()
df_closed_st_rc = rsc.fit_transform(df_closed_enc[['age_18', 'estimated_household_income_18', 'premium_amount_18']])
df_closed_st_rc = pd.DataFrame(df_closed_st_rc, columns=df_columns)
df_closed_st_rc['leadID'] = list(df_closed_enc['leadID'])
df_closed_st_rc = df_closed_st_rc >> left_join(df_closed_enc, by='leadID') >> drop(['leadID', 'age_18_y', 'estimated_household_income_18_y', 'premium_amount_18_y'])
df_closed_st_rc.head(10)


# In[35]:


clusters_range = [2,3,4,5,6,7,8,9,10,11,12,13,14]
inertias =[] 
for c in clusters_range:
    kmeans = KMeans(n_clusters=c, random_state=0).fit(df_closed_st_rc)
    inertias.append(kmeans.inertia_)
plt.figure()
plt.plot(clusters_range,inertias, marker='o')


# In[36]:


# Running K-means cluster on the encoded dataframe with 4 clusters based on elbow method
kmens = KMeans(n_clusters=4, random_state=0).fit(df_closed_st_rc)
# Adding cluster variable to closed lead dataframe 
df_closed['Clusters_rc'] = kmens.labels_
df_closed.head(10)


# In[37]:


# Checking the clusters characteristics

#Creating cluster df
cluster_df_rc = df_closed >> group_by(X.Clusters_rc) >> summarize(Gender_dist = X.gender_18.mean(), 
                                                            Avg_Age = X.age_18.mean(), 
                                                            Avg_Inc = X.estimated_household_income_18.mean(), 
                                                            Avg_Contr_size = X.premium_amount_18.mean(),
                                                            Avg_MBids = X.max_bid.mean(),
                                                            SD_MBids = sd(X.max_bid),
                                                            N=n(X.leadID))

# Adding info about what states existing customers are coming from 
df_freq_state = df_closed >> group_by(X.state_18, X.Clusters_rc) >> summarize(State_N = n(X.state_18))
df_freq_state = df_freq_state.sort_values(by=['Clusters_rc', 'State_N'], ascending = [True, False])
df_freq_state = df_freq_state.pivot(index='Clusters_rc', columns='state_18', values='State_N')
cluster_df_rc['Most_Freq_States'] = pd.Series()
for rows in range(0,len(df_freq_state)):
    string = ", "
    string = string.join(df_freq_state.loc[rows].sort_values(ascending=False).index[0:5])
    cluster_df_rc.loc[rows, 'Most_Freq_States'] = string
cluster_df_rc


# In[38]:


print(cluster_df_mm)
print(cluster_df_rc)


# In[54]:


# Applying hierarchical clustering
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(df_closed_st_rc, method='ward'))
plt.axhline(y=10, color='r', linestyle='--')


# In[55]:


cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')  
# Adding cluster variable to closed lead dataframe 
df_closed['Clusters_hc'] = cluster.fit_predict(df_closed_st_rc)
df_closed.head(10)


# In[56]:


# Checking the clusters characteristics

#Creating cluster df
cluster_df_hc = df_closed >> group_by(X.Clusters_hc) >> summarize(Gender_dist = X.gender_18.mean(), 
                                                            Avg_Age = X.age_18.mean(), 
                                                            Avg_Inc = X.estimated_household_income_18.mean(), 
                                                            Avg_Contr_size = X.premium_amount_18.mean(),
                                                            Avg_MBids = X.max_bid.mean(),
                                                            SD_MBids = sd(X.max_bid),
                                                            N=n(X.leadID))

# Adding info about what states existing customers are coming from 
df_freq_state = df_closed >> group_by(X.state_18, X.Clusters_hc) >> summarize(State_N = n(X.state_18))
df_freq_state = df_freq_state.sort_values(by=['Clusters_hc', 'State_N'], ascending = [True, False])
df_freq_state = df_freq_state.pivot(index='Clusters_hc', columns='state_18', values='State_N')
cluster_df_rc['Most_Freq_States'] = pd.Series()
for rows in range(0,len(df_freq_state)):
    string = ", "
    string = string.join(df_freq_state.loc[rows].sort_values(ascending=False).index[0:5])
    cluster_df_hc.loc[rows, 'Most_Freq_States'] = string
cluster_df_hc


# #### <span style="color:darkred"> Question 2: Analysis on customer segments that Abc LLC should target to maximize their income? </span>

# #### <span style="color:sienna"> <i> Description of clusters of current customers:</i></span>
# -> <b>Cluster 1:</b> Mixed men and women (with more men), older adults, with HH income close to lower middle, having average contract size the highest<br/>
# -> <b>Cluster 2:</b> Mixed men and women (with slightly more men), middle aged, from upper middle income group, having contract size in the fourth quartile <br/>
# -> <b>Cluster 3:</b> Exclusively men, youngest average age, from low income HH, having contract size in the first quartile <br/>
# -> <b>Cluster 4:</b> Exclusively women, middle aged, from low income HH, having contract size around the median  <br/>  

# #### <span style="color:steelblue"> Customer segmentation on lost leads </span>

# In[ ]:


# Creating dataframe for lost leads
df_lost = df >> mask(X.cpa_status_18 == 0)
print(len(df_lost))


# In[ ]:


# Checking dataframe
df_lost.describe()


# In[ ]:


# Checking HH income for detecting outliers
sns.boxplot(x="estimated_household_income_18", data=df_lost, orient="v", palette="BuPu")


# In[ ]:


indexes_to_drop = list(outliers_iqr(df_lost['estimated_household_income_18'], 3.5))

df_lost = df_lost.drop(df_lost.index[indexes_to_drop])


# In[ ]:


df_lost.hist('estimated_household_income_18', bins=30, color = "lightsteelblue", ec="teal")
plt.title('Distribution of HH Income', fontsize=14)
plt.suptitle('Lost deals')
plt.xlabel('HH Income')


# In[ ]:


# One-hot-encoding categorical variables

# Selecting relevant features
df_lost_enc = df_lost >> select(X.leadID, 
                                    X.gender_18, 
                                    X.age_18, 
                                    X.estimated_household_income_18,
                                    X.state_18)

cat_columns = ["gender_18", "state_18"]
df_lost_enc = pd.get_dummies(df_lost_enc, 
                               prefix_sep="__",
                               columns=cat_columns)


df_lost_enc.head(10)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

# Using standardization because of presence of possible outliers: 
df_columns = ['age_18', 'estimated_household_income_18']
mms = MinMaxScaler()
df_lost_st = mms.fit_transform(df_lost_enc[['age_18', 'estimated_household_income_18']])
df_lost_st = pd.DataFrame(df_lost_st, columns=df_columns)
df_lost_st['leadID'] = list(df_lost_enc['leadID'])
df_lost_st = df_lost_st >> left_join(df_lost_enc, by='leadID') >> drop(['leadID', 'age_18_y', 'estimated_household_income_18_y'])
df_lost_st.head(10)


# In[ ]:


# Normalizing/Standardizing data to ensure that unit of dimension does not distort relative near-ness of observations

# Using standardization because of presence of possible outliers: 
df_columns = ['age_18', 'estimated_household_income_18']
rsc = RobustScaler()
df_lost_st = rsc.fit_transform(df_lost_enc[['age_18', 'estimated_household_income_18']])
df_lost_st = pd.DataFrame(df_lost_st, columns=df_columns)
df_lost_st['leadID'] = list(df_lost_enc['leadID'])
df_lost_st = df_lost_st >> left_join(df_lost_enc, by='leadID') >> drop(['leadID', 'age_18_y', 'estimated_household_income_18_y'])
df_lost_st.head(10)


# In[ ]:


# Determining the number of clusters for K-means
clusters_range = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in clusters_range]
score = [kmeans[i].fit(df_lost_st).score(df_lost_st) for i in range(len(kmeans))]

plt.plot(clusters_range,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Number of clusters by score')
plt.show()


# In[ ]:


clusters_range = [2,3,4,5,6,7,8,9,10,11,12,13,14]
inertias =[] 
for c in clusters_range:
    kmeans = KMeans(n_clusters=c, random_state=0).fit(df_lost_st)
    inertias.append(kmeans.inertia_)
plt.figure()
plt.plot(clusters_range,inertias, marker='o')


# In[ ]:


# Running K-means cluster on the encoded dataframe with 4 clusters based on elbow method
kmens = KMeans(n_clusters=4, random_state=0).fit(df_lost_st)
# Adding cluster variable to closed lead dataframe 
df_lost['Clusters'] = kmens.labels_
df_lost.head(10)


# In[ ]:


print(len(df_lost))
print(len(df_lost_enc))
print(len(df_lost_st))
print(len(kmens.labels_))


# In[ ]:


# Checking the clusters characteristics

#Creating cluster df
cluster_df_lost = df_lost >> group_by(X.Clusters) >> summarize(Gender_dist = X.gender_18.mean(), 
                                                            Avg_Age = X.age_18.mean(), 
                                                            Avg_Inc = X.estimated_household_income_18.mean(), 
                                                            Avg_Contr_size = X.premium_amount_18.mean(),
                                                            Avg_MBids = X.max_bid.mean(),
                                                            SD_MBids = sd(X.max_bid),
                                                            N=n(X.leadID))

# Adding info about what states lost possible customers are coming from 
df_freq_state = df_lost >> group_by(X.state_18, X.Clusters) >> summarize(State_N = n(X.state_18))
df_freq_state = df_freq_state.sort_values(by=['Clusters', 'State_N'], ascending = [True, False])
df_freq_state = df_freq_state.pivot(index='Clusters', columns='state_18', values='State_N')
cluster_df_lost['Most_Freq_States'] = pd.Series()
for rows in range(0,len(df_freq_state)):
    string = ", "
    string = string.join(df_freq_state.loc[rows].sort_values(ascending=False).index[0:5])
    cluster_df_lost.loc[rows, 'Most_Freq_States'] = string
cluster_df_lost


# In[ ]:


# Checking the clusters characteristics

#Creating cluster df
cluster_df_lost = df_lost >> group_by(X.Clusters) >> summarize(Gender_dist = X.gender_18.mean(), 
                                                            Avg_Age = X.age_18.mean(), 
                                                            Avg_Inc = X.estimated_household_income_18.mean(), 
                                                            Avg_Contr_size = X.premium_amount_18.mean(),
                                                            Avg_MBids = X.max_bid.mean(),
                                                            SD_MBids = sd(X.max_bid),
                                                            N=n(X.leadID))

# Adding info about what states lost possible customers are coming from 
df_freq_state = df_lost >> group_by(X.state_18, X.Clusters) >> summarize(State_N = n(X.state_18))
df_freq_state = df_freq_state.sort_values(by=['Clusters', 'State_N'], ascending = [True, False])
df_freq_state = df_freq_state.pivot(index='Clusters', columns='state_18', values='State_N')
cluster_df_lost['Most_Freq_States'] = pd.Series()
for rows in range(0,len(df_freq_state)):
    string = ", "
    string = string.join(df_freq_state.loc[rows].sort_values(ascending=False).index[0:5])
    cluster_df_lost.loc[rows, 'Most_Freq_States'] = string
cluster_df_lost

