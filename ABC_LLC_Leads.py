#!/usr/bin/env python
# coding: utf-8

# # <span style="color:darkslategray">ABC LLC - Leads for Health Insurance Services </span>

# ### <span style="color:dimgray">Importing packages</span>

# In[165]:


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

# Importing packages for modeling
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm


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


# Finding out how winning a bid relates to not having info on contract size
filtered_df = df[df['cpa_status_18'].isna()] 
filtered_df >> group_by(X.won) >> summarize(N = n(X.leadID))


# In[7]:


# Describing continuous variables
df.describe()


# In[8]:


#Missing values computation
df.isnull().sum()


# #### <span style="color:steelblue"> Univariate analysis on closed leads </span>

# In[9]:


df_closed = df >> mask(X.cpa_status_18 == 1)


# In[10]:


df_closed.dtypes


# In[11]:


df_closed.describe()


# In[12]:


# Distribution of gender 
sns.countplot(x='gender_18', data=df_closed, palette="BuPu")
plt.title('Distribution of Gender')


# In[13]:


sns.boxplot(x="age_18", data=df_closed, orient="v", palette="BuPu")


# In[14]:


# Distribution of ages

df_closed.hist('age_18', bins=30, color = "lightsteelblue", ec="teal")
plt.title('Distribution of Age')
plt.xlabel('Age')


# In[15]:


sns.boxplot(x="estimated_household_income_18", data=df_closed, orient="v", palette="BuPu")


# In[16]:


#Excluding the outlier cases 

def outliers_iqr(x, mplyr):
    quartile_1, quartile_3 = np.percentile(x, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * mplyr)
    upper_bound = quartile_3 + (iqr * mplyr)
    return np.where((x > upper_bound) | (x < lower_bound))
indexes_to_drop = list(outliers_iqr(df_closed['estimated_household_income_18'], 3.5))

df_closed = df_closed.drop(df_closed.index[indexes_to_drop])


# In[17]:


df_closed.hist('estimated_household_income_18', bins=20, color = "lightsteelblue", ec="teal")
plt.title('Distribution of HH Income')
plt.xlabel('HH Income')


# In[18]:


sns.boxplot(x="premium_amount_18", data=df_closed, orient="v", palette="BuPu")


# In[19]:


#Excluding the outlier case 
df_closed = df_closed >> mask(X.premium_amount_18 < max(df_closed['premium_amount_18'])) 


# In[20]:


df_closed = df_closed >> mask(X.premium_amount_18 > 0) 

df_closed.hist(column='premium_amount_18', bins=25, color = "lightsteelblue", ec="teal")
plt.title('Value of the contract')
plt.xlabel('Contract Value')


# In[21]:


# Distribution of closed leads by states 
df_state = df_closed >> group_by(X.state_18, X.gender_18) >> summarize(N=n(X.leadID))


plt.figure(figsize=(16, 16))
g = sns.catplot(x='state_18', y='N', hue='gender_18', data=df_state, palette="BuPu", kind='bar', legend=False)
plt.title('Distribution of leads by states')
plt.legend(title='Gender', loc='upper left', labels=['Male', 'Female'])
g.fig.set_figwidth(10)
g.fig.set_figheight(7)


# ### <span style="color:dimgray"> I. Abc LLC's current most typical client </span>

# In[22]:


# Creating new categorical age variable
#bins = pd.IntervalIndex.from_tuples([(0, 17), (18, 44), (45, 64), (65, 150)])
bins=[0, 18, 45, 65, np.inf]
df_closed['Age_c'] = pd.cut(round(df_closed['age_18'], 0), bins=bins, labels=['Under_18', '18-44', '45-64', '65_and_Above'])


# In[23]:


# Checking classes of age
df_closed >> group_by(X.Age_c) >> summarize(N=n(X.leadID), Min=X.age_18.min(), Max=X.age_18.max())


# In[24]:


# Creating new categorical income variable
df_closed['Income_c'] = pd.cut(df_closed['estimated_household_income_18'], bins=4, labels=['Lowest', 'Lower_middle', 'Upper_middle', 'Highest'])


# In[25]:


# Checking classes of income
df_closed >> group_by(X.Income_c) >> summarize(N=n(X.leadID), Min=X.estimated_household_income_18.min(), Max=X.estimated_household_income_18.max())


# In[26]:


#Finding the 5 most typical customer groups
df_typ_cust = df_closed >> group_by(X.state_18, X.gender_18, X.Age_c, X.Income_c) >> summarize(N = n(X.leadID), Avg_Contr_Size = X.premium_amount_18.mean())
df_typ_cust.sort_values(by='N', ascending=False).head(5)


# #### <span style="color:darkred"> Question 1: Analysis on most typical client </span>

# #### <span style="color:sienna"> <i> The typical client of ABC LLC is a young adult male who lives in New York, whose household income belongs to the lowest segment of the sample and average contract size is around 740 dollars. </i></span>

# ### <span style="color:dimgray"> II. Customer segmentation using K-means and Hierarchical clustering </span>

# ### <span style="color:steelblue"> 1. Customer segmentation on closed leads </span>

# #### <span style="color:darkgray"> One-hot-encoding </span>

# In[27]:


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


# #### <span style="color:darkgray"> Running K-Means on data standardized with MinMaxScaler </span>

# In[28]:


# Standardizing data to ensure that unit of dimension does not distort relative near-ness of observations

# Using MinMaxScale considering the presence of high number of binnary features: 
df_columns = ['age_18', 'estimated_household_income_18', 'premium_amount_18']
mms = MinMaxScaler()
df_closed_st_mm = mms.fit_transform(df_closed_enc[['age_18', 'estimated_household_income_18', 'premium_amount_18']])
df_closed_st_mm = pd.DataFrame(df_closed_st_mm, columns=df_columns)
df_closed_st_mm['leadID'] = list(df_closed_enc['leadID'])
df_closed_st_mm = df_closed_st_mm >> left_join(df_closed_enc, by='leadID') >> drop(['leadID', 'age_18_y', 'estimated_household_income_18_y', 'premium_amount_18_y'])
df_closed_st_mm.head(10)


# In[29]:


# Determining the number of clusters for K-means
clusters_range = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in clusters_range]
score = [kmeans[i].fit(df_closed_st_mm).score(df_closed_st_mm) for i in range(len(kmeans))]

plt.plot(clusters_range,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Number of clusters by score')
plt.show()


# In[30]:


clusters_range = [2,3,4,5,6,7,8,9,10,11,12,13,14]
inertias =[] 
for c in clusters_range:
    kmeans = KMeans(n_clusters=c, random_state=0).fit(df_closed_st_mm)
    inertias.append(kmeans.inertia_)
plt.figure()
plt.plot(clusters_range,inertias, marker='o')


# In[31]:


# Running K-means cluster on the encoded dataframe with 4 clusters based on elbow method
kmens = KMeans(n_clusters=4, random_state=0).fit(df_closed_st_mm)
# Adding cluster variable to closed lead dataframe 
df_closed['Clusters_mm'] = kmens.labels_
df_closed.head(10)


# In[32]:


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


# #### <span style="color:darkgray"> Running K-Means on data standardized with RobustScaler </span>

# In[33]:


# Standardizing data to ensure that unit of dimension does not distort relative near-ness of observations

# Using RobustScaler considering the presence of possible outliers: 
df_columns = ['age_18', 'estimated_household_income_18', 'premium_amount_18']
rsc = RobustScaler()
df_closed_st_rc = rsc.fit_transform(df_closed_enc[['age_18', 'estimated_household_income_18', 'premium_amount_18']])
df_closed_st_rc = pd.DataFrame(df_closed_st_rc, columns=df_columns)
df_closed_st_rc['leadID'] = list(df_closed_enc['leadID'])
df_closed_st_rc = df_closed_st_rc >> left_join(df_closed_enc, by='leadID') >> drop(['leadID', 'age_18_y', 'estimated_household_income_18_y', 'premium_amount_18_y'])
df_closed_st_rc.head(10)


# In[34]:


clusters_range = [2,3,4,5,6,7,8,9,10,11,12,13,14]
inertias =[] 
for c in clusters_range:
    kmeans = KMeans(n_clusters=c, random_state=0).fit(df_closed_st_rc)
    inertias.append(kmeans.inertia_)
plt.figure()
plt.plot(clusters_range,inertias, marker='o')


# In[35]:


# Running K-means cluster on the encoded dataframe with 4 clusters based on elbow method
kmens = KMeans(n_clusters=4, random_state=0).fit(df_closed_st_rc)
# Adding cluster variable to closed lead dataframe 
df_closed['Clusters_rc'] = kmens.labels_
df_closed.head(10)


# In[36]:


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


# In[37]:


print(cluster_df_mm)
print(cluster_df_rc)


# #### <span style="color:darkgray"> Running Hierarchical clustering on data standardized with MinMaxScaler </span>

# In[38]:


# Applying hierarchical clustering
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(df_closed_st_mm, method='ward'))
plt.axhline(y=10, color='r', linestyle='--')


# In[39]:


cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
# Adding cluster variable to closed lead dataframe 
df_closed['Clusters_hc'] = cluster.fit_predict(df_closed_st_rc)
df_closed.head(10)


# In[69]:


# Checking the clusters characteristics

#Creating cluster df
cluster_df_hc = df_closed >> group_by(X.Clusters_hc) >> summarize(Gender_dist = X.gender_18.mean(), 
                                                            Median_Age = median(X.age_18),
                                                            FirstQ_Age = median(X.age_18)- IQR(X.age_18)/2, 
                                                            ThirdQ_Age = median(X.age_18)+ IQR(X.age_18)/2,
                                                            Median_Inc = median(X.estimated_household_income_18),
                                                            FirstQ_Inc = median(X.estimated_household_income_18)- IQR(X.estimated_household_income_18)/2,
                                                            ThirdQ_Inc = median(X.estimated_household_income_18)+ IQR(X.estimated_household_income_18)/2,
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
cluster_df_hc[['Median_Age', 'FirstQ_Age', 'ThirdQ_Age', 'Median_Inc', 'FirstQ_Inc', 'ThirdQ_Inc', 'Avg_Contr_size']] = round(cluster_df_hc[['Median_Age', 'FirstQ_Age', 'ThirdQ_Age', 'Median_Inc', 'FirstQ_Inc', 'ThirdQ_Inc', 'Avg_Contr_size']], 0)
cluster_df_hc


# #### <span style="color:darkred"> Question 2: Analysis on customer segments that Abc LLC should target to maximize their income? </span>

# #### <span style="color:sienna"> <i> Considerations on Cluster analysis </i></span>

# 1. Clusters' characteristics are highly depends on encoding and standardization method. Mixed binnary and continous feature space makes the analysis challenging. Using Kmeans and hierarchical cluster algortihms provided fairly different outcomes.
# 2. It is likely that the importance of features varies depending on standardization method. Variance between groups' by features varies by chosen method.
# 3. For this analysis, hierarchical cluster approach was chosen considering it provides failry good result regarding difference between clusters for each feature

# #### <span style="color:sienna"> <i> Description of clusters of current customers:</i></span>
# -> <b>Cluster 1:</b> Mixed men and women (with more men), older adults, with middle HH income, having average contract size the highest, mostly from NY, MA, WA<br/>
# -> <b>Cluster 2:</b> Exclusively men, middle aged, their HH income is around the median, having contract size above the median, mostly from MA, WA, NJ <br/>
# -> <b>Cluster 3:</b> Exclusively women, middle aged, from low income HH, having contract size around the median, mostly from NY, WA, MA <br/>
# -> <b>Cluster 4:</b> Exclusively men, youngest average age, more HH in the low income group, having contract less than average, mostly from NY, IL, OH <br/>  

# ##### <span style="color:sienna"> <i> Targeting strategy:</i></span>
# In general, the first and forth clusters could be targeted considering the lower number of successfull sales there. Plus, first cluster's high average contract size can be a very good starting point to maximize income. In this group, we can see the largest amount of purchasing power.
# The company is strong in NY, MA and WA, in almost every segment. Other states like NJ and CT could be also in the point of interest, since they are present in case of more than one segment.
# Gender specificity can be recognized in the sample meaning that, it may be a good idea to target gender separately.
# As regards age, the company is successfull in the middle aged customers, while may want to increase the number of successfull sales in the younger and older adult groups.

# ### <span style="color:steelblue"> Customer segmentation on lost leads </span>

# In[41]:


# Creating dataframe for lost leads
df_lost = df >> mask(X.cpa_status_18 == 0)


# In[42]:


# Checking dataframe
df_lost.describe()


# In[43]:


# Checking HH income for detecting outliers
sns.boxplot(x="estimated_household_income_18", data=df_lost, orient="v", palette="BuPu")


# In[44]:


indexes_to_drop = list(outliers_iqr(df_lost['estimated_household_income_18'], 3.5))

df_lost = df_lost.drop(df_lost.index[indexes_to_drop])


# In[45]:


df_lost.hist('estimated_household_income_18', bins=30, color = "lightsteelblue", ec="teal")
plt.title('Distribution of HH Income', fontsize=14)
plt.suptitle('Lost deals')
plt.xlabel('HH Income')


# #### <span style="color:darkgray"> One-hot-encoding </span>

# In[46]:


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


# #### <span style="color:darkgray"> Running Hierarchical Clustering on data standardized with MinMaxScaler </span>

# In[47]:


# Using MinMaxScaler:
df_columns = ['age_18', 'estimated_household_income_18']
mms = MinMaxScaler()
df_lost_st = mms.fit_transform(df_lost_enc[['age_18', 'estimated_household_income_18']])
df_lost_st = pd.DataFrame(df_lost_st, columns=df_columns)
df_lost_st['leadID'] = list(df_lost_enc['leadID'])
df_lost_st = df_lost_st >> left_join(df_lost_enc, by='leadID') >> drop(['leadID', 'age_18_y', 'estimated_household_income_18_y'])
df_lost_st.head(10)


# In[48]:


df_lost['Clusters_hc'] = cluster.fit_predict(df_lost_st)
df_lost.head(10)


# In[49]:


# Checking the clusters characteristics

#Creating cluster df
cluster_df_lost = df_lost >> group_by(X.Clusters_hc) >> summarize(Gender_dist = X.gender_18.mean(), 
                                                            Avg_Age = X.age_18.mean(), 
                                                            Avg_Inc = X.estimated_household_income_18.mean(),
                                                            Avg_MBids = X.max_bid.mean(),
                                                            SD_MBids = sd(X.max_bid),
                                                            N=n(X.leadID))

# Adding info about what states lost possible customers are coming from 
df_freq_state = df_lost >> group_by(X.state_18, X.Clusters_hc) >> summarize(State_N = n(X.state_18))
df_freq_state = df_freq_state.sort_values(by=['Clusters_hc', 'State_N'], ascending = [True, False])
df_freq_state = df_freq_state.pivot(index='Clusters_hc', columns='state_18', values='State_N')
cluster_df_lost['Most_Freq_States'] = pd.Series()
for rows in range(0,len(df_freq_state)):
    string = ", "
    string = string.join(df_freq_state.loc[rows].sort_values(ascending=False).index[0:5])
    cluster_df_lost.loc[rows, 'Most_Freq_States'] = string
cluster_df_lost


# ### <span style="color:steelblue"> Current market potential for customer segments </span>

# <b>Market potential:</b> The entire size of the market for a product at a specific time (in this case last one year counted from the latest lead). It represents the upper limits of the market for a product. Market potential is usually measured either by sales value or sales volume. 

# #### <span style="color:darkgray"> Analyzing Potential Customer Base </span> 

# In[72]:


# Checking customer segments again
cluster_df_hc


# In[137]:


# Converting created feature in df to timestamp
df['created'] = pd.to_datetime(df.created) 

# Market potential for customer segment 1
mp_cs1_df = df >> mask(X.age_18 >=53, 
                       X.age_18 <= 60, 
                       X.estimated_household_income_18 >= 71225, 
                       X.estimated_household_income_18 <= 102575,
                       X.created >= pd.to_datetime('2018-02-15 12:55:28'),
           (X.state_18 == 'NY') | (X.state_18 == 'MA') | (X.state_18 == 'WA'))
print('Customer base for Segment 1: ' + str(len(mp_cs1_df)))
print('Estimated income potential based on average contract size for segment 1: ' + str(len(mp_cs1_df)*cluster_df_hc.loc[0,'Avg_Contr_size']) + str(' USD'))


# In[138]:


# Market potential for customer segment 2
mp_cs2_df = df >> mask(X.gender_18 == 1,
                       X.age_18 >=33, 
                       X.age_18 <= 59, 
                       X.estimated_household_income_18 >= 38126, 
                       X.estimated_household_income_18 <= 53876,
                       X.created >= pd.to_datetime('2018-02-15 12:55:28'),
                       (X.state_18 == 'NJ') | (X.state_18 == 'MA') | (X.state_18 == 'WA'))
print('Customer base for Segment 2: ' + str(len(mp_cs2_df)))
print('Estimated income potential based on average contract size for segment 2: ' + str(len(mp_cs2_df)*cluster_df_hc.loc[1,'Avg_Contr_size']) + str(' USD'))


# In[139]:


# Market potential for customer segment 3
mp_cs3_df = df >> mask(X.gender_18 == 2,
                       X.age_18 >=33, 
                       X.age_18 <= 59, 
                       X.estimated_household_income_18 >= 37000, 
                       X.estimated_household_income_18 <= 57000,
                       X.created >= pd.to_datetime('2018-02-15 12:55:28'),
                       (X.state_18 == 'NY') | (X.state_18 == 'MA') | (X.state_18 == 'WA'))
print('Customer base for Segment 3: ' + str(len(mp_cs3_df)))
print('Estimated income potential based on average contract size for segment 3: ' + str(len(mp_cs3_df)*cluster_df_hc.loc[2,'Avg_Contr_size']) + str(' USD'))


# In[140]:


# Market potential for customer segment 4
mp_cs4_df = df >> mask(X.gender_18 == 1,
                       X.age_18 >=26, 
                       X.age_18 <= 48, 
                       X.estimated_household_income_18 >= 35780, 
                       X.estimated_household_income_18 <= 47420,
                       X.created >= pd.to_datetime('2018-02-15 12:55:28'),
                       (X.state_18 == 'NY') | (X.state_18 == 'MA') | (X.state_18 == 'WA'))
print('Customer base for Segment 4: ' + str(len(mp_cs4_df)))
print('Estimated income potential based on average contract size for segment 4: ' + str(len(mp_cs4_df)*cluster_df_hc.loc[3,'Avg_Contr_size']) + str(' USD'))


# ### <span style="color:steelblue"> How much more Abc LLC should bid on average on the segments to increase their income by 30% </span>

# <b> Strategy for analysis: </b> Running logistic regression to predict whether a lead will be won. 

# <b><i>Input variables:</b></i> <br/>
# - gender_18<br/>
# - age_18<br/>
# - estimated_household_income_18<br/>
# - max_bid <br/>
# 
# <b><i>Target variable:</b></i> <br/>
# - won

# In[159]:


# Creating new dataframe for the regression
df_log_enc = df >> select(X.leadID, X.gender_18, X.age_18, X.estimated_household_income_18, X.state_18, X.max_bid )

# One-hot-encoding categorical variables

cat_columns = ["gender_18", "state_18"]
df_log_enc = pd.get_dummies(df_log_enc,
                            prefix_sep="__",
                            columns=cat_columns,
                            drop_first=True)

df_log_enc.head(10)


# In[162]:


# Using MinMaxScaler for Standardization: 
mms = MinMaxScaler()

# Scaling dataframe
df_columns = ['age_18', 'estimated_household_income_18', 'max_bid']
df_log_mm = mms.fit_transform(df_log_enc[['age_18', 'estimated_household_income_18', 'max_bid']])
df_log_mm = pd.DataFrame(df_log_mm, columns=df_columns)
df_log_mm['leadID'] = list(df_log_enc['leadID'])
df_log_mm = df_log_mm >> left_join(df_log_enc, by='leadID') >> drop(['leadID', 'age_18_y', 'estimated_household_income_18_y', 'max_bid_y'])
df_log_mm.head(10)


# In[167]:


logmodel = sm.Logit(df['won'], df_log_mm)
result = logmodel.fit()
result.summary()


# In[168]:


np.exp(result.params)


# In[ ]:





# In[ ]:





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

