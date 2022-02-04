#!/usr/bin/env python
# coding: utf-8

# In[5]:


import warnings
warnings.filterwarnings('ignore')


# In[6]:


import pandas as pd
customers_df=pd.read_csv("customers.csv")


# In[7]:


customers_df.head(5)


# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


sn.lmplot("age","income",data=customers_df, fit_reg=False,size =4)
# plt.title ("fig1: customers segments based on income and age")


# In[10]:


from sklearn.cluster import KMeans
clusters = KMeans(3)
clusters.fit(customers_df)


# In[11]:


customers_df["clusterid"] = clusters.labels_


# In[12]:


customers_df[0:5]


# In[15]:


markers = ['+','^','.']
sn.lmplot("age","income",
data=customers_df,
hue="clusterid",
fit_reg=False,
markers=markers,
size=4);


# In[17]:


from sklearn.preprocessing import StandardScaler


# In[18]:


scaler = StandardScaler()
scaled_customers_df = scaler.fit_transform(
customers_df[["age","income"]])
scaled_customers_df[0:5]


# In[23]:


from sklearn.cluster import KMeans
clusters_new =KMeans (3, random_state=42)
clusters_new.fit(scaled_customers_df)
customers_df["clusterid_new"]= clusters_new.labels_

markers = ['+','^','.']
sn.lmplot("age","income",
data=customers_df,
hue="clusterid",
fit_reg=False,
markers=markers,
size=4);


# In[24]:


clusters.cluster_centers_


# In[25]:


customers_df.groupby('clusterid')['age',
'income'].agg(["mean",
'std']).reset_index()


# In[ ]:




