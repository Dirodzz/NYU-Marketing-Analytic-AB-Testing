#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
from scipy import stats


# In[2]:


df = pd.read_csv('AB_test_data.csv')


# In[3]:


df


# In[5]:


df['Variant'].value_counts()


# In[7]:


df_pop = df.loc[df['Variant'] == 'A']
df_sample = df.loc[df['Variant'] == 'B']


# In[22]:


df_pop['purchase_TF'].value_counts()


# In[19]:


# Setting up null hypo

pop_t = df_pop['purchase_TF'].value_counts()[1]
pop_f = df_pop['purchase_TF'].value_counts()[0]

pop_port = pop_t/df_pop.shape[0]


# In[23]:


pop_port


# In[17]:


df_sample['purchase_TF'].value_counts()


# In[18]:


samp_t = df_sample['purchase_TF'].value_counts()[1]
samp_f = df_sample['purchase_TF'].value_counts()[0]


# In[21]:


# Calculate p value

stats.binom_test((samp_t, samp_f), p=pop_port, alternative='two-sided')


# In[ ]:




