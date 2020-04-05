#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from scipy import stats
import math
import matplotlib.pyplot as plt
from random import sample


# In[2]:


# Read csv file
df = pd.read_csv('AB_test_data.csv')


# In[3]:


df


# In[4]:


df['Variant'].value_counts()


# In[5]:


df_pop = df.loc[df['Variant'] == 'A']
df_sample = df.loc[df['Variant'] == 'B']


# In[6]:


df_pop['purchase_TF'].value_counts()


# #### Check if Time matters

# In[7]:


df.groupby(['date','Variant','purchase_TF']).count()


# In[8]:


df.dtypes


# In[9]:


df['date'] = pd.to_datetime(df['date'])


# In[10]:


# Since the AB Test starts at 2020-01-01, create dataframe under this time period
df_date_count = df.loc[df['date'] >= '2020-01-01'].groupby(['date','Variant','purchase_TF']).count().reset_index()


# In[11]:


df_date_count


# In[12]:


groups = df_date_count[df_date_count['purchase_TF'] == True].groupby("Variant")
for name, group in groups:
    plt.plot(group["date"], group["id"], marker="o", linestyle="", label=name)
plt.legend()
plt.show()


# In[13]:


groups = df_date_count[df_date_count['purchase_TF'] == False].groupby("Variant")
for name, group in groups:
    plt.plot(group["date"], group["id"], marker="o", linestyle="", label=name)
plt.legend()
plt.show()


# #### Did not find a clear seasonality behaviour

# ### Question 1. A/B Test

# In[14]:


df_date_count


# In[15]:


# Obtain the number of bookings for control group
df.loc[(df['Variant'] == 'A') & (df['date'] >= '2020-01-01')]['purchase_TF'].shape[0]


# In[16]:


# Obtain the number of True & False for control group
df.loc[(df['Variant'] == 'A') & (df['date'] >= '2020-01-01')]['purchase_TF'].value_counts()


# In[17]:


# Calculate possibility of success for control group
df.loc[(df['Variant'] == 'A') & (df['date'] >= '2020-01-01')]['purchase_TF'].value_counts()[1]/df.loc[(df['Variant'] == 'A') & (df['date'] >= '2020-01-01')]['purchase_TF'].shape[0]


# In[18]:


# Calculate possibility of success for treatment group

df.loc[(df['Variant'] == 'B') & (df['date'] >= '2020-01-01')]['purchase_TF'].value_counts()[1]/df.loc[(df['Variant'] == 'B') & (df['date'] >= '2020-01-01')]['purchase_TF'].shape[0]


# In[19]:


# Calculate possibility of success prior to the experiment

df.loc[(df['Variant'] == 'A') & (df['date'] < '2020-01-01')]['purchase_TF'].value_counts()[1]/df.loc[(df['Variant'] == 'A') & (df['date'] < '2020-01-01')]['purchase_TF'].shape[0]


# In[20]:


def hypo_test(df0, df1):
    '''
    Inputs:
        df0 -> (pandas dataframe) dataframe with control group information
        df1 -> (pandas dataframe) dataframe with treatment group information
    
    Return:
        result -> (float) p-value of the two-tailed test
    '''
    
    # Sample size for both groups
    n0 = df0.shape[0]
    n1 = df1.shape[0]
    
    # success rate for both groups
    p0 = df0['purchase_TF'].value_counts()[1]/n0
    p1 = df1['purchase_TF'].value_counts()[1]/n1
    
    # Assuming both groups are samples, calculate standard deviation 
    std = math.sqrt(((p0 * (1-p0))/n0) + ((p1 * (1-p1))/n1)) 
    
    # Z score
    z = (p1 - p0)/std
    
    print('z score is ' + str(z))
    
    # calculate p_value associated with the calculated z score -- under 2 tailed test situation
    result = (1 - stats.norm(0,1).cdf(z))*2
    print ('P-value is ' + str(result))
    
    return result


# In[21]:


hypo_test(df.loc[(df['Variant'] == 'A') & (df['date'] >= '2020-01-01')], df.loc[(df['Variant'] == 'B') & (df['date'] >= '2020-01-01')])


# ### Question 2. Optimal Sample Size

# In[22]:


# Assume variant A and B are two samples
# Reference:
# https://www.itl.nist.gov/div898/handbook/prc/section2/prc242.htm


def optimal_sample_size(alpha, p0, delta, power):
    '''
    Inputs:
        alpha -> (float) equal to 1 - confidence level
        p0 -> (float) success rate under null hypothesis
        delta -> (float) minimum detectable difference
        power -> (float) desired power of the test
        
    Return:
        result -> (float) optimal sample size, not rounded
    
    '''
    # Calculate p1 based on p0 and delta given.
    p1 = p0 + delta 
    
    # Calculate average success rate
    p_bar = (p0 + p1)/2
    
    # Obtain t_alpha and t_beta -- 2 tailed test
    t_type1 = stats.norm(0,1).ppf(alpha/2) * -1
    t_type2 = stats.norm(0,1).ppf(power)
    
    # Optimal sample size calculation
    result = ((t_type1 * ((2 * p_bar * (1 - p_bar)) ** 0.5)) + (t_type2 * ((p0 * (1 - p0) + p1 * (1 - p1)) ** 0.5)))** 2
    
    if delta != 0:
        result = result / (delta**2)
    
    return float(result)


# In[23]:


optimal_sample_size(0.05, 0.1519305311938327, 0.03, 0.8)


# In[24]:


def sample_simulation(n, df):
    '''
    Inputs:
        n -> (float) desired sample size for treatment and control groups
        df -> (pandas dataframe) dataframe that sample should be selected from
        
    return:
    result -> (pandas dataframe) a dataframe with sample selected
    
    '''
    # Separate the dataframe into two 
    df_a = df[df['Variant'] == 'A']
    df_b = df[df['Variant'] == 'B']
    
    # Upper limit of sample selection
    max_a = df_a.shape[0]
    max_b = df_b.shape[0]
    
    # Data point picked for each group
    pick_a = list(sample(range(0, max_a),round(n)))
    pick_b = list(sample(range(0, max_b),round(n)))
    
    # Create sample dataframes
    df_a_picked = df_a.iloc[pick_a]
    df_b_picked = df_b.iloc[pick_b]
    
    # Merge two sample dataframes together
    result = df_a_picked.append(df_b_picked, ignore_index = True)
    
    return result


# In[25]:


test_1 = sample_simulation(optimal_sample_size(0.05, 0.1519305311938327, 0.03, 0.8), df[df['date'] >= '2020-01-01'])


# In[26]:


test_1


# In[27]:


hypo_test(test_1.loc[test_1['Variant'] == 'A'], test_1.loc[test_1['Variant'] == 'B'])


# In[28]:


# Obtain 10 samples based on the optimal sample size calculated above

l_df = []

for i in range(10):
    l_df.append(sample_simulation(optimal_sample_size(0.05, 0.1519305311938327, 0.03, 0.8), df[df['date'] >= '2020-01-01']))


# In[29]:


# For each sample, calculate the p-value 

result_q2 = []

n_iter = 0
for sim_df in l_df:
    n_iter += 1
    print('Iteration # {} \n -------------------------'.format(n_iter))
    sim_r = hypo_test(sim_df[sim_df['Variant'] == 'A'], sim_df[sim_df['Variant'] == 'B'])
    print('\n')
    result_q2.append(sim_r)
    


# In[30]:


# Check how many time the null hypo was rejected

null_rejected = 0

for index in range(len(result_q2)):
    if float(result_q2[index]) <= 0.05:
        null_rejected += 1
    else:
        print('Sample #' + str(index + 1) + ' failed to reject Null hypothesis')
        
print('Out of 10 simulations based on the optimal sample size calculated, the null hypothesis was rejected {} times.'.format(null_rejected))


# ### Question 3. Sequential Test

# In[31]:


def SPRT(p0, p1, alpha, power, df):
    '''
    Inputs:
        p0 -> (float) success rate under null hypothesis
        p1 -> (float) success rate under alternative hypothesis
        alpha -> (float) equal to 1 - confidence level
        power -> (float) desired power of the test
        df -> (pandas dataframe) dataframe contains treatment group data
    
    Returns:
        count -> (int) number of data point used to reach to a conclusion
        choice -> (str) conclusion of the SPRT
    
    '''
    # Calculate the upper and lower boundary of the test
    log_a = math.log(1/(alpha))
    log_b = math.log(1 - power)
    
    result = 0
    
    count = 0
            
    for index, row in df.iterrows():
        if row['purchase_TF'] == True:
            result = result + math.log(p1/p0)
        else:
            result = result + math.log((1 - p1)/(1 - p0))
        count += 1
        
        if (result >= log_a) or (result <= log_b):
            break
    
    
    if result >= log_a:
        print('alternative is true')
        choice = 'Alternative'

    if result <= log_b:
        print('null hypo is true')
        choice = 'Null'
        
    if count == df.shape[0]:
        print('no solution reached')
        choice = 'No_solution'
    
    return (count, choice)


# In[32]:


# p0 take the average p of A group throughout 2019 while p1 take the sum of p0 and delta 0.03


result_q3 = {'Alternative':[], 'Null':[], 'No_solution':[]}

for index in range(len(l_df)):
    sim_df = l_df[index]
    sim_r = SPRT(0.1519305311938327, 0.1819305311938327, 
                 0.05, 0.8, sim_df.loc[(sim_df['Variant'] == 'B') & (sim_df['date'] >= '2020-01-01')])
    n_of_iteration = sim_r[0]
    decision = sim_r[1]
    
    # Obtain which sample failed to reject the Null hypothesis
    if decision != 'Alternative':
        print('Sample #' + str(index + 1) + ' failed to reject Null hypothesis')
    result_q3[decision].append(n_of_iteration)
    
result_q3

