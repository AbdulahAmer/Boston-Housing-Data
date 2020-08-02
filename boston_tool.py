#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_boston 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error

import pandas as pd 
import numpy as np 


# In[2]:


# Gather Data 
boston_dataset =load_boston()
data=pd.DataFrame(data=boston_dataset.data, 
                  columns=boston_dataset.feature_names)
features =data.drop(['INDUS', 'AGE'], axis=1)

log_prices = np.log(boston_dataset.target)
#features are 506,11 
#log prices are 506,
# turn log prices into data frame 

target=pd.DataFrame(log_prices, columns=['PRICE'])


# In[11]:


CRIME_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX =4
PTRATIO_IDX = 8

# property_stats =np.ndarray(shape=(1,11))
# property_stats[0][CRIME_IDX] = features['CRIM'].mean()
# property_stats[0][ZN_IDX] =
# property_stats[0][CHAS_IDX] =

#dont do that just get values of our features. 
features.mean() # series
features.mean().values #array , shape is 11, we need 11,1

property_stats = features.mean().values.reshape(1,11)


# In[12]:


property_stats


# In[43]:


regr= LinearRegression().fit(features, target)
#calculating predicted values
fitted_vals =regr.predict(features) 

MSE = mean_squared_error(target, fitted_vals)
RMSE =np.sqrt(MSE)

#units for MSE and RMSE are in log dollar *1000


# In[44]:


def get_log_estimate(nr_rooms, 
                    students_per_classroom, 
                    next_to_river= False, 
                    high_confidence=True): 
 

    if next_to_river: 
        property_stats[0][CHAS_IDX]=1
    else: 
        property_stats[0][CHAS_IDX]=0
    
    property_stats[0][RM_IDX]=nr_rooms 
    property_stats[0][PTRATIO_IDX]=students_per_classroom
    log_estimate =regr.predict(property_stats)[0][0]
    
    
    # calc range 
    
    if high_confidence: 
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68
    return log_estimate, upper_bound, lower_bound, interval


# In[45]:


get_log_estimate(3,20, next_to_river=True)


# In[49]:


median_70s=np.median(boston_dataset.target)


# In[76]:


Zillow_median_price = 583.3
scale=Zillow_median_price/median_70s

def get_dollar(rm, ptratio, chas=False, conf=True):
    """
    Estimate price of a boston home. 
    rm---num of rooms 
    ptratio--num of students per teacher in area
    chas--True if near Charles River, Otherwise False
    conf=True for 95% confidence , False for 68% confidence
    
    """
    if rm<1 or ptratio <1: 
        print('not possible. Try again.')
        return 
    est, upper, lower, conf = get_log_estimate(rm, 
                                               ptratio, 
                                               chas, 
                                               conf)
    #convert to approximately todays dollars 
    dollars = np.e**est *1000 * scale 
    upper=np.e**upper*1000*scale 
    lower=np.e**lower*1000*scale
    # round
    rounded= np.around(dollars, -3)
    rupper=np.around(upper, -3)
    rlower=np.around(lower, -3)

    print(f'Estimated property value $ {rounded}')
    print(f'At {conf}% confidence range is')
    print(f'between $ {rlower}, and $ {rupper}')


# In[77]:


get_dollar(5,30)


# In[ ]:




