# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 23:55:50 2018

@author: User
"""
# grab dataset from https://archive.ics.uci.edu/ml/datasets.html to ilusstrate python machine learning
#part 1 to demonstrate data analysis
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data= pd.read_csv('energydata_complete.csv')

d = data.iloc[:,:27]

d.iloc[0,1] 
d.head(3)
d.describe()
#see null values in dataset
sns.heatmap(d.isnull(),cmap='viridis',cbar=False,yticklabels= False)

#fill in missing values in dataset my mean of set
d[d['Appliances'].isnull()==True]=97.696868
#check if values is ok
d.iloc[0,1] 

d[d.isnull()==True]

# see the general view of dataset 
sns.pairplot(d)

d.iloc[:,0] = pd.to_datetime(d.iloc[:,0])
# add time 
d['month'] = d.iloc[:,0].apply( lambda x : x.month )
d['year'] = d.iloc[:,0].apply( lambda x : x.year )
d['month'].head()
d['year'].head()

# find most powers among months
d1= d.groupby(['month','year'])['Appliances'].sum().to_frame().sort_values('Appliances',ascending=0)
d1

#visualization 
d1.plot.bar(stacked=True)
d1.reset_index().plot.scatter(x='month',y= 'Appliances',s=d1['Appliances']/200, cmap='coolwarm',figsize=(12,8))
d.head(3)

#base con pairplot, dive into RH1-RH2,month-windspeed , month-press mm
# RH1-RH2, put a constrant on upler limit to remove anormalies 
sns.lmplot(x='RH_1',y='RH_2',data= d[d['RH_2']<90])
# see the relationship in Humidity in kitchen area and in living room

#month-windspeed  Windspeed Press_mm_hg
sns.lmplot(x='month',y='Windspeed',data= d)
sns.lmplot(x='month',y='Press_mm_hg',data= d)
# there is no significant relationsnhip by time

#Part 2, with machine learning
from sklearn.preprocessing import Imputer
im = Imputer(missing_values="NaN",axis=0,strategy='mean')
im = im.fit(d[:,1])
d[:,1] = im.transform(d[:,1])

