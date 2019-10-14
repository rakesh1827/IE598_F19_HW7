#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Reference 1: Codes and hints provided by Prof. Matthew D Murphy

# Importing necessary libraries

import sklearn
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score


# In[2]:


pd.options.display.max_columns = 2000


# In[3]:


# loading the credit card default dataset

cc_df = pd.read_csv("ccdefault.csv")


# In[4]:


# droppinng "ID" column from the dataset

cc_df = cc_df.drop(["ID"],axis =1)


# In[5]:


# Top 5 samples of the dataset

cc_df.head()


# In[6]:


# Summary Statistics of the dataset

cc_df.describe()


# In[7]:


# Checking for missing values

cc_df.isnull().describe()


# In[8]:


# Creating feature variables and target variable

X = cc_df.drop(["DEFAULT"], axis =1)
y = cc_df["DEFAULT"]


# In[9]:


# Checking dimension of X and y

print(X.shape,"\n",y.shape)


# In[10]:


# printing top 5 samples of X and y

print(X.head(),"\n",y.head())


# In[11]:


from sklearn.preprocessing   import StandardScaler


# In[12]:


# normalisation of data

sc_x  = StandardScaler()

X = sc_x.fit_transform(X)


# In[13]:


# printing samples after normalisation

print(X.shape,"\n",y.shape)

print(X[0:5],"\n",y[0:5])


# In[14]:


import time


# In[15]:


# splitting into training and test data

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.1, stratify = y, random_state = 42)


# In[19]:


from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier()

param_range = [10,25,50,100,200,300,500]

param_grid  = [{'n_estimators': param_range}]


# In[20]:


gs = GridSearchCV(RF,param_grid, scoring = 'accuracy', cv = 10, n_jobs = -1)

gs.fit(X_train,y_train)


# In[21]:


gs.best_score_


# In[22]:


gs.best_params_


# In[23]:


gs.cv_results_


# In[24]:


rf_best = RandomForestClassifier(n_estimators = 500,random_state = 1)

rf_best.fit(X_train,y_train)

print("train accuracy: ",rf_best.score(X_train,y_train),"\n")
print("test accuracy: ",rf_best.score(X_test,y_test),"\n")


# In[39]:


feat_labels = cc_df.columns[:-1]

importances = rf_best.feature_importances_

indices     = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print ("%2d %-*s %f" % (f+1,30,feat_labels[indices[f]],importances[indices[f]]))


# In[40]:


plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),importances[indices],align ='center')
plt.xticks(range(X_train.shape[1]),feat_labels[indices],rotation=90)
plt.xlim([-1, X_train.shape[1]])


# In[41]:


print("My name is Rakesh Reddy Mudhireddy")
print("My NetID is: rmudhi2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[ ]:




