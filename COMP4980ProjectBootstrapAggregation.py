#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("sapimouse_ABS_dx_dy_1min.csv",header=None)


# In[3]:


#get total distance
df.insert(len(df.columns)-1, 'x_distance', df.iloc[:, :128].sum(axis=1))
df.insert(len(df.columns)-1, 'y_distance', df.iloc[:, 128:256].sum(axis=1))


# In[4]:


#get amount of zeros
count_zeros = lambda row: (row == 0).sum()
df.insert(len(df.columns) - 1, 'Num_Zeros', df.apply(count_zeros, axis=1))


# In[5]:


#get max speed
get_max_x = lambda row: row[:128].max()
get_max_y = lambda row: row[128:256].max()
df.insert(len(df.columns) - 1, 'max_x', df.apply(get_max_x, axis=1))
df.insert(len(df.columns) - 1, 'max_y', df.apply(get_max_y, axis=1))


# In[6]:


df.iloc[:, 256:261]


# In[7]:


df.columns = df.columns.astype(str)


# In[8]:


#divide into train and test sets
random_state = 42
df_train,df_test=df.sample(frac=0.8,random_state=random_state),df.sample(frac=0.2,random_state=random_state)


# In[33]:


df


# In[10]:


#divide non type and type
#df_train_x=df_train.iloc[:, :261]
#df_train_y=df_train.iloc[:, -1]
#df_test_x=df_test.iloc[:, :261]
#df_test_y=df_test.iloc[:, -1]
df_train_x=df_train.iloc[:, 256:261]
df_train_y=df_train.iloc[:, -1]
df_test_x=df_test.iloc[:, 256:261]
df_test_y=df_test.iloc[:, -1]


# In[11]:


from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# In[12]:


models = [RandomForestClassifier(), 
          DecisionTreeClassifier(),
          SVC(),
          KNeighborsClassifier(),
          MLPClassifier()]
models_str = ['RandomForestClassifier',
              'DecisionTreeClassifier',
             'SVC',
             'KNeighborsClassifier',
             'MLPClassifier']


# In[13]:


accuracies_y = []
estimators_x = range(1,9) #range of estimators
for n in estimators_x:
    print(n)
    i=0
    temp=[]
    for m in models:
        print(models_str[i])
        i+=1
        bagging_model = BaggingClassifier(base_estimator=m, n_estimators=n, random_state=random_state)
        bagging_model.fit(df_train_x, df_train_y)
        y_pred = bagging_model.predict(df_test_x)
        temp.append(round(accuracy_score(df_test_y, y_pred),2))
    accuracies_y.append(temp)


# In[14]:


accuracies_y


# In[30]:


plt.plot(estimators_x, accuracies_y, label=models_str)
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Estimators')
plt.legend(loc='center right', bbox_to_anchor=(1, 0.6))

