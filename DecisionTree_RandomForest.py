#!/usr/bin/env python
# coding: utf-8

#%% 
# Importing libraries


import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,confusion_matrix



#%%  
# Reading the data set


iris_data=pd.read_csv('Iris_data.csv')


# ### 

#%%
# Data preprocessing

iris_data=iris_data.drop(['Id'],axis=1)

iris_data['Species']=iris_data['Species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})


#%%

# Splitting data as test and train

iris_data_columns_list=list(iris_data.columns)

features=list(set(iris_data_columns_list)-set(['Species']))

x = iris_data[features].values
y = iris_data['Species'].values

train_x, test_x, train_y, test_y = train_test_split( x, y, test_size=0.25, random_state=43)


#%% 

# Building Decision Tree model

dt = DecisionTreeClassifier()
dt = dt.fit(train_x,train_y)

# Predicting the model on test data
pred_dt=dt.predict(test_x)

#Plotting the tree
from sklearn import tree
plt.figure(figsize=(8,8))

#Alternate option to set figure size
plt.rcParams['figure.figsize'] = (8, 8)
fig.set_size_inches(8, 8)

tree.plot_tree(dt) 

# Confusion matrix and accuracy
print(confusion_matrix(test_y, pred_dt))
print(accuracy_score(test_y, pred_dt))

#%%
# Building Random Forest model

rf = RandomForestClassifier(n_estimators=200, random_state=0)
rf.fit(train_x, train_y)

# Predicting the model on test data
predictions_rf = rf.predict(test_x)

# Confusion matrix and accuracy
print(confusion_matrix(test_y, predictions_rf))
print(accuracy_score(test_y, predictions_rf))




