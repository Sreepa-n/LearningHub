# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 12:37:02 2021

@author: sreepa
"""

import os
os.chdir('C:/Users/sreepa/Downloads')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

data=pd.read_csv('bank_marketing.csv',index_col=0)

from sklearn.impute import SimpleImputer
cat_col=[n for n in data.columns if data[n].dtypes=='object']
imp = SimpleImputer(missing_values='unknown', strategy='most_frequent')
for col in cat_col:
    data[col]=imp.fit_transform(data[[col]])
    


#for col in cat_col:
  #  mode=data[col].mode()[0]
   # data[col]=data[col].replace('unknown',mode)
#imp.fit(data)
#data1=imp.transform(data)
#data2=pd.DataFrame(data1,columns=col)
#=================================================
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

print(IQR)
num_columns = ['age','balance', 'day','duration', 'campaign', 'pdays', 'previous']

for col in num_columns:
    plt.boxplot(data[col])
    plt.title(col)
    plt.show()
    
data.boxplot(column='age',by='deposit')
data.boxplot(column='balance',by='deposit')
data.boxplot(column='day',by='deposit')
data.boxplot(column='duration',by='deposit')
data.boxplot(column='campaign',by='deposit')
data.boxplot(column='pdays',by='deposit')
data.boxplot(column='previous',by='deposit')

data1 = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
data=data1    

data1['age'].value_counts()
data1['balance'].value_counts()
data1['day'].value_counts()
data1['duration'].value_counts()
data1['campaign'].value_counts()
data1['pdays'].value_counts()
data1['previous'].value_counts()
#data1 = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
#data1.shape


    

#==============================================
data.info()
data.describe().T
data.isnull().sum()
data['deposit'].value_counts()
pd.crosstab(index=data['deposit'],columns=data['deposit'],normalize=True)
#pd.crosstab(data.deposit, data.deposit, normalize=True)

cat_col=[n for n in data.columns if data[n].dtypes=='object']
for col in cat_col:
    pd.crosstab(data[col], data.deposit).plot(kind='bar')
    plt.title(col)
    
cat_col=[n for n in data.columns if data[n].dtypes=='object']
for col in cat_col:
    plt.hist(x=data[col],bins='auto')
    plt.title(col)

cat_col=[n for n in data.columns if data[n].dtypes=='object']
plt.style.use('ggplot')
for col in cat_col:
    plt.figure(figsize=(20,4))
    plt.subplot(121)
    data[col].value_counts().plot(kind='bar')
    plt.title(col)
  
    
num_col=[n for n in data.columns if data[n].dtypes!='object']
for col in num_col:
    plt.figure(figsize=(20,5))
    plt.subplot(121)
    sns.distplot(data[col])
    plt.title(col)
data['pdays'].value_counts()

data= data.drop(columns = ['pdays'])

data['deposit']=data['deposit'].map({'yes':0,'no':1})
data['deposit']

# Build correlation matrix
corr = data.corr()
#from pandas_profiling import ProfileReport
#prof=ProfileReport(data)
#Dummy encoding#
data=pd.get_dummies(data,drop_first=True)
#Seperating the input names from data
features=list(set(data.columns)-set(['deposit']))

y=data['deposit'].values
print(features)
print(y)
x=data[features].values
print(x)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

#Splitting the data into train and test data
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3,random_state=0)

train_x=sc.fit_transform(train_x)
test_x=sc.transform(test_x)
#Making the instance of the  model
logistic=LogisticRegression(max_iter=1000)
#Fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_
#Prediction from test data
prediction=logistic.predict(test_x)
print(prediction)
#Confusion matrix
conf_matrix = confusion_matrix(test_y, prediction)
print(conf_matrix)
acc_score=accuracy_score(test_y, prediction)
print(acc_score)
#Printing the misclassified 
print('Misclassified samples: %d' % (test_y!=prediction).sum())

#logistic.score(train_x,train_y)
#logistic.score(test_x,test_y)

############KNN Classifier==============================

from sklearn.neighbors import KNeighborsClassifier


Misclassified_sample = []
# Calculating error for K values between 1 and 40
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x, train_y)
    pred_i=knn.predict(test_x)
    Misclassified_sample.append((test_y != pred_i).sum())
    
print(Misclassified_sample)

#Plotting the effect of K
plt.figure(figsize = (7,7))
plt.plot(range(1,20,1),Misclassified_sample,
         color='red',linestyle='dashed' , marker='o',
         markerfacecolor='blue' , markersize=10)

plt.title('Effect of K value on Misclassification')
plt.xlabel('K value')
plt.ylabel('Misclassified samples')
plt.show()

#====================================================================
#====================================================================
num_columns = ['age','balance', 'day','duration', 'campaign', 'pdays', 'previous']
fig, axs = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(20, 15))

counter = 0
for num_column in num_columns:
    
    trace_x = counter // 3
    trace_y = counter % 3
    
    axs[trace_x, trace_y].hist(data[num_column])
    
    axs[trace_x, trace_y].set_title(num_column)
    
    counter += 1

plt.show()

len (data[data['pdays'] > 400] ) / len(data) * 100

len (data[data['pdays'] == -1] )

