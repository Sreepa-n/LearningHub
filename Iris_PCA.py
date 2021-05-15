# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 13:37:16 2021

@author: sreepa
"""

import os
os.chdir('C:/Users/sreepa/Downloads')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score,confusion_matrix
#from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

iris_data=pd.read_csv('Iris_data.csv')
iris_data.info()
iris_data.describe()
iris_data.head()

iris_data=iris_data.drop(['Id'],axis=1)
iris_data.head()
input_columns=list(iris_data.iloc[:,:4].columns)
input_columns.sort()
input_data=iris_data[input_columns]
input_data.head()

plt.subplots(figsize=(8,6))
sns.scatterplot(x='SepalLengthCm',y='SepalWidthCm',hue='Species',data=iris_data)
plt.show()


scaler=StandardScaler(with_std=False)
input_data=scaler.fit_transform(input_data)
type(input_data)
input_data=pd.DataFrame(input_data,columns=input_columns)
input_data.head()

u, s, v = np.linalg.svd(input_data)
pc=iris_data[input_columns].dot(v.T)
pc.columns=['PC1','PC2','PC3','PC4']
pc['Species']=iris_data.Species
pc.head()

exp_var=s**2/np.sum(s**2)*100
exp_var

plt.subplots(figsize=(8,6))
sns.scatterplot(x='PC1',y='PC2',hue='Species',data=pc)
plt.show()

#================================================

PCA_Sklearn = PCA(n_components=0.95)

Projected_data_sklearn=PCA_Sklearn.fit_transform(iris_data.iloc[:,:4])

Projected_data_sklearn_df=pd.DataFrame(Projected_data_sklearn,columns=['PC1','PC2'])

Projected_data_sklearn_df_with_class_info=pd.concat([Projected_data_sklearn_df,iris_data.Species],axis=1)

print('Explained variance :\n')
print(PCA_Sklearn.explained_variance_ratio_)

plt.subplots(figsize=(8,6))
sns.scatterplot(x='PC1',y='PC2',hue='Species',data=Projected_data_sklearn_df_with_class_info)
plt.show()

#======================================================


pd.crosstab(index=iris_data['Species'],columns=iris_data['Species'],normalize=True)

sc=StandardScaler()

iris_data['Species']=iris_data['Species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
features=list(set(iris_data.columns)-set(['Species']))
y=iris_data['Species'].values
print(features)
x=iris_data[features].values

x=sc.fit_transform(x)

pca=PCA(n_components=2)
pca.fit(input_data)
pca.components_
z=pca.transform(input_data)
plt.scatter(z[:,0],z[:,1], c=y)

