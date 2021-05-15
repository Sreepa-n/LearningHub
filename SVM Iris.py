# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 13:56:11 2021

@author: sreepa
"""
import time
start_time = time.time()
import os
os.chdir('C:/Users/sreepa/Downloads')

# Importing libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_val_predict,cross_validate
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, chi2
X, y = datasets.load_iris(return_X_y=True)
target_names=['setosa','versicolor','virginica']

df=pd.DataFrame(np.concatenate((X,y.reshape(-1,1)),axis=1),columns=["sepal length","sepal width","petal length","petal width","species"])
print(df.head())
plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
target_names=['setosa','versicolor','virginica']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], alpha=.8, color=color,
    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('IRIS dataset')

plt.show()

# Splitting the dataset into train and test sets
train_x, test_x, train_y, test_y = train_test_split(X,y,test_size=0.3, random_state=0)
svc=SVC(C=0.001,degree=0,gamma=0.0001,kernel='linear')
train_y=train_y.reshape(-1,)
svc.fit(train_x,train_y)
n_folds=5
scores_svc=cross_val_score(svc,train_x,train_y,cv=n_folds,scoring='accuracy')
scores_cv_svc=cross_validate(svc,train_x,train_y,cv=n_folds,scoring='accuracy',return_estimator=True,return_train_score=True)
y_cv_pred_svc=cross_val_predict(svc,train_x,train_y,cv=n_folds)
y_test_pred_svc=svc.predict(test_x)


def evaluate(yt,yp):
    cf=confusion_matrix(yt,yp)
    acc=accuracy_score(yt,yp)
    return cf,acc
# Display metrics
def display(yt,yp,model):
    cf,acc = evaluate(yt,yp)
    print('Model=',model,'\ncf=',cf,'\nacc=',acc,'\n')
display(test_y,y_test_pred_svc,'SVC:Testing')

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(svc, test_x, test_y)
print(target_names)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
train_x_sc=sc.fit_transform(train_x)
test_x_sc=sc.transform(test_x)
svc.fit(train_x_sc,train_y)
y_test_pred_svc_sc = svc.predict(test_x_sc)
plot_confusion_matrix(svc, test_x_sc, test_y)
c_range=[1*10**i for i in range(-5,5)]
g_range=[1*10**i for i in range(-5,3)]


#%% Gridsearchcv
from sklearn.model_selection import GridSearchCV

#gscv = GridSearchCV(SVC(), {'kernel':['rbf','linear'],'C':c_range,'gamma':g_range}, cv=5,verbose=False,n_jobs=8)
gscv = GridSearchCV(SVC(), {'C':c_range,'gamma':g_range}, cv=5,verbose=False,n_jobs=8)
gscv.fit(train_x_sc,train_y)
gscv.best_params_

svc=SVC(kernel='linear',C=10,gamma=0.01)
svc.fit(train_x_sc,train_y)
y_test_pred_svc_sc = svc.predict(test_x_sc)
plot_confusion_matrix(svc, test_x_sc, test_y)

np.random.seed(0)
X = np.hstack((X, 2 * np.random.random((X.shape[0], 36))))
X.shape

###############################
# Create a feature-selection transform, a scaler and an instance of SVM that we
# combine together to have an full-blown estimator
clf = Pipeline([('anova', SelectPercentile(chi2)),
      ('scaler', StandardScaler()),
      ('svc', SVC(gamma="auto"))])


# Plot the cross-validation score as a function of percentile of features
score_means = list()
score_stds = list()
percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

for percentile in percentiles:
    clf.set_params(anova__percentile=percentile)
    this_scores = cross_val_score(clf, X, y)
    score_means.append(this_scores.mean())
    score_stds.append(this_scores.std())
    
plt.errorbar(percentiles, score_means, np.array(score_stds))
plt.title('Performance of the SVM-Anova varying the percentile of features selected')
plt.xticks(np.linspace(0, 100, 11, endpoint=True))
plt.xlabel('Percentile')
plt.ylabel('Accuracy Score')
plt.axis('tight')
plt.show()
###############**********************
##########*************not req**************
# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
# plot the decision function for each datapoint on the grid

Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.imshow(Z, interpolation='nearest',
extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
linestyles='dashed')
plt.scatter(X[:, 0], X[:, 1], s=30, c=5, cmap=plt.cm.Paired,
edgecolors='k')
plt.xticks(())
plt.yticks(())
plt.axis([-3, 3, -3, 3])
plt.show()
#####################not req#####*************************
###########################************************

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = .02 # step size in the mesh
svc=SVC(kernel='linear',C=100)
svc.fit(train_x[:,:2],train_y)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.title('3-Class classification using Support Vector Machine with linear'
' kernel')
plt.axis('tight')
plt.show()

seconds = time.time() - start_time
print('Time Taken:', time.strftime("%H:%M:%S",time.gmtime(seconds)))
