#!/usr/bin/python

# Inluence in Social Networks
#
# We investigated a standard, pairwise preference learning problem to find the
# more influential of a pair of candidates in a social network. Our approach was
# to compare multiple regression models using a grid search algorithm to find
# the most suitable parameters for each model. We then used the ROC curve to
# estimate the accuracies of each model.
#
# written by: Brett Boehmer and Vyyom Kelkar
# date written: April 20, 2018

import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle
from scipy import interp
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import svm, datasets
from sklearn.svm import SVR
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Loading the training Data
trainfile = open('../train.csv')
header = trainfile.next().rstrip().split(',')

y_train = []
X_train_A = []
X_train_B = []

for line in trainfile:
    splitted = line.rstrip().split(',')
    label = int(splitted[0])
    A_features = [float(item) for item in splitted[1:12]]
    B_features = [float(item) for item in splitted[12:]]
    y_train.append(label)
    X_train_A.append(A_features)
    X_train_B.append(B_features)
trainfile.close()

# These lines double the amount of training data by entering the the opposite of every entry
trainfile = open('../train.csv')
header = trainfile.next().rstrip().split(',')
for line in trainfile:
    splitted = line.rstrip().split(',')
    label = int(splitted[0])
    A_features = [float(item) for item in splitted[1:12]]
    B_features = [float(item) for item in splitted[12:]]
    if label == 0:
        y_train.append(1)
    else:
        y_train.append(0)
    X_train_A.append(B_features)
    X_train_B.append(A_features)
trainfile.close()

y_train = np.array(y_train)
X_train_A = np.array(X_train_A)
X_train_B = np.array(X_train_B)

# Use logarithmic cmoothing to transorm the features and combine into one list
X_train = np.concatenate((np.log(1+X_train_A),np.log(1+X_train_B)),1)

# Create the five classification models
classifier = OneVsRestClassifier(svm.SVR(kernel='linear'))
classifier2 = OneVsRestClassifier(svm.SVR(kernel='rbf'))
randFor = RandomForestRegressor()
gradientBoost = GradientBoostingRegressor()
elasticNet = ElasticNet()

# Training the models on the training data and predicting on the training data.
y_score = classifier.fit(X_train, y_train).predict(X_train)
y_score1 = classifier2.fit(X_train, y_train).predict(X_train)

# We use the GridSearchCV method to test mutiple parameters and find the best parameters
# This is for the Random Forest Regression Model
# y_scoreG = GridSearchCV(randFor,{'n_estimators': [10, 20, 30, 40, 50, 60, 70], 'max_features': [5, 6, 7, 8, 'auto'], 'max_depth': [1, 2, 3]}, n_jobs=16, verbose=3, cv=5).fit(X_train, y_train).predict(X_test)
y_score2 = GridSearchCV(randFor,{'n_estimators': [20], 'max_features': ['auto'], 'max_depth': [3]}, n_jobs=16, cv=5).fit(X_train, y_train).predict(X_train)

# This is the code for the Gradient Boosted Regression Model
# y_score4 = GridSearchCV(gradientBoost,{'n_estimators': [50, 100, 150, 200, 250, 300],'learning_rate': [0.5, 0.6, 0.7, 0.8, 0.9], 'max_depth': [1, 2, 3]}, n_jobs=16).fit(X_train, y_train).predict(X_test)
y_score3 = GridSearchCV(gradientBoost,{'n_estimators': [100], 'learning_rate': [0.5], 'max_depth': [2]}, n_jobs=16, cv=5).fit(X_train, y_train).predict(X_train)

# This is the code for the Elastic Net Model
# y_score5 = GridSearchCV(elasticNet,{'l1_ratio': [0, .25, .5, .75, 1],'alpha': [.01, .5, 1, 1.5, 2]}, n_jobs=16).fit(X_train, y_train).predict(X_test)
y_score4 = GridSearchCV(elasticNet,{'l1_ratio': [0],'alpha': [.01]}, n_jobs=16).fit(X_train, y_train).predict(X_train)


# Declaring dictionaries used for graphing the training accuracy of the models
fpr = dict()
tpr = dict()
fpr2 = dict()
tpr2 = dict()
fpr3 = dict()
tpr3 = dict()
fpr4 = dict()
tpr4 = dict()
fpr5 = dict()
tpr5 = dict()

plt.figure()
lw = 2
for i in range(len(X_train)):
    fpr[i], tpr[i], _ = roc_curve(y_train, y_score)
    fpr2[i], tpr2[i], _ = roc_curve(y_train, y_score1)
    fpr3[i], tpr3[i], _ = roc_curve(y_train, y_score2)
    fpr4[i], tpr4[i], _ = roc_curve(y_train, y_score3)
    fpr5[i], tpr5[i], _ = roc_curve(y_train, y_score4)

# Plotting the accuracy on the training data of the 5 different models using the roc_auc_score
plt.plot(fpr[2], tpr[2], color='purple',
         lw=lw, label='Linear SVR (area = %0.5f)' % roc_auc_score(y_train,y_score, average='macro',sample_weight=None))
plt.plot(fpr2[2], tpr2[2], color='red',
         lw=lw, label='RBF SVR (area = %0.5f)' % roc_auc_score(y_train,y_score1, average='macro',sample_weight=None))
plt.plot(fpr3[2], tpr3[2], color='darkorange',
         lw=lw, label='Random Forest Regressor (area = %0.5f)' % roc_auc_score(y_train,y_score2, average='macro',sample_weight=None))
plt.plot(fpr4[2], tpr4[2], color='blue',
         lw=lw, label='Gradient Boosting Regression (area = %0.5f)' % roc_auc_score(y_train,y_score3, average='macro',sample_weight=None))
plt.plot(fpr5[2], tpr5[2], color='yellow',
         lw=lw, label='Elastic Net (area = %0.5f)' % roc_auc_score(y_train,y_score4, average='macro',sample_weight=None))

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Classifying Training Data')
plt.legend(loc="lower right")
plt.show()

# Reading in the test data
testfile = open('../test.csv')
testfile.next()

X_test_A = []
X_test_B = []
for line in testfile:
    splitted = line.rstrip().split(',')
    A_features = [float(item) for item in splitted[0:11]]
    B_features = [float(item) for item in splitted[11:]]
    X_test_A.append(A_features)
    X_test_B.append(B_features)
testfile.close()

X_test_A = np.array(X_test_A)
X_test_B = np.array(X_test_B)

# transform features in the same way as for training to ensure consistency
X_test = np.concatenate((np.log(1+X_test_A),np.log(1+X_test_B)),1)

# Create the classification for the Gradient Boost Regression model
gradientBoost = GradientBoostingRegressor()

# This is the code for the Gradient Boosted Regression Model
y_score = GridSearchCV(gradientBoost,{'n_estimators': [100], 'learning_rate': [0.5], 'max_depth': [2]}, n_jobs=16, cv=5).fit(X_train, y_train).predict(X_test)


#Creating the submission file using the Gradient Boost Regression Model
predfile = open('GradientBoostingRegressor.csv','w')
print >> predfile, 'Id,Choice'
for i in range(len(y_score)):
    if y_score[i]>1:
        print >>predfile, '%d,%d' % (i+1, 1)
    elif y_score[i]<0:
        print >>predfile, '%d,%d' % (i+1, 0)
    else:
        print >>predfile, '%d,%f' % (i+1, y_score[i])

predfile.close()
