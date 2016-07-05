#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

data = np.load('data_styletransfer.npz')

#(Examples, Time frames, joints)
clips = data['clips']

clips = np.swapaxes(clips, 1, 2)
X = clips[:,:-4]

#(Motion, Styles)
classes = data['classes']


# get mean and std
preprocessed = np.load('styletransfer_preprocessed.npz')
Xmean = preprocessed['Xmean']
Xmean = Xmean.reshape(1,len(Xmean),1)
Xstd  = preprocessed['Xstd']
Xstd = Xstd.reshape(1,len(Xstd),1)

Xstd[np.where(Xstd == 0)] = 1

X = (X - Xmean) / Xstd

# Discard the temporal dimension
X = X.reshape(X.shape[0], -1)
Y = classes[:,0]

shuffled = zip(X,Y)
np.random.shuffle(shuffled)

# 80% training, 20% test
split = int(X.shape[0] * 0.8)

X, Y = zip(*shuffled)
X_train = np.array(X)[:split]
Y_train = np.array(Y)[:split]

X_test = np.array(X)[split:]
Y_test = np.array(Y)[split:]

print 'Naive Bayes:'
nb = GaussianNB()
nb.fit(X_train, Y_train)
Z = nb.predict(X_test)
print accuracy_score(Z, Y_test)

print 'Logistic regression:'
logreg = linear_model.LogisticRegression()
logreg.fit(X_train, Y_train)
Z = logreg.predict(X_test)
print accuracy_score(Z, Y_test)

print 'SVM:'
svm = LinearSVC()
svm.fit(X_train, Y_train)
Z = svm.predict(X_test)
print accuracy_score(Z, Y_test)

print 'DT:'
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
Z = dt.predict(X_test)
print accuracy_score(Z, Y_test)

print 'KNN:'
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, Y_train)
Z = neigh.predict(X_test)
print accuracy_score(Z, Y_test)
