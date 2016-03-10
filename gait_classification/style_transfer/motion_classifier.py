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
X  = data['clips']
#(Motion, Styles)
classes = data['classes']

# import some data to play with
X = X.reshape(X.shape[0], -1)
Y = classes[:,0]

shuffled = zip(X,Y)
np.random.shuffle(shuffled)

split = int(X.shape[0] * 0.7)

X, Y = zip(*shuffled)
X_train = np.array(X)[:split]
Y_train = np.array(Y)[:split]

print 'Naive Bayes:'
nb = GaussianNB()
nb.fit(X_train, Y_train)
Z = nb.predict(X[split:])
print accuracy_score(Z, Y[split:])

print 'Logistic regression:'
logreg = linear_model.LogisticRegression()
logreg.fit(X_train, Y_train)
Z = logreg.predict(X[split:])
print accuracy_score(Z, Y[split:])

print 'SVM:'
svm = LinearSVC()
svm.fit(X_train, Y_train)
Z = svm.predict(X[split:])
print accuracy_score(Z, Y[split:])

print 'DT:'
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
Z = dt.predict(X[split:])
print accuracy_score(Z, Y[split:])

print 'KNN:'
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, Y_train)
Z = neigh.predict(X[split:])
print accuracy_score(Z, Y[split:])
