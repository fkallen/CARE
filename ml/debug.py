#!/usr/bin/env python3

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

clf : RandomForestClassifier = pickle.load(open(sys.argv[1], "rb"))

X, y = map(np.array, zip(*( ([float(x) for x in splt[:clf.n_features_in_]], float(splt[-1])) for splt in (line.split() for line in open(sys.argv[2])))))

# y_ = clf.predict_proba(X)[:,1]
# print(y)
# print(y_)
# print(np.all(y==y_))
# print(y[y!=y_])
# print(y_[y!=y_])
# print(np.max(np.abs(y-y_)))
# print(y[np.argmax(np.abs(y-y_))])
# print(y_[np.argmax(np.abs(y-y_))])
# # for i in y-y_:
# #     if i>0:
# #         print(i)
# print(np.sum(y!=y_), y.shape[0], np.sum(np.abs(y-y_)>0.00) / y.shape[0])

# print(np.argmax(np.abs(y-y_)), X[np.argmax(np.abs(y-y_))])

################################################################################

# probs = 0
# for i, tree in enumerate(clf.estimators_):
#     proba = tree.predict_proba(X)[0,1]
#     if proba>0:
#         print(i, proba)
#     probs += proba
# print(probs/clf.n_estimators)

plot_tree(clf.estimators_[42])
plt.savefig("plot.pdf")


