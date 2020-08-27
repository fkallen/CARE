#!/usr/bin/python3

from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn import tree

import numpy as np
from matplotlib import pyplot as plt

def onehot(base):
    if base == "A":
        return [1,0,0,0]
    elif base == "C":
        return [0,1,0,0]
    elif base == "G":
        return [0,0,1,0]
    elif base == "T":
        return [0,0,0,1]
    else:
        print("ASDASDASDASD!!!!!!!!!!!!!!!!!", base)
        return [0,0,0,0]

def onehot_(enc):
    return np.array(["A","C","G","T"])[[bool(x) for x in enc]][0]

### get X

linecount = sum(1 for line in open("ml/samples", "r"))
X = np.zeros((linecount, 19))
for i, line in enumerate(open("ml/samples", "r")):
    splt = line.split()
    X[i] = splt[:2] + onehot(splt[2]) + onehot(splt[3]) + splt[4:]

print(X[0])
print(X[1])
print(X.shape)

print("sorting...")
X = X[X[:,0].argsort()]
print(X.shape)
print("done.")
print(X[0])
print(X[1])

### get y
y = np.zeros(X.shape[0])

# with open("/home/jc/extra/ec/athaliana60cov_errFree.fq", "r") as truthfile:
#     file_index = 0
#     # infile.readline()
#     truthfile.readline()
#     for i in sorted(rawfeatures):
#         for _ in range(4*(i-file_index)-1):
#             # infile.readline()
#             truthfile.readline()
#         file_index = i
#         # seq = infile.readline()[:-1]
#         # print(seq)
#         trueseq = truthfile.readline()[:-1]
#         new = [rawfeatures[i][j][0]==trueseq[j] for j in rawfeatures[i]]
#         y+=new

with open("/home/jc/extra/ec/athaliana60cov_errFree.fq", "r") as truthfile:
    filepos = 0
    for i, x in enumerate(X):
        if filepos != int(x[0])*4+2:
            while filepos<int(x[0])*4+1:
                truthfile.readline()
                filepos += 1
            trueseq = truthfile.readline()
            filepos += 1
        # print(trueseq, filepos, filepos%4, x[0], x[1])
        y[i] = onehot_(x[2:6])==trueseq[int(x[1])]

print(y[0])
### classifier
X, y = shuffle(np.array(X), np.array(y))
num_samples = len(y)

# test_ratio = 0.1
# num_test = int(num_samples*test_ratio)
# num_train = num_samples-num_test
# X_train, y_train = X[:-num_test], y[:-num_test]
# X_test, y_test = X[-num_test:], y[-num_test:]
#------------------------------------------------------------------------------------------------------------------------------------------
small_class = sum(y==True)<len(y)//2
small_idx, big_idx = np.arange(num_samples)[y==small_class], np.arange(num_samples)[y!=small_class]
print(len(small_idx), len(big_idx))

bootstrap = np.random.choice(len(small_idx), len(small_idx))
train_small_idx = small_idx[bootstrap]
test_small_idx = np.delete(small_idx, bootstrap)

bootstrap = np.random.choice(len(big_idx), len(small_idx)) # we want equally sized class groups
train_big_idx = big_idx[bootstrap]
test_big_idx = np.random.choice(np.delete(big_idx, bootstrap), len(test_small_idx), replace=False)
#------------------------------------------------------------------------------------------------------------------------------------------
# small_class = sum(y==True)<len(y)//2
# train_idx = np.random.choice(num_samples, num_samples)
# test_idx = np.delete(np.arange(num_samples), train_idx)

# train_small_idx = train_idx[y[train_idx]==small_class]
# train_big_idx = np.random.choice(train_idx[y[train_idx]!=small_class], len(train_small_idx), replace=False)

# test_small_idx = test_idx[y[test_idx]==small_class]
# test_big_idx = np.random.choice(test_idx[y[test_idx]!=small_class], len(test_small_idx), replace=False)
#------------------------------------------------------------------------------------------------------------------------------------------
X_train, y_train = shuffle(np.concatenate([X[train_small_idx], X[train_big_idx]]), np.concatenate([y[train_small_idx], y[train_big_idx]]))
X_test, y_test = shuffle(np.concatenate([X[test_small_idx], X[test_big_idx]]), np.concatenate([y[test_small_idx], y[test_big_idx]]))
#------------------------------------------------------------------------------------------------------------------------------------------

print("\nTest Ratio:")
print(len(y_test), "/", len(y_test)+len(y_train), "=" , 100*len(y_test)/(len(y_train)+len(y_test)), "%\n")

print("Class Balance:")
print(sum(y_train)+sum(y_test), "/", len(y_train)+len(y_test), "=" , 100*(sum(y_train)+sum(y_test))/(len(y_train)+len(y_test)), "%\n")

print("Stratification:")
print(sum(y_train), "/", len(y_train), "=" , 100*sum(y_train)/len(y_train), "%")
print(sum(y_test), "/", len(y_test), "=" , 100*sum(y_test)/len(y_test), "%\n")

print("training...")

# clf = LogisticRegression(random_state=20, n_jobs=16).fit(X_train, y_train)
clf = RandomForestClassifier(max_depth=8, n_jobs=16, n_estimators=100, criterion="entropy").fit(X_train, y_train)
# clf = tree.DecisionTreeClassifier(max_depth=2, criterion="entropy").fit(X_train, y_train) 
# print(tree.export_graphviz(clf))

print("predicting...")

probs_ = clf.predict_proba(X_test)
probs = probs_[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, probs, pos_label=True)
plt.plot(fpr, tpr, label="ROC curve (area = %.2f)" % metrics.roc_auc_score(y_test, probs))
plt.plot([0, 1], [0, 1], linestyle="dashed", color='gray')
plt.title("Receiver Operating Characteristic")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
