#!/usr/bin/python3

import sys

from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import tree

import numpy as np
from matplotlib import pyplot as plt

def onehot(base):
    if base == "A":
        return (1,0,0,0)
    elif base == "C":
        return (0,1,0,0)
    elif base == "G":
        return (0,0,1,0)
    elif base == "T":
        return (0,0,0,1)
    else:
        print("ASDASDASDASD!!!!!!!!!!!!!!!!!", base)
        return (0,0,0,0)

def onehot_(enc):
    return np.array(["A","C","G","T"])[[bool(x) for x in enc]][0]

def retrieve_data(xpath, ypath, outpath):
    ### get X
    row_t = np.dtype([("readId", "u4"), ("col", "u4"), ('atts', '(17,)f4'), ('class', bool)], align=False)

    linecount = sum(1 for line in open(xpath, "r"))
    # linecount = 1000000
    samples = np.zeros(linecount, row_t)
    for i, line in enumerate(open(xpath, "r")):
        # if i==1000000:
        #     break
        if (i%1000000==0):
            print(i, "/", linecount)
        splt = line.split()
        samples[i]['readId'] = splt[0]
        samples[i]['col'] = splt[1]
        samples[i]['atts'] = onehot(splt[2]) + onehot(splt[3]) + tuple(splt[4:])

    print(samples[0:10])
    print(samples.shape)

    print("sorting...")
    samples.sort(axis=0, order='readId')
    print(samples.shape)
    print("done.")
    print(samples[0:10])

    ### get y
    with open(ypath, "r") as truthfile:
        filepos = 0
        for i, s in enumerate(samples):
            if (i%1000000==0):
                print(i, "/", linecount)
            if filepos != int(s['readId'])*4+2:
                while filepos<int(s['readId'])*4+1:
                    truthfile.readline()
                    filepos += 1
                trueseq = truthfile.readline()
                filepos += 1
            s['class'] = onehot_(s['atts'][:4])==trueseq[s['col']]

    print(samples[0:10])
    np.save(outpath, samples)


def train(train_data, test_data):
    #np.random.shuffle(train)

    X_train, y_train = train_data['atts'], train_data['class']
    X_test, y_test = test_data['atts'], test_data['class']

    print("\nTest Ratio:")
    print(y_test.shape[0], "/", y_test.shape[0]+y_train.shape[0], "=" , 100*y_test.shape[0]/(y_train.shape[0]+y_test.shape[0]), "%\n")

    print("Class Balance:")
    print(np.sum(y_train)+np.sum(y_test), "/", y_train.shape[0]+y_test.shape[0], "=" , 100*(np.sum(y_train)+np.sum(y_test))/(y_train.shape[0]+y_test.shape[0]), "%\n")

    print("Stratification:")
    print(np.sum(y_train), "/", y_train.shape[0], "=" , 100*np.sum(y_train)/y_train.shape[0], "%")
    print(np.sum(y_test), "/", y_test.shape[0], "=" , 100*np.sum(y_test)/y_test.shape[0], "%\n")

    print("training...")
    # clf = LogisticRegression(n_jobs=88).fit(X_train, y_train)

    clf = RandomForestClassifier(n_jobs=32).fit(X_train, y_train)
    # clf = tree.DecisionTreeClassifier(max_depth=3).fit(X_train, y_train) 
    # tree.export_graphviz(clf, out_file='tree.dot')

    print("predicting...")

    probs = clf.predict_proba(X_test)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probs[:,1], pos_label=True)
    plt.plot(fpr, tpr, label="0ROC curve (area = %.2f)" % metrics.roc_auc_score(y_test, probs[:,1]))

    plt.plot([0, 1], [0, 1], linestyle="dashed", color='gray')
    plt.title("Receiver Operating Characteristic")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


### main ###

if len(sys.argv)==5 and sys.argv[1] == "read":
    retrieve_data(*sys.argv[2:5])

if len(sys.argv)==4 and sys.argv[1] == "train":
    train_data = np.load(sys.argv[2])
    test_data = np.load(sys.argv[3])
    train(train_data, test_data)


### stuff ###

## bootstrapping class balance schemes
#------------------------------------------------------------------------------------------------------------------------------------------
# small_class = sum(y==True)<y.shape[0]//2
# small_idx, big_idx = np.arange(num_samples)[y==small_class], np.arange(num_samples)[y!=small_class]
# print(small_idx.shape[0], big_idx.shape[0])

# bootstrap = np.random.choice(small_idx.shape[0], small_idx.shape[0])
# train_small_idx = small_idx[bootstrap]
# test_small_idx = np.delete(small_idx, bootstrap)

# bootstrap = np.random.choice(big_idx.shape[0], small_idx.shape[0]) # we want equally sized class groups
# train_big_idx = big_idx[bootstrap]
# test_big_idx = np.random.choice(np.delete(big_idx, bootstrap), test_small_idx.shape[0], replace=False)
#------------------------------------------------------------------------------------------------------------------------------------------
# small_class = sum(y==True)<y.shape[0]//2
# train_idx = np.random.choice(num_samples, num_samples)
# test_idx = np.delete(np.arange(num_samples), train_idx)

# train_small_idx = train_idx[y[train_idx]==small_class]
# train_big_idx = np.random.choice(train_idx[y[train_idx]!=small_class], train_small_idx.shape[0], replace=False)

# test_small_idx = test_idx[y[test_idx]==small_class]
# test_big_idx = np.random.choice(test_idx[y[test_idx]!=small_class], test_small_idx.shape[0], replace=False)
#------------------------------------------------------------------------------------------------------------------------------------------
# X_train, y_train = shuffle(np.concatenate([X[train_small_idx], X[train_big_idx]]), np.concatenate([y[train_small_idx], y[train_big_idx]]))
# X_test, y_test = shuffle(np.concatenate([X[test_small_idx], X[test_big_idx]]), np.concatenate([y[test_small_idx], y[test_big_idx]]))
#------------------------------------------------------------------------------------------------------------------------------------------
