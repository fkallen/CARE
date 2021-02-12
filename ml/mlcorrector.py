#!/usr/bin/python3

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import tree
import numpy as np
import struct
from itertools import accumulate
from tqdm import tqdm
import os
from time import sleep


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

def read_data(paths):    
    ### get X
    if "np" in paths[0] and os.path.isfile(paths[0]["np"]):
        num_features = np.load(paths[0]["np"]).dtype['atts'].shape[0]
    else:
        num_features = len(open(paths[0]["X"], "r").readline().split()) - 2
    
    row_t = np.dtype([("fileId", "u1"), ("readId", "u4"), ("col", "i2"), ('atts', '('+str(int(num_features))+',)f4'), ('class', bool)])

    linecounts = [np.load(path["np"]).shape[0] if "np" in path and os.path.isfile(path["np"]) else sum(1 for line in open(path["X"], "r")) for path in paths]
    tqdm.write("# files: "+str(len(linecounts)))
    tqdm.write("lengths: "+str(linecounts))
    tqdm.write("total: "+str(sum(linecounts)))
    tqdm.write("# features: "+str(num_features))
    offsets = list(accumulate(linecounts, initial=0))
    samples = np.zeros(sum(linecounts), row_t)
    tqdm.write("reading files:")
    for file_id, path in tqdm(enumerate(paths), total=len(paths), colour="blue", miniters=1, mininterval=0, leave=False):
        if "np" in path and os.path.isfile(path["np"]):
            tqdm.write("load: "+path["np"])
            samples[offsets[file_id]:offsets[file_id+1]] = np.load(path["np"])
        else:
            tqdm.write("parse: "+path["X"])
            for i, line in tqdm(enumerate(open(path["X"], "r")), total=linecounts[file_id], colour="cyan", leave=False):
                s = samples[i+offsets[file_id]]
                splt = line.split()
                s['fileId'] = file_id
                s['readId'] = splt[0]
                s['col'] = splt[1]
                s['atts'] = tuple(splt[2:])

    # print(samples[0:10])
    # print(samples.shape)

    tqdm.write("sorting...")
    samples.sort(axis=0, order=['fileId', 'readId'])
    # print(samples.shape)
    # print(samples[0:10])

    ### get y
    tqdm.write("reading classes...")
    for file_id, path in tqdm(enumerate(paths), total=len(paths), colour="red", miniters=1, mininterval=0, leave=False):
        if "np" in path and os.path.isfile(path["np"]):
            tqdm.write("skip: "+path["np"])
        else:
            with open(path["y"], "r") as truthfile:
                tqdm.write("parse: "+path["y"])
                filepos = 0
                for i in tqdm(range(linecounts[file_id]), total=linecounts[file_id], colour="magenta", leave=False):
                    s = samples[i+offsets[file_id]]
                    if filepos != int(s['readId'])*4+2:
                        while filepos<int(s['readId'])*4+1:
                            truthfile.readline()
                            filepos += 1
                        trueseq = truthfile.readline()
                        filepos += 1
                    try:
                        if s['col']>=0:
                            s['class'] = onehot_(s['atts'][4:8])==trueseq[s['col']]
                        else:
                            s['class'] = onehot_(s['atts'][7:3:-1])==trueseq[s['col']-1] # -1 because last character is newline
                    except:
                        print("!!!!##!#!#!#!#!#!#!")
                        print(s, s.dtype)
                        print(len(trueseq), trueseq)
                        print()
                        return

            if "np" in path:
                tqdm.write("save: "+path["np"])
                np.save(path["np"], samples[offsets[file_id]:offsets[file_id]+linecounts[file_id]])

    # print(samples[0:10])
    return samples

def train(train_data, clf="rf"):
    X_train, y_train = train_data['atts'], train_data['class']

    # print("\nTest Ratio:")
    # print(y_test.shape[0], "/", y_test.shape[0]+y_train.shape[0], "=" , 100*y_test.shape[0]/(y_train.shape[0]+y_test.shape[0]), "%\n")

    # print("Class Balance:")
    # print(np.sum(y_train)+np.sum(y_test), "/", y_train.shape[0]+y_test.shape[0], "=" , 100*(np.sum(y_train)+np.sum(y_test))/(y_train.shape[0]+y_test.shape[0]), "%\n")

    # print("Stratification:")
    # print(np.sum(y_train), "/", y_train.shape[0], "=" , 100*np.sum(y_train)/y_train.shape[0], "%")
    # print(np.sum(y_test), "/", y_test.shape[0], "=" , 100*np.sum(y_test)/y_test.shape[0], "%\n")

    print("training...")

    if clf == "rf":
        return RandomForestClassifier(n_jobs=44).fit(X_train, y_train)
    elif clf == "lr":
        return LogisticRegression(n_jobs=44).fit(X_train, y_train)
        # return LogisticRegression(solver='saga', penalty='l1').fit(X_train, y_train)


    # clf = tree.DecisionTreeClassifier(max_depth=3).fit(X_train, y_train) 
    # extract_forest(clf, out_file='ml/forest.bin')
    # tree.export_graphviz(clf.estimators_[0], out_file='ml/tree.dot')

def test(test_data, clf, roc_file):
    print("predicting...")

    X_test, y_test = test_data['atts'], test_data['class']
    probs = clf.predict_proba(X_test)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probs[:,1], pos_label=True)
    plt.plot(fpr, tpr, label="ROC curve (area = %.2f)" % metrics.roc_auc_score(y_test, probs[:,1]))
    # print(thresholds)
    # for i, txt in enumerate(thresholds):
    #     plt.annotate("{:4.3f}".format(txt), (fpr[i], tpr[i]))

    plt.plot([0, 1], [0, 1], linestyle="dashed", color='gray')
    plt.title("Receiver Operating Characteristic")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(roc_file)

def extract_node(tree_, i, out_file):
    if tree_.children_left[i] == tree._tree.TREE_LEAF:
        out_file.write(struct.pack("f", (tree_.value[i][0][1]/(tree_.value[i][0][0]+tree_.value[i][0][1]))))
    else:
        out_file.write(struct.pack("B", tree_.feature[i]))
        out_file.write(struct.pack("f", tree_.threshold[i]))
        
        lhs, rhs = tree_.children_left[i], tree_.children_right[i]
        
        flag = 0
        if tree_.children_left[lhs] == tree._tree.TREE_LEAF:
            flag += 2
        if tree_.children_left[rhs] == tree._tree.TREE_LEAF:
            flag += 1

        out_file.write(struct.pack("B", flag))

        extract_node(tree_, lhs, out_file)
        extract_node(tree_, rhs, out_file)

def extract_forest(clf, out_file):
    with open(out_file, "wb") as out_file:
        out_file.write(struct.pack("I", len(clf.estimators_)))
        for i, tree in enumerate(clf.estimators_):
            out_file.write(struct.pack("I", tree.get_n_leaves()-1))
            print("Tree", i, "Nodes:", tree.get_n_leaves()-1)
            extract_node(tree.tree_, 0, out_file)

def extract_lr(clf, out_file):
    with open(out_file, "wb") as out_file:
        out_file.write(struct.pack("I", clf.coef_.shape[-1]))
        out_file.write(struct.pack("I", clf.intercept_[0]))
        for coef in clf.coef_[0]:
            out_file.write(struct.pack("I", coef))

### main ###
# def main():
#     plans = json.load(open(sys.argv[1]))
#     for plan in plans:
#         data = {}
#         disp = {
#             "load": read_data,
#             "save": lambda d, f: np.save(f,data[d]),
#             "extract": lambda clf, f: extract_forest(data[clf], f),
#             "train": lambda trn, *args: train(data[trn], *args),
#             "test": lambda tst, clf: test(data[tst], clf)
#         }
#         for ins in plan:
#             ret, cmd, args = ins[0], ins[1], ins[2:]
#             data[ret] = disp[cmd](*args)

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
