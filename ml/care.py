#!/usr/bin/python3

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import tree
import numpy as np
import struct
from itertools import accumulate
from tqdm import tqdm
import os
import pickle
import zipfile

def npz_headers(npz): # https://stackoverflow.com/a/43223420
    """Takes a path to an .npz file, which is a Zip archive of .npy files.
    Generates a sequence of (name, shape, np.dtype).
    """
    with zipfile.ZipFile(npz) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):
                continue

            npy = archive.open(name)
            version = np.lib.format.read_magic(npy)
            shape, fortran, dtype = np.lib.format._read_array_header(npy, version)
            yield name[:-4], shape, dtype

def npz_samples_metadata(path):
    for i in npz_headers(path):
        if i[0] == 'samples':
            n, dtype = i[1][0], i[2]
            return n, dtype

def check_descr(old, new):
    if old is not None and old != new:
        raise ValueError("Data descrs do not match!")
    return new


# def onehot(base):
#     if base == "A":
#         return (1,0,0,0)
#     elif base == "C":
#         return (0,1,0,0)
#     elif base == "G":
#         return (0,0,1,0)
#     elif base == "T":
#         return (0,0,0,1)
#     else:
#         print("ASDASDASDASD!!!!!!!!!!!!!!!!!", base)
#         return (0,0,0,0)

# def onehot_(enc):
#     return np.array(["A","C","G","T"])[[bool(x) for x in enc]][0]

def nucleo_id(enc):
    return "ACGT".index(enc)

def nucleotide(id):
    return "ACGT"[id]

def read_data(paths):    
    ### get X
    if "np" in paths[0] and os.path.isfile(paths[0]["np"]):
        descr = None
        num_features = npz_samples_metadata(paths[0]["np"])[1]['atts'].shape[0]
    else:
        with open(paths[0]["X"], "r", encoding="utf-8") as infile:
            descr = infile.readline()[:-1]
            num_features = len(infile.readline().split()) - 3
            if num_features != int(descr.split()[0]):
                raise ValueError("Data descriptor does not fit data shape!")

    row_t = np.dtype([("fileId", "u1"), ("readId", "u4"), ("col", "i2"), ('atts', '('+str(num_features)+',)f4'), ('class', "u1")])

    linecounts = [npz_samples_metadata(path["np"])[0] if "np" in path and os.path.isfile(path["np"]) else sum(1 for _ in open(path["X"], "r"))-1 for path in tqdm(paths, total=len(paths), colour="blue", miniters=1, mininterval=0, leave=False)]
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
            with np.load(path["np"]) as infile:
                descr = check_descr(descr, infile["desc"])
                if infile['samples'].dtype != row_t:
                    raise ValueError("Data dtype does not match!")
                samples[offsets[file_id]:offsets[file_id+1]] = infile["samples"]
        else:
            tqdm.write("parse: "+path["X"])
            with open(path["X"], "r", encoding="utf-8") as infile:
                descr = check_descr(descr, infile.readline()[:-1])
                for i, line in tqdm(enumerate(infile), total=linecounts[file_id], colour="cyan", leave=False):
                    s = samples[i+offsets[file_id]]
                    splt = line.split()
                    s['fileId'] = file_id
                    s['readId'] = splt[0]
                    s['col'] = splt[1]
                    s['atts'] = tuple(splt[3:])
                    s['class'] = nucleo_id(splt[2])

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
                    if s['col']>=0:
                        s['class'] = s['class']==nucleo_id(trueseq[s['col']])
                    else:
                        s['class'] = s['class']==3-nucleo_id(trueseq[s['col']-1]) # -1 because last character is newline

            if "np" in path:
                tqdm.write("save: "+path["np"])
                np.savez_compressed(path["np"], desc=descr, samples=samples[offsets[file_id]:offsets[file_id]+linecounts[file_id]])
                os.remove(path["X"])

    return descr, samples

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
    out_file.write(struct.pack("I", len(clf.estimators_)))
    for i, tree in enumerate(clf.estimators_):
        out_file.write(struct.pack("I", tree.get_n_leaves()-1))
        print("Tree", i, "Nodes:", tree.get_n_leaves()-1)
        extract_node(tree.tree_, 0, out_file)

def extract_lr(clf: LogisticRegression, out_file):
    print(clf.coef_.shape[-1])
    out_file.write(struct.pack("I", clf.coef_.shape[-1]))
    for coef in clf.coef_[0]:
        print(coef)
        out_file.write(struct.pack("f", coef))
    print(clf.intercept_[0])
    out_file.write(struct.pack("f", clf.intercept_[0]))

def extract_clf(clf, out_file_path):
    with open(out_file_path, "wb") as out_file:
        desc = clf.CARE_desc.encode("utf-8")
        out_file.write(struct.pack("Q", len(desc)))
        out_file.write(desc)
        if isinstance(clf, RandomForestClassifier):
            extract_forest(clf, out_file)
        elif isinstance(clf, LogisticRegression):
            extract_lr(clf, out_file)

def process(clf_t, clf_args, train_map, test_map, name):
    train_desc, train_data = read_data(train_map)
    test_desc, test_data = read_data(test_map)

    if test_desc != train_desc:
        raise ValueError('Train and test data descriptors do not match!')

    X_train, y_train = train_data['atts'], train_data['class']
    X_test, y_test = test_data['atts'], test_data['class']

    tqdm.write("### training classifier(s):")
    clf = clf_t(**clf_args).fit(X_train, y_train)
    clf.CARE_desc = str(train_desc)

    probs = clf.predict_proba(X_test)
    auroc = metrics.roc_auc_score(y_test, probs[:,1])
    avgps = metrics.average_precision_score(y_test, probs[:,1])
    tqdm.write("AUROC (test)  : "+str(auroc))
    tqdm.write("AVGPS (test)  : "+str(avgps))
    
    probs_train = clf.predict_proba(X_train)
    auroc_train = metrics.roc_auc_score(y_train, probs_train[:,1])
    avgps_train = metrics.average_precision_score(y_train, probs_train[:,1])
    tqdm.write("AUROC (train)  : "+str(auroc_train))
    tqdm.write("AVGPS (train)  : "+str(avgps_train))
    
    pickle.dump(clf, open(name+".rf.p", "wb"))

    # extract_clf(clf, name+".rf")

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
