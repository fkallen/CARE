#!/usr/bin/env python3
from mlcorrector import *
import pickle
from tqdm import tqdm

def main(prefixes, effiles, test_index):
    anchor_map = [{"X":prefix+"_anchor.samples", "y":effile, "np":prefix+"_anchor.npy"} for prefix, effile in zip(prefixes, effiles)]
    cands_map = [{"X":prefix+"_cands.samples", "y":effile, "np":prefix+"_cands.npy"} for prefix, effile in zip(prefixes, effiles)]

    NJOBS = 16
    
    process(anchor_map, RandomForestClassifier, {"n_jobs":NJOBS}, "_anchor.rf", test_index)
    process(cands_map, RandomForestClassifier, {"n_jobs":NJOBS}, "_cands.rf", test_index)

def process(data_map, clf_t, clf_args, suffix, test_index):
    tqdm.write("### data sets: "+str(len(data_map))+"\n")
    tqdm.write("### leave index "+str(test_index)+" out:\n")
    train_map = list(data_map)
    test_map = [train_map.pop(test_index)]
    train_data = read_data(train_map)
    test_data = read_data(test_map)

    X_train, y_train = train_data['atts'], train_data['class']
    X_test, y_test = test_data['atts'], test_data['class']

    tqdm.write("### training classifier(s):")
    clf = clf_t(**clf_args).fit(X_train, y_train)
    extract_clf(clf, str(test_index)+suffix)
    # probs = clf.predict_proba(X_test)
    # auroc = metrics.roc_auc_score(y_test, probs[:,1])
    # probs_train = clf.predict_proba(X_train)
    # auroc_train = metrics.roc_auc_score(y_train, probs_train[:,1])
    # tqdm.write("AUROC (test)  : "+str(auroc))
    # pickle.dump(probs, open("probs_"+str(test_index)+suffix+".p", "wb"))
    # pickle.dump(auroc, open("aurocs_"+str(test_index)+suffix+".p", "wb"))
    # tqdm.write("AUROC (train)  : "+str(auroc_train))
    # pickle.dump(probs, open("probs_train_"+str(test_index)+suffix+".p", "wb"))
    # pickle.dump(auroc, open("aurocs_train_"+str(test_index)+suffix+".p", "wb"))

    for i in range(len(data_map)):
        test_map = [data_map[i]]
        test_data = read_data(test_map)
        X_test, y_test = test_data['atts'], test_data['class']
        tqdm.write("AUROC ("+str(i+1)+")  : "+str(metrics.roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])))

