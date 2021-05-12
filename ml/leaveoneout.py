#!/usr/bin/env python3
from mlcorrector import *
import pickle
from tqdm import tqdm

def main(prefixes, effiles):
    data_map = [{"X":prefix+"_anchor.samples", "y":effile, "np":prefix+"_anchor.npy"} for prefix, effile in zip(prefixes, effiles)]
    leave_one_out(data_map, None, "_anchor")
    
    data_map = [{"X":prefix+"_cands.samples", "y":effile, "np":prefix+"_cands.npy"} for prefix, effile in zip(prefixes, effiles)]
    leave_one_out(data_map, None, "_cands")

def leave_one_out(data_map, max_depth=22, suffix=""):
    tqdm.write("### data sets: "+str(len(data_map))+"\n")
    for i in tqdm(range(len(data_map)), colour="green", leave=False):
        tqdm.write("### leave index "+str(i+1)+" out:\n")
        train_map = list(data_map)
        test_map = [train_map.pop(i)]
        train_data = read_data(train_map)
        test_data = read_data(test_map)

        X_train, y_train = train_data['atts'], train_data['class']
        X_test, y_test = test_data['atts'], test_data['class']

        tqdm.write("### training classifiers:")
        clf = RandomForestClassifier(n_jobs=88, max_depth=max_depth).fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        auroc = metrics.roc_auc_score(y_test, probs[:,1])
        tqdm.write("AUROC (test)  : "+str(auroc))
        pickle.dump(probs, open("probs_"+str(i+1)+suffix+".p", "wb"))
        pickle.dump(auroc, open("aurocs_"+str(i+1)+suffix+".p", "wb"))
        extract_forest(clf, str(i+1)+suffix+".rf")

