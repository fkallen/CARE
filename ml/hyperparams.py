#!/usr/bin/env python3
from mlcorrector import *
import pickle
from tqdm import tqdm

def main(data_map):
    tqdm.write("### data sets: "+str(len(data_map))+"\n")
    for i in tqdm(range(len(data_map)), colour="green"):
        tqdm.write("### leave index "+str(i)+" out:\n")
        train_map = list(data_map)
        test_map = [train_map.pop(i)]
        train_data = read_data(37, train_map, hide_pbar=True)
        test_data = read_data(37, test_map, hide_pbar=True)

        X_train, y_train = train_data['atts'], train_data['class']
        X_test, y_test = test_data['atts'], test_data['class']

        probs = {}
        aurocs = {}
        probs_train = {}
        aurocs_train = {}
        tqdm.write("### training classifiers:")
        for k in tqdm(range(50,0,-1), colour="yellow", leave=False):
            tqdm.write("n: "+str(k))
            clf = RandomForestClassifier(n_jobs=44, n_estimators=44, max_depth=k).fit(X_train, y_train)
            probs[k]=clf.predict_proba(X_test)
            probs_train[k]=clf.predict_proba(X_train)
            auroc = metrics.roc_auc_score(y_test, probs[k][:,1])
            auroc_train = metrics.roc_auc_score(y_train, probs_train[k][:,1])
            tqdm.write("AUROC (train) : "+str(auroc_train))
            tqdm.write("AUROC (test)  : "+str(auroc))
            aurocs[k]=auroc
            aurocs_train[k]=auroc_train
            
        pickle.dump(probs, open("probs_depth_"+str(i)+".p", "wb"))
        pickle.dump(aurocs, open("aurocs_depth_"+str(i)+".p", "wb"))
        pickle.dump(probs_train, open("probs_depth_"+str(i)+"_train.p", "wb"))
        pickle.dump(aurocs_train, open("aurocs_depth_"+str(i)+"_train.p", "wb"))
