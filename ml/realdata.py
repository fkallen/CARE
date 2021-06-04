#!/usr/bin/env python3
from mlcorrector import *
import pickle
from tqdm import tqdm

def main(prefixes, effiles):
    data_map = [{"X":prefix+"_anchor.samples", "y":effile, "np":prefix+"_anchor.npy"} for prefix, effile in zip(prefixes, effiles)]
    job(data_map, None, "_anchor")
    
    data_map = [{"X":prefix+"_cands.samples", "y":effile, "np":prefix+"_cands.npy"} for prefix, effile in zip(prefixes, effiles)]
    job(data_map, None, "_cands")

def job(data_map, max_depth=None, suffix=""):
    tqdm.write("### data sets: "+str(len(data_map))+"\n")
    train_data = read_data(data_map)
    X_train, y_train = train_data['atts'], train_data['class']
    tqdm.write("### training classifiers:")
    clf = RandomForestClassifier(n_jobs=88, max_depth=max_depth).fit(X_train, y_train)
    extract_forest(clf, "all6"+suffix+".rf")

