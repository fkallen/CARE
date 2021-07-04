#!/usr/bin/env python3
from mlcorrector import *
import pickle
from tqdm import tqdm

def main(prefixes, effiles, test_index):
    anchor_map = [{"X":prefix+"_anchor.samples", "y":effile, "np":prefix+"_anchor.npz"} for prefix, effile in zip(prefixes, effiles)]
    cands_map = [{"X":prefix+"_cands.samples", "y":effile, "np":prefix+"_cands.npz"} for prefix, effile in zip(prefixes, effiles)]

    NJOBS = 16

    process(anchor_map, RandomForestClassifier, {"n_jobs":NJOBS, "max_depth":2}, "_anchor.rf", test_index)
    process(cands_map, RandomForestClassifier, {"n_jobs":NJOBS, "max_depth":2}, "_cands.rf", test_index)
    
    # process(anchor_map, RandomForestClassifier, {"n_jobs":NJOBS}, "_anchor.rf", test_index)
    # process(cands_map, RandomForestClassifier, {"n_jobs":NJOBS}, "_cands.rf", test_index)
    
    # process(anchor_map, LogisticRegression, {"n_jobs":NJOBS}, "_cands.lr", test_index)
    # process(cands_map, LogisticRegression, {"n_jobs":NJOBS}, "_cands.lr", test_index)


