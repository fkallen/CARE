#!/usr/bin/env python3
from mlcorrector import *
import pickle
from tqdm import tqdm

def main(feature_str, prefixes, effiles):
    anchor_map = [{"X":prefix+"_"+feature_str+"_anchor.samples", "y":effile, "np":prefix+"_"+feature_str+"_anchor.npz", "prefix":prefix} for prefix, effile in zip(prefixes, effiles)]
    cands_map = [{"X":prefix+"_"+feature_str+"_cands.samples", "y":effile, "np":prefix+"_"+feature_str+"_cands.npz", "prefix":prefix} for prefix, effile in zip(prefixes, effiles)]

    NJOBS = 8

    for i in range(len(prefixes)):
        
        process(anchor_map, RandomForestClassifier, {"n_jobs":NJOBS, "max_depth":22}, feature_str+"_md22_anchor", i)
        process(cands_map, RandomForestClassifier, {"n_jobs":NJOBS, "max_depth":22}, feature_str+"_md22_cands", i)

        process(anchor_map, RandomForestClassifier, {"n_jobs":NJOBS}, feature_str+"_anchor", i)
        process(cands_map, RandomForestClassifier, {"n_jobs":NJOBS}, feature_str+"_cands", i)
    
        # process(anchor_map, LogisticRegression, {"n_jobs":NJOBS}, "_cands.lr", test_index)
        # process(cands_map, LogisticRegression, {"n_jobs":NJOBS}, "_cands.lr", test_index)