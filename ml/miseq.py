#!/usr/bin/env python3
from mlcorrector import *
import pickle
from tqdm import tqdm

def main(prefixes, effiles):
    anchor_map = [{"X":prefix+"_anchor.samples", "y":effile, "np":prefix+"_anchor.npz", "prefix":prefix} for prefix, effile in zip(prefixes, effiles)]
    cands_map = [{"X":prefix+"_cands.samples", "y":effile, "np":prefix+"_cands.npz", "prefix":prefix} for prefix, effile in zip(prefixes, effiles)]

    NJOBS = 11

    for i in range(len(prefixes)):
        process(anchor_map, RandomForestClassifier, {"n_jobs":NJOBS, "verbose":42}, "_anchor", i)
        process(cands_map, RandomForestClassifier, {"n_jobs":NJOBS, "verbose":42}, "_cands", i)
    
