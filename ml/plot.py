#!/usr/bin/env python3.jc

import os
from typing import DefaultDict
from matplotlib import pyplot as plt
import pickle
import sys
import numpy as np

res = {}
def add_eval(ext, atct, sample, tp, fp, fn, tn):
    cand = {'tp':tp, 'fp':fp, 'fn':fn, 'tn':tn}
    if ext in res:
        if atct in res[ext]:
            if sample in res[ext][atct]:
                if res[ext][atct][sample] != cand:
                    print(f"{ext}, {atct}, {sample} data point already seen! Old: {res[ext][atct][sample]} New: {cand}", file=sys.stderr)
                    res[ext][atct][sample] = cand

            else:
                res[ext][atct][sample] = cand
        else:
            res[ext][atct] = {sample:cand}
    else:
        res[ext] = {atct:{sample:cand}}

def reduce():
    shared_samples = set.intersection(*(set(res[e][a].keys()) for e in res for a in res[e]))
    for ext in res:
        for atct in res[ext]:
            for sample in set(res[ext][atct]):
                if sample not in shared_samples:
                    res[ext][atct].pop(sample, None)
    print(shared_samples)


def make_frontier(elements):
    keep = []
    frontx, fronty = [], []
    for key, (x, y) in elements:
        pos = np.searchsorted(frontx, x)
        if pos < len(keep):
            if (frontx[pos], fronty[pos]) == (x,y):
                keep.insert(pos, key)
                frontx.insert(pos, x)
                fronty.insert(pos, y)
                continue
            elif fronty[pos] <= y:
                continue
        b = np.searchsorted(fronty, y)
        e = np.searchsorted(frontx, x, side="right")
        keep[b:e] = [key]
        frontx[b:e] = [x]
        fronty[b:e] = [y]
    return keep

def frontierize():
    for ext in res:
        frontier = make_frontier((atct,np.sum([(res[ext][atct][s]['tp'],res[ext][atct][s]['fp']) for s in res[ext][atct]], axis=0)) for atct in res[ext])
        for atct in list(res[ext].keys()):
            if atct not in frontier:
                if ext == "cc" or ext == "t" and atct == (90,30):
                    continue
                res[ext].pop(atct)

def frontierize_global():
    frontier = make_frontier(((ext,atct),np.sum([(res[ext][atct][s]['tp'],res[ext][atct][s]['fp']) for s in res[ext][atct]], axis=0)) for ext in res for atct in res[ext])
    for ext in res:
        for atct in list(res[ext].keys()):
            if ext == "cc" or ext == "t" and atct == (90,30):
                continue
            if (ext, atct) not in frontier:
                res[ext].pop(atct)

extdirs = ["paired/t", "paired/v2", "paired/v3", "paired/cc"]
def load():
    for extdir in extdirs:
        extdir = f"/home/jcascitt/care/{extdir}/out"
        if not os.path.exists(extdir):
            continue
        for inpath in os.listdir(extdir):
            if not inpath.endswith("eval"):
                continue
            ext = inpath.split('_')[-2]
            sample = inpath.split('_')[0]
            print(inpath, ext, sample)
            for line in open(f"{extdir}/{inpath}"):
                if line.startswith("out/"):
                    at, ct = (int(x) for x in line.split('.')[-2].split('_')[-1].split('-')[-2:]) if ext!='cc' else (0,0)
                elif line.startswith("TP"):
                    tp = int(line.split()[-1])
                elif line.startswith("FP"):
                    fp = int(line.split()[-1])
                elif line.startswith("FN"):
                    fn = int(line.split()[-1])
                elif line.startswith("TN"):
                    tn = int(line.split()[-1])
                    add_eval(ext, (at, ct), sample, tp, fp, fn, tn)    

    reduce()
    frontierize()
    print(res)
    pickle.dump(res, open("plot.p", "wb"))

def show():
    res = pickle.load(open("plot.p", "rb"))
    for ext in res:
        try:
            plt.scatter(*zip(*sorted((np.sum([(res[ext][atct][s]['tp'],res[ext][atct][s]['fp']) for s in res[ext][atct]], axis=0) for atct in res[ext]), key=lambda x:x[0])), marker='x' if ext=="cc" else 'o', label=ext)
        except TypeError:
            pass
        for atct in res[ext]:
            plt.annotate(str(atct), np.sum([(res[ext][atct][s]['tp'],res[ext][atct][s]['fp']) for s in res[ext][atct]], axis=0))
        # plt.annotate("(90, 30)", np.sum([(res["t"][(90,30)][s]['tp'],res["t"][(90,30)][s]['fp']) for s in res["t"][(90,30)]], axis=0))
    plt.legend()
    plt.show()

if sys.argv[1] == "load":
    load()
elif sys.argv[1] == "show":
    show()

    
    