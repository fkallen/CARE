#!/usr/bin/env python3

import os
from typing import DefaultDict
from matplotlib import pyplot as plt
import pickle
import sys

extractors = ["s", "snw", "t", "tnw", "v2"]
totals = {}
def load():
    res = {}
    for ext_ in ["s", "snw", "t", "tnw", "v2"]:
        res[ext_] = {}
        res[ext_+"_md22"] = {}
        extdir = f"/home/jcascitt/care/{ext_}/out"
        for inpath in os.listdir(extdir):
            if not inpath.endswith("eval") or "melan" in inpath or "ele" in inpath:
                continue
            ext = ext_+"_md22" if "md22" in inpath else ext_
            sample = inpath.split("-")[0]
            for line in open(f"{extdir}/{inpath}"):
                if line.startswith("out/"):
                    at, ct = (int(x) for x in line.split('.')[-2].split('_')[-1].split('-')[-2:])
                    res[ext][(at,ct)] = [0,0]
                elif line.startswith("TP"):
                    tp = int(line.split()[-1])
                elif line.startswith("FP"):
                    fp = int(line.split()[-1])
                elif line.startswith("FN"):
                    fn = int(line.split()[-1])
                elif line.startswith("TN"):
                    tn = int(line.split()[-1])
                    res[ext][(at,ct)][0] += tp
                    res[ext][(at,ct)][1] += fp
                    total = tp+fp+tn+fn
                    # print(total)
                    if sample in totals:
                        if total != totals[sample]:
                            print("TOTALS DO NOT MATCH!!!", total, totals[sample], file=sys.stderr)
                    else:
                        totals[sample] = total
        print(ext_, len(res[ext_]))
        print(ext_+"_md22", len(res[ext_+"_md22"]))
    print(totals)
    pickle.dump(res, open("plot.p", "wb"))


def show():
    res = pickle.load(open("plot.p", "rb"))
    for ext in extractors+[x+"_md22" for x in extractors if x != "v2"]:
        # print(*zip(*res[ext].values()))
        # plt.plot(*zip(*res[ext].values()), marker='x', label=ext)
        plt.scatter(*zip(*res[ext].values()), marker='x', label=ext)
        # for atct, tpfp in res[ext].items():
        #     plt.annotate(atct, tpfp)
    plt.legend()
    plt.show()

if sys.argv[1] == "load":
    load()
elif sys.argv[1] == "show":
    show()

    
    