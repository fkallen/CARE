#!/usr/bin/env python3.jc

import os
from typing import DefaultDict
from matplotlib import pyplot as plt
import pickle
import sys

def dd():
    return [0,0]
extractors = ["v1", "v1nw", "t", "tnw", "v2"]
totals = {}
def load():
    res = {}
    for ext in extractors:
        res[ext] = DefaultDict(dd)
        extdir = f"/home/jcascitt/care/{ext}/out"
        for inpath in os.listdir(extdir):
            if not (inpath.endswith("eval") and inpath.startswith("human")):
                continue
            print(inpath)
            sample = inpath.split("-")[0]
            for line in open(f"{extdir}/{inpath}"):
                if line.startswith("out/"):
                    at, ct = (int(x) for x in line.split('.')[-2].split('_')[-1].split('-')[-2:])
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
                            print("TOTALS DO NOT MATCH!!!", total, totals[sample], totals, sample, inpath, at, ct, file=sys.stderr)
                            break
                    else:
                        totals[sample] = total
        print(ext, len(res[ext]))
    print(totals)
    pickle.dump(res, open("plot.p", "wb"))


def show():
    res = pickle.load(open("plot.p", "rb"))
    for ext in extractors:
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

    
    