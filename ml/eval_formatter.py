#!/usr/bin/env python3

from collections import defaultdict
import sys

data = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: defaultdict(lambda: ""))))
for f in sys.argv[1:]:
    for line in open(f, "r"):
        if line.endswith(".fq\n"):
            anchor, cands = (int(x) for x in line[:-4].split('-')[-2:])
        if line.startswith("TP"):
            data[f][anchor][cands]["TP"] = int(line.split()[1])
        if line.startswith("FP"):
            data[f][anchor][cands]["FP"] = int(line.split()[1])

for anchor in range(0,101,5):
    for file in sys.argv[1:]:
        print("\t".join(str(x) for x in (data[file][anchor][cands]["TP"] for cands in range(0,101,5))))
        print("\t".join(str(x) for x in (data[file][anchor][cands]["FP"] for cands in range(0,101,5))))
            