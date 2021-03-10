#!/usr/bin/env python3

from collections import defaultdict
import sys

data = {}
for f in sys.argv[1:]:
    for line in open(f, "r"):
        if line.endswith(".fq\n"):
            anchor, cands = (int(x) for x in line[:-4].split('-')[-2:])
        elif line.startswith("TP"):
            tp = int(line.split()[1])
        elif line.startswith("FP"):
            fp = int(line.split()[1])
            
            to_delete = []
            add = True
            for setting in data:
                if data[setting]["TP"]>=tp and data[setting]["FP"]<=fp:
                    add = False
                    break
                elif data[setting]["TP"]<=tp and data[setting]["FP"]>=fp:
                    to_delete.append(setting)
            if add:
                data[(f, anchor, cands)]={"TP":tp, "FP":fp}
                for setting in to_delete:
                    del data[setting]

min_thresh=60
max_thresh=91
step_thresh=5

print("\t\t"+"\t".join(str(cands) for cands in range(min_thresh,max_thresh,step_thresh)))
for anchor in range(min_thresh,max_thresh,step_thresh):
    print(str(anchor), end="")
    for file in sys.argv[1:]:
        print("\t\t"+"\t".join(str((data[(file,anchor,cands)]["TP"]) if (file,anchor,cands) in data else "") for cands in range(min_thresh,max_thresh,step_thresh)))
        print("\t\t"+"\t".join(str((data[(file,anchor,cands)]["FP"]) if (file,anchor,cands) in data else "") for cands in range(min_thresh,max_thresh,step_thresh)))
            