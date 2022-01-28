#!/usr/bin/python3

from random import Random
from click import pass_context
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.tree._tree import TREE_LEAF
import numpy as np
import struct
from itertools import accumulate
from tqdm import tqdm
import os
import pickle
import zipfile


class Dataset(object):

    @staticmethod
    def _check_desc(old, new):
        if old is not None and old != new:
            raise ValueError("Data descriptors do not match!")
        return new

    @staticmethod
    def _npz_headers(npz): # https://stackoverflow.com/a/43223420
        """Takes a path to an .npz file, which is a Zip archive of .npy files.
        Generates a sequence of (name, shape, np.dtype).
        """
        with zipfile.ZipFile(npz) as archive:
            for name in archive.namelist():
                if not name.endswith('.npy'):
                    continue

                npy = archive.open(name)
                version = np.lib.format.read_magic(npy)
                shape, fortran, dtype = np.lib.format._read_array_header(npy, version)
                yield name[:-4], shape, dtype

    @staticmethod
    def _npz_samples_metadata(path):
        for i in Dataset._npz_headers(path):
            if i[0] == 'samples':
                n, dtype = i[1][0], i[2]
                return n, dtype

    @staticmethod
    def _nucleo_id(enc):
        return "ACGT".index(enc)

    @staticmethod
    def _nucleotide(id):
        return "ACGT"[id]

    def __init__(self, paths):
        ### get X
        if "np" in paths[0] and os.path.isfile(paths[0]["np"]):
            desc = str(np.load(paths[0]["np"])["desc"])
            num_features = Dataset._npz_samples_metadata(paths[0]["np"])[1]['atts'].shape[0]
        else:
            with open(paths[0]["X"], "r", encoding="utf-8") as infile:
                desc = infile.readline()[:-1]
                num_features = len(infile.readline().split()) - 3
                if num_features != int(desc.split()[0]):
                    raise ValueError("Data descriptor does not fit data shape!")

        row_t = np.dtype([("fileId", "u1"), ("readId", "u4"), ("col", "i2"), ('atts', '('+str(num_features)+',)f4'), ('class', "u1")])

        linecounts = [Dataset._npz_samples_metadata(path["np"])[0] if "np" in path and os.path.isfile(path["np"]) else sum(1 for _ in open(path["X"], "r"))-1 for path in paths]
        tqdm.write(f"Initializing dataset:")
        tqdm.write(f"Files: {len(linecounts)}")
        tqdm.write(f"Lengths: {linecounts}")
        tqdm.write(f"Total: {sum(linecounts)}")
        tqdm.write(f"Features: {num_features}")
        tqdm.write(f"Descriptor: {desc if desc else '-'}\n")
        offsets = list(accumulate(linecounts, initial=0))
        samples = np.zeros(sum(linecounts), row_t)
        for file_id, path in tqdm(enumerate(paths), total=len(paths), miniters=1, desc="Loading features", colour="blue"):
            if "np" in path and os.path.isfile(path["np"]):
                tqdm.write(f"Loading {os.path.basename(path['np'])}")
                with np.load(path["np"]) as infile:
                    desc = Dataset._check_desc(desc, str(infile["desc"]))
                    if infile['samples'].dtype != row_t:
                        raise ValueError("Data dtype does not match!")
                    infile["samples"]["fileId"] = file_id
                    samples[offsets[file_id]:offsets[file_id+1]] = infile["samples"]
            else:
                with open(path["X"], "r", encoding="utf-8") as infile:
                    desc = Dataset._check_desc(desc, infile.readline()[:-1])
                    for i, line in tqdm(enumerate(infile), total=linecounts[file_id], desc=f"Parsing {os.path.basename(path['X'])}", leave=False, colour="black"):
                        s = samples[i+offsets[file_id]]
                        splt = line.split()
                        s['fileId'] = file_id
                        s['readId'] = splt[0]
                        s['col'] = splt[1]
                        s['atts'] = tuple(splt[3:])
                        s['class'] = Dataset._nucleo_id(splt[2])
                    tqdm.write(f"Parsed {os.path.basename(path['X'])}")

        tqdm.write("Sorting by coordinates...",end='')
        samples.sort(axis=0, order=['fileId', 'readId'])
        tqdm.write("done.")

        ### get y
        for file_id, path in tqdm(enumerate(paths), total=len(paths), miniters=1, desc="Loading classes", colour="blue"):
            if "np" in path and os.path.isfile(path["np"]):
                continue
            else:
                with open(path["y"], "r") as truthfile:
                    filepos = 0
                    for i in tqdm(range(linecounts[file_id]), total=linecounts[file_id], desc=f"Parsing {os.path.basename(path['y'])}", leave=False, colour="black"):
                        s = samples[i+offsets[file_id]]
                        if filepos != int(s['readId'])*4+2:
                            while filepos<int(s['readId'])*4+1:
                                truthfile.readline()
                                filepos += 1
                            trueseq = truthfile.readline()
                            filepos += 1
                        if s['col']>=0:
                            s['class'] = s['class']==Dataset._nucleo_id(trueseq[s['col']])
                        else:
                            s['class'] = s['class']==3-Dataset._nucleo_id(trueseq[s['col']-1]) # -1 because last character is newline
                tqdm.write(f"Parsed {os.path.basename(path['y'])}")

                if "np" in path:
                    np.savez_compressed(path["np"], desc=desc, samples=samples[offsets[file_id]:offsets[file_id]+linecounts[file_id]])
                    # os.remove(path["X"])

        self.paths = paths
        self.desc = str(desc)
        self.data = samples
        tqdm.write("Dataset initialized.\n")

class Classifier(object):

    def __init__(self, dataset):
        self.datapaths = dataset.paths
        self.desc = dataset.desc

    def extract(self, path):
        print(f"Serializing classifier into {path} ...")
        with open(path, "wb") as out_file:
            desc = self.desc.encode("utf-8")
            out_file.write(struct.pack("Q", len(desc)))
            out_file.write(desc)
            self._extract(out_file)
        print("Classifier serialized.\n")

    def verify(self, dataset):
        tqdm.write("Verifying classifier...")
        if dataset.desc != self.desc:
            raise ValueError('Train and test data descriptors do not match!')
        X_test, y_test = dataset.data['atts'], dataset.data['class']
        probs = self._clf.predict_proba(X_test)
        auroc = metrics.roc_auc_score(y_test, probs[:,1])
        avgps = metrics.average_precision_score(y_test, probs[:,1])
        tqdm.write(f"AUROC: {auroc}")
        tqdm.write(f"AVGPS: {avgps}")
        tqdm.write("\n")

class RandomForest(Classifier):

    def __init__(self, dataset, **clf_args):
        super().__init__(dataset)
        clf_args.setdefault("n_jobs", os.cpu_count())
        print("Training Random Forest...")
        self._clf = RandomForestClassifier(**clf_args).fit(dataset.data["atts"], dataset.data["class"])
        print("Random Forest trained.\n")

    @staticmethod
    def _extract_node(tree, i, out_file):
        if tree.children_left[i] == TREE_LEAF:
            out_file.write(struct.pack("f", (tree.value[i][0][1]/(tree.value[i][0][0]+tree.value[i][0][1]))))
        else:
            out_file.write(struct.pack("B", tree.feature[i]))
            out_file.write(struct.pack("d", tree.threshold[i]))
            
            lhs, rhs = tree.children_left[i], tree.children_right[i]
            
            flag = 0
            if tree.children_left[lhs] == TREE_LEAF:
                flag += 2
            if tree.children_left[rhs] == TREE_LEAF:
                flag += 1

            out_file.write(struct.pack("B", flag))

            RandomForest._extract_node(tree, lhs, out_file)
            RandomForest._extract_node(tree, rhs, out_file)

    def _extract(self, out_file):
        out_file.write(struct.pack("I", len(self._clf.estimators_)))
        for i, tree in tqdm(enumerate(self._clf.estimators_), total=len(self._clf.estimators_), miniters=1, desc="Extracting trees", colour="black"):
            out_file.write(struct.pack("I", tree.get_n_leaves()-1))
            self._extract_node(tree.tree_, 0, out_file)


class LogReg(Classifier):

    def __init__(self, data, **clf_args):
        raise NotImplementedError("LogReg classifier it not implemented yet.")

    def _extract(self, out_file):
        print(self._clf.coef_.shape[-1])
        out_file.write(struct.pack("I", self._clf.coef_.shape[-1]))
        for coef in self._clf.coef_[0]:
            print(coef)
            out_file.write(struct.pack("f", coef))
        print(self._clf.intercept_[0])
        out_file.write(struct.pack("f", self._clf.intercept_[0]))

def train(dataset, clf_t_str="RF", **clf_args):
    _clfs = {"RF":RandomForest}
    if clf_t_str not in _clfs:
        raise ValueError(f"Unknown CARE classifier type: {clf_t_str}")
    else:
        return _clfs[clf_t_str](dataset, **clf_args)

def _process(clf_t_str, clf_args, train_paths, test_paths, clf_path):
    trainset = Dataset(train_paths)
    testset = Dataset(test_paths)
    clf = train(trainset, clf_t_str, **clf_args)
    clf.verify(trainset)
    clf.verify(testset)
    pickle.dump(clf, open(clf_path+".p", "wb"))
    clf.extract(clf_path)
    tqdm.write("\n\n")

