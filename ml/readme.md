# Training Classifiers for CARE Error Correction



## Requirements:


- Python 3.8+ (lower version might work)
- CARE Python Module
  - scikit-learn
  - numpy
  - tqdm
- Training Data
  - Containing Errors + Error-Free copy


## Step 1: Sampling Data

CARE provides a "Print" mode for the extraction of features for training CARE-compatible classifiers.

Run CARE with the appropriate "Print" mode paramters:

```bash
# anchor correction
care-cpu [...] \ # Print mode only available in CPU version.
--correctionType 2 \ # Print mode
--ml-print-forestfile anchor.samples \ # "Output Path"
--samplingRateAnchor 0.25 \ # Optional. Subsampling to control file size.

# candidate correction
care-cpu [...] \
--candidateCorrection --correctionTypeCands 2 \ # Candidate correction and print mode
--samplingRateCands 0.01 \ # Optional
--ml-cands-print-forestfile cands.samples 
```

## Step 2: The CARE Python Module

The care python module provides a simple interface for training CARE-compatible classifiers.

### Example Usage

```python
import sys
sys.path.append(".../care.py") # configure appropriately
import care

train_paths = [
    {"X" : "anchor_1.samples", # samples from print mode
     "y" : "error_free_reads_1.fq", # corresponds to print-mode input
     "np": "anchor_1.npz"} # optional: samples compression
    ,
    {"X" : "anchor_2.samples",
     "y" : "error_free_reads_2.fq",
    }
    ,
    {"np": "anchor_3.npz"} # must exist
]

trainset = care.Dataset(train_paths) # Trains clf on all 3 of datasets
                                     # stores samples in "np" paths if provided
                                     # reads from "np" paths if file exists

clf = care.Classifier(trainset, n_jobs=128) # number of threads
clf.extract("anchor.rf") # serializes classifier in CARE-compatible binary format (see below)
```
See advanced section for more details.

## Step 3: Running classifier-based correction
```bash
care-gpu -i error_reads.fq -c 30 -o corrected_reads.fq [...] \
--correctionType 1 --ml-forestfile anchor.rf \
--candidateCorrection --correctionTypeCands 1 --ml-cands-forestfile cands.rf \

# optional: # confidence thresholds for correction
--thresholdAnchor 0.90 --thresholdCands 0.15 
```

## Advanced Usage

### Verifying Classifiers

The CARE python module provides an easy interface for testing classifiers on datasets, calculating the AUROC (area under ROC curve) and
average precision (area under precision-recall curve)
statistics.

```python
test_paths = [
    {"X" : "anchor_test_1.samples", 
     "y" : "error_free_reads_test_1.fq"}
]
testset = care.Dataset(test_paths)
auroc, avgps = my_clf.verify(testset)

# Example Output:
# AUROC: 0.9649405580281372
# AVGPS: 0.9992003048989536
```

For further analysis, the internal classifier object can be accessed through:
`._clf`

(Currently, all classifiers are internally scikit-learn objects.)

### Classifier hyperparamters:

The `Classifier()` call accepts several arguments to allow fine-tuning of parameters.
By default, these are all the paramters of the underlying `sklearn.RandomForestClassifier`, such as `n_estimators`, `n_jobs`, `max_depth`, etc.


### [BETA] Selecting feature extractor:
Features for classification-based correction are extracted using feature extractor functors, which transform the MSA information of considered nucleotides into a vector of numeric features.

Feature extraction can be modified by selecting different feature extractors by recompiling CARE with appropriate flags.

The CARE Python Module will automatically adjust to any shape of the feature vector.
At runtime, CARE will check assert that provided classifiers are based on the same extraction function that it was compiled with.

### [BETA] Selecting classification algorithm:

The classifier type (random-forest, logistic-regression, etc.) to be used
can be selected by providing the `Classifier()` call 
with a classifier type string as in `Classifier(clf_t_str="RF")`.

Currently, the only supported modes are `RF` (default) and `LR`, which correspond to scikit-learn's `RandomForestClassifier` and `LogisticRegressionClassifier`.

Changing the classificaton method requires recompilation of CARE using appropriate make flags.

