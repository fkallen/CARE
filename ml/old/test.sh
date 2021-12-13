#!/bin/bash

auto_preprocess() {
	if [ -f "${1}_ht" ]; then
    	echo "--load-hashtables-from ${1}_ht"
	else
		echo "--save-hashtables-to ${1}_ht"
	fi
	if [ -f "${1}_pr" ]; then
    	echo "--load-preprocessedreads-from ${1}_pr"
	else
		echo "--save-preprocessedreads-to ${1}_pr"
	fi
}

# $1: File
# $2: File error-free
# $3: cov
# $4: file-prefix
# $5: classifier-prefix
# $6: threshold
full_eval() {

    $CAREGPU -i $1 -c $3 -o ${4}_c_c $FLAGS \
        --correctionType 0 \
        --candidateCorrection \
        $(auto_preprocess $4)

    $CARE -i $1 -c $3 -o ${4}_${5}-a${6}_${5}-c${6} $FLAGS \
        --correctionType 1 --ml-forestfile ${5}_anchor.rf \
        --candidateCorrection --correctionTypeCands 1 --ml-cands-forestfile ${5}_cands.rf \
        $(auto_preprocess $4)

    EVALFILE=${4}_${5}_eval

    $ARTEVAL $1 $2 ${4}_c_c >> $EVALFILE
    $ARTEVAL $1 $2 ${4}_${5}-a${6}_${5}-c${6} >> $EVALFILE
}

SCRIPTDIR=$(readlink -nf $0)

MLCDIR=/home/jcascitt/errorcorrector/ml
EVALDIR=/home/jcascitt/ec/test

CARE=/home/jcascitt/errorcorrector/care-cpu
CAREGPU="/home/jcascitt/errorcorrector/care-gpu -g 1"
FLAGS="-d . -h 48 -q --excludeAmbiguous --minalignmentoverlap 30 --minalignmentoverlapratio 0.3 -m 100G -p -t 88"

ARTEVAL=/home/jcascitt/errorcorrector/evaluationtool/arteval


### train file

FILE1=/share/errorcorrection/datasets/arthiseq2000elegans/elegans30cov.fq
FILE1EF=/share/errorcorrection/datasets/arthiseq2000elegans/elegans30cov_errFree.fq
COV1=30
PREFIX1=ele-30

CLF1=testclf

### test file
TESTFILE1=/share/errorcorrection/datasets/arthiseq2000ecoli/ecoli30cov.fq
TESTFILE1EF=/share/errorcorrection/datasets/arthiseq2000ecoli/ecoli30cov_errFree.fq
TESTPREFIX1=ecoli-30
TESTCOV1=30

### run

mkdir $EVALDIR
cd $EVALDIR
SCRIPTNAME=$(basename $SCRIPTDIR)
num=0
while [ -f ${SCRIPTNAME}.log.${num} ]; do
    num=$(( $num + 1 ))
done
cp $SCRIPTDIR $SCRIPTNAME.log.${num}

# $CARE -i $FILE1 -c $COV1 -o null $FLAGS \
# 	--correctionType 2 --ml-forestfile ${PREFIX1}_anchor.samples \
# 	--candidateCorrection --correctionTypeCands 2 --ml-cands-forestfile ${PREFIX1}_cands.samples \
# 	$(auto_preprocess $PREFIX1)

$CARE -i $TESTFILE1 -c $TESTCOV1 -o null $FLAGS \
	--correctionType 2 --ml-forestfile ${TESTPREFIX1}_anchor.samples \
	--candidateCorrection --correctionTypeCands 2 --ml-cands-forestfile ${TESTPREFIX1}_cands.samples \
	$(auto_preprocess $TESTPREFIX1)

# python3 - <<EOF
# import sys
# sys.path.append("$MLCDIR")
# from mlcorrector import *

# data = read_data(37, [{"X":"${PREFIX1}_anchor.samples", "y":"$FILE1EF"}])
# np.save("${PREFIX1}_anchor.npy", data)
# clf = train(data, "rf")
# extract_forest(clf, "${CLF1}_anchor.rf")

# data = read_data(37, [{"X":"${TESTPREFIX1}_anchor.samples", "y":"$TESTFILE1EF"}])
# np.save("${TESTPREFIX1}_anchor.npy", data)
# test(data, clf, "${CLF1}_anchor_${TESTPREFIX1}_anchor.roc.png")

# data = read_data(42, [{"X":"${PREFIX1}_cands.samples", "y":"$FILE1EF"}])
# np.save("${PREFIX1}_cands.npy", data)
# clf = train(data, "rf")
# extract_forest(clf, "${CLF1}_cands.rf")

# data = read_data(42, [{"X":"${TESTPREFIX1}_cands.samples", "y":"$TESTFILE1EF"}])
# np.save("${TESTPREFIX1}_cands.npy", data)
# test(data, clf, "${CLF1}_cands_${TESTPREFIX1}_cands.roc.png")

# EOF

# # $1: File
# # $2: File error-free
# # $3: cov
# # $4: file-prefix
# # $5: classifier-prefix
# # $6: threshold
# full_eval $TESTFILE1 $TESTFILE1EF $TESTCOV1 $TESTPREFIX1 $CLF1 73



