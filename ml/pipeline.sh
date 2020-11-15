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
    $CAREGPU -i $1 -c $3 -o ${4}_c $FLAGS \
        --correctionType 0 \
        $(auto_preprocess $4)

    $CAREGPU -i $1 -c $3 -o ${4}_c_c $FLAGS \
        --correctionType 0 \
        --candidateCorrection \
        $(auto_preprocess $4)

    $CARE -i $1 -c $3 -o ${4}_${5}-a${6} $FLAGS \
        --correctionType 1 --ml-forestfile ${5}_anchor.rf \
        $(auto_preprocess $4)

    $CARE -i $1 -c $3 -o ${4}_${5}-a${6}_c $FLAGS \
        --correctionType 1 --ml-forestfile ${5}_anchor.rf \
        --candidateCorrection \
        $(auto_preprocess $4)

    $CARE -i $1 -c $3 -o ${4}_${5}-a${6}_${5}-c${6} $FLAGS \
        --correctionType 1 --ml-forestfile ${5}_anchor.rf \
        --candidateCorrection --correctionTypeCands 1 --ml-cands-forestfile ${5}_cands.rf \
        $(auto_preprocess $4)

    $CARE -i $1 -c $3 -o ${4}_${5}-a${6}_${5}-a${6} $FLAGS \
        --correctionType 1 --ml-forestfile ${5}_anchor.rf \
        --candidateCorrection --correctionTypeCands 1 --ml-cands-forestfile ${5}_anchor.rf \
        $(auto_preprocess $4)

    EVALFILE=${4}_${5}_allcombi

    $ARTEVAL $1 $2 ${4}_c > $EVALFILE
    $ARTEVAL $1 $2 ${4}_c_c >> $EVALFILE
    $ARTEVAL $1 $2 ${4}_${5}-a${6} >> $EVALFILE
    $ARTEVAL $1 $2 ${4}_${5}-a${6}_c >> $EVALFILE
    $ARTEVAL $1 $2 ${4}_${5}-a${6}_${5}-c${6} >> $EVALFILE
    $ARTEVAL $1 $2 ${4}_${5}-a${6}_${5}-a${6} >> $EVALFILE
}

SCRIPTDIR=$(readlink -nf $0)

MLCDIR=/home/jcascitt/errorcorrector/
EVALDIR=/home/jcascitt/ec/mixdata_new

CARE=/home/jcascitt/errorcorrector/care-cpu
CAREGPU="/home/jcascitt/errorcorrector/care-gpu -g 0"
FLAGS="-d . -h 48 -q --excludeAmbiguous --minalignmentoverlap 30 --minalignmentoverlapratio 0.3 -m 100G -p -t 88"

ARTEVAL=/home/jcascitt/errorcorrector/evaluationtool/arteval


### train files

FILE1=/share/errorcorrection/datasets/arthiseq2000humanchr14/humanchr1430cov.fq
FILE1EF=/share/errorcorrection/datasets/arthiseq2000humanchr14/humanchr1430cov_errFree.fq
COV1=30
PREFIX1=humanchr14-30

FILE2=/share/errorcorrection/datasets/arthiseq2000athaliana/athaliana60cov.fq
FILE2EF=/share/errorcorrection/datasets/arthiseq2000athaliana/athaliana60cov_errFree.fq
COV2=60
PREFIX2=atha-60

FILE3=/share/errorcorrection/datasets/arthiseq2000elegans/elegans30cov.fq
FILE3EF=/share/errorcorrection/datasets/arthiseq2000elegans/elegans30cov_errFree.fq
COV3=30
PREFIX3=ele-30

CLF1=mixed


### test files
TESTFILE1=/share/errorcorrection/datasets/arthiseq2000humanchr14/humanchr1460cov.fq
TESTFILE1EF=/share/errorcorrection/datasets/arthiseq2000humanchr14/humanchr1460cov_errFree.fq
TESTPREFIX1=humanchr14-60
TESTCOV1=60

TESTFILE2=/share/errorcorrection/datasets/arthiseq2000elegans/elegans60cov.fq
TESTFILE2EF=/share/errorcorrection/datasets/arthiseq2000elegans/elegans60cov_errFree.fq
TESTPREFIX2=ele-60
TESTCOV2=60

TESTFILE3=/share/errorcorrection/datasets/arthiseq2000athaliana/athaliana30cov.fq
TESTFILE3EF=/share/errorcorrection/datasets/arthiseq2000athaliana/athaliana30cov_errFree.fq
TESTPREFIX3=atha-30
TESTCOV3=30

### run

mkdir $EVALDIR
cd $EVALDIR
cp $SCRIPTDIR $(basename $SCRIPTDIR).log.2

$CARE -i $FILE1 -c $COV1 -o null $FLAGS \
	--correctionType 2 --ml-forestfile ${PREFIX1}_anchor.samples \
	--candidateCorrection --correctionTypeCands 2 --ml-cands-forestfile ${PREFIX1}_cands.samples \
	$(auto_preprocess $PREFIX1)

$CARE -i $FILE2 -c $COV2 -o null $FLAGS \
	--correctionType 2 --ml-forestfile ${PREFIX2}_anchor.samples \
	--candidateCorrection --correctionTypeCands 2 --ml-cands-forestfile ${PREFIX2}_cands.samples \
	$(auto_preprocess $PREFIX2)

$CARE -i $FILE3 -c $COV3 -o null $FLAGS \
	--correctionType 2 --ml-forestfile ${PREFIX3}_anchor.samples \
	--candidateCorrection --correctionTypeCands 2 --ml-cands-forestfile ${PREFIX3}_cands.samples \
	$(auto_preprocess $PREFIX3)

$CARE -i $TESTFILE1 -c $TESTCOV1 -o null $FLAGS \
	--correctionType 2 --ml-forestfile ${TESTPREFIX1}_anchor.samples \
	--candidateCorrection --correctionTypeCands 2 --ml-cands-forestfile ${TESTPREFIX1}_cands.samples \
	$(auto_preprocess $TESTPREFIX1)

$CARE -i $TESTFILE2 -c $TESTCOV2 -o null $FLAGS \
	--correctionType 2 --ml-forestfile ${TESTPREFIX2}_anchor.samples \
	--candidateCorrection --correctionTypeCands 2 --ml-cands-forestfile ${TESTPREFIX2}_cands.samples \
	$(auto_preprocess $TESTPREFIX2)

$CARE -i $TESTFILE3 -c $TESTCOV3 -o null $FLAGS \
	--correctionType 2 --ml-forestfile ${TESTPREFIX3}_anchor.samples \
	--candidateCorrection --correctionTypeCands 2 --ml-cands-forestfile ${TESTPREFIX3}_cands.samples \
	$(auto_preprocess $TESTPREFIX3)

python3 - <<EOF
import sys
sys.path.append("$MLCDIR")
from mlcorrector import *

data = read_data(37, [{"X":"${PREFIX1}_anchor.samples", "y":"$FILE1EF"}, {"X":"${PREFIX2}_anchor.samples", "y":"$FILE2EF"}, {"X":"${PREFIX3}_anchor.samples", "y":"$FILE3EF"}])
np.save("${PREFIX1}+${PREFIX2}+${PREFIX3}_anchor.npy", data)
clf = train(data, "rf")
extract_forest(clf, "${CLF1}_anchor.rf")

data = read_data(37, [{"X":"${TESTPREFIX1}_anchor.samples", "y":"$TESTFILE1EF"}, {"X":"${TESTPREFIX2}_anchor.samples", "y":"$TESTFILE2EF"}, {"X":"${TESTPREFIX3}_anchor.samples", "y":"$TESTFILE3EF"}])
np.save("${TESTPREFIX1}+${TESTPREFIX2}+${TESTPREFIX3}_anchor.npy", data)
test(data, clf, "${CLF1}_anchor_${TESTPREFIX1}+${TESTPREFIX2}+${TESTPREFIX3}_anchor.roc.png")

data = read_data(42, [{"X":"${PREFIX1}_cands.samples", "y":"$FILE1EF"}, {"X":"${PREFIX2}_cands.samples", "y":"$FILE2EF"}, {"X":"${PREFIX3}_cands.samples", "y":"$FILE3EF"}])
np.save("${PREFIX1}+${PREFIX2}+${PREFIX3}_cands.npy", data)
clf = train(data, "rf")
extract_forest(clf, "${CLF1}_cands.rf")

data = read_data(42, [{"X":"${TESTPREFIX1}_cands.samples", "y":"$TESTFILE1EF"}, {"X":"${TESTPREFIX2}_cands.samples", "y":"$TESTFILE2EF"}, {"X":"${TESTPREFIX3}_cands.samples", "y":"$TESTFILE3EF"}])
np.save("${TESTPREFIX1}+${TESTPREFIX2}+${TESTPREFIX3}_cands.npy", data)
test(data, clf, "${CLF1}_cands_${TESTPREFIX1}+${TESTPREFIX2}+${TESTPREFIX3}_cands.roc.png")

EOF

$1: File
$2: File error-free
$3: cov
$4: file-prefix
$5: classifier-prefix
$6: threshold
full_eval $TESTFILE1 $TESTFILE1EF $TESTCOV1 $TESTPREFIX1 $CLF1 73
full_eval $TESTFILE2 $TESTFILE2EF $TESTCOV2 $TESTPREFIX2 $CLF1 73
full_eval $TESTFILE3 $TESTFILE3EF $TESTCOV3 $TESTPREFIX3 $CLF1 73


