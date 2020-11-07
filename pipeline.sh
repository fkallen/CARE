#!/bin/bash

save_prep () {
	echo "--save-hashtables-to ${1}_ht --save-preprocessedreads-to ${1}_pr"
}

load_prep () {
	echo "--load-hashtables-from ${1}_ht --load-preprocessedreads-from ${1}_pr"
}

MLCDIR=/home/jcascitt/errorcorrector/
EVALDIR=/home/jcascitt/ec/humanchr14-30-60

CARE=/home/jcascitt/errorcorrector/care-cpu
CAREGPU="/home/jcascitt/errorcorrector/care-cpu -g 0"
FLAGS="-d . -h 48 -q --excludeAmbiguous --minalignmentoverlap 30 --minalignmentoverlapratio 0.3 -m 150G -p -t 88"

ARTEVAL=/home/jcascitt/errorcorrector/evaluationtool/arteval

FILE1=/share/errorcorrection/datasets/arthiseq2000humanchr14/humanchr1430cov.fq
# FILE1=/home/jcascitt/ec/1m.fq
FILE1EF=/share/errorcorrection/datasets/arthiseq2000humanchr14/humanchr1430cov_errFree.fq
COV1=30
PREFIX1=humanchr14-30

FILE2=/share/errorcorrection/datasets/arthiseq2000humanchr14/humanchr1460cov.fq
# FILE2=/home/jcascitt/ec/1m.fq
FILE2EF=/share/errorcorrection/datasets/arthiseq2000humanchr14/humanchr1460cov_errFree.fq
COV2=30
PREFIX2=humanchr14-60

mkdir $EVALDIR
cd $EVALDIR

$CARE -i $FILE1 -c $COV1 -o null $FLAGS \
	--correctionType 2 --ml-forestfile ${PREFIX1}_anchor.samples \
	--candidateCorrection --correctionTypeCands 2 --ml-cands-forestfile ${PREFIX1}_cands.samples \
	$(save_prep $PREFIX1)

python3 - <<EOF
import sys
sys.path.append("$MLCDIR")
from mlcorrector import *

data = read_data(36, [{"X":"${PREFIX1}_anchor.samples", "y":"$FILE1EF"}])
np.save("${PREFIX1}_anchor.npy", data)
clf = train(data, "rf")
extract_forest(clf, "${PREFIX1}_anchor.rf")

data = read_data(36, [{"X":"${PREFIX1}_cands.samples", "y":"$FILE1EF"}])
np.save("${PREFIX1}_cands.npy", data)
clf = train(data, "rf")
extract_forest(clf, "${PREFIX1}_cands.rf")

EOF

$CAREGPU -i $FILE2 -c $COV2 -o ${PREFIX2}_c $FLAGS \
	--correctionType 0 \
	$(save_prep $PREFIX2)

$CAREGPU -i $FILE2 -c $COV2 -o ${PREFIX2}_c_c $FLAGS \
	--correctionType 0 \
	--candidateCorrection \
	$(load_prep $PREFIX2)

$CARE -i $FILE2 -c $COV2 -o ${PREFIX2}_${PREFIX1}-rfa73 $FLAGS \
	--correctionType 1 --ml-forestfile ${PREFIX1}_anchor.rf \
	$(load_prep $PREFIX2)

$CARE -i $FILE2 -c $COV2 -o ${PREFIX2}_${PREFIX1}-rfa73_c $FLAGS \
	--correctionType 1 --ml-forestfile ${PREFIX1}_anchor.rf \
	--candidateCorrection \
	$(load_prep $PREFIX2)

$CARE -i $FILE2 -c $COV2 -o ${PREFIX2}_${PREFIX1}-rfa73_${PREFIX1}-rfc73 $FLAGS \
	--correctionType 1 --ml-forestfile ${PREFIX1}_anchor.rf \
	--candidateCorrection --correctionTypeCands 1 --ml-cands-forestfile ${PREFIX1}_cands.rf \
	$(load_prep $PREFIX2)

$CARE -i $FILE2 -c $COV2 -o ${PREFIX2}_${PREFIX1}-rfa73_${PREFIX1}-rfa73 $FLAGS \
	--correctionType 1 --ml-forestfile ${PREFIX1}_anchor.rf \
	--candidateCorrection --correctionTypeCands 1 --ml-cands-forestfile ${PREFIX1}_anchor.rf \
	$(load_prep $PREFIX2)

EVALFILE=${PREFIX1}_${PREFIX2}_allcombi

$ARTEVAL $FILE2 $FILE2EF ${PREFIX2}_c > $EVALFILE
$ARTEVAL $FILE2 $FILE2EF ${PREFIX2}_c_c >> $EVALFILE
$ARTEVAL $FILE2 $FILE2EF ${PREFIX2}_${PREFIX1}-rfa73 >> $EVALFILE
$ARTEVAL $FILE2 $FILE2EF ${PREFIX2}_${PREFIX1}-rfa73_c >> $EVALFILE
$ARTEVAL $FILE2 $FILE2EF ${PREFIX2}_${PREFIX1}-rfa73_${PREFIX1}-rfc73 >> $EVALFILE
$ARTEVAL $FILE2 $FILE2EF ${PREFIX2}_${PREFIX1}-rfa73_${PREFIX1}-rfa73 >> $EVALFILE

