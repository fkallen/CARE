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

SCRIPTDIR=$(readlink -nf $0)

MLCDIR=/home/jcascitt/errorcorrector/ml
EVALDIR=/home/jcascitt/ec/hyperparams

CARE=/home/jcascitt/errorcorrector/care-cpu
CAREGPU="/home/jcascitt/errorcorrector/care-gpu -g 0"
FLAGS="-d . -h 48 -q --excludeAmbiguous --minalignmentoverlap 30 --minalignmentoverlapratio 0.3 -m 100G -p -t 88"

ARTEVAL=/home/jcascitt/errorcorrector/evaluationtool/arteval


### train files

FILE1=/share/errorcorrection/datasets/arthiseq2000humanchr14/humanchr1430cov.fq
FILE1EF=/share/errorcorrection/datasets/arthiseq2000humanchr14/humanchr1430cov_errFree.fq
COV1=30
PREFIX1=humanchr14-30

FILE2=/share/errorcorrection/datasets/arthiseq2000humanchr15/humanchr1530cov.fq
FILE2EF=/share/errorcorrection/datasets/arthiseq2000humanchr15/humanchr1530cov_errFree.fq
COV2=30
PREFIX2=humanchr15-30

FILE3=/share/errorcorrection/datasets/arthiseq2000athaliana/athaliana30cov.fq
FILE3EF=/share/errorcorrection/datasets/arthiseq2000athaliana/athaliana30cov_errFree.fq
COV3=30
PREFIX3=atha-30

FILE4=/share/errorcorrection/datasets/arthiseq2000elegans/elegans30cov.fq
FILE4EF=/share/errorcorrection/datasets/arthiseq2000elegans/elegans30cov_errFree.fq
COV4=30
PREFIX4=ele-30

FILE5=/share/errorcorrection/datasets/arthiseq2000melanogaster/melanogaster30cov.fq
FILE5EF=/share/errorcorrection/datasets/arthiseq2000melanogaster/melanogaster30cov_errFree.fq
COV5=30
PREFIX5=melan-30

FILE6=/share/errorcorrection/datasets/arthiseq2000mus/mus_chr15_30cov.fq
FILE6EF=/share/errorcorrection/datasets/arthiseq2000mus/mus_chr15_30cov_errFree.fq
COV6=30
PREFIX6=muschr15-30

### run

if [ ! -d $EVALDIR ]; then
	mkdir $EVALDIR
fi
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

# $CARE -i $FILE2 -c $COV2 -o null $FLAGS \
# 	--correctionType 2 --ml-forestfile ${PREFIX2}_anchor.samples \
# 	--candidateCorrection --correctionTypeCands 2 --ml-cands-forestfile ${PREFIX2}_cands.samples \
# 	$(auto_preprocess $PREFIX2)

# $CARE -i $FILE3 -c $COV3 -o null $FLAGS \
# 	--correctionType 2 --ml-forestfile ${PREFIX3}_anchor.samples \
# 	--candidateCorrection --correctionTypeCands 2 --ml-cands-forestfile ${PREFIX3}_cands.samples \
# 	$(auto_preprocess $PREFIX3)

# $CARE -i $FILE4 -c $COV4 -o null $FLAGS \
# 	--correctionType 2 --ml-forestfile ${PREFIX4}_anchor.samples \
# 	--candidateCorrection --correctionTypeCands 2 --ml-cands-forestfile ${PREFIX4}_cands.samples \
# 	$(auto_preprocess $PREFIX4)

$CARE -i $FILE5 -c $COV5 -o null $FLAGS \
	--correctionType 2 --ml-forestfile ${PREFIX5}_anchor.samples \
	--candidateCorrection --correctionTypeCands 2 --ml-cands-forestfile ${PREFIX5}_cands.samples \
	$(auto_preprocess $PREFIX5)

$CARE -i $FILE6 -c $COV6 -o null $FLAGS \
	--correctionType 2 --ml-forestfile ${PREFIX6}_anchor.samples \
	--candidateCorrection --correctionTypeCands 2 --ml-cands-forestfile ${PREFIX6}_cands.samples \
	$(auto_preprocess $PREFIX6)

python3 - <<EOF

import sys
sys.path.append("$MLCDIR")
from hyperparams import main

prefixes = ["${PREFIX1}", "${PREFIX2}", "${PREFIX3}", "${PREFIX4}", "${PREFIX5}", "${PREFIX6}"]
effiles = ["${FILE1EF}", "${FILE2EF}", "${FILE3EF}", "${FILE4EF}", "${FILE5EF}", "${FILE6EF}"]

### anchors only for now
data_map = [{"X":prefix+"_anchor.samples", "y":effile, "np":prefix+"_anchor.npy"} for prefix, effile in zip(prefixes, effiles)]
main(data_map)

EOF

echo "done."


