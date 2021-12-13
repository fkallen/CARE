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
grid_search() {

    EVALFILE=${4}_${5}_eval

    for (( THRESH="$6"; THRESH<="$7"; THRESH+="$8" )); do
        $CARE -i $1 -c $3 -o ${4}_${5}-a${THRESH}_${5}-c${THRESH} $FLAGS \
        --correctionType 1 --ml-forestfile ${5}_anchor.rf \
        --candidateCorrection --correctionTypeCands 1 --ml-cands-forestfile ${5}_cands.rf \
        --threshold $THRESH \
        $(auto_preprocess $4)
    done

    # for (( THRESH="$6"; THRESH<="$7"; THRESH+="$8" )); do
    #     $ARTEVAL $1 $2 ${4}_${5}-a${THRESH}_${5}-c${THRESH} >> $EVALFILE
    #     rm ${4}_${5}-a${THRESH}_${5}-c${THRESH}
    # done
}

SCRIPTDIR=$(readlink -nf $0)

MLCDIR=/home/jcascitt/errorcorrector/ml
EVALDIR=/home/jcascitt/ec/realdata

CARE=/home/jcascitt/errorcorrector/care-cpu
CAREGPU="/home/jcascitt/errorcorrector/care-gpu -g 0"
FLAGS="-d . -h 48 -q --excludeAmbiguous --minalignmentoverlap 30 --minalignmentoverlapratio 0.3 -m 100G -p -t 88 --samplingRateAnchor 0.2 --samplingRateCands 0.008"

ARTEVAL=/home/jcascitt/errorcorrector/evaluationtool/arteval


### files

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


FILE_ELE=/share/errorcorrection/datasets/C.elegans_SRR543736.fastq
COV_ELE=58
PREFIX_ELE=real-ele

FILE_MEL=/share/errorcorrection/datasets/D.melanogaster_SRR988075.fastq
COV_MEL=64
PREFIX_MEL=real-mel

FILE_AIP=/share/errorcorrection/datasets/Aiptasia_SRR606428_20cov.fastq.gz
COV_AIP=20
PREFIX_AIP=real-aip

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


### comparison runs

module load cuda/11.1

# $CAREGPU -i $FILE_ELE -c $COV_ELE -o ${PREFIX_ELE}_cc $FLAGS \
# 	--candidateCorrection \
# 	$(auto_preprocess ${PREFIX_ELE})

$CAREGPU -i $FILE_MEL -c $COV_MEL -o ${PREFIX_MEL}_cc $FLAGS \
	--candidateCorrection \
	$(auto_preprocess ${PREFIX_MEL})

$CAREGPU -i $FILE_AIP -c $COV_AIP -o ${PREFIX_AIP}_cc $FLAGS \
	--candidateCorrection \
	$(auto_preprocess ${PREFIX_AIP})

python3 - <<EOF

import sys
sys.path.append("$MLCDIR")
from realdata import main

prefixes = ["${PREFIX1}", "${PREFIX2}", "${PREFIX3}", "${PREFIX4}", "${PREFIX5}", "${PREFIX6}"]
effiles = ["${FILE1EF}", "${FILE2EF}", "${FILE3EF}", "${FILE4EF}", "${FILE5EF}", "${FILE6EF}"]

main(prefixes, effiles)

EOF

grid_search $FILE_ELE ASDF $COV_ELE $PREFIX_ELE 4 80 85 5
grid_search $FILE_MEL ASDF $COV_MEL $PREFIX_MEL 5 80 85 5

grid_search $FILE_AIP ASDF $COV_AIP $PREFIX_AIP all6 80 85 5

#echo "done."


