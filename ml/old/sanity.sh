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

        $ARTEVAL $1 $2 ${4}_${5}-a${THRESH}_${5}-c${THRESH} >> $EVALFILE
        rm ${4}_${5}-a${THRESH}_${5}-c${THRESH}

    done
}

SCRIPTDIR=$(readlink -nf $0)

MLCDIR=/home/jcascitt/errorcorrector/ml
EVALDIR=/home/jcascitt/ec/sanitycheck

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

# $1: File
# $2: File error-free
# $3: cov
# $4: file-prefix
# $5: classifier-prefix


# $CAREGPU -i $FILE1 -c $COV1 -o ${PREFIX1}_cc $FLAGS \
# 	--candidateCorrection \
# 	$(auto_preprocess ${PREFIX1})

# $ARTEVAL $FILE1 $FILE1EF ${PREFIX1}_cc >> ${PREFIX1}_cc_eval

# $CAREGPU -i $FILE2 -c $COV2 -o ${PREFIX2}_cc $FLAGS \
# 	--candidateCorrection \
# 	$(auto_preprocess ${PREFIX2})

# $ARTEVAL $FILE2 $FILE2EF ${PREFIX2}_cc >> ${PREFIX2}_cc_eval

# $CAREGPU -i $FILE3 -c $COV3 -o ${PREFIX3}_cc $FLAGS \
# 	--candidateCorrection \
# 	$(auto_preprocess ${PREFIX3})

# $ARTEVAL $FILE3 $FILE3EF ${PREFIX3}_cc >> ${PREFIX3}_cc_eval

# $CAREGPU -i $FILE4 -c $COV4 -o ${PREFIX4}_cc $FLAGS \
# 	--candidateCorrection \
# 	$(auto_preprocess ${PREFIX4})

# $ARTEVAL $FILE4 $FILE4EF ${PREFIX4}_cc >> ${PREFIX4}_cc_eval

$CARE -i $FILE5 -c $COV5 -o ${PREFIX5}_cc $FLAGS \
	--candidateCorrection \
	$(auto_preprocess ${PREFIX5})

$ARTEVAL $FILE5 $FILE5EF ${PREFIX5}_cc >> ${PREFIX5}_cc_eval

# $CAREGPU -i $FILE6 -c $COV6 -o ${PREFIX6}_cc $FLAGS \
# 	--candidateCorrection \
# 	$(auto_preprocess ${PREFIX6})

# $ARTEVAL $FILE6 $FILE6EF ${PREFIX6}_cc >> ${PREFIX6}_cc_eval



# ### print runs

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

# $CARE -i $FILE5 -c $COV5 -o null $FLAGS \
# 	--correctionType 2 --ml-forestfile ${PREFIX5}_anchor.samples \
# 	--candidateCorrection --correctionTypeCands 2 --ml-cands-forestfile ${PREFIX5}_cands.samples \
# 	$(auto_preprocess $PREFIX5)

# $CARE -i $FILE6 -c $COV6 -o null $FLAGS \
# 	--correctionType 2 --ml-forestfile ${PREFIX6}_anchor.samples \
# 	--candidateCorrection --correctionTypeCands 2 --ml-cands-forestfile ${PREFIX6}_cands.samples \
# 	$(auto_preprocess $PREFIX6)

# python3 - <<EOF

# import sys
# sys.path.append("$MLCDIR")
# from leaveoneout import main

# prefixes = ["${PREFIX1}", "${PREFIX2}", "${PREFIX3}", "${PREFIX4}", "${PREFIX5}", "${PREFIX6}"]
# effiles = ["${FILE1EF}", "${FILE2EF}", "${FILE3EF}", "${FILE4EF}", "${FILE5EF}", "${FILE6EF}"]

# main(prefixes, effiles)

# EOF

# grid_search $FILE1 $FILE1EF $COV1 $PREFIX1 1 85 85 5
# grid_search $FILE2 $FILE2EF $COV2 $PREFIX2 2 85 85 5
# grid_search $FILE3 $FILE3EF $COV3 $PREFIX3 3 85 85 5
# grid_search $FILE4 $FILE4EF $COV4 $PREFIX4 4 85 85 5
# grid_search $FILE5 $FILE5EF $COV5 $PREFIX5 5 85 85 5
# grid_search $FILE6 $FILE6EF $COV6 $PREFIX6 6 85 85 5

echo "done."


