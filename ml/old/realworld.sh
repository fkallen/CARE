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

run_classic() {
	$CAREGPU -i ${files[${1}]} -c ${cov[${1}]} -o out/${prefixes[${1}]}_cc${3}.fq ${CARE_FLAGS} \
		--candidateCorrection ${2} \
		$(auto_preprocess ${prefixes[${1}]})
	
	$ARTEVAL ${files[${1}]} ${files_ef[${1}]} out/${prefixes[${1}]}_cc${3}.fq >> out/${prefixes[${1}]}_cc_eval
	rm out/${prefixes[${1}]}_cc${3}.fq

}

# run_classic_cpu() {
# 	$CARE -i ${files[${1}]} -c ${cov[${1}]} -o out/${prefixes[${1}]}_cc.fq ${CARE_FLAGS} \
# 		--candidateCorrection \
# 		$(auto_preprocess ${prefixes[${1}]})
	
# 	$ARTEVAL ${files[${1}]} ${files_ef[${1}]} out/${prefixes[${1}]}_cc.fq >> out/${prefixes[${1}]}_cc_eval
# 	rm out/${prefixes[${1}]}_cc.fq
# }

run_print() {
	$CARE -i ${files[${1}]} -c ${cov[${1}]} -o null $CARE_FLAGS \
		--correctionType 2 --ml-forestfile ${prefixes[${1}]}_${EVALDIRNAME}_anchor.samples \
		--candidateCorrection --correctionTypeCands 2 --ml-cands-forestfile ${prefixes[${1}]}_${EVALDIRNAME}_cands.samples \
		$(auto_preprocess ${prefixes[${1}]})
}

run_clf() {
	$CAREGPU -i ${files[${1}]} -c ${cov[${1}]} -o out/${prefixes[${1}]}_${2}-${3}-${4}${6}.fq $CARE_FLAGS \
		--correctionType 1 --ml-forestfile ${2}_anchor.rf \
		--candidateCorrection --correctionTypeCands 1 --ml-cands-forestfile ${2}_cands.rf \
		--thresholdAnchor ${3} --thresholdCands ${4} ${5} \
		$(auto_preprocess ${prefixes[${1}]})

	$ARTEVAL ${files[${1}]} ${files_ef[${1}]} out/${prefixes[${1}]}_${2}-${3}-${4}${6}.fq >> out/${prefixes[${1}]}_${2}_eval
	rm out/${prefixes[${1}]}_${2}-${3}-${4}${6}.fq
}

grid_search() {
    for (( THRESH="${3}"; THRESH<="${4}"; THRESH+="${5}" )); do
        for (( THRESHC="${6}"; THRESHC<="${7}"; THRESHC+="${8}" )); do
			run_clf ${1} ${2} ${THRESH} ${THRESHC}
        done
    done
}



### settings

SCRIPTPATH=$(readlink -nf $0)

MLCDIR=/home/jcascitt/errorcorrector/ml
EVALDIRNAME=${1}
EVALDIR=/home/jcascitt/care/${EVALDIRNAME}

CARE=/home/jcascitt/errorcorrector/care-cpu
CAREGPU="/home/jcascitt/errorcorrector/care-gpu -g 0"
CARE_FLAGS="-d . -h 48 -q --excludeAmbiguous --minalignmentoverlap 30 --minalignmentoverlapratio 0.3 -m 64G -p -t 88 --samplingRateAnchor 1 --samplingRateCands 0.04 --enforceHashmapCount --warpcore 1"

ARTEVAL=/home/jcascitt/errorcorrector/evaluationtool/arteval


### files

files[0]=/share/errorcorrection/datasets/artpairedelegans/elegans30cov_500_10.fq
files_ef[0]=/share/errorcorrection/datasets/artpairedelegans/elegans30cov_500_10_errFree.fastq
cov[0]=30
prefixes[0]=pele-30
clf[0]=ele-30_t


files[1]=/share/errorcorrection/datasets/artpairedhumanchr14/humanchr1430cov_500_10.fastq
files_ef[1]=/share/errorcorrection/datasets/artpairedhumanchr14/humanchr1430cov_500_10_errFree.fastq
cov[1]=30
prefixes[1]=phumanchr14-30
clf[1]=humanchr15-30

files[2]=/share/errorcorrection/datasets/artpairedmelanogaster/melanogaster30cov_500_10.fastq
files_ef[2]=/share/errorcorrection/datasets/artpairedmelanogaster/melanogaster30cov_500_10_errFree.fastq
cov[2]=30
prefixes[2]=pmelan-30
clf[2]=melan-30_t

###

### run

if [ ! -d $EVALDIR ]; then
	mkdir $EVALDIR
	ln -s /dev/null $EVALDIR/null
	mkdir $EVALDIR/out
fi
cd $EVALDIR
SCRIPTNAME=$(basename $SCRIPTPATH)
num=0
while [ -f ${SCRIPTNAME}.log.${num} ]; do
    num=$(( $num + 1 ))
done
cp $SCRIPTPATH $SCRIPTNAME.log.${num}

# run

for (( j="0"; j<="2"; j+="1" )); do
	run_classic $j "--pairmode SE"
	run_clf $j ${clf[${j}]} 90 30 "--pairmode SE"
	for (( i="1"; i<="9"; i+="1" )); do
		run_classic $j "--pairmode PE --pairedthreshold1 0.0${i}" "-0.0${i}"
		run_clf $j ${clf[${j}]} 90 30 "--pairmode PE --pairedthreshold1 0.0${i}" "-0.0${i}"
	done
	run_classic $j "--pairmode PE --pairedthreshold1 0.1" "-0.1"
	run_clf $j ${clf[${j}]} 90 30 "--pairmode PE --pairedthreshold1 0.1" "-0.1"
done

echo "done."


