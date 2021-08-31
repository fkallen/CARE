#!/bin/bash

auto_preprocess() {
	# if [ -f "${1}_ht" ]; then
    # 	echo "--load-hashtables-from ${1}_ht"
	# else
	# 	echo "--save-hashtables-to ${1}_ht"
	# fi
	# if [ -f "${1}_pr" ]; then
    # 	echo "--load-preprocessedreads-from ${1}_pr"
	# else
	# 	echo "--save-preprocessedreads-to ${1}_pr"
	# fi
	echo "" #warpcore
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
		--correctionType 2 --ml-forestfile ${clf[${1}]}_anchor.samples \
		--candidateCorrection --correctionTypeCands 2 --ml-cands-forestfile ${clf[${1}]}_cands.samples \
		--pairmode SE \
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
			run_clf ${1} ${2} ${THRESH} ${THRESHC} "--pairmode SE"
        done
    done
}



### settings

SCRIPTPATH=$(readlink -nf $0)

MLCDIR=/home/jcascitt/errorcorrector/ml
EVALDIRNAME=${1}
EVALDIR=/home/jcascitt/care/${EVALDIRNAME}

CARE=/home/jcascitt/errorcorrector/care-cpu
CAREGPU="/home/jcascitt/errorcorrector/care-gpu -g 0  --warpcore 1"
CARE_FLAGS="-d . -h 48 -q --excludeAmbiguous --minalignmentoverlap 30 --minalignmentoverlapratio 0.3 -m 64G -p -t 88 --samplingRateAnchor 1 --samplingRateCands 0.3 --enforceHashmapCount"

ARTEVAL=/home/jcascitt/errorcorrector/evaluationtool/arteval


### files

files[1]=/share/errorcorrection/datasets/artpairedathaliana_miseqv3/athaliana30cov_500_10.fq
files_ef[1]=/share/errorcorrection/datasets/artpairedathaliana_miseqv3/athaliana30cov_500_10_errFree.fq
cov[1]=30
prefixes[1]=mps-atha-30
clf[1]=mps-atha-30_t

files[2]=/share/errorcorrection/datasets/artpairedelegans_miseqv3/elegans30cov_500_10.fq
files_ef[2]=/share/errorcorrection/datasets/artpairedelegans_miseqv3/elegans30cov_500_10_errFree.fq
cov[2]=30
prefixes[2]=mps-ele-30
clf[2]=mps-ele-30_t

files[3]=/share/errorcorrection/datasets/artpairedhumanchr15_miseqv3/humanchr1530cov_500_10.fq
files_ef[3]=/share/errorcorrection/datasets/artpairedhumanchr15_miseqv3/humanchr1530cov_500_10_errFree.fq
cov[3]=30
prefixes[3]=mps-humanchr15-30
clf[3]=mps-humanchr15-30_t

files[4]=/share/errorcorrection/datasets/artpairedmelanogaster_miseqv3/melanogaster30cov_500_10.fq
files_ef[4]=/share/errorcorrection/datasets/artpairedmelanogaster_miseqv3/melanogaster30cov_500_10_errFree.fq
cov[4]=30
prefixes[4]=mps-melan-30
clf[4]=mps-melan-30_t

files[5]=/share/errorcorrection/datasets/artpairedmuschr15_miseqv3/muschr1530cov_500_10.fq
files_ef[5]=/share/errorcorrection/datasets/artpairedmuschr15_miseqv3/muschr1530cov_500_10_errFree.fq
cov[5]=30
prefixes[5]=mps-muschr15-30
clf[5]=mps-muschr15-30_t

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


# # print
# for (( j="1"; j<="5"; j+="1" )); do
# 	# run_classic $j "--pairmode SE" # for comparison
# 	run_print $j
# done

# train clfs

clfprefixes_joined=$(printf ",\"%s\"" "${clf[@]:1}")
clfprefixes_joined=${clfprefixes_joined:1}

files_ef_joined=$(printf ",\"%s\"" "${files_ef[@]:1}")
files_ef_joined=${files_ef_joined:1}

python3 - <<EOF

import sys
sys.path.append("$MLCDIR")
from miseq import main

clfprefixes = [${clfprefixes_joined}]
effiles = [${files_ef_joined}]

main(clfprefixes, effiles)

EOF

# grid search

# comparison
for (( j="1"; j<="5"; j+="1" )); do
	run_classic $j "--pairmode SE"
done

for (( i="1"; i<="5"; i+="1" )); do
	grid_search ${i} ${clf[${i}]} 0 100 10 0 100 10
done

echo "done."


