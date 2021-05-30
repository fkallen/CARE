#!/bin/bash

module load cuda/11.1


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
    echo ""
}

run_classic() {
	$CAREGPU -i ${files[${1}]} -c ${cov[${1}]} -o ${prefixes[${1}]}_cc${CARE_FILEENDING} ${CARE_FLAGS} \
		--candidateCorrection \
		$(auto_preprocess ${prefixes[${1}]})
	
	$ARTEVAL ${files[${1}]} ${files_ef[${1}]} ${prefixes[${1}]}_cc${CARE_FILEENDING} >> ${prefixes[${1}]}_cc_eval
	rm ${prefixes[${1}]}_cc${CARE_FILEENDING}
}

run_classic_cpu() {
	$CARE -i ${files[${1}]} -c ${cov[${1}]} -o ${prefixes[${1}]}_cc${CARE_FILEENDING} ${CARE_FLAGS} \
		--candidateCorrection \
		$(auto_preprocess ${prefixes[${1}]})
	
	$ARTEVAL ${files[${1}]} ${files_ef[${1}]} ${prefixes[${1}]}_cc${CARE_FILEENDING} >> ${prefixes[${1}]}_cc_eval
	rm ${prefixes[${1}]}_cc${CARE_FILEENDING}
}

run_print() {
	$CARE -i ${files[${1}]} -c ${cov[${1}]} -o null $CARE_FLAGS \
		--correctionType 2 --ml-forestfile ${prefixes[${1}]}_anchor.samples \
		--candidateCorrection --correctionTypeCands 2 --ml-cands-forestfile ${prefixes[${1}]}_cands.samples \
		$(auto_preprocess ${prefixes[${1}]})
}

run_clf() {
	$CAREGPU -i ${files[${1}]} -c ${cov[${1}]} -o ${prefixes[${1}]}_${2}-${3}-${4}$CARE_FILEENDING $CARE_FLAGS \
		--correctionType 1 --ml-forestfile ${2}_anchor.rf \
		--candidateCorrection --correctionTypeCands 1 --ml-cands-forestfile ${2}_cands.rf \
		--thresholdAnchor ${3} --thresholdCands ${4} \
		$(auto_preprocess ${prefixes[${1}]})

	$ARTEVAL ${files[${1}]} ${files_ef[${1}]} ${prefixes[${1}]}_${2}-${3}-${4}${CARE_FILEENDING} >> ${prefixes[${1}]}_${2}_eval
	rm ${prefixes[${1}]}_${2}-${3}-${4}${CARE_FILEENDING}
}

run_clf_cpu() {
	$CARE -i ${files[${1}]} -c ${cov[${1}]} -o ${prefixes[${1}]}_${2}-${3}-${4}$CARE_FILEENDING $CARE_FLAGS \
		--correctionType 1 --ml-forestfile ${2}_anchor.rf \
		--candidateCorrection --correctionTypeCands 1 --ml-cands-forestfile ${2}_cands.rf \
		--thresholdAnchor ${3} --thresholdCands ${4} \
		$(auto_preprocess ${prefixes[${1}]})

	$ARTEVAL ${files[${1}]} ${files_ef[${1}]} ${prefixes[${1}]}_${2}-${3}-${4}${CARE_FILEENDING} >> ${prefixes[${1}]}_${2}_eval
	rm ${prefixes[${1}]}_${2}-${3}-${4}${CARE_FILEENDING}
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

MLCDIR=/home/jcascitt/errorcorrector_wip/ml
EVALDIR=/home/jcascitt/ec/gpucheck_new

CARE=/home/jcascitt/errorcorrector/care-cpu
CAREGPU="/home/jcascitt/errorcorrector/care-gpu -g 0"
CARE_FLAGS="-d . -h 48 -q --excludeAmbiguous --minalignmentoverlap 30 --minalignmentoverlapratio 0.3 -m 64G -p -t 64 --samplingRateAnchor 1 --samplingRateCands 1"

ARTEVAL=/home/jcascitt/errorcorrector/evaluationtool/arteval


### files

CARE_FILEENDING=.fq

files[0]=/share/errorcorrection/datasets/arthiseq2000humanchr14/humanchr1430cov.fq
files_ef[0]=/share/errorcorrection/datasets/arthiseq2000humanchr14/humanchr1430cov_errFree.fq
cov[0]=30
prefixes[0]=humanchr14-30

files[1]=/share/errorcorrection/datasets/arthiseq2000athaliana/athaliana30cov.fq
files_ef[1]=/share/errorcorrection/datasets/arthiseq2000athaliana/athaliana30cov_errFree.fq
cov[1]=30
prefixes[1]=atha-30


### run

if [ ! -d $EVALDIR ]; then
	mkdir $EVALDIR
	ln -s /dev/null $EVALDIR/null
fi
cd $EVALDIR
SCRIPTNAME=$(basename $SCRIPTPATH)
num=0
while [ -f ${SCRIPTNAME}.log.${num} ]; do
    num=$(( $num + 1 ))
done
cp $SCRIPTPATH $SCRIPTNAME.log.${num}



## comparison

# run_classic 0
# run_classic_cpu 0

### print runs

# for (( i="0"; i<="1"; i+="1" )); do
# 	run_print $i
# done

prefixes_joined=$(printf ",\"%s\"" "${prefixes[@]}")
prefixes_joined=${prefixes_joined:1}

files_ef_joined=$(printf ",\"%s\"" "${files_ef[@]}")
files_ef_joined=${files_ef_joined:1}

python3 - <<EOF

import sys
print("$MLCDIR")
sys.path.append("$MLCDIR")
from gpucheck import main

prefixes = [${prefixes_joined}]
effiles = [${files_ef_joined}]

main(prefixes, effiles, 0)

EOF

# run_clf 0 0 70 70
# run_clf_cpu 0 0 70 70


#echo "done."


