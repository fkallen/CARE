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
	# echo ""
}

run_classic() {
	$CAREGPU -i ${files[${1}]} -c ${cov[${1}]} -o ${prefixes[${1}]}_cc${3}.fq ${CARE_FLAGS} \
		--candidateCorrection ${2} \
		$(auto_preprocess ${prefixes[${1}]})
	
	$ARTEVAL ${files[${1}]} ${files_ef[${1}]} out/${prefixes[${1}]}_cc${3}.fq >> out/${prefixes[${1}]}_cc_eval
	rm out/${prefixes[${1}]}_cc${3}.fq

}

# run_classic_cpu() {
# 	$CARE -i ${files[${1}]} -c ${cov[${1}]} -o ${prefixes[${1}]}_cc.fq ${CARE_FLAGS} \
# 		--candidateCorrection \
# 		$(auto_preprocess ${prefixes[${1}]})
	
# 	$ARTEVAL ${files[${1}]} ${files_ef[${1}]} out/${prefixes[${1}]}_cc.fq >> out/${prefixes[${1}]}_cc_eval
# 	rm out/${prefixes[${1}]}_cc.fq
# }

run_print() {
	$CARE -i ${files[${1}]} -c ${cov[${1}]} -o null $CARE_FLAGS \
		--correctionType 2 --ml-print-forestfile ${prefixes[${1}]}_${EVALDIRNAME}_anchor.samples \
		--candidateCorrection --correctionTypeCands 2 --ml-cands-print-forestfile ${prefixes[${1}]}_${EVALDIRNAME}_cands.samples ${2} \
		$(auto_preprocess ${prefixes[${1}]})
}

run_clf() {
	$CAREGPU -i ${files[${1}]} -c ${cov[${1}]} -o ${prefixes[${1}]}_${2}-${3}-${4}${6}.fq $CARE_FLAGS \
		--correctionType 1 --ml-forestfile ${2}_anchor.rf \
		--candidateCorrection --correctionTypeCands 1 --ml-cands-forestfile ${2}_cands.rf \
		--thresholdAnchor ${3} --thresholdCands ${4} ${5}
		# $(auto_preprocess ${prefixes[${1}]})

	$ARTEVAL ${files[${1}]} ${files_ef[${1}]} out/${prefixes[${1}]}_${2}-${3}-${4}${6}.fq >> out/${prefixes[${1}]}_${2}${6}_eval
	rm out/${prefixes[${1}]}_${2}-${3}-${4}${6}.fq
}

grid_search() {
    for (( THRESH="${3}"; THRESH<="${4}"; THRESH+="${5}" )); do
        for (( THRESHC="${6}"; THRESHC<="${7}"; THRESHC+="${8}" )); do
			run_clf ${1} ${2} ${THRESH} ${THRESHC} "${9}" ${10}
        done
    done
}


### settings

SCRIPTPATH=$(readlink -nf $0)

MLCDIR=/home/jcascitt/errorcorrector/ml
EVALDIRNAME=${1}
EVALDIR=/home/jcascitt/care/paired/${EVALDIRNAME}

CARE="/home/jcascitt/errorcorrector/care-cpu-v3 -t 88"
CAREGPU="/home/jcascitt/errorcorrector/care-gpu-v3 -g ${2} --warpcore 1 -t 88"
CARE_FLAGS="-d out -h 48 -q --excludeAmbiguous --minalignmentoverlap 30 --minalignmentoverlapratio 0.3 -m 64G -p --samplingRateAnchor 0.25 --samplingRateCands 0.01 --enforceHashmapCount"

ARTEVAL=/home/jcascitt/errorcorrector/evaluationtool/arteval

### files

files[0]=/share/errorcorrection/datasets/artpairedhumanchr15/humanchr15_500_10.fastq
files_ef[0]=/share/errorcorrection/datasets/artpairedhumanchr15/humanchr15_500_10_errFree.fastq
cov[0]=30
prefixes[0]=humanchr15-30p

files[1]=/share/errorcorrection/datasets/artpairedathaliana/athaliana_500_10.fastq
files_ef[1]=/share/errorcorrection/datasets/artpairedathaliana/athaliana_500_10_errFree.fastq
cov[1]=30
prefixes[1]=atha-30p

files[2]=/share/errorcorrection/datasets/artpairedelegans/elegans30cov_500_10.fq
files_ef[2]=/share/errorcorrection/datasets/artpairedelegans/elegans30cov_500_10_errFree.fastq
cov[2]=30
prefixes[2]=ele-30p

# files[3]=/share/errorcorrection/datasets/artpairedmuschr15/muschr15_500_10.fastq
# files_ef[3]=/share/errorcorrection/datasets/artpairedmuschr15/muschr15_500_10_errFree.fastq
# cov[3]=30
# prefixes[3]=muschr15-30p

# files[4]=/share/errorcorrection/datasets/artpairedmelanogaster/melanogaster30cov_500_10.fastq
# files_ef[4]=/share/errorcorrection/datasets/artpairedmelanogaster/melanogaster30cov_500_10_errFree.fastq
# cov[4]=30
# prefixes[4]=melan-30p

###########################################################

spack load cuda@11.5.0

if [ ! -d $EVALDIR ]; then
	mkdir $EVALDIR
	mkdir $EVALDIR/out
	ln -s /dev/null $EVALDIR/out/null

fi
cd $EVALDIR
SCRIPTNAME=$(basename $SCRIPTPATH)
num=0
while [ -f ${SCRIPTNAME}.log.${num} ]; do
    num=$(( $num + 1 ))
done
cp $SCRIPTPATH $SCRIPTNAME.log.${num}

CARE_TESTIDX=0


# ###########################################################

# run_classic ${CARE_TESTIDX} "--pairmode PE --pairedthreshold1 0.06"

# ###########################################################

# for (( i="0"; i<"3"; i+="1" )); do
# 	run_print $i "--pairmode PE --pairedthreshold1 0.06"
# done

# ###########################################################

python3 - <<EOF


import sys
sys.path.append("$MLCDIR")
import care
from tqdm import tqdm
prefixes = [$(printf "\"%s\"," "${prefixes[@]}")]
effiles = [$(printf "\"%s\"," "${files_ef[@]}")]
anchor_map = [{"X":prefix+"_${EVALDIRNAME}_anchor.samples", "y":effile, "np":prefix+"_${EVALDIRNAME}_anchor.npz"} for prefix, effile in zip(prefixes, effiles)]
cands_map = [{"X":prefix+"_${EVALDIRNAME}_cands.samples", "y":effile, "np":prefix+"_${EVALDIRNAME}_cands.npz"} for prefix, effile in zip(prefixes, effiles)]

for i in range(0, 3):
	anchor_map_train, cands_map_train = list(anchor_map), list(cands_map)
	anchor_map_test, cands_map_test = [anchor_map_train.pop(i)], [cands_map_train.pop(i)]
	care._process("RF", {"n_estimators":16, "max_depth":4}, anchor_map_train, anchor_map_test, prefixes[i]+"_${EVALDIRNAME}_anchor.rf")
	care._process("RF", {"n_estimators":16, "max_depth":4}, cands_map_train, cands_map_test, prefixes[i]+"_${EVALDIRNAME}_cands.rf")

EOF

# # ############################################################

for (( i="0"; i<"1"; i+="1" )); do
	echo "starting search" ${i}
	grid_search ${i} ${prefixes[${i}]}_${EVALDIRNAME} 94 94 1 20 20 5 "--maxForestTreesAnchor 4 --maxForestTreesCands 1 --pairmode PE --pairedthreshold1 0.06"
done

############################################################

echo "done."


