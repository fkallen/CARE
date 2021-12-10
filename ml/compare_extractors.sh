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
	echo ""
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
		--candidateCorrection --correctionTypeCands 2 --ml-cands-forestfile ${prefixes[${1}]}_${EVALDIRNAME}_cands.samples ${2} \
		$(auto_preprocess ${prefixes[${1}]})
}

run_clf() {
	$CAREGPU -i ${files[${1}]} -c ${cov[${1}]} -o out/${prefixes[${1}]}_${2}-${3}-${4}${6}.fq $CARE_FLAGS \
		--correctionType 1 --ml-forestfile ${2}_anchor.rf \
		--candidateCorrection --correctionTypeCands 1 --ml-cands-forestfile ${2}_cands.rf \
		--thresholdAnchor ${3} --thresholdCands ${4} ${5} \
		$(auto_preprocess ${prefixes[${1}]})

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
EVALDIR=/home/jcascitt/care/${EVALDIRNAME}

CARE=/home/jcascitt/errorcorrector/care-cpu
CAREGPU="/home/jcascitt/errorcorrector/care-gpu -g 0 --warpcore 1"
CARE_FLAGS="-d . -h 48 -q --excludeAmbiguous --minalignmentoverlap 30 --minalignmentoverlapratio 0.3 -m 64G -p -t 128 --samplingRateAnchor 0.25 --samplingRateCands 0.01 --enforceHashmapCount"

ARTEVAL=/home/jcascitt/errorcorrector/evaluationtool/arteval

### files

files[0]=/share/errorcorrection/datasets/arthiseq2000humanchr14/humanchr1430cov.fq
files_ef[0]=/share/errorcorrection/datasets/arthiseq2000humanchr14/humanchr1430cov_errFree.fq
cov[0]=30
prefixes[0]=humanchr14-30

files[1]=/share/errorcorrection/datasets/arthiseq2000humanchr15/humanchr1530cov.fq
files_ef[1]=/share/errorcorrection/datasets/arthiseq2000humanchr15/humanchr1530cov_errFree.fq
cov[1]=30
prefixes[1]=humanchr15-30

files[2]=/share/errorcorrection/datasets/arthiseq2000athaliana/athaliana30cov.fq
files_ef[2]=/share/errorcorrection/datasets/arthiseq2000athaliana/athaliana30cov_errFree.fq
cov[2]=30
prefixes[2]=atha-30

files[3]=/share/errorcorrection/datasets/arthiseq2000elegans/elegans30cov.fq
files_ef[3]=/share/errorcorrection/datasets/arthiseq2000elegans/elegans30cov_errFree.fq
cov[3]=30
prefixes[3]=ele-30

files[4]=/share/errorcorrection/datasets/arthiseq2000melanogaster/melanogaster30cov.fq
files_ef[4]=/share/errorcorrection/datasets/arthiseq2000melanogaster/melanogaster30cov_errFree.fq
cov[4]=30
prefixes[4]=melan-30

files[5]=/share/errorcorrection/datasets/arthiseq2000mus/mus_chr15_30cov.fq
files_ef[5]=/share/errorcorrection/datasets/arthiseq2000mus/mus_chr15_30cov_errFree.fq
cov[5]=30
prefixes[5]=muschr15-30

###########################################################

spack load cuda@11.5.0

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

# ###########################################################

# for (( i="0"; i<="5"; i+="1" )); do
#     run_print $i "--pairmode SE"
# done

###########################################################

python3 - <<EOF
import sys
sys.path.append("$MLCDIR")
import care
from tqdm import tqdm
prefixes = [$(printf "\"%s\"," "${prefixes[@]}")]
effiles = [$(printf "\"%s\"," "${files_ef[@]}")]
anchor_map = [{"X":prefix+"_${EVALDIRNAME}_anchor.samples", "y":effile, "np":prefix+"_${EVALDIRNAME}_anchor.npz"} for prefix, effile in zip(prefixes, effiles)]
cands_map = [{"X":prefix+"_${EVALDIRNAME}_cands.samples", "y":effile, "np":prefix+"_${EVALDIRNAME}_cands.npz"} for prefix, effile in zip(prefixes, effiles)]

NJOBS = 100
for i in tqdm(range(len(prefixes)), total=len(prefixes), miniters=1, mininterval=0, leave=False):
	anchor_map_train, cands_map_train = list(anchor_map), list(cands_map)
	anchor_map_test, cands_map_test = [anchor_map_train.pop(i)], [cands_map_train.pop(i)]

	care.process(care.RandomForestClassifier, {"n_jobs":NJOBS}, anchor_map_train, anchor_map_test, prefixes[i]+"_${EVALDIRNAME}_anchor")
	care.process(care.RandomForestClassifier, {"n_jobs":NJOBS}, cands_map_train, cands_map_test, prefixes[i]+"_${EVALDIRNAME}_cands")
	
	# care.process(care.RandomForestClassifier, {"n_jobs":NJOBS, "max_depth":22}, anchor_map_train, anchor_map_test, prefixes[i]+"_${EVALDIRNAME}_md22_anchor")
	# care.process(care.RandomForestClassifier, {"n_jobs":NJOBS, "max_depth":22}, cands_map_train, cands_map_test, prefixes[i]+"_${EVALDIRNAME}_md22_cands")

EOF

############################################################

for (( i="0"; i<="5"; i+="1" )); do
    grid_search $i ${prefixes[${i}]}_${EVALDIRNAME} 10 90 10 10 90 10 "--pairmode SE"
done

############################################################

echo "done."


