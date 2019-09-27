#!/bin/bash




if [ $# -lt 3 ]
then
	echo "Usage: ./run2.sh exetype datainfofile outputfolder [num_threads candidatecorrection extractfeatures]"
	exit
fi

executable=./errorcorrector_$1
datainfofile=$2
outdir=$3

#outdir=$datapath/correctedcandidates/

#max number of threads to use
threads=16

#max number of threads which use a gpu
threadsgpu=0

candidateCorrection=true
candidateCorrectionNewColumns=15
extractFeatures=false

if [ $# -gt 3 ]
then
	threads=$4
fi

if [ $# -gt 4 ]
then
	candidateCorrection=$5
fi

if [ $# -gt 5 ]
then
	extractFeatures=$6
fi


if $extractFeatures;
then
	candidateCorrection=false
fi

deviceIds="--deviceIds=0"
datapath=/home/fekallen/storage/evaluationtool
nowstring=$(date +"%Y-%m-%d_%H-%M-%S")

#0: Classic, 1: Forest, 2: Convnet
correctionType=0

forest="./forests/combinedforestaligncov.so"
#forest="./forests/combinedecolielegansmelanogaster_clipmin_forest.so"
#forest="./forests/combinedforestwithconsensus.so"
#forest="./forests/C.elegans_SRX218989_with_consforest.so"
#forest="./forests/eleganslabeledsimple.so"
nnmodel="./nn_models/conv_ele736/model_conv_03-27_0006.ckpt"

fileformat=fastq
useQualityScores=true


batchsize=1000
showProgress=true

#minhashing parameters
#kmer length
k=16
#hashmaps (one kmer hash value per map)
maps=48

num_hits=1

#properties of good alignment
maxmismatchratio=0.20
#minimum overlap size
minalignmentoverlap=30
#minimum relative overlap size
minalignmentoverlapratio=0.30

IFS=$'\n' array=($(cat $datainfofile ))

inputfile=${array[0]}${array[1]}
num_reads=${array[2]}
coverage=${array[3]}
max_readlength=${array[4]}
min_readlength=0
#candidates=${array[5]}
candidates=$(echo "($maps * 2.5 * $coverage)/1" | bc)
#candidates=$(($maps * 2.5 * $coverage))

bin_reads=${array[8]}${array[9]}
bin_tables=${array[8]}${array[10]}

errorrate=0.07
m=0.6

inputfilename=$(basename -- "$inputfile")
inputfileextension="${inputfilename##*.}"
outputfilename="${inputfilename%.*}_"$nowstring"."$inputfileextension


echo $inputfile
echo $executable --fileformat=$fileformat --inputfile=$inputfile --outdir=$outdir --outfile=$outputfilename --threads=$threads\
                 --hashmaps=$maps --kmerlength=$k --batchsize=$batchsize \
                 --maxmismatchratio=$maxmismatchratio --minalignmentoverlap=$minalignmentoverlap --minalignmentoverlapratio=$minalignmentoverlapratio\
                 --useQualityScores=$useQualityScores --coverage=$coverage --errorrate=$errorrate --m_coverage=$m --candidateCorrection=$candidateCorrection\
                 --candidateCorrectionNewColumns=$candidateCorrectionNewColumns\
                 --extractFeatures=$extractFeatures $deviceIds --correctionType=$correctionType --maxCandidates=$candidates --progress=$showProgress\
                 --nReads=$num_reads --min_length=$min_readlength --max_length=$max_readlength --hits_per_candidate=$num_hits --forest=$forest\
		 --nnmodel=$nnmodel\
                 #--load-binary-reads-from=$bin_reads # --load-hashtables-from=$bin_tables

$executable --fileformat=$fileformat --inputfile=$inputfile --outdir=$outdir --outfile=$outputfilename --threads=$threads\
                 --hashmaps=$maps --kmerlength=$k --batchsize=$batchsize \
                 --maxmismatchratio=$maxmismatchratio --minalignmentoverlap=$minalignmentoverlap --minalignmentoverlapratio=$minalignmentoverlapratio\
                 --useQualityScores=$useQualityScores --coverage=$coverage --errorrate=$errorrate --m_coverage=$m --candidateCorrection=$candidateCorrection\
                 --candidateCorrectionNewColumns=$candidateCorrectionNewColumns\
                 --extractFeatures=$extractFeatures $deviceIds --correctionType=$correctionType --maxCandidates=$candidates --progress=$showProgress\
                 --nReads=$num_reads --max_length=$max_readlength --hits_per_candidate=$num_hits --forest=$forest\
		 --nnmodel=$nnmodel\
                 #--load-binary-reads-from=$bin_reads # --load-hashtables-from=$bin_tables
