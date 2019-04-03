#!/bin/bash




if [ $# -lt 2 ]
then
	echo "Usage: ./run2.sh exetype datainfofile [num_threads num_threads_for_gpus]"
	exit
fi

executable=./errorcorrector_$1
datainfofile=$2

#max number of threads to use
threads=16

#max number of threads which use a gpu
threadsgpu=0

candidateCorrection=true
extractFeatures=false

if [ $# -gt 2 ]
then
	threads=$3
fi

if [ $# -gt 3 ]
then
	threadsgpu=$4
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
classicMode=false
forest="./forests/combinedforestaligncov.so"
nnmodel="./nn_models/model_conv_04-01_0006.ckpt"
fileformat=fastq
useQualityScores=true


batchsize=50000
showProgress=true

#minhashing parameters
#kmer length
k=16
#hashmaps (one kmer hash value per map)
maps=8

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
candidates=${array[5]}

bin_reads=${array[8]}${array[9]}
bin_tables=${array[8]}${array[10]}

errorrate=0.03
m=0.6

outdir=$datapath/correcteddatasets/

#output file
#outputfile="readscorrectednew.fq"
outputfile=$nowstring"_"${array[1]}


echo $inputfile
echo $executable --fileformat=$fileformat --inputfile=$inputfile --outdir=$outdir --outfile=$outputfile --threads=$threads\
                 --threadsForGPUs=$threadsgpu --hashmaps=$maps --kmerlength=$k --batchsize=$batchsize \
                 --maxmismatchratio=$maxmismatchratio --minalignmentoverlap=$minalignmentoverlap --minalignmentoverlapratio=$minalignmentoverlapratio\
                 --useQualityScores=$useQualityScores --coverage=$coverage --errorrate=$errorrate --m_coverage=$m --candidateCorrection=$candidateCorrection\
                 --extractFeatures=$extractFeatures $deviceIds --classicMode=$classicMode --maxCandidates=$candidates --progress=$showProgress\
                 --nReads=$num_reads --max_length=$max_readlength --hits_per_candidate=$num_hits --forest=$forest\
		 --nnmodel=$nnmodel\
                 --load-binary-reads-from=$bin_reads --load-hashtables-from=$bin_tables

$executable --fileformat=$fileformat --inputfile=$inputfile --outdir=$outdir --outfile=$outputfile --threads=$threads\
                 --threadsForGPUs=$threadsgpu --hashmaps=$maps --kmerlength=$k --batchsize=$batchsize \
                 --maxmismatchratio=$maxmismatchratio --minalignmentoverlap=$minalignmentoverlap --minalignmentoverlapratio=$minalignmentoverlapratio\
                 --useQualityScores=$useQualityScores --coverage=$coverage --errorrate=$errorrate --m_coverage=$m --candidateCorrection=$candidateCorrection\
                 --extractFeatures=$extractFeatures $deviceIds --classicMode=$classicMode --maxCandidates=$candidates --progress=$showProgress\
                 --nReads=$num_reads --max_length=$max_readlength --hits_per_candidate=$num_hits --forest=$forest\
		 --nnmodel=$nnmodel\
                 --load-binary-reads-from=$bin_reads --load-hashtables-from=$bin_tables
