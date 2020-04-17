#!/bin/bash

care=/home/fekallen/arbeit/errorcorrector/errorcorrector_gpu

if [ $# -lt 2 ]
then
	echo "Usage: $0 inputfile outputfile coverage [numthreads]"
	exit
fi

inputfile=$1
outputfile=$2
coverage=$3
numthreads=1

if [ $# -gt 3 ]
then
	numthreads=$4
fi

echo "run.sh $inputfile $outputfile $coverage $numthreads"

filename=$(basename -- "$outputfile")
extension="${filename##*.}"
filename="${filename%.*}"

outputfilenamenopath=$filename"."$extension

outputdir=$(dirname $outputfile)
tempdir="/home/fekallen/storage/temp/"

mkdir -p $outputdir
mkdir -p $tempdir

k=20

echo "$care -i $inputfile --tempdir=$tempdir -d $outputdir -o $outputfilenamenopath --threads=$numthreads \
      --hashmaps=48 --kmerlength=$k --batchsize=1000 --maxmismatchratio=0.20 --minalignmentoverlap=20 --minalignmentoverlapratio=0.20 \
      --useQualityScores --coverage=$coverage --errorfactortuning=0.06 --coveragefactortuning=0.6 \
      --candidateCorrection --candidateCorrectionNewColumns=15 --deviceIds=0 \
      --showProgress --nReads=0 --min_length=0 --max_length=0 --memTotal 27G"

$care -i $inputfile --tempdir=$tempdir -d $outputdir -o $outputfilenamenopath --threads=$numthreads \
      --hashmaps=48 --kmerlength=$k --batchsize=1000 --maxmismatchratio=0.20 --minalignmentoverlap=20 --minalignmentoverlapratio=0.20 \
      --useQualityScores --coverage=$coverage --errorfactortuning=0.06 --coveragefactortuning=0.6 \
      --candidateCorrection --candidateCorrectionNewColumns=15 --deviceIds=0 \
      --showProgress --nReads=0 --min_length=0 --max_length=0 --memTotal 27G
