#!/bin/bash


#gpu version does NOT work for long reads at the moment. we tested with reads of fixed length 100
executable=./errorcorrector_gpu
#executable=./errorcorrector_cpu

#max number of threads to use. in gpu version, when using N gpus each gpu will be used by threads / N threads
threads=16

#input file
#inputfile=/home/fekallen/arbeit/evaluationtool/datasets/E.coli_SRR1191655_1M.fastq
#coverage=21

inputfile=/home/fekallen/arbeit/evaluationtool/datasets/E.coli_SRR1191655.fastq
coverage=255

#inputfile=/home/fekallen/arbeit/evaluationtool/datasets/C.elegans_SRX218989.fastq
#coverage=31

#inputfile=/home/fekallen/arbeit/evaluationtool/datasets/D.melanogaster_SRR823377.fastq
#coverage=52

#estimated error rate
errorrate=0.03
m=0.6

#output path. this is used as temporary storage, too
outdir=/home/fekallen/arbeit/evaluationtool/correcteddatasets2/

#output file
outputfile="readscorrectednew.fq"
#outputfile="ecolisrr11_m06_e001_h8_k16_hq_qscores.fq"

#absolute output file path = outdir/outputfile
outfile="--outfile=$outputfile"
#outfile=

#fastq
fileformat=fastq

#only valid for fastq fileformat
useQualityScores=--useQualityScores=true
#useQualityScores=

candidateCorrection=--candidateCorrection=true

#if indels should be corrected, too
indels=--indels=true

#minhashing parameters
#kmer length
k=16
#hashmaps (one kmer hash value per map)
maps=8

#alignment scores for semiglobal alignment. only used if indels=true.
#we use a high indel penalty to focus on substitutions only. you may want to change this to include indel correction
matchscore=1
subscore=-1
insertscore=-100
deletionscore=-100

#batchsize reads are aligned simultaneously per thread. batchsize > 1 is useful for gpu alignment to increase gpu utilization
batchsize=3

#properties of good alignment
maxmismatchratio=0.20
#minimum overlap size
minalignmentoverlap=30
#minimum relative overlap size
minalignmentoverlapratio=0.30

#correction parameters
#during the voting phase, if at a fixed position in the read base B from the original read occurs N times and base D occurs M times,
#then B is corrected into D if  M-N >= aa*pow(xx,N)
xx=$(echo 'scale=2; 12/10' | bc)
aa=$(echo 'scale=2; 10/10' | bc)


echo $executable --fileformat=$fileformat --inputfile=$inputfile --outdir=$outdir $outfile --threads=$threads $indels --hashmaps=$maps --kmerlength=$k --batchsize=$batchsize --base=$xx --alpha=$aa --matchscore=$matchscore --subscore=$subscore --insertscore=$insertscore --deletionscore=$deletionscore --maxmismatchratio=$maxmismatchratio --minalignmentoverlap=$minalignmentoverlap --minalignmentoverlapratio=$minalignmentoverlapratio $useQualityScores --coverage=$coverage --errorrate=$errorrate --m_coverage=$m $candidateCorrection

time $executable --fileformat=$fileformat --inputfile=$inputfile --outdir=$outdir $outfile --threads=$threads $indels --hashmaps=$maps --kmerlength=$k --batchsize=$batchsize --base=$xx --alpha=$aa --matchscore=$matchscore --subscore=$subscore --insertscore=$insertscore --deletionscore=$deletionscore --maxmismatchratio=$maxmismatchratio --minalignmentoverlap=$minalignmentoverlap --minalignmentoverlapratio=$minalignmentoverlapratio $useQualityScores --coverage=$coverage --errorrate=$errorrate --m_coverage=$m $candidateCorrection
