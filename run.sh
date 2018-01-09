#!/bin/bash


#gpu version does NOT work for long reads at the moment. we tested with reads of fixed length 100
executable=./errorcorrector_gpu
#executable=./errorcorrector_cpu

#number of threads to hash the reads
ithreads=8

#number of threads to correct reads. in gpu version, when using N gpus each gpu will be used by cthreads / N threads
cthreads=8

#input file
#inputfile=/ssd/fkallenb/eccomparison/datasets/E.coli_SRR1191655_1M.fastq
#inputfile=/ssd/fkallenb/eccomparison/datasets/E.coli_SRR1191655.fastq
#inputfile=/ssd/fkallenb/eccomparison/correcteddatasets2/readscorrected.fq
inputfile=/ssd/fkallenb/eccomparison/datasets/C.elegans_SRX218989.fastq

#output path. this is used as temporary storage, too
outdir=/ssd/fkallenb/eccomparison/correcteddatasets2/

#output file
outputfile="readscorrected.fq"

#absolute output file path = outdir/outputfile
#if --outfile is not used, absolute path will be outdir/inputfilenameWithoutEnding+_#k_#maps_1_alpha_#alpha_x_#x_corrected
outfile="--outfile $outputfile"
#outfile=

# fasta or fastq
fileformat=fastq 

#only valid for fastq fileformat
useQualityScores=--useQualityScores
#useQualityScores=

#minhashing parameters
#kmer length
k=16
#hashmaps (one kmer hash value per map)
maps=4

#alignment scores for semiglobal alignment
#we use a high indel penalty to focus on substitutions only. you may want to change this to include indel correction
matchscore=1
subscore=-1
insertscore=-100
deletionscore=-100

#batchsize reads are aligned simultaneously per thread. batchsize > 1 is useful for gpu alignment to increase gpu utilization
batchsize=3

#properties of good alignment
#overlap must contain minimum of 80% matches
maxmismatchratio=0.10
#minimum overlap size
minalignmentoverlap=35
#minimum relative overlap size
minalignmentoverlapratio=0.35

#correction parameters
#during the voting phase, if at a fixed position in the read base B from the original read occurs N times and base D occurs M times,
#then B is corrected into D if  M-N >= aa*pow(xx,N)
xx=$(echo 'scale=2; 15/10' | bc)
aa=$(echo 'scale=2; 10/10' | bc)


echo $executable --fileformat=$fileformat --inputfile $inputfile --outdir $outdir $outfile --hashmaps $maps --kmerlength $k --insertthreads $ithreads --correctorthreads $cthreads --batchsize $batchsize -x $xx -a $aa --matchscore=$matchscore --subscore=$subscore --insertscore=$insertscore --deletionscore=$deletionscore --maxmismatchratio=$maxmismatchratio --minalignmentoverlap=$minalignmentoverlap --minalignmentoverlapratio=$minalignmentoverlapratio $useQualityScores

time $executable --fileformat=$fileformat --inputfile $inputfile --outdir $outdir $outfile --hashmaps $maps --kmerlength $k --insertthreads $ithreads --correctorthreads $cthreads --batchsize $batchsize -x $xx -a $aa --matchscore=$matchscore --subscore=$subscore --insertscore=$insertscore --deletionscore=$deletionscore --maxmismatchratio=$maxmismatchratio --minalignmentoverlap=$minalignmentoverlap --minalignmentoverlapratio=$minalignmentoverlapratio $useQualityScores


