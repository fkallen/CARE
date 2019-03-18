#!/bin/bash

datapath=/home/fekallen/storage/evaluationtool
outdir=$datapath/correcteddatasets/

outputfile="readscorrectednew.fq"

count=0

for indexfile in ~/storage/evaluationtool/index/*; do
	if [ $count -gt 0 ]
	then
		./run2.sh gpu $indexfile 16 1 false true

		IFS=$'\n' array=($(cat $indexfile ))

		inputfilename=${array[1]}
	
		rm $outdir$outputfile
		mv $outdir$outputfile"_features" $outdir/newfeatures/$inputfilename"_features"

	fi

	count=$(($count+1))
done
