#!/bin/bash

files_ef[1]=/share/errorcorrection/datasets/arthiseq2000humanchr15/humanchr1530cov_errFree.fq
files_ef[2]=/share/errorcorrection/datasets/arthiseq2000athaliana/athaliana30cov_errFree.fq
files_ef[3]=/share/errorcorrection/datasets/arthiseq2000elegans/elegans30cov_errFree.fq
files_ef[4]=/share/errorcorrection/datasets/arthiseq2000melanogaster/melanogaster30cov_errFree.fq
files_ef[5]=/share/errorcorrection/datasets/arthiseq2000mus/mus_chr15_30cov_errFree.fq

for (( i="1"; i<="5"; i+="1" )); do
	scp monster2:${files_ef[${i}]} /home/jc/care
done
