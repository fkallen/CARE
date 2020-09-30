# Evaluation of corrected simulated reads

The program arteval is used to collect per-nucleotide correction metrics given 
the uncorrected reads, error free reads, and corrected reads. Reads must appear in the same order in every file.

artevalsmallercorrected additionally allows some reads to be missing in the corrected file. However, the read order must 
still be preserved.

Run make arteval to build.

# Lost true k-mers statistics

Requires Jellyfish: https://github.com/gmarcais/Jellyfish

Requires seqan3 library: https://github.com/seqan/seqan3

The path to the seqan3 library on your system must be set manually in the Makefile

Run make kmertools to build

``` 
jellyfish count -m 21 -C -s 100M -t 16 uncorrectedreads.fasta
jellyfish dump mer_counts.jf > uncorrectedkmers.fasta

./findkmersingenome uncorrectedkmers.fasta 21 genomefile.fasta uncorrectedkmersInGenome.fasta

jellyfish count -m 21 -C -s 100M -t 16 tool_correctedreads.fasta
jellyfish dump mer_counts.jf > tool_correctedkmers.fasta

./findmissingkmers uncorrectedkmersInGenome.fasta tool_correctedreads.fasta 21 10
```