# CARE: Context-Aware Read Error correction for Illumina reads.

## Prerequisites
* GCC with c++14 support
* OpenMP
* Zlib
* GNU Make

# Additional prerequisites for GPU version
* CUDA Toolkit 10 or newer
* A CUDA capable Pascal or Volta card. Other cards may work, but have not been tested.
* CUB Version 1.8.0 or newer.
* Thrust 1.9 or newer. Thrust is shipped together with the CUDA Toolkit



## Build
CPU version: This produces executable file errorcorrector_cpu
```
make / make cpu
```

GPU version: This produces executable file errorcorrector_gpu
```
make gpu
```

# Run   
The simplest command which only includes mandatory options is

```
./errorcorrector_cpu -i reads.fastq -d outputdir -o correctedreads.fastq -c 30 
```

This command will attempt to correct the reads from file reads.fastq, assuming a read coverage of 30.
The outputfile named correctedreads.fastq will be placed in the directory outputdir.

Available program options:
```
 Mandatory options:
  -d, --outdir arg           The output directory. Will be created if it does
                             not exist yet
  -c, --coverage arg         Estimated coverage of input file. (i.e.
                             number_of_reads * read_length / genome_size)
  -i, --inputfiles arg       The file(s) to correct. Fasta or Fastq format.
                             May be gzip'ed. Repeat this option for each input
                             file (e.g. -i file1.fastq -i file2.fastq). Must
                             not mix fasta and fastq files. Input files are
                             treated as unpaired.
  -o, --outputfilenames arg  The names of outputfiles. Repeat this option for
                             each output file (e.g. -o file1_corrected.fastq
                             -o file2_corrected.fastq). If a single output
                             file is specified, it will contain the concatenated
                             results of all input files. If multiple output
                             files are specified, the number of output files
                             must be equal to the number of input files. In this
                             case, output file i will contain the results of
                             input file i. Output files are uncompressed.

 Optional options:
  -h, --help                    Show this help message

      --tempdir arg             Directory to store temporary files. Default
                                is output directory
      --hashmaps arg            The number of hash maps. Must be greater than
                                0.
      --kmerlength arg          The kmer length for minhashing. Must be
                                greater than 0.
      --threads arg             Maximum number of thread to use. Must be
                                greater than 0
      --batchsize arg           Number of reads to correct in a single batch.
                                Must be greater than 0.
      --useQualityScores        If set, quality scores (if any) are
                                considered during read correction
      --candidateCorrection     If set, candidate reads will be
                                corrected,too.
      --candidateCorrectionNewColumns arg
                                If candidateCorrection is set, a candidates
                                with an absolute shift of
                                candidateCorrectionNewColumns compared to anchor are corrected
      --maxmismatchratio arg    Overlap between anchor and candidate must
                                contain at most maxmismatchratio * overlapsize
                                mismatches
      --minalignmentoverlap arg
                                Overlap between anchor and candidate must be
                                at least this long
      --minalignmentoverlapratio arg
                                Overlap between anchor and candidate must be
                                at least as long as minalignmentoverlapratio *
                                querylength
      --errorfactortuning arg   errorfactortuning
      --coveragefactortuning arg
                                coveragefactortuning
      --nReads arg              Upper bound for number of reads in the
                                inputfile. If missing or set 0, the input file is
                                parsed to find the exact number of reads before
                                any work is done.
      --min_length arg          Lower bound for read length in file. If
                                missing or set negative, the input file is parsed
                                to find the exact minimum length before any
                                work is done.
      --max_length arg          Upper bound for read length in file. If
                                missing or set 0, the input file is parsed to find
                                the exact maximum length before any work is
                                done.
      --showProgress            If set, progress bar is shown during
                                correction
      --save-preprocessedreads-to arg
                                Save binary dump of data structure which
                                stores input reads to disk
      --load-preprocessedreads-from arg
                                Load binary dump of read data structure from
                                disk
      --save-hashtables-to arg  Save binary dump of hash tables to disk
      --load-hashtables-from arg
                                Load binary dump of hash tables from disk
      --memHashtables arg       Memory limit in bytes for hash tables and
                                hash table construction. Can use suffix K,M,G ,
                                e.g. 20G means 20 gigabyte. This option is not
                                a hard limit. Default: A bit less than
                                memTotal.
      --memTotal arg            Total memory limit in bytes. Can use suffix
                                K,M,G , e.g. 20G means 20 gigabyte. This option
                                is not a hard limit. Default: All free
                                memory.
```

# Algorithm
Please refer to the description in the paper: 



