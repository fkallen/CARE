# CARE: Context-Aware Read Error correction for Illumina reads.

## Prerequisites
* GCC with c++14 support
* OpenMP
* Zlib
* GNU Make
* Thrust 1.9 or newer.

# Additional prerequisites for GPU version
* CUDA Toolkit 10 or newer
* A CUDA capable Pascal or Volta card. Other cards may work, but have not been tested.
* CUB Version 1.8.0 or newer.
* Thrust 1.9 or newer. Thrust is shipped together with the CUDA Toolkit



## Build
First, run the configure script to specify include paths and directories.
``` 
./configure --help

    --prefix=PREFIX          make install will copy executables to PREFIX/bin/ [/usr/local]
    --with-cuda-dir=DIR      The installation directory of the CUDA toolkit. [/usr/local/cuda/]
    --with-cub-incdir=DIR    use the copy of CUB in DIR. DIR/cub/cub.cuh must exist [/usr/local/cuda/include/]
    --with-thrust-incdir=DIR use the copy of THRUST in DIR. DIR/thrust/version.h must exist [/usr/local/cuda/include/]
```

Then, run make to generate the executables.

CPU version: This produces executable file care-cpu in the top-level directory of care
```
make / make cpu
```

GPU version: This produces executable file care-gpu in the top-level directory of care
```
make gpu
```

Optionally, after executables have been built they can be copied to the installation directory via make install

# Run   
The simplest command which only includes mandatory options is

```
./care-cpu -i reads.fastq -d outputdir -o correctedreads.fastq -c 30 
```

This command will attempt to correct the reads from file reads.fastq, assuming a read coverage of 30.
The outputfile named correctedreads.fastq will be placed in the directory outputdir.

Available program parameters:
```
 Mandatory options:
  -d, --outdir arg           The output directory. Will be created if it does
                             not exist yet.
  -c, --coverage arg         Estimated coverage of input file. (i.e.
                             number_of_reads * read_length / genome_size)
  -i, --inputfiles arg       The file(s) to correct. Fasta or Fastq format.
                             May be gzip'ed. Repeat this option for each input
                             file (e.g. -i file1.fastq -i file2.fastq). Must
                             not mix fasta and fastq files. Input files are
                             treated as unpaired. The collection of input files
                             is treated as a single read library.
  -o, --outputfilenames arg  The names of outputfiles. Repeat this option for
                             each output file (e.g. -o file1_corrected.fastq
                             -o file2_corrected.fastq). If a single output
                             file is specified, it will contain the concatenated
                             results of all input files. If multiple output
                             files are specified, the number of output files
                             must be equal to the number of input files. In this
                             case, output file i will contain the results of
                             input file i. Output files are uncompressed.

 Additional options:
      --help                    Show this help message
      --tempdir arg             Directory to store temporary files. Default:
                                output directory
  -h, --hashmaps arg            The requested number of hash maps. Must be
                                greater than 0. The actual number of used hash
                                maps may be lower to respect the set memory
                                limit. Default: 48
  -k, --kmerlength arg          The kmer length for minhashing. If 0 or
                                missing, it is automatically determined.
      --enforceHashmapCount     If the requested number of hash maps cannot
                                be fullfilled, the program terminates without
                                error correction. Default: false
  -t, --threads arg             Maximum number of thread to use. Must be
                                greater than 0
      --batchsize arg           Number of reads to correct in a single batch.
                                Must be greater than 0. In CARE CPU, one
                                batch per thread is used. In CARE GPU, two batches
                                per GPU are used. Default: 1000
  -q, --useQualityScores        If set, quality scores (if any) are
                                considered during read correction. Default: false
      --excludeAmbiguous        If set, reads which contain at least one
                                ambiguous nucleotide will not be corrected.
                                Default: false
      --candidateCorrection     If set, candidate reads will be
                                corrected,too. Default: false
      --candidateCorrectionNewColumns arg
                                If candidateCorrection is set, a candidates
                                with an absolute shift of
                                candidateCorrectionNewColumns compared to anchor are corrected.
                                Default: 15
      --maxmismatchratio arg    Overlap between anchor and candidate must
                                contain at most (maxmismatchratio * overlapsize)
                                mismatches. Default: 0.200000
      --minalignmentoverlap arg
                                Overlap between anchor and candidate must be
                                at least this long. Default: 20
      --minalignmentoverlapratio arg
                                Overlap between anchor and candidate must be
                                at least as long as (minalignmentoverlapratio
                                * candidatelength). Default: 0.200000
      --errorfactortuning arg   errorfactortuning. Default: 0.060000
      --coveragefactortuning arg
                                coveragefactortuning. Default: 0.600000
  -g, --gpu arg                 One or more GPU device ids to be used for
                                correction. When running the CARE GPU, at least
                                one valid device id is required.
      --nReads arg              Upper bound for number of reads in the
                                inputfile. If missing or set 0, the input file is
                                parsed to find the exact number of reads before
                                any work is done.
      --min_length arg          Lower bound for read length in file. If
                                missing or set 0, the input file is parsed to find
                                the exact minimum length before any work is
                                done.
      --max_length arg          Upper bound for read length in file. If
                                missing or set 0, the input file is parsed to find
                                the exact maximum length before any work is
                                done.
  -p, --showProgress            If set, progress bar is shown during
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
  -m, --memTotal arg            Total memory limit in bytes. Can use suffix
                                K,M,G , e.g. 20G means 20 gigabyte. This option
                                is not a hard limit. Default: All free
                                memory.

```

If an option allows multiple values to be specified, the option can be repeated with different values.
As an alternative, multiple values can be separated by comma (,). Both ways can be used simulatneously.
For example, to specify three input files the following options are equivalent:

```
-i file1,file2,file3
-i file1 -i file2 -i file3
-i file1 -i file2,file3
```

# Algorithm
Please refer to the description in the paper: 



