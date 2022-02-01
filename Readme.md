# CARE: Context-Aware Read Error correction for Illumina reads.

## Prerequisites
* C++17 
* OpenMP
* Zlib
* GNU Make

## Additional prerequisites for GPU version
* CUDA Toolkit 11 or newer
* A CUDA-capable graphics card with Pascal architecture (e.g. Nvidia GTX 1080) or newer.

# Download
Clone the repository and initialize submodules: `git clone --recurse-submodules https://github.com/fkallen/CARE.git`


# Build
The build process assumes that the required compilers are available in your PATH.

## Make
Run make to generate the executables.

CPU version: This produces an executable file care-cpu in the top-level directory of CARE
```
make / make cpu
```

GPU version: This produces an executable file care-gpu in the top-level directory of CARE
```
make gpu
```

Optionally, after executables have been built they can be copied to an installation directory via `make install`.
This will copy available executables to the directory PREFIX/bin. The default value for PREFIX is `/usr/local`.
A custom prefix can be set as follows:

```
make install PREFIX=/my/custom/prefix
```



# Run   
The simplest command which only includes mandatory options is

```
./care-cpu -i reads.fastq -d outputdir -o correctedreads.fastq -c 30 --pairmode PE
```

This command will attempt to correct the reads from file reads.fastq, assuming a read coverage of 30. The parameter `--pairmode PE` is used to execute the paired-end correction path.
The outputfile named correctedreads.fastq will be placed in the directory outputdir. The available program parameters are listed below.

Input files must be in fasta or fastq format, and may be gzip'ed. Specifying both fasta files and fastq files together is not allowed.
If the input files are unpaired, the setting `--pairmode SE` must be used, which selects the single-end correction path.
If the input files are paired instead, either `--pairmode SE` or `--pairmode PE` may be used.
Output files will be uncompressed. The order of reads will be preserved. Read headers and quality scores (if fastq) remain unchanged.


# Specifying input files
## Single-end library
For a single-end library consisting of one or more files, repeat argument `-i` for each file

## Paired-end library
A paired-end library must be either a single file in interleaved format, or two files in split format.

### Interleaved
Two consecutive reads form a read pair. Use `-i reads_interleaved` .

### Split
Read number N in file 1 and read number N in file 2 form a read pair. Use `-i reads_1 -i reads_2`.

# Available program parameters
Please execute `./care-cpu --help` or `./care-gpu --help` to print a list of available parameters. Both versions share a common subset of parameters.

The following list is a selection of usefull options.

```
-h, --hashmaps arg            The requested number of hash maps. Must be
                              greater than 0. The actual number of used hash
                              maps may be lower to respect the set memory
                              limit. Default: 48

-t, --threads arg             Maximum number of thread to use. Default: 1

-q, --useQualityScores        If set, quality scores (if any) are
                                considered during read correction. Default: false

--candidateCorrection         If set, candidate reads will be corrected,too. Default: false

-p, --showProgress            If set, progress bar is shown during correction

-m, --memTotal arg            Total memory limit in bytes. Can use suffix
                              K,M,G , e.g. 20G means 20 gigabyte. This option
                              is not a hard limit. Default: All free
                              memory.
```

If an option allows multiple values to be specified, the option can be repeated with different values.
As an alternative, multiple values can be separated by comma (,). Both ways can be used simultaneously.
For example, to specify three single-end input files the following options are equivalent:

```
-i file1,file2,file3
-i file1 -i file2 -i file3
-i file1 -i file2,file3
```



# Algorithm

This work is presented in the following paper.

Felix Kallenborn, Andreas Hildebrandt, Bertil Schmidt, CARE: Context-Aware Sequencing Read Error Correction, Bioinformatics, , btaa738, [https://doi.org/10.1093/bioinformatics/btaa738](https://doi.org/10.1093/bioinformatics/btaa738)



