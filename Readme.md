# CARE 2.0: Context-Aware Read Error correction for Illumina reads.

## Prerequisites
* C++17 
* OpenMP
* Zlib
* GNU Make
* Python 3.8 (optional. See section 'Forests for CARE 2.0')

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

A more advanced usage could look like the following command. It enables progress counter `-p` and uses quality scores `-q` which are stored in a lossy compressed 2-bit format `--qualityScoreBits 2`. The program should use 16 threads `-t 16` with a memory limit of 22 gigabyte `-m 22G`. Sequences which contain other letters than A,C,G,T, e.g. N, will be skipped `--excludeAmbiguous`. `-k` and `-h` specify the parameters of the hashing, namely the k-mer size and the number of hash tables. With `--candidateCorrection`, additional sequence corrections may be computed per read which are then used to either accept or reject the primary correction. This can improve correction quality (reduces FP, but also TP) at the expense of greater memory usage to store the additional corrections.

```
./care-cpu -i reads.fastq -d . -o correctedreads.fastq -c 30 --pairmode PE -p -q --qualityScoreBits 2 --excludeAmbiguous -m 22G -t 16 -k 20 -h 32 --candidateCorrection
```

The equivalent execution of the GPU version using two GPUs would be:

```
./care-gpu -i reads.fastq -d . -o correctedreads.fastq -c 30 --pairmode PE -p -q --qualityScoreBits 2 --excludeAmbiguous -m 22G -t 16 -k 20 -h 32 --candidateCorrection -g 0,1
```
Note the additional mandatory parameter `-g` which accepts a comma-separated list of integers to indicate which GPUs can be used. The integers must be between 0 and N-1, where N is the number of available GPUs in the system.


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

## Forests for CARE 2.0
CARE 2.0 introduces a Random-Forest-based error correction mode. To use this mode, trained random forest classifiers need to be supplied using the program parameters
`--correctionType 1 --ml-forestfile anchorforest.rf ` and `--correctionTypeCands 1 --ml-cands-forestfile candsforest.rf` for anchor correction and candidate correction, respectively.
The same forest files can be used for both the CPU version and the GPU version.

A small collection of pre-trained forests is available [here](https://seafile.rlp.net/d/e784b6f809a240d095c8/)

For more information about training the random forests please see the descriptions in [ml/readme.md](ml/readme.md)


# Algorithm

This work is presented in the following paper.

Felix Kallenborn, Andreas Hildebrandt, Bertil Schmidt, CARE: Context-Aware Sequencing Read Error Correction, Bioinformatics, , btaa738, [https://doi.org/10.1093/bioinformatics/btaa738](https://doi.org/10.1093/bioinformatics/btaa738)



