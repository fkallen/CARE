# CARE 2.3
* Faster merge step during construction of output file
* Use CCCL 2.2.0
* Update program options for GPU version
* GPUs are now specified via CUDA_VISIBLE_DEVICES environment variable
* Massive refactoring to improve both single-GPU performance and greatly improve multi-GPU performance
* Bugfixes:
  * Fix potential race in gpu msa
  * Fix memory consumption of gpu hash tables
# CARE 2.2
* CPU / GPU performance improvements
* Add option to specify data layout for hash tables and reads when using multiple gpus
* Add option to specify detailed thread configuration for gpu version
* Bugfixes
* Refactoring
# CARE 2.1
* .gz output
* Avoid some non-deterministic floating-point operations in gpu version
* Faster hash table construction in gpu version
* Improve multi-gpu performance
* Update dependencies
* Bugfixes
* Refactoring
# CARE 2.0
* Add a dedicated code path for paired-end inputs
* Add a Random Forest-based correction mode
* Add an option to compress quality scores
* GPU version can use GPU hashtables
* Big code refactoring
* Performance and memory improvements
