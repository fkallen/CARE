#include <gpu/minhashkernels.hpp>
#include <gpu/nvtxtimelinemarkers.hpp>
#include <gpu/kernellaunch.hpp>
#include <hpc_helpers.cuh>
#include <gpu/utility_kernels.cuh>
#include <config.hpp>

#include <sequence.hpp>

#include <nvToolsExt.h>

#include <cub/cub.cuh>
#include <cub/iterator/discard_output_iterator.cuh>

#include <algorithm>
#include <cassert>
#include <vector>


namespace care{




    // HASHING


    __global__
    void minhashSignaturesKernel(
            std::uint64_t* __restrict__ signatures,
            size_t signaturesRowPitchElements,
            const unsigned int* __restrict__ sequences2Bit,
            size_t sequenceRowPitchElements,
            int numSequences,
            const int* __restrict__ sequenceLengths,
            int k,
            int numHashFuncs,
            int firstHashFunc){
                
        //constexpr int blocksize = 128;
        constexpr int maximum_kmer_length = 32;

        auto murmur3_fmix = [](std::uint64_t x) {
            x ^= x >> 33;
            x *= 0xff51afd7ed558ccd;
            x ^= x >> 33;
            x *= 0xc4ceb9fe1a85ec53;
            x ^= x >> 33;
            return x;
        };

        auto make_reverse_complement = [](std::uint64_t s){
            s = ((s >> 2)  & 0x3333333333333333ull) | ((s & 0x3333333333333333ull) << 2);
            s = ((s >> 4)  & 0x0F0F0F0F0F0F0F0Full) | ((s & 0x0F0F0F0F0F0F0F0Full) << 4);
            s = ((s >> 8)  & 0x00FF00FF00FF00FFull) | ((s & 0x00FF00FF00FF00FFull) << 8);
            s = ((s >> 16) & 0x0000FFFF0000FFFFull) | ((s & 0x0000FFFF0000FFFFull) << 16);
            s = ((s >> 32) & 0x00000000FFFFFFFFull) | ((s & 0x00000000FFFFFFFFull) << 32);
            return ((std::uint64_t)(-1) - s) >> (8 * sizeof(s) - (32 << 1));
        };

        const int tid = threadIdx.x + blockIdx.x * blockDim.x;

        if(tid < numSequences * numHashFuncs){
            const int mySequenceIndex = tid / numHashFuncs;
            const int myNumHashFunc = tid % numHashFuncs;
            const int hashFuncId = myNumHashFunc + firstHashFunc;

            const unsigned int* const mySequence = sequences2Bit + mySequenceIndex * sequenceRowPitchElements;
            const int myLength = sequenceLengths[mySequenceIndex];

            std::uint64_t* const mySignature = signatures + mySequenceIndex * signaturesRowPitchElements;

            std::uint64_t minHashValue = std::numeric_limits<std::uint64_t>::max();

            auto handlekmer = [&](auto fwd, auto rc){
                const auto smallest = min(fwd, rc);
                const auto hashvalue = murmur3_fmix(smallest + hashFuncId);
                minHashValue = min(minHashValue, hashvalue);
            };

            if(myLength >= k){
                //const int numKmers = myLength - k + 1;
                const std::uint64_t kmer_mask = std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - k) * 2);
                const int rcshiftamount = (maximum_kmer_length - k) * 2;

                std::uint64_t kmer_encoded = mySequence[0];
                if(k <= 16){
                    kmer_encoded >>= (16 - k) * 2;
                }else{
                    kmer_encoded = (kmer_encoded << 32) | mySequence[1];
                    kmer_encoded >>= (32 - k) * 2;
                }

                kmer_encoded >>= 2; //k-1 bases, allows easier loop

                std::uint64_t rc_kmer_encoded = make_reverse_complement(kmer_encoded);

                auto addBase = [&](std::uint64_t encBase){
                    kmer_encoded <<= 2;
                    rc_kmer_encoded >>= 2;

                    const std::uint64_t revcBase = (~encBase) & 3;
                    kmer_encoded |= encBase;
                    rc_kmer_encoded |= revcBase << 62;
                };

                const int itersend1 = min(SDIV(k-1, 16) * 16, myLength);

                for(int nextSequencePos = k - 1; nextSequencePos < itersend1; nextSequencePos++){
                    const int nextIntIndex = nextSequencePos / 16;
                    const int nextPositionInInt = nextSequencePos % 16;

                    const std::uint64_t nextBase = mySequence[nextIntIndex] >> (30 - 2 * nextPositionInInt);

                    addBase(nextBase);

                    handlekmer(
                        kmer_encoded & kmer_mask, 
                        rc_kmer_encoded >> rcshiftamount
                    );
                }

                const int full16Iters = (myLength - itersend1) / 16;

                for(int iter = 0; iter < full16Iters; iter++){
                    const int intIndex = (itersend1 + iter * 16) / 16;
                    const unsigned int data = mySequence[intIndex];

                    #pragma unroll
                    for(int posInInt = 0; posInInt < 16; posInInt++){
                        const std::uint64_t nextBase = data >> (30 - 2 * posInInt);

                        addBase(nextBase);

                        handlekmer(
                            kmer_encoded & kmer_mask, 
                            rc_kmer_encoded >> rcshiftamount
                        );
                    }
                }


                for(int nextSequencePos = full16Iters * 16 + itersend1; nextSequencePos < myLength; nextSequencePos++){
                    const int nextIntIndex = nextSequencePos / 16;
                    const int nextPositionInInt = nextSequencePos % 16;

                    const std::uint64_t nextBase = mySequence[nextIntIndex] >> (30 - 2 * nextPositionInInt);

                    addBase(nextBase);

                    handlekmer(
                        kmer_encoded & kmer_mask, 
                        rc_kmer_encoded >> rcshiftamount
                    );
                }

                mySignature[myNumHashFunc] = minHashValue;

            }else{
                mySignature[myNumHashFunc] = std::numeric_limits<std::uint64_t>::max();
            }
        }
    }


    void callMinhashSignaturesKernel_async(
            std::uint64_t* d_signatures,
            size_t signaturesRowPitchElements,
            const unsigned int* d_sequences2Bit,
            size_t sequenceRowPitchElements,
            int numSequences,
            const int* d_sequenceLengths,
            int k,
            int numHashFuncs,
            int firstHashFunc,
            cudaStream_t stream){

        constexpr int blocksize = 128;

        if(numSequences <= 0){
            return;
        }

        dim3 block(blocksize, 1, 1);
        dim3 grid(SDIV(numSequences * numHashFuncs, blocksize), 1, 1);
        size_t smem = 0;

        minhashSignaturesKernel<<<grid, block, smem, stream>>>(
            d_signatures,
            signaturesRowPitchElements,
            d_sequences2Bit,
            sequenceRowPitchElements,
            numSequences,
            d_sequenceLengths,
            k,
            numHashFuncs,
            firstHashFunc
        );
    }

    void callMinhashSignaturesKernel_async(
            std::uint64_t* d_signatures,
            size_t signaturesRowPitchElements,
            const unsigned int* d_sequences2Bit,
            size_t sequenceRowPitchElements,
            int numSequences,
            const int* d_sequenceLengths,
            int k,
            int numHashFuncs,
            cudaStream_t stream){
                
        constexpr int blocksize = 128;

        if(numSequences <= 0){
            return;
        }
                
        dim3 block(blocksize, 1, 1);
        dim3 grid(SDIV(numSequences * numHashFuncs, blocksize), 1, 1);
        size_t smem = 0;
        
        const int firstHashFunc = 0;

        minhashSignaturesKernel<<<grid, block, smem, stream>>>(
            d_signatures,
            signaturesRowPitchElements,
            d_sequences2Bit,
            sequenceRowPitchElements,
            numSequences,
            d_sequenceLengths,
            k,
            numHashFuncs,
            firstHashFunc
        );
    }


    __global__
    void minhashSignaturesKernel(
            std::uint64_t* __restrict__ signatures, // numSequences * numHashFunc
            std::uint64_t* __restrict__ kmers, // numSequences * numHashFunc
            size_t signaturesRowPitchElements,
            const unsigned int* __restrict__ sequences2Bit,
            size_t sequenceRowPitchElements,
            int numSequences,
            const int* __restrict__ sequenceLengths,
            int k,
            int numHashFuncs
    ){
                
        //constexpr int blocksize = 128;
        constexpr int maximum_kmer_length = 32;

        auto murmur3_fmix = [](std::uint64_t x) {
            x ^= x >> 33;
            x *= 0xff51afd7ed558ccd;
            x ^= x >> 33;
            x *= 0xc4ceb9fe1a85ec53;
            x ^= x >> 33;
            return x;
        };

        auto make_reverse_complement = [](std::uint64_t s){
            s = ((s >> 2)  & 0x3333333333333333ull) | ((s & 0x3333333333333333ull) << 2);
            s = ((s >> 4)  & 0x0F0F0F0F0F0F0F0Full) | ((s & 0x0F0F0F0F0F0F0F0Full) << 4);
            s = ((s >> 8)  & 0x00FF00FF00FF00FFull) | ((s & 0x00FF00FF00FF00FFull) << 8);
            s = ((s >> 16) & 0x0000FFFF0000FFFFull) | ((s & 0x0000FFFF0000FFFFull) << 16);
            s = ((s >> 32) & 0x00000000FFFFFFFFull) | ((s & 0x00000000FFFFFFFFull) << 32);
            return ((std::uint64_t)(-1) - s) >> (8 * sizeof(s) - (32 << 1));
        };

        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        constexpr int firstHashFunc = 0;

        if(tid < numSequences * numHashFuncs){
            const int mySequenceIndex = tid / numHashFuncs;
            const int myNumHashFunc = tid % numHashFuncs;
            const int hashFuncId = myNumHashFunc + firstHashFunc;

            const unsigned int* const mySequence = sequences2Bit + mySequenceIndex * sequenceRowPitchElements;
            const int myLength = sequenceLengths[mySequenceIndex];

            std::uint64_t* const mySignature = signatures + mySequenceIndex * signaturesRowPitchElements;
            std::uint64_t* const myKmers = kmers + mySequenceIndex * signaturesRowPitchElements;            

            std::uint64_t minHashValue = std::numeric_limits<std::uint64_t>::max();
            std::uint64_t minKmer = 0;

            auto handlekmer = [&](auto fwd, auto rc){
                const auto smallest = min(fwd, rc);
                const auto hashvalue = murmur3_fmix(smallest + hashFuncId);
                if(hashvalue < minHashValue){
                    minHashValue = hashvalue;
                    minKmer = smallest;
                }
            };

            if(myLength >= k){
                //const int numKmers = myLength - k + 1;
                const std::uint64_t kmer_mask = std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - k) * 2);
                const int rcshiftamount = (maximum_kmer_length - k) * 2;

                std::uint64_t kmer_encoded = mySequence[0];
                if(k <= 16){
                    kmer_encoded >>= (16 - k) * 2;
                }else{
                    kmer_encoded = (kmer_encoded << 32) | mySequence[1];
                    kmer_encoded >>= (32 - k) * 2;
                }

                kmer_encoded >>= 2; //k-1 bases, allows easier loop

                std::uint64_t rc_kmer_encoded = make_reverse_complement(kmer_encoded);

                auto addBase = [&](std::uint64_t encBase){
                    kmer_encoded <<= 2;
                    rc_kmer_encoded >>= 2;

                    const std::uint64_t revcBase = (~encBase) & 3;
                    kmer_encoded |= encBase;
                    rc_kmer_encoded |= revcBase << 62;
                };

                const int itersend1 = min(SDIV(k-1, 16) * 16, myLength);

                for(int nextSequencePos = k - 1; nextSequencePos < itersend1; nextSequencePos++){
                    const int nextIntIndex = nextSequencePos / 16;
                    const int nextPositionInInt = nextSequencePos % 16;

                    const std::uint64_t nextBase = mySequence[nextIntIndex] >> (30 - 2 * nextPositionInInt);

                    addBase(nextBase);

                    handlekmer(
                        kmer_encoded & kmer_mask, 
                        rc_kmer_encoded >> rcshiftamount
                    );
                }

                const int full16Iters = (myLength - itersend1) / 16;

                for(int iter = 0; iter < full16Iters; iter++){
                    const int intIndex = (itersend1 + iter * 16) / 16;
                    const unsigned int data = mySequence[intIndex];

                    #pragma unroll
                    for(int posInInt = 0; posInInt < 16; posInInt++){
                        const std::uint64_t nextBase = data >> (30 - 2 * posInInt);

                        addBase(nextBase);

                        handlekmer(
                            kmer_encoded & kmer_mask, 
                            rc_kmer_encoded >> rcshiftamount
                        );
                    }
                }


                for(int nextSequencePos = full16Iters * 16 + itersend1; nextSequencePos < myLength; nextSequencePos++){
                    const int nextIntIndex = nextSequencePos / 16;
                    const int nextPositionInInt = nextSequencePos % 16;

                    const std::uint64_t nextBase = mySequence[nextIntIndex] >> (30 - 2 * nextPositionInInt);

                    addBase(nextBase);

                    handlekmer(
                        kmer_encoded & kmer_mask, 
                        rc_kmer_encoded >> rcshiftamount
                    );
                }

                mySignature[myNumHashFunc] = minHashValue;
                myKmers[myNumHashFunc] = minKmer;

            }else{
                mySignature[myNumHashFunc] = std::numeric_limits<std::uint64_t>::max();
                myKmers[myNumHashFunc] = std::numeric_limits<std::uint64_t>::max();
            }
        }
    }


template<int blocksize, int elemsPerThread>
__global__
void keepUniqueKmerKernel(
    std::uint64_t* __restrict__ signatures, 
    std::size_t signaturesRowPitchElements,
    int* __restrict__ hashFuncIds,
    std::size_t hashFuncIdsRowPitchElements,
    int* __restrict__ sizesOfUniqueSignatures, 
    const std::uint64_t* __restrict__ kmers,
    std::size_t kmersRowPitchElements,
    int numSequences,
    int numHashFuncs,
    int k
){
    constexpr int maxNumHashFuncs = 64;

    assert(numHashFuncs <= maxNumHashFuncs); 

    using BlockRadixSort = cub::BlockRadixSort<std::uint64_t, blocksize, elemsPerThread, int>;
    using BlockRadixSortInt = cub::BlockRadixSort<int, blocksize, elemsPerThread, std::uint64_t>;

    using BlockDiscontinuity = cub::BlockDiscontinuity<std::uint64_t, blocksize>;
    using BlockScan = cub::BlockScan<int, blocksize>; 

    using BlockStore = cub::BlockStore<std::uint64_t, blocksize, elemsPerThread, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreInt = cub::BlockStore<int, blocksize, elemsPerThread, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    
    using BlockLoad = cub::BlockLoad<std::uint64_t, blocksize, elemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadInt = cub::BlockLoad<int, blocksize, elemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

    __shared__ union{
        typename BlockRadixSort::TempStorage sort;
        typename BlockRadixSortInt::TempStorage sort_int;
        typename BlockDiscontinuity::TempStorage discontinuity;
        typename BlockScan::TempStorage scan;
        typename BlockLoad::TempStorage load;
        typename BlockLoadInt::TempStorage load_int;
        typename BlockStore::TempStorage store;
        typename BlockStoreInt::TempStorage store_int;
    } temp_storage;

    // if(threadIdx.x + blockIdx.x * blockDim.x == 0){
    //     printf("typename BlockRadixSort::TempStorage %llu\n", sizeof(typename BlockRadixSort::TempStorage));
    //     printf("typename BlockRadixSortInt::TempStorage %llu\n", sizeof(typename BlockRadixSortInt::TempStorage));
    //     printf("typename BlockDiscontinuity::TempStorage %llu\n", sizeof(typename BlockDiscontinuity::TempStorage));
    //     printf("typename BlockScan::TempStorage %llu\n", sizeof(typename BlockScan::TempStorage));
    //     printf("typename BlockLoad::TempStorage %llu\n", sizeof(typename BlockLoad::TempStorage));
    //     printf("typename BlockLoadInt::TempStorage %llu\n", sizeof(typename BlockLoadInt::TempStorage));
    //     printf("typename BlockStore::TempStorage %llu\n", sizeof(typename BlockStore::TempStorage));
    //     printf("typename BlockStoreInt::TempStorage %llu\n", sizeof(typename BlockStoreInt::TempStorage));
    // }

    __shared__ int uniquehashnums[maxNumHashFuncs];
    __shared__ std::uint64_t uniquehashvalues[maxNumHashFuncs];

    const int kmerBits = 2 * k;

    for(int sequenceIndex = blockIdx.x; sequenceIndex < numSequences; sequenceIndex += gridDim.x){

        std::uint64_t regkmers[elemsPerThread]; 
        std::uint64_t reghashvalues[elemsPerThread]; 
        int reghashnums[elemsPerThread];

        const int sizeOfRange = numHashFuncs;
        if(sizeOfRange == 0){
            if(threadIdx.x == 0){
                sizesOfUniqueSignatures[sequenceIndex] = 0;
            }
        }else{
        
            const std::uint64_t* const myKmers = kmers + sequenceIndex * kmersRowPitchElements;
            int* const myHashFuncIds = hashFuncIds + sequenceIndex * hashFuncIdsRowPitchElements;
            std::uint64_t* const mySignatures = signatures + sequenceIndex * signaturesRowPitchElements;

            assert(sizeOfRange <= elemsPerThread * blocksize);            

            // load kmers
            BlockLoad(temp_storage.load).Load(
                myKmers, 
                regkmers, 
                sizeOfRange,
                std::numeric_limits<std::uint64_t>::max()
            );

            __syncthreads();

            // BlockLoad(temp_storage.load).Load(
            //     mySignatures, 
            //     reghashvalues, 
            //     sizeOfRange,
            //     std::numeric_limits<std::uint64_t>::max()
            // );

            // if(sequenceIndex == 0){

            //     if(threadIdx.x == 0){
            //         printf("expected kmers;\n");

            //         for(int i = 0; i < sizeOfRange; i++){
            //             printf("%llu ", myKmers[i]);
            //         }
            //         printf("\n");

            //         printf("expected signatures;\n");

            //         for(int i = 0; i < sizeOfRange; i++){
            //             printf("%llu ", mySignatures[i]);
            //         }
            //         printf("\n");
            //     }
            //     __syncthreads();

            //     if(threadIdx.x == 0){
            //         printf("loaded kmers;\n");
            //     }
            //     __syncthreads();

            //     for(int x = 0; x < blocksize; x++){
            //         if(x == threadIdx.x){
            //             #pragma unroll 
            //             for(int i = 0; i < elemsPerThread; i++){
            //                 printf("%d %llu\n", threadIdx.x * elemsPerThread + i, regkmers[i]);
            //             }
            //         }
            //         __syncthreads();
            //     }
            // }

            //set hash func numbers
            #pragma unroll 
            for(int i = 0; i < elemsPerThread; i++){
                reghashnums[i] = threadIdx.x * elemsPerThread + i;
            }

            __syncthreads();

            //sort kmers (and hash func numbers)
            BlockRadixSort(temp_storage.sort).Sort(regkmers, reghashnums, 0, kmerBits);

            __syncthreads();


            // if(sequenceIndex == 0){
            //     if(threadIdx.x == 0){
            //         printf("hashnums sorted by kmer\n");
            //     }
            //     __syncthreads();
            //     for(int x = 0; x < blocksize; x++){
            //         if(x == threadIdx.x){
            //             #pragma unroll 
            //             for(int i = 0; i < elemsPerThread; i++){
            //                 printf("%2d %20llu\n", reghashnums[i], regkmers[i]);
            //             }
            //         }
            //         __syncthreads();
            //     }
            // }



            int head_flags[elemsPerThread];

            //mark first occurence of each kmer
            BlockDiscontinuity(temp_storage.discontinuity).FlagHeads(
                head_flags, 
                regkmers, 
                cub::Inequality()
            );

            __syncthreads();            

            // don't use out of bounds elements
            #pragma unroll
            for(int i = 0; i < elemsPerThread; i++){
                if(threadIdx.x * elemsPerThread + i >= sizeOfRange){
                    head_flags[i] = 0;
                }
            }

            int prefixsum[elemsPerThread];
            int numberOfSetHeadFlags = 0;

            //calculate output positions of unique kmers
            BlockScan(temp_storage.scan).ExclusiveSum(head_flags, prefixsum, numberOfSetHeadFlags);

            __syncthreads();

            //store unique hashnums in smem
            #pragma unroll
            for(int i = 0; i < elemsPerThread; i++){
                if(threadIdx.x * elemsPerThread + i < sizeOfRange && head_flags[i] == 1){
                    uniquehashnums[prefixsum[i]] = reghashnums[i];
                    uniquehashvalues[prefixsum[i]] = mySignatures[reghashnums[i]];
                }
            }

            //store number of unique kmers
            if(threadIdx.x == 0){
                sizesOfUniqueSignatures[sequenceIndex] = numberOfSetHeadFlags;
            }

            // if(sequenceIndex == 0){
            //     if(threadIdx.x == 0){
            //         printf("unique hashnums\n");
            //         for(int x = 0; x < numberOfSetHeadFlags; x++){
            //             printf("%d ", uniquehashnums[x]);
            //         }
            //         printf("\n");
            //     }
            //     __syncthreads();                
            // }

            __syncthreads(); //uniquehashnums

            //store signatures of unique kmers in smem
            // for(int i = threadIdx.x; i < numberOfSetHeadFlags; i += blocksize){
            //     uniquehashvalues[i] = mySignatures[uniquehashnums[i]];
            // }


            // __syncthreads();

            // if(sequenceIndex == 0){
            //     if(threadIdx.x == 0){
            //         printf("unique hash funcs\n");
            //         for(int x = 0; x < numberOfSetHeadFlags; x++){
            //             printf("%llu ", uniquehashvalues[x]);
            //         }
            //         printf("\n");
            //     }
            //     __syncthreads();                
            // }

            // if(sequenceIndex == 0){
            //     if(threadIdx.x == 0){
            //         printf("unique hashnums | unique hash funcs\n");
            //         for(int x = 0; x < numberOfSetHeadFlags; x++){
            //             printf("%d %llu\n", uniquehashnums[x], uniquehashvalues[x]);
            //         }
            //         printf("\n");
            //     }
            //     __syncthreads();                
            // }

            //load unique signatures into regs
            BlockLoad(temp_storage.load).Load(
                uniquehashvalues, 
                reghashvalues, 
                numberOfSetHeadFlags,
                std::numeric_limits<std::uint64_t>::max()
            );

            __syncthreads();

            //load unique hash nums into regs
            BlockLoadInt(temp_storage.load_int).Load(
                uniquehashnums, 
                reghashnums, 
                numberOfSetHeadFlags,
                std::numeric_limits<int>::max()
            );

            __syncthreads();

            //sort unique signatures by unique hash nums
            //numhashfuncs is limited to 64, so each hash num is in range [0...63] and occupies 6 bits.
            BlockRadixSortInt(temp_storage.sort_int).Sort(reghashnums, reghashvalues, 0, 6); 

            __syncthreads();

            //finally, store unique signatures back to gmem
            BlockStore(temp_storage.store).Store(mySignatures, reghashvalues, numberOfSetHeadFlags);

            __syncthreads();

            
            BlockStoreInt(temp_storage.store_int).Store(myHashFuncIds, reghashnums, numberOfSetHeadFlags);

            __syncthreads();

            // if(sequenceIndex == 0){
            //     if(threadIdx.x == 0){
            //         printf("unique hashnums | unique hash funcs after sort\n");
            //         for(int x = 0; x < numberOfSetHeadFlags; x++){
            //             printf("%d %llu\n", myHashFuncIds[x], mySignatures[x]);
            //         }
            //         printf("\n");
            //     }
            //     __syncthreads();                
            // }
            
        }
    }
}



void callUniqueMinhashSignaturesKernel_async(
    std::uint64_t* d_temp,
    std::uint64_t* d_signatures,
    std::size_t signaturesRowPitchElements,
    int* d_hashFuncIds,
    std::size_t hashFuncIdsRowPitchElements,
    int* d_signatureSizePerSequence,
    const unsigned int* d_sequences2Bit,
    std::size_t sequenceRowPitchElements,
    int numSequences,
    const int* d_sequenceLengths,
    int k,
    int numHashFuncs,
    cudaStream_t stream
){
        
    constexpr int blocksize = 128;

    if(numSequences <= 0){
        return;
    }

    std::uint64_t* d_kmers = d_temp;
            
    dim3 block(blocksize, 1, 1);
    dim3 grid(SDIV(numSequences * numHashFuncs, blocksize), 1, 1);
    size_t smem = 0;

    minhashSignaturesKernel<<<grid, block, smem, stream>>>(
        d_signatures,
        d_kmers,
        signaturesRowPitchElements,
        d_sequences2Bit,
        sequenceRowPitchElements,
        numSequences,
        d_sequenceLengths,
        k,
        numHashFuncs
    );

    const std::size_t kmersRowPitchElements = numHashFuncs;

    keepUniqueKmerKernel<64, 1><<<numSequences, 64, 0, stream>>>(
        d_signatures, 
        signaturesRowPitchElements,
        d_hashFuncIds,
        hashFuncIdsRowPitchElements,
        d_signatureSizePerSequence, 
        d_kmers,
        kmersRowPitchElements,
        numSequences,
        numHashFuncs,
        k
    );
}



//use one block per sequence

template<int blocksize, int elemsPerThread>
__global__
void minhashSignaturesOfUniqueKmersKernel128(
    std::uint64_t* __restrict__ signatures, // numSequences * numHashFunc
    size_t signaturesRowPitchElements,
    const unsigned int* __restrict__ sequences2Bit,
    size_t sequenceRowPitchElements,
    int numSequences,
    const int* __restrict__ sequenceLengths,
    int k,
    int numHashFuncs
){

    using BlockLoad = cub::BlockLoad<std::uint64_t, blocksize, elemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStore = cub::BlockStore<std::uint64_t, blocksize, elemsPerThread, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockRadixSort = cub::BlockRadixSort<std::uint64_t, blocksize, elemsPerThread>;
    using BlockDiscontinuity = cub::BlockDiscontinuity<std::uint64_t, blocksize>;
    using BlockScan = cub::BlockScan<int, blocksize>; 
    //BlockStore(temp_storage.store).Store(mySignatures, reghashvalues, numberOfSetHeadFlags);

    __shared__ union{
        typename BlockLoad::TempStorage load;
        typename BlockStore::TempStorage store;
        typename BlockRadixSort::TempStorage sort;
        typename BlockDiscontinuity::TempStorage discontinuity;
        typename BlockScan::TempStorage scan;
    } temp_storage;

    __shared__ unsigned int sharedsequence[8];
    __shared__ std::uint64_t sharedkmers[128];

    auto make_reverse_complement = [](std::uint64_t s){
        s = ((s >> 2)  & 0x3333333333333333ull) | ((s & 0x3333333333333333ull) << 2);
        s = ((s >> 4)  & 0x0F0F0F0F0F0F0F0Full) | ((s & 0x0F0F0F0F0F0F0F0Full) << 4);
        s = ((s >> 8)  & 0x00FF00FF00FF00FFull) | ((s & 0x00FF00FF00FF00FFull) << 8);
        s = ((s >> 16) & 0x0000FFFF0000FFFFull) | ((s & 0x0000FFFF0000FFFFull) << 16);
        s = ((s >> 32) & 0x00000000FFFFFFFFull) | ((s & 0x00000000FFFFFFFFull) << 32);
        return ((std::uint64_t)(-1) - s) >> (8 * sizeof(s) - (32 << 1));
    };

    auto murmur3_fmix = [](std::uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccd;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53;
        x ^= x >> 33;
        return x;
    };

    auto hashfunc = murmur3_fmix;
    const std::uint64_t kmer_mask = (std::uint64_t(1) << ((2*k) - 1)) | (std::uint64_t(1) << (((2*k) - 1))) - 1;

    for(int sequenceIndex = blockIdx.x; sequenceIndex < numSequences; sequenceIndex += gridDim.x){

        std::uint64_t* const mySignature = signatures + sequenceIndex * signaturesRowPitchElements;

        // if(threadIdx.x == 0 && sequenceIndex < 10){
        //     printf("%d mysig %p\n", sequenceIndex, mySignature);
        // }

        const int sequenceLength = sequenceLengths[sequenceIndex];
        assert(sequenceLength <= 128);

        if(sequenceLength >= k){
            
            const unsigned int* const mySequencePtr = sequences2Bit + sequenceIndex * sequenceRowPitchElements;
            //load sequence into shared memory.
            const int numInts = getEncodedNumInts2Bit(sequenceLength);

            if(threadIdx.x < numInts){
                sharedsequence[threadIdx.x] = mySequencePtr[threadIdx.x];
                // if(sequenceIndex < 2){
                //     printf("%d %d enc: %u\n", sequenceIndex, threadIdx.x, sharedsequence[threadIdx.x]);
                // }
            }

            __syncthreads();

            //kmerize , store canonical kmers into shared memory
            const int numKmers = sequenceLength - k + 1;
            assert(numKmers > 0);

            for(int i = threadIdx.x; i < numKmers; i += blockDim.x){
                const int firstPos = i;
                const int lastPos = i + k - 1; // inclusive
                const int firstIntPos = firstPos / 16;
                const int lastIntPos = lastPos / 16;

                assert(firstIntPos < numInts);
                assert(lastIntPos < numInts);

                
                const std::uint64_t l = sharedsequence[firstIntPos];
                const std::uint64_t r = sharedsequence[lastIntPos];
                std::uint64_t kmer = (l << 32) | r;
                // ((unsigned int*)&kmer)[1] = sharedsequence[lastIntPos];

                // ((unsigned int*)&kmer)[0] = sharedsequence[firstIntPos];
                // ((unsigned int*)&kmer)[1] = sharedsequence[lastIntPos];

                //const int leftFirstPosInInt = firstPos % 16;
                const int rightFirstPosInInt = lastPos % 16;

                kmer = kmer >> 2*(15 - rightFirstPosInInt);
                kmer = kmer & kmer_mask;

                std::uint64_t revc = make_reverse_complement(kmer);
                sharedkmers[i] = min(kmer, revc);
            }

            __syncthreads();

            int numberOfUniqueKmers = 0;

            // find unique kmers and store them into shared memory, replacing original kmers. sets numberOfUniqueKmers

            {

                std::uint64_t regkmers[elemsPerThread];

                BlockLoad(temp_storage.load).Load(
                    sharedkmers, 
                    regkmers, 
                    numKmers,
                    std::numeric_limits<std::uint64_t>::max()
                );

                __syncthreads();

                BlockRadixSort(temp_storage.sort).Sort(regkmers, 0, (2*k));

                __syncthreads();

                int head_flags[elemsPerThread];

                //mark first occurence of each kmer
                BlockDiscontinuity(temp_storage.discontinuity).FlagHeads(
                    head_flags, 
                    regkmers, 
                    cub::Inequality()
                );

                __syncthreads();            

                // don't use out of bounds elements
                #pragma unroll
                for(int i = 0; i < elemsPerThread; i++){
                    if(threadIdx.x * elemsPerThread + i >= numKmers){
                        head_flags[i] = 0;
                    }
                }

                int prefixsum[elemsPerThread];            

                //calculate output positions of unique kmers
                BlockScan(temp_storage.scan).ExclusiveSum(head_flags, prefixsum, numberOfUniqueKmers);

                #pragma unroll
                for(int i = 0; i < elemsPerThread; i++){
                    if(threadIdx.x * elemsPerThread + i < numKmers && head_flags[i] == 1){
                        sharedkmers[prefixsum[i]] = regkmers[i];
                    }
                }
            }

            __syncthreads();

            // if(numberOfUniqueKmers != 81 && threadIdx.x == 0){
            //     printf("numberOfUniqueKmers %d\n", numberOfUniqueKmers);
            // }

            // perform minhashing of kmers

            for(int h = threadIdx.x; h < numHashFuncs; h += blockDim.x){

                std::uint64_t smallesthash = std::numeric_limits<std::uint64_t>::max();

                for(int i = 0; i < numberOfUniqueKmers; i++){
                    const std::uint64_t kmer = sharedkmers[i];
                    const std::uint64_t hash = hashfunc(kmer + h);
                    if(hash < smallesthash){
                        smallesthash = hash;
                    }
                }

                mySignature[h] = smallesthash;

                // if(sequenceIndex < 2){
                //     printf("%d %d %llu numberOfUniqueKmers %d\n", sequenceIndex, h, smallesthash, numberOfUniqueKmers);
                // }
            }

            __syncthreads();


        }else{
            for(int i = threadIdx.x; i < numHashFuncs; i += blockDim.x){
                mySignature[i] = std::numeric_limits<std::uint64_t>::max();
            }
        }
    }
}


void callMinhashSignaturesOfUniqueKmersKernel128_async(
        std::uint64_t* d_signatures,
        size_t signaturesRowPitchElements,
        const unsigned int* d_sequences2Bit,
        size_t sequenceRowPitchElements,
        int numSequences,
        const int* d_sequenceLengths,
        int k,
        int numHashFuncs,
        cudaStream_t stream){
            
    constexpr int blocksize = 128;

    if(numSequences <= 0){
        return;
    }
            
    dim3 block(blocksize, 1, 1);
    dim3 grid(numSequences, 1, 1);
    size_t smem = 0;

    minhashSignaturesOfUniqueKmersKernel128<128,1><<<grid, block, smem, stream>>>(
        d_signatures,
        signaturesRowPitchElements,
        d_sequences2Bit,
        sequenceRowPitchElements,
        numSequences,
        d_sequenceLengths,
        k,
        numHashFuncs
    );
}








// ############## SET_UNION ###############





template<int blocksize>
__global__
void compactDataOfUniqueRanges(
        read_number* __restrict__ output,
        const read_number* __restrict__ input,
        const int* __restrict__ sizesOfUniqueRangesPrefixsum,
        const int* __restrict__ rangesPerSequenceBegins,
        int numSequences){

    for(int sequenceIndex = blockIdx.x; sequenceIndex < numSequences; sequenceIndex += gridDim.x){

        const read_number* blockinput = input + rangesPerSequenceBegins[sequenceIndex];
        read_number* blockoutput = output + sizesOfUniqueRangesPrefixsum[sequenceIndex];
        const int numElements = sizesOfUniqueRangesPrefixsum[sequenceIndex + 1] - sizesOfUniqueRangesPrefixsum[sequenceIndex];

        for(int index = threadIdx.x; index < numElements; index += blocksize){
            blockoutput[index] = blockinput[index];
        }
    }

}



// perform set-union of inoutData on each range identified by offsets
// assumes input data is sorted
// removes anchorIds[i] from set_union range[i]
// returns number of remaining elements if range[i] in sizesOfUniqueRanges[i]
// template<int blocksize, int elemsPerThread>
// __global__
// void makeUniqueRangesFromSortedWithOnlyCUBKernel(
//         read_number* __restrict__ inoutDataSorted, 
//         int* __restrict__ sizesOfUniqueRanges, 
//         int numSequences,
//         const read_number* __restrict__ anchorIds,
//         const int* __restrict__ offsets, // inoutData[offsets[i] - globalOffset] to inoutData[offsets[i+1] - globalOffset] (exclusive) belong to sequence i
//         int globalOffset
// ){

//     using BlockLoad = cub::BlockLoad<read_number, blocksize, elemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
//     using BlockDiscontinuity = cub::BlockDiscontinuity<read_number, blocksize>;
//     using BlockScan = cub::BlockScan<int, blocksize>; 

//     __shared__ union{
//         typename BlockLoad::TempStorage load;
//         typename BlockDiscontinuity::TempStorage discontinuity;
//         typename BlockScan::TempStorage scan;
//     } temp_storage;

//     __shared__ read_number anchorId;

//     for(int sequenceIndex = blockIdx.x; sequenceIndex < numSequences; sequenceIndex += gridDim.x){

//         read_number tempregs[elemsPerThread];   

//         #pragma unroll
//         for(int i = 0; i <elemsPerThread; i++){
//             tempregs[i] = std::numeric_limits<read_number>::max();
//         }

//         const int sizeOfRange = offsets[sequenceIndex + 1] - offsets[sequenceIndex];
//         if(sizeOfRange == 0){
//             if(threadIdx.x == 0){
//                 sizesOfUniqueRanges[sequenceIndex] = 0;
//             }
//         }else{
        
//             read_number* const myRange = inoutDataSorted + offsets[sequenceIndex] - globalOffset;

//             assert(sizeOfRange <= elemsPerThread * blocksize);            

//             BlockLoad(temp_storage.load).Load(
//                 myRange, 
//                 tempregs, 
//                 sizeOfRange
//             );

//             if(threadIdx.x == 0){
//                 anchorId = anchorIds[sequenceIndex];
//             }

//             __syncthreads();

//             int head_flags[elemsPerThread];

//             BlockDiscontinuity(temp_storage.discontinuity).FlagHeads(
//                 head_flags, 
//                 tempregs, 
//                 cub::Inequality()
//             );

//             __syncthreads();            

//             #pragma unroll
//             for(int i = 0; i < elemsPerThread; i++){
//                 if(threadIdx.x * elemsPerThread + i >= sizeOfRange){
//                     head_flags[i] = 0;
//                 }else{
//                     if(tempregs[i] == anchorId){
//                         head_flags[i] = 0;
//                     }
//                 }
//             }

//             int prefixsum[elemsPerThread];
//             int numberOfSetHeadFlags = 0;

//             BlockScan(temp_storage.scan).ExclusiveSum(head_flags, prefixsum, numberOfSetHeadFlags);

//             __syncthreads();

//             #pragma unroll
//             for(int i = 0; i < elemsPerThread; i++){
//                 if(threadIdx.x * elemsPerThread + i < sizeOfRange && head_flags[i] == 1){
//                     myRange[prefixsum[i]] = tempregs[i];
//                 }
//             }

//             if(threadIdx.x == 0){
//                 sizesOfUniqueRanges[sequenceIndex] = numberOfSetHeadFlags;
//             }
//         }
//     }

// }


// void callSegmentedMakeUniqueWithGmemSortKernel(
//     void* temp_storage,
//     std::size_t& temp_storage_bytes,
//     MergeRangesGpuHandle<read_number>& handle,
//     const read_number* d_data,
//     read_number d_results,
//     int* __restrict__ sizesOfUniqueRanges, 
//     int numSegments,
//     int* d_segmentsBeginOffsets,
//     int* d_segmentsEndOffsets,
//     int numItems,
//     const read_number* __restrict__ anchorIds,
//     int globalOffset,
//     cudaStream_t stream
// ){
//     std::size_t requiredbytes = 0;

//     cub::DeviceSegmentedRadixSort::SortKeys(
//         nullptr, 
//         requiredbytes, 
//         d_data, 
//         d_results,
//         numItems, 
//         numSegments, 
//         d_segmentsBeginOffsets, 
//         d_segmentsEndOffsets,
//         0,
//         sizeof(read_number) * 8,
//         stream
//     );

//     if(temp_storage == nullptr){
//         temp_storage_bytes = requiredbytes;
//         return;
//     }

//     assert(requiredbytes <= temp_storage_bytes);

//     cub::DeviceSegmentedRadixSort::SortKeys(
//         temp_storage, 
//         temp_storage_bytes, 
//         d_data, 
//         d_results,
//         numItems, 
//         numSegments, 
//         d_segmentsBeginOffsets, 
//         d_segmentsEndOffsets,
//         0,
//         sizeof(read_number) * 8,
//         stream
//     );

// }






// perform set-union of inoutData on each range identified by offsets
// removes anchorIds[i] from set_union range[i]
// returns number of remaining elements if range[i] in sizesOfUniqueRanges[i]
template<int blocksize, int elemsPerThread>
__global__
void makeUniqueRangesWithOnlyCUBKernel(
        read_number* __restrict__ inoutData, 
        int* __restrict__ sizesOfUniqueRanges, 
        int numSequences,
        const read_number* __restrict__ anchorIds,
        const int* __restrict__ offsets, // inoutData[offsets[i] - globalOffset] to inoutData[offsets[i+1] - globalOffset] (exclusive) belong to sequence i
        int globalOffset
){

    using BlockRadixSort = cub::BlockRadixSort<read_number, blocksize, elemsPerThread>;
    using BlockLoad = cub::BlockLoad<read_number, blocksize, elemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockDiscontinuity = cub::BlockDiscontinuity<read_number, blocksize>;
    using BlockScan = cub::BlockScan<int, blocksize>; 

    __shared__ union{
        typename BlockRadixSort::TempStorage sort;
        typename BlockLoad::TempStorage load;
        typename BlockDiscontinuity::TempStorage discontinuity;
        typename BlockScan::TempStorage scan;
    } temp_storage;

    __shared__ read_number anchorId;

    for(int sequenceIndex = blockIdx.x; sequenceIndex < numSequences; sequenceIndex += gridDim.x){

        read_number tempregs[elemsPerThread];   

        #pragma unroll
        for(int i = 0; i <elemsPerThread; i++){
            tempregs[i] = std::numeric_limits<read_number>::max();
        }

        const int sizeOfRange = offsets[sequenceIndex + 1] - offsets[sequenceIndex];
        if(sizeOfRange == 0){
            if(threadIdx.x == 0){
                sizesOfUniqueRanges[sequenceIndex] = 0;
            }
        }else{
        
            read_number* const myRange = inoutData + offsets[sequenceIndex] - globalOffset;

            assert(sizeOfRange <= elemsPerThread * blocksize);            

            BlockLoad(temp_storage.load).Load(
                myRange, 
                tempregs, 
                sizeOfRange
            );

            if(threadIdx.x == 0){
                anchorId = anchorIds[sequenceIndex];
            }

            __syncthreads();

            BlockRadixSort(temp_storage.sort).Sort(tempregs);

            __syncthreads();

            int head_flags[elemsPerThread];

            BlockDiscontinuity(temp_storage.discontinuity).FlagHeads(
                head_flags, 
                tempregs, 
                cub::Inequality()
            );

            __syncthreads();            

            #pragma unroll
            for(int i = 0; i < elemsPerThread; i++){
                if(threadIdx.x * elemsPerThread + i >= sizeOfRange){
                    head_flags[i] = 0;
                }else{
                    if(tempregs[i] == anchorId){
                        head_flags[i] = 0;
                    }
                }
            }

            int prefixsum[elemsPerThread];
            int numberOfSetHeadFlags = 0;

            BlockScan(temp_storage.scan).ExclusiveSum(head_flags, prefixsum, numberOfSetHeadFlags);

            __syncthreads();

            #pragma unroll
            for(int i = 0; i < elemsPerThread; i++){
                if(threadIdx.x * elemsPerThread + i < sizeOfRange && head_flags[i] == 1){
                    myRange[prefixsum[i]] = tempregs[i];
                }
            }

            if(threadIdx.x == 0){
                sizesOfUniqueRanges[sequenceIndex] = numberOfSetHeadFlags;
            }
        }
    }

}



template<int blocksize, int elemsPerThread>
__device__
void makeUniqueRangeSingleWarp(
        read_number* __restrict__ myRange, 
        int sizeOfRange, 
        int* __restrict__ sizeOfUniqueRange,
        read_number anchorId,
        typename cub::BlockRadixSort<read_number, blocksize, elemsPerThread>::TempStorage& sorttemp,
        typename cub::BlockLoad<read_number, blocksize, elemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>::TempStorage& loadtemp){

    using BlockRadixSort = cub::BlockRadixSort<read_number, blocksize, elemsPerThread>;
    using BlockLoad = cub::BlockLoad<read_number, blocksize, elemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;


    read_number tempregs[elemsPerThread];   

    #pragma unroll
    for(int i = 0; i <elemsPerThread; i++){
        tempregs[i] = std::numeric_limits<read_number>::max();
    }

    BlockLoad(loadtemp).Load(
        myRange, 
        tempregs, 
        sizeOfRange
    );

    __syncthreads();

    BlockRadixSort(sorttemp).SortBlockedToStriped(tempregs);

    int numUniqueElements = 0;
 
    #pragma unroll
    for(int i = 0; i < elemsPerThread; i++){
        const read_number curElement = tempregs[i];

        read_number nextElement = threadIdx.x == 0 ? tempregs[(i+1) % elemsPerThread] : tempregs[i];
        nextElement = __shfl_sync(0xFFFFFFFF, nextElement, threadIdx.x+1);               

        //find elements which are not equal to their right neighbor, not equal to the anchor id and not out of range
        const bool predicate = (curElement != nextElement) && (curElement != anchorId) && (i * blocksize + threadIdx.x < sizeOfRange);

        const uint32_t mask = __ballot_sync(0xFFFFFFFF, predicate);
        const uint32_t count = __popc(mask);

        //get position
        const uint32_t numPre = __popc(mask & ((1 << threadIdx.x) -1));
        const int position = numUniqueElements + numPre;

        if(predicate){
            myRange[position] = curElement;
        }
        numUniqueElements += count;
    }
    
    if(threadIdx.x == 0){
        *sizeOfUniqueRange = numUniqueElements;
    }
}


template<int blocksize, int elemsPerThread>
__global__
void makeUniqueRangesKernelWithIntrinsicsSingleWarp(
        read_number* __restrict__ inoutData, 
        int* __restrict__ sizesOfUniqueRanges, 
        int numSequences,
        const read_number* __restrict__ anchorIds,
        const int* __restrict__ rangesPerSequenceBegins,
        int globalOffset){

    static_assert(blocksize == 32, "blocksize must be 32 for SingleWarp kernel");

    using BlockRadixSort = cub::BlockRadixSort<read_number, blocksize, elemsPerThread>;
    using BlockLoad = cub::BlockLoad<read_number, blocksize, elemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    
    __shared__ union{
        typename BlockRadixSort::TempStorage sort;
        typename BlockLoad::TempStorage load;
    } temp_storage;

    for(int sequenceIndex = blockIdx.x; sequenceIndex < numSequences; sequenceIndex += gridDim.x){

        const int sizeOfRange = rangesPerSequenceBegins[sequenceIndex + 1] - rangesPerSequenceBegins[sequenceIndex];
        if(sizeOfRange > elemsPerThread * blocksize){
            assert(sizeOfRange <= elemsPerThread * blocksize);
        }

        if(sizeOfRange == 0){
            if(threadIdx.x == 0){
                sizesOfUniqueRanges[sequenceIndex] = 0;
            }
        }else{
        
            const read_number anchorId = anchorIds[sequenceIndex];
            read_number* const myRange = inoutData + rangesPerSequenceBegins[sequenceIndex] - globalOffset;

            makeUniqueRangeSingleWarp<blocksize, elemsPerThread>(
                myRange, 
                sizeOfRange, 
                &sizesOfUniqueRanges[sequenceIndex],
                anchorId,
                temp_storage.sort,
                temp_storage.load
            );
        }
    }

}





template<int blocksize, int elemsPerThread>
__global__
void makeUniqueRangesKernelWithIntrinsicsSingleWarpChunked(
        read_number* __restrict__ inoutData, 
        int* __restrict__ sizesOfUniqueRanges, 
        int numSequences,
        const read_number* __restrict__ anchorIds,
        const int* __restrict__ rangesPerSequenceBegins,
        int globalOffset){

    static_assert(blocksize == 32, "blocksize must be 32 for SingleWarp kernel");

    using BlockRadixSortFull = cub::BlockRadixSort<read_number, blocksize, elemsPerThread>;
    using BlockLoadFull = cub::BlockLoad<read_number, blocksize, elemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

    using BlockRadixSort1 = cub::BlockRadixSort<read_number, blocksize, 1>;
    using BlockLoad1 = cub::BlockLoad<read_number, blocksize, 1, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

    using BlockRadixSort2 = cub::BlockRadixSort<read_number, blocksize, 2>;
    using BlockLoad2 = cub::BlockLoad<read_number, blocksize, 2, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

    using BlockRadixSort4 = cub::BlockRadixSort<read_number, blocksize, 4>;
    using BlockLoad4 = cub::BlockLoad<read_number, blocksize, 4, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

    using BlockRadixSort8 = cub::BlockRadixSort<read_number, blocksize, 8>;
    using BlockLoad8 = cub::BlockLoad<read_number, blocksize, 8, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

    using BlockRadixSort16 = cub::BlockRadixSort<read_number, blocksize, 16>;
    using BlockLoad16 = cub::BlockLoad<read_number, blocksize, 16, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

    using BlockRadixSort32 = cub::BlockRadixSort<read_number, blocksize, 32>;
    using BlockLoad32 = cub::BlockLoad<read_number, blocksize, 32, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

    using BlockRadixSort64 = cub::BlockRadixSort<read_number, blocksize, 64>;
    using BlockLoad64 = cub::BlockLoad<read_number, blocksize, 64, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    
    __shared__ union{
        typename BlockRadixSortFull::TempStorage sortFull;
        typename BlockLoadFull::TempStorage loadFull;

        typename BlockRadixSort1::TempStorage sort1;
        typename BlockLoad1::TempStorage load1;

        typename BlockRadixSort2::TempStorage sort2;
        typename BlockLoad2::TempStorage load2;

        typename BlockRadixSort4::TempStorage sort4;
        typename BlockLoad4::TempStorage load4;

        typename BlockRadixSort8::TempStorage sort8;
        typename BlockLoad8::TempStorage load8;

        typename BlockRadixSort16::TempStorage sort16;
        typename BlockLoad16::TempStorage load16;

        typename BlockRadixSort32::TempStorage sort32;
        typename BlockLoad32::TempStorage load32;

        typename BlockRadixSort64::TempStorage sort64;
        typename BlockLoad64::TempStorage load64;
    } temp_storage;

    for(int sequenceIndex = blockIdx.x; sequenceIndex < numSequences; sequenceIndex += gridDim.x){

        const int sizeOfRange = rangesPerSequenceBegins[sequenceIndex + 1] - rangesPerSequenceBegins[sequenceIndex];
        assert(sizeOfRange <= elemsPerThread * blocksize);

        if(sizeOfRange == 0){
            if(threadIdx.x == 0){
                sizesOfUniqueRanges[sequenceIndex] = 0;
            }
        }else{
            const read_number anchorId = anchorIds[sequenceIndex];

            read_number* const myRange = inoutData + rangesPerSequenceBegins[sequenceIndex] - globalOffset;
            int* const sizeOfUniqueRange = sizesOfUniqueRanges + sequenceIndex;

            if(sizeOfRange <= blocksize * 1){
                makeUniqueRangeSingleWarp<blocksize, 1>(
                    myRange, 
                    sizeOfRange, 
                    sizeOfUniqueRange,
                    anchorId,
                    temp_storage.sort1,
                    temp_storage.load1
                );
            }else if(sizeOfRange <= blocksize * 2){
                makeUniqueRangeSingleWarp<blocksize, 2>(
                    myRange, 
                    sizeOfRange, 
                    sizeOfUniqueRange,
                    anchorId,
                    temp_storage.sort2,
                    temp_storage.load2
                );
            }else if(sizeOfRange <= blocksize * 4){
                makeUniqueRangeSingleWarp<blocksize, 4>(
                    myRange, 
                    sizeOfRange, 
                    sizeOfUniqueRange,
                    anchorId,
                    temp_storage.sort4,
                    temp_storage.load4
                );
            }else if(sizeOfRange <= blocksize * 8){
                makeUniqueRangeSingleWarp<blocksize, 8>(
                    myRange, 
                    sizeOfRange, 
                    sizeOfUniqueRange,
                    anchorId,
                    temp_storage.sort8,
                    temp_storage.load8
                );
            }else if(sizeOfRange <= blocksize * 16){
                makeUniqueRangeSingleWarp<blocksize, 16>(
                    myRange, 
                    sizeOfRange, 
                    sizeOfUniqueRange,
                    anchorId,
                    temp_storage.sort16,
                    temp_storage.load16
                );
            }else if(sizeOfRange <= blocksize * 32){
                makeUniqueRangeSingleWarp<blocksize, 32>(
                    myRange, 
                    sizeOfRange, 
                    sizeOfUniqueRange,
                    anchorId,
                    temp_storage.sort32,
                    temp_storage.load32
                );
            }else if(sizeOfRange <= blocksize * 64){
                makeUniqueRangeSingleWarp<blocksize, 64>(
                    myRange, 
                    sizeOfRange, 
                    sizeOfUniqueRange,
                    anchorId,
                    temp_storage.sort64,
                    temp_storage.load64
                );
            }else{
                makeUniqueRangeSingleWarp<blocksize, elemsPerThread>(
                    myRange, 
                    sizeOfRange, 
                    sizeOfUniqueRange,
                    anchorId,
                    temp_storage.sortFull,
                    temp_storage.loadFull
                );
            }

            
        }
    }

}




template<int blocksize, int elemsPerThread>
__global__
void makeUniqueRangesKernelWithIntrinsicsMultiWarp(
        read_number* __restrict__ inoutData, 
        int* __restrict__ sizesOfUniqueRanges, 
        int numSequences,
        const read_number* __restrict__ anchorIds,
        const int* __restrict__ rangesPerSequenceBegins,
        int globalOffset){

    using BlockRadixSort = cub::BlockRadixSort<read_number, blocksize, elemsPerThread>;
    using BlockLoad = cub::BlockLoad<read_number, blocksize, elemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using WarpScan = cub::WarpScan<int>;
    
    constexpr int warpsize = 32;
    constexpr int numWarpsPerBlock = blocksize / warpsize;
    static_assert(blocksize % warpsize == 0, "blocksize not multiple of warpsize");

    __shared__ union{
        typename BlockRadixSort::TempStorage sort;
        typename BlockLoad::TempStorage load;
        typename WarpScan::TempStorage warpscan[numWarpsPerBlock];  
    } temp_storage;

    __shared__ read_number rightNeighborPerWarp[numWarpsPerBlock];
    __shared__ int countsPerWarp[numWarpsPerBlock];
    __shared__ int countsPrefixsum[numWarpsPerBlock+1];

    const int warpId = threadIdx.x / warpsize;
    const int laneId = threadIdx.x % warpsize;

    for(int sequenceIndex = blockIdx.x; sequenceIndex < numSequences; sequenceIndex += gridDim.x){

        
        read_number tempregs[elemsPerThread];   

        #pragma unroll
        for(int i = 0; i <elemsPerThread; i++){
            tempregs[i] = std::numeric_limits<read_number>::max();
        }

        const int sizeOfRange = rangesPerSequenceBegins[sequenceIndex + 1] - rangesPerSequenceBegins[sequenceIndex];
        if(sizeOfRange == 0){
            if(threadIdx.x == 0){
                sizesOfUniqueRanges[sequenceIndex] = 0;
            }
        }else{
            const read_number anchorId = anchorIds[sequenceIndex];
            read_number* const myRange = inoutData + rangesPerSequenceBegins[sequenceIndex] - globalOffset;

            assert(sizeOfRange <= elemsPerThread * blocksize);

            BlockLoad(temp_storage.load).Load(
                myRange, 
                tempregs, 
                sizeOfRange
            );

            __syncthreads();

            BlockRadixSort(temp_storage.sort).SortBlockedToStriped(tempregs);

            int numUniqueElements = 0;
            #pragma unroll
            for(int i = 0; i < elemsPerThread; i++){                

                //save input for previous warp in shared memory
                if(laneId == 0){
                    if(warpId == 0){
                        rightNeighborPerWarp[numWarpsPerBlock-1] = tempregs[(i+1) % elemsPerThread];
                    }else{
                        rightNeighborPerWarp[warpId-1] = tempregs[i];
                    }
                    countsPerWarp[warpId] = 0;
                }
                __syncthreads();

                const read_number curElement = tempregs[i];

                read_number nextElement = laneId == 0 ? rightNeighborPerWarp[warpId] : tempregs[i];
                nextElement = __shfl_sync(0xFFFFFFFF, nextElement, laneId+1);               

                //find elements which are not equal to their right neighbor, not equal to anchorId, and not out of range
                const bool predicate = (curElement != nextElement) && (curElement != anchorId) && (i * blocksize + threadIdx.x < sizeOfRange);

                const uint32_t mask = __ballot_sync(0xFFFFFFFF, predicate);
                const int count = __popc(mask);

                if(laneId == 0){
                    countsPerWarp[warpId] = count;
                }
                __syncthreads();

                if(warpId == 0){
                    int c = laneId < numWarpsPerBlock ? countsPerWarp[laneId] : 0;
                    int warp_aggregate = 0;
                    WarpScan(temp_storage.warpscan[warpId]).ExclusiveSum(c, c, warp_aggregate);
                    //__syncwarp();

                    if(laneId < numWarpsPerBlock){
                        countsPrefixsum[laneId] = c;
                    }
                    if(laneId == 0){
                        countsPrefixsum[numWarpsPerBlock] = warp_aggregate;
                    }
                }

                const int numPre = __popc(mask & ((1 << laneId) -1));

                __syncthreads();

                //get position                
                const int position = numUniqueElements + countsPrefixsum[warpId] + numPre;

                if(predicate){
                    myRange[position] = curElement;
                }
                numUniqueElements += countsPrefixsum[numWarpsPerBlock];
            }

            if(threadIdx.x == 0){
                sizesOfUniqueRanges[sequenceIndex] = numUniqueElements;
            }
        }
    }

}









void makeCompactUniqueRangesGmem(
        MergeRangesGpuHandle<read_number>& handle, 
        const std::pair<const read_number*, const read_number*>* ranges,
        int numRanges, 
        const read_number* d_anchorIds, 
        int rangesPerSequence, 
        int totalNumElements, 
        bool onlyAlloc,
        cudaStream_t stream){

    assert(false && "cannot use gmem merge currently");

    const int numSequences = numRanges / rangesPerSequence;
    if(numSequences == 0){
        return;
    }
    
    
    {
        size_t temp_storage_bytes = 0;
        size_t temp_storage_bytes2 = 0;

        cub::DeviceSegmentedRadixSort::SortKeys(
            nullptr, 
            temp_storage_bytes, 
            handle.d_data.get(), 
            handle.d_results.get(),
            totalNumElements, 
            numSequences, 
            handle.d_rangesBeginPerSequence.get(), 
            handle.d_rangesBeginPerSequence.get() + 1,
            0,
            sizeof(read_number) * 8,
            stream
        );

        using CountIter = cub::CountingInputIterator<int>;
        read_number* d_data = handle.d_data.get();
        auto toHeadFlag = [=] __device__ (int index){
            if(index == 0 || d_data[index-1] != d_data[index]){
                return 1;
            }else{
                return 0;
            }
        };

        using TransformOp = decltype(toHeadFlag);
        using HeadFlagsIter = cub::TransformInputIterator<int, TransformOp, CountIter>;

        HeadFlagsIter head_flags(CountIter{0}, toHeadFlag);

        cub::DeviceSegmentedReduce::Sum(
            nullptr, 
            temp_storage_bytes2, 
            head_flags, 
            handle.d_uniqueRangeLengths.get(),
            numSequences, 
            handle.d_rangesBeginPerSequence.get(), 
            handle.d_rangesBeginPerSequence.get() + 1,
            stream
        );

        temp_storage_bytes = std::max(temp_storage_bytes, temp_storage_bytes2);
    
        cub::DeviceScan::InclusiveSum(
            nullptr, 
            temp_storage_bytes2, 
            handle.d_uniqueRangeLengths.get(), 
            handle.d_uniqueRangeLengthsPrefixsum.get() + 1, 
            numSequences,
            stream
        );

        temp_storage_bytes = std::max(temp_storage_bytes, temp_storage_bytes2);
    
        cub::DeviceSelect::Flagged(
            nullptr, 
            temp_storage_bytes2,
            handle.d_data.get(), 
            head_flags, 
            handle.d_results.get(), 
            cub::DiscardOutputIterator<int>{}, //d_num_selected_out
            totalNumElements,
            stream
        );

        temp_storage_bytes = std::max(temp_storage_bytes, temp_storage_bytes2);
        handle.cubTempStorage.resize(temp_storage_bytes);
    }

    if(onlyAlloc){
        return;
    }

    cudaMemcpyAsync(
        handle.d_rangesBeginPerSequence.get(), 
        handle.h_rangesBeginPerSequence.get(),
        sizeof(int) * (numSequences+1),
        H2D,
        stream
    ); CUERR;

    //copy data of ranges into contiguous pinned memory, then to device async
    {
        auto begin = handle.h_data.get();
        auto end = handle.h_data.get();
        for(int rangeIndex = 0; rangeIndex < numRanges; rangeIndex++){
            end = std::copy(
                ranges[rangeIndex].first, 
                ranges[rangeIndex].second,
                begin
            );
            
            begin = end;
        }
    }

    cudaMemcpyAsync(
        handle.d_data.get(), 
        handle.h_data.get(),
        sizeof(read_number) * totalNumElements,
        H2D,
        stream
    ); CUERR;

    size_t cubTempStorageSize = handle.cubTempStorage.sizeInBytes();

    cub::DeviceSegmentedRadixSort::SortKeys(
        handle.cubTempStorage.get(), 
        cubTempStorageSize, 
        handle.d_data.get(), 
        handle.d_results.get(),
        totalNumElements, 
        numSequences, 
        handle.d_rangesBeginPerSequence.get(), 
        handle.d_rangesBeginPerSequence.get() + 1,
        0,
        sizeof(read_number) * 8,
        stream
    );

    std::swap(handle.d_data, handle.d_results);

    using CountIter = cub::CountingInputIterator<int>;
    read_number* d_data = handle.d_data.get();
    auto toHeadFlag = [=] __device__ (int index){
        if(index == 0 || d_data[index-1] != d_data[index]){
            return 1;
        }else{
            return 0;
        }
    };

    using TransformOp = decltype(toHeadFlag);
    using HeadFlagsIter = cub::TransformInputIterator<int, TransformOp, CountIter>;


    HeadFlagsIter head_flags(CountIter{0}, toHeadFlag);

    cub::DeviceSegmentedReduce::Sum(
        handle.cubTempStorage.get(), 
        cubTempStorageSize, 
        head_flags, 
        handle.d_uniqueRangeLengths.get(),
        numSequences, 
        handle.d_rangesBeginPerSequence.get(), 
        handle.d_rangesBeginPerSequence.get() + 1,
        stream
    );

    cub::DeviceScan::InclusiveSum(
        handle.cubTempStorage.get(), 
        cubTempStorageSize, 
        handle.d_uniqueRangeLengths.get(), 
        handle.d_uniqueRangeLengthsPrefixsum.get() + 1, 
        numSequences,
        stream
    );

    cub::DeviceSelect::Flagged(
        handle.cubTempStorage.get(), 
        cubTempStorageSize,
        handle.d_data.get(), 
        head_flags, 
        handle.d_results.get(), 
        cub::DiscardOutputIterator<int>{}, //d_num_selected_out == uniqueRangelengthsPrefixSum[numSequences]
        totalNumElements,
        stream
    );

       
}



bool makeCompactUniqueRangesSmem(
        MergeRangesGpuHandle<read_number>& handle, 
        read_number* d_compactUniqueCandidateIds,
        int* d_candidatesPerAnchor,
        int* d_candidatesPerAnchorPrefixSum,
        read_number* d_candidateIds,
        const std::pair<const read_number*, const read_number*>* ranges,
        int numRanges, 
        const read_number* d_anchorIds, 
        int rangesPerSequence, 
        int totalNumElements, 
        bool onlyAlloc,
        MergeRangesKernelType kernelType,
        cudaStream_t stream){

    const int numSequences = numRanges / rangesPerSequence;
    if(numSequences == 0){
        return true;
    }

    {
        size_t temp_storage_bytes = 0;

        cub::DeviceScan::InclusiveSum(
            nullptr, 
            temp_storage_bytes,
            d_candidatesPerAnchor,
            d_candidatesPerAnchorPrefixSum + 1,
            numSequences,
            stream
        ); CUERR;
    
        handle.cubTempStorage.resize(temp_storage_bytes);
    }

    if(onlyAlloc){
        return true;
    }

    cudaMemcpyAsync(
        handle.d_rangesBeginPerSequence.get(), 
        handle.h_rangesBeginPerSequence.get(),
        sizeof(int) * (numSequences+1),
        H2D,
        stream
    ); CUERR;

    cudaEventRecord(handle.events.back(), stream); CUERR;

    for(auto& pipelinestream : handle.streams){
        cudaStreamWaitEvent(pipelinestream, handle.events.back(), 0); CUERR;
    }

    {
        const int numstreams = handle.streams.size();

        int elementOffset = 0;
        int sequenceOffset = 0;

        bool error = false;

        for(int i = 0; i < numstreams; i++){
            const int sequenceBegin = i * numSequences / numstreams;
            const int sequenceEnd = std::min((i+1) * numSequences / numstreams, numSequences);

            const int mynumSequences = sequenceEnd - sequenceBegin;

            int mynumElements = 0;
            int largestNumElementsPerSequence = 0;
            for(int sequenceIndex = sequenceBegin; sequenceIndex < sequenceEnd; sequenceIndex++){
                
                int numElementsForSequence = 0;

                for(int k = 0; k < rangesPerSequence; k++){
                    const int rangeIndex = sequenceIndex * rangesPerSequence + k;    
                    const int sizeOfRange = std::distance(ranges[rangeIndex].first, ranges[rangeIndex].second);
                    numElementsForSequence += sizeOfRange;
                }

                largestNumElementsPerSequence = std::max(largestNumElementsPerSequence, numElementsForSequence);                
            } 

            // read_number* d_candidateIds_tmp;
            // cudaMalloc(&d_candidateIds_tmp, sizeof(read_number) * totalNumElements); CUERR;
            // cudaMemcpy(d_candidateIds_tmp, d_candidateIds, sizeof(read_number) * totalNumElements, D2D); CUERR;

            // int* d_candidatesPerAnchor_tmp;
            // cudaMalloc(&d_candidatesPerAnchor_tmp, sizeof(int) * numSequences);

            #define processData(blocksize, elemsPerThread) \
            { \
                switch(kernelType){ \
                case MergeRangesKernelType::allcub: \
                    makeUniqueRangesWithOnlyCUBKernel<(blocksize), (elemsPerThread)><<<mynumSequences, (blocksize), 0, handle.streams[i]>>>( \
                        d_candidateIds + elementOffset,  \
                        d_candidatesPerAnchor + sequenceOffset,  \
                        mynumSequences, \
                        d_anchorIds, \
                        handle.d_rangesBeginPerSequence.get() + sequenceOffset, \
                        elementOffset \
                    ); CUERR; \
                    break; \
                case MergeRangesKernelType::popcmultiwarp: \
                    makeUniqueRangesKernelWithIntrinsicsMultiWarp<(blocksize), (elemsPerThread)><<<mynumSequences, (blocksize), 0, handle.streams[i]>>>( \
                        d_candidateIds + elementOffset,  \
                        d_candidatesPerAnchor + sequenceOffset,  \
                        mynumSequences, \
                        d_anchorIds, \
                        handle.d_rangesBeginPerSequence.get() + sequenceOffset, \
                        elementOffset \
                    ); CUERR; \
                    break; \
                case MergeRangesKernelType::popcsinglewarp: \
                    makeUniqueRangesKernelWithIntrinsicsSingleWarp<32, (elemsPerThread)><<<mynumSequences, 32, 0, handle.streams[i]>>>( \
                        d_candidateIds + elementOffset,  \
                        d_candidatesPerAnchor + sequenceOffset,  \
                        mynumSequences, \
                        d_anchorIds, \
                        handle.d_rangesBeginPerSequence.get() + sequenceOffset, \
                        elementOffset \
                    ); CUERR; \
                    break; \
                case MergeRangesKernelType::popcsinglewarpchunked: \
                default: std::cerr << "unknown kernel type\n"; \
                } \
            }


            if(largestNumElementsPerSequence <= 32){
                constexpr int blocksize = 32;
                constexpr int elemsPerThread = 1;
                assert(largestNumElementsPerSequence <= blocksize * elemsPerThread);

                processData(blocksize, elemsPerThread);
            }else if(largestNumElementsPerSequence <= 64){
                constexpr int blocksize = 64;
                constexpr int elemsPerThread = 1;
                assert(largestNumElementsPerSequence <= blocksize * elemsPerThread);

                processData(blocksize, elemsPerThread);
            }else if(largestNumElementsPerSequence <= 96){
                constexpr int blocksize = 96;
                constexpr int elemsPerThread = 1;
                assert(largestNumElementsPerSequence <= blocksize * elemsPerThread);

                processData(blocksize, elemsPerThread);
            }else if(largestNumElementsPerSequence <= 128){
                constexpr int blocksize = 128;
                constexpr int elemsPerThread = 1;
                assert(largestNumElementsPerSequence <= blocksize * elemsPerThread);

                processData(blocksize, elemsPerThread);
            }else if(largestNumElementsPerSequence <= 256){
                constexpr int blocksize = 128;
                constexpr int elemsPerThread = 2;
                assert(largestNumElementsPerSequence <= blocksize * elemsPerThread);

                processData(blocksize, elemsPerThread);
            }else if(largestNumElementsPerSequence <= 512){
                constexpr int blocksize = 128;
                constexpr int elemsPerThread = 4;
                assert(largestNumElementsPerSequence <= blocksize * elemsPerThread);

                processData(blocksize, elemsPerThread);
            }else if(largestNumElementsPerSequence <= 1024){
                constexpr int blocksize = 128;
                constexpr int elemsPerThread = 8;
                assert(largestNumElementsPerSequence <= blocksize * elemsPerThread);

                processData(blocksize, elemsPerThread);
            }else if(largestNumElementsPerSequence <= 2048){
                constexpr int blocksize = 128;
                constexpr int elemsPerThread = 16;
                assert(largestNumElementsPerSequence <= blocksize * elemsPerThread);

                processData(blocksize, elemsPerThread);
            }else if(largestNumElementsPerSequence <= 4096){
                constexpr int blocksize = 128;
                constexpr int elemsPerThread = 32;
                assert(largestNumElementsPerSequence <= blocksize * elemsPerThread);

                processData(blocksize, elemsPerThread);
            }else{
                constexpr int blocksize = 128;
                constexpr int elemsPerThread = 64;


                if(largestNumElementsPerSequence <= blocksize * elemsPerThread){
                
                    processData(blocksize, elemsPerThread);

                }else{

                    std::cerr << largestNumElementsPerSequence << " > " << (blocksize * elemsPerThread) << " , cannot use smem set_union \n";

                    error = true;
                    break;
                }
            }

            cudaEventRecord(handle.events[i], handle.streams[i]); CUERR;
            cudaStreamWaitEvent(stream, handle.events[i], 0); CUERR;

            elementOffset += mynumElements;
            sequenceOffset += mynumSequences;

            if(error){
                return false;
            }
        }
        
    }    

    size_t tmpsize = handle.cubTempStorage.sizeInBytes();

    cub::DeviceScan::InclusiveSum(
        handle.cubTempStorage.get(), 
        tmpsize,
        d_candidatesPerAnchor,
        d_candidatesPerAnchorPrefixSum + 1,
        numSequences,
        stream
    ); CUERR;

    compactDataOfUniqueRanges<256><<<numSequences, 256, 0, stream>>>(
        d_compactUniqueCandidateIds,
        d_candidateIds,
        d_candidatesPerAnchorPrefixSum,
        handle.d_rangesBeginPerSequence.get(),
        numSequences
    ); CUERR;

    //cudaEventRecord(handle.events.back(), stream); CUERR;

    return true;
}



void mergeRangesGpuAsync(
        MergeRangesGpuHandle<read_number>& handle, 
        read_number* d_compactUniqueCandidateIds,
        int* d_candidatesPerAnchor,
        int* d_candidatesPerAnchorPrefixSum,
        read_number* d_candidateIds,
        const std::pair<const read_number*, const read_number*>* h_ranges, 
        int numRanges, 
        const read_number* d_anchorIds, 
        int rangesPerSequence, 
        cudaStream_t stream,
        MergeRangesKernelType kernelType){
    
    const int numSequences = numRanges / rangesPerSequence;
    if(numSequences == 0){
        return;
    }

    handle.d_rangesBeginPerSequence.resize(numSequences+1);
    handle.h_rangesBeginPerSequence.resize(numSequences+1);

    handle.h_rangesBeginPerSequence[0] = 0;

    //int longestRange = 0;

    //nvtx::push_range("longestrange", 4);
    int maxNumResults = 0;
    for(int i = 0; i < numSequences; i++){   
        //int rangeOfSequence = 0;     
        for(int k = 0; k < rangesPerSequence; k++){
            const int rangeIndex = i * rangesPerSequence + k;
            maxNumResults += std::distance(h_ranges[rangeIndex].first, h_ranges[rangeIndex].second);
            //rangeOfSequence += std::distance(h_ranges[rangeIndex].first, h_ranges[rangeIndex].second);
        }
        //longestRange = std::max(longestRange, rangeOfSequence);
        handle.h_rangesBeginPerSequence[i+1] = maxNumResults;
    }
    //nvtx::pop_range();

    //std::cerr << "longestRange = " << longestRange << "\n";
    handle.d_data.resize(maxNumResults);
    handle.h_data.resize(maxNumResults);

    handle.d_results.resize(maxNumResults);
    handle.h_results.resize(maxNumResults);

    handle.h_numresults.resize(1);

    handle.d_uniqueRangeLengths.resize(numSequences);
    handle.h_uniqueRangeLengths.resize(numSequences);

    handle.d_uniqueRangeLengthsPrefixsum.resize(numSequences + 1);
    cudaMemsetAsync(handle.d_uniqueRangeLengthsPrefixsum, 0, sizeof(int), stream); CUERR;


    cudaMemsetAsync(d_candidatesPerAnchorPrefixSum, 0, sizeof(int), stream); CUERR;


    //TIMERSTARTCPU(makeCompactUniqueRanges);

    if(kernelType == MergeRangesKernelType::devicewide){
        makeCompactUniqueRangesGmem(
            handle, 
            h_ranges,
            numRanges, 
            d_anchorIds,
            rangesPerSequence, 
            maxNumResults, 
            false,
            stream
        );
        
    }else{
        bool success = makeCompactUniqueRangesSmem(
            handle, 
            d_compactUniqueCandidateIds,
            d_candidatesPerAnchor,
            d_candidatesPerAnchorPrefixSum,
            d_candidateIds,
            h_ranges,
            numRanges, 
            d_anchorIds,
            rangesPerSequence, 
            maxNumResults, 
            false,
            kernelType,
            stream
        );
    }

}



OperationResult mergeRangesGpu(
        MergeRangesGpuHandle<read_number>& handle, 
        const std::pair<const read_number*, const read_number*>* h_ranges, 
        int numRanges, 
        const read_number* d_anchorIds, 
        int rangesPerSequence, 
        cudaStream_t stream,
        MergeRangesKernelType kernelType){
    
    const int numSequences = numRanges / rangesPerSequence;
    if(numSequences == 0){
        return OperationResult{};
    }

    // mergeRangesGpuAsync(
    //     handle, 
    //     h_ranges, 
    //     numRanges, 
    //     d_anchorIds, 
    //     rangesPerSequence, 
    //     stream,
    //     kernelType
    // );

    cudaMemcpyAsync(
        handle.h_uniqueRangeLengths.get(), 
        handle.d_uniqueRangeLengths.get(), 
        sizeof(int) * numSequences, 
        D2H, 
        stream
    ); CUERR;
    cudaMemcpyAsync(
        handle.h_numresults.get(), 
        handle.d_uniqueRangeLengthsPrefixsum.get() + numSequences, 
        sizeof(int), 
        D2H, 
        stream
    ); CUERR;
    cudaMemcpyAsync(
        handle.h_results.get(), 
        handle.d_results.get(), 
        sizeof(read_number) * handle.d_results.size(), 
        D2H, 
        stream
    ); CUERR; 


    cudaStreamSynchronize(stream); CUERR;

    //TIMERSTOPCPU(makeCompactUniqueRanges);

    OperationResult result;

    result.candidateIds.clear();
    result.candidateIds.resize(*handle.h_numresults.get());

    result.candidateIdsPerSequence.clear();
    result.candidateIdsPerSequence.resize(numSequences, 0);

    std::copy_n(
        handle.h_results.get(), 
        *handle.h_numresults.get(), 
        result.candidateIds.begin()
    );

    std::copy_n(
        handle.h_uniqueRangeLengths.get(), 
        numSequences, 
        result.candidateIdsPerSequence.begin()
    );

    return result;

}



} //namespace care