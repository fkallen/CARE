#ifndef CARE_MINHASHINGKERNELS_CUH
#define CARE_MINHASHINGKERNELS_CUH

#include <cstdint>
#include <config.hpp>
#include <sequencehelpers.hpp>
#include <gpu/cudaerrorcheck.cuh>

#include <gpu/cuda_block_select.cuh>
#include <cub/cub.cuh>

#include <hpc_helpers.cuh>

namespace care{
namespace gpu{

    template<class HashValueType>
    __global__
    void minhashSignatures3264Kernel(
        HashValueType* __restrict__ signatures,
        std::size_t signaturesRowPitchElements,
        const unsigned int* __restrict__ sequences2Bit,
        std::size_t sequenceRowPitchElements,
        int numSequences,
        const int* __restrict__ sequenceLengths,
        int k,
        int numHashFuncs,
        const int* __restrict__ hashFunctionNumbers
    ){
                
        constexpr int maximum_kmer_length = max_k<std::uint64_t>::value;
        const std::uint64_t kmer_mask = std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - k) * 2);

        const int tid = threadIdx.x + blockIdx.x * blockDim.x;

        if(tid < numSequences * numHashFuncs){
            const int mySequenceIndex = tid / numHashFuncs;
            const int myNumHashFunc = tid % numHashFuncs;
            const int hashFuncId = hashFunctionNumbers[myNumHashFunc];

            const unsigned int* const mySequence = sequences2Bit + mySequenceIndex * sequenceRowPitchElements;
            const int myLength = sequenceLengths[mySequenceIndex];

            HashValueType* const mySignature = signatures + mySequenceIndex * signaturesRowPitchElements;

            if(myLength >= k){
                std::uint64_t minHashValue = std::numeric_limits<std::uint64_t>::max();

                SequenceHelpers::forEachEncodedCanonicalKmerFromEncodedSequence(
                    mySequence,
                    myLength,
                    k,
                    [&](std::uint64_t kmer, int /*pos*/){
                        using hasher = hashers::MurmurHash<std::uint64_t>;

                        const auto hashvalue = hasher::hash(kmer + hashFuncId);
                        minHashValue = min(minHashValue, hashvalue);
                    }
                );

                mySignature[myNumHashFunc] = HashValueType(minHashValue & kmer_mask);

            }else{
                mySignature[myNumHashFunc] = std::numeric_limits<HashValueType>::max();
            }
        }
    } 


    template<class HashValueType>
    void callMinhashSignatures3264Kernel(
        HashValueType* __restrict__ signatures,
        std::size_t signaturesRowPitchElements,
        const unsigned int* __restrict__ sequences2Bit,
        std::size_t sequenceRowPitchElements,
        int numSequences,
        const int* __restrict__ sequenceLengths,
        int k,
        int numHashFuncs,
        const int* __restrict__ hashFunctionNumbers,
        cudaStream_t stream
    ){
        dim3 block(128,1,1);
        dim3 grid(SDIV(numHashFuncs * numSequences, block.x),1,1);

        minhashSignatures3264Kernel<<<grid, block, 0, stream>>>(
            signatures,
            signaturesRowPitchElements,
            sequences2Bit,
            sequenceRowPitchElements,
            numSequences,
            sequenceLengths,
            k,
            numHashFuncs,
            hashFunctionNumbers
        ); CUDACHECKASYNC;
    }


    template<int blocksize>
    void transformSignatureToKmers(
        kmer_type* kmers,
        std::size_t kmersRowPitchElements,
        const std::uint64_t* signatures,
        std::size_t signaturesRowPitchElements,
        int numHashFuncs,
        int k,
        int numSequences
    ){
        constexpr int maximum_kmer_length = max_k<kmer_type>::value;
        const kmer_type kmer_mask = std::numeric_limits<kmer_type>::max() >> ((maximum_kmer_length - k) * 2);

        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        for(int i = tid; i < numHashFuncs * numSequences; i += stride){
            const int s = i / numHashFuncs;
            const int h = i % numHashFuncs;

            //narrowing intentional
            kmer_type kmer = signatures[s * signaturesRowPitchElements + h];
            kmer &= kmer_mask;
            kmers[s * kmersRowPitchElements + h] = kmer;
        }
    }

    void callMinhashSignaturesKernel(
        std::uint64_t* d_signatures,
        std::size_t signaturesRowPitchElements,
        const unsigned int* d_sequences2Bit,
        std::size_t sequenceRowPitchElements,
        int numSequences,
        const int* d_sequenceLengths,
        int k,
        int numHashFuncs,
        int firstHashFunc,
        cudaStream_t stream
    );

    void callMinhashSignaturesKernel(
        std::uint64_t* d_signatures,
        std::size_t signaturesRowPitchElements,
        const unsigned int* d_sequences2Bit,
        std::size_t sequenceRowPitchElements,
        int numSequences,
        const int* d_sequenceLengths,
        int k,
        int numHashFuncs,
        const int* d_hashFunctionNumbers,
        cudaStream_t stream
    );

    /*
    If toFind[s] exists in segment s, remove it from this segment s 
    by shifting remaining elements to the left.
    segmentSizes[s] will be updated.
    */
    template<class T, int BLOCKSIZE, int ITEMS_PER_THREAD>
    __global__ 
    void findAndRemoveFromSegmentKernel(
        const T* __restrict__ toFind,
        T* items,
        int numSegments,
        int* __restrict__ segmentSizes,
        const int* __restrict__ segmentBeginOffsets
    ){
        constexpr int itemsPerIteration = ITEMS_PER_THREAD * BLOCKSIZE;

        assert(BLOCKSIZE == blockDim.x);

        using BlockLoad = cub::BlockLoad<T, BLOCKSIZE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
        using MyBlockSelect = BlockSelect<T, BLOCKSIZE>;

        __shared__ union{
            typename BlockLoad::TempStorage load;
            typename MyBlockSelect::TempStorage select;
        } temp_storage;

        for(int s = blockIdx.x; s < numSegments; s += gridDim.x){
            const int segmentsize = segmentSizes[s];
            const int beginOffset = segmentBeginOffsets[s];
            const T idToRemove = toFind[s];

            const int numIterations = SDIV(segmentsize, itemsPerIteration);
            T myitems[ITEMS_PER_THREAD];
            int flags[ITEMS_PER_THREAD];

            int numSelectedTotal = 0;
            int remainingItems = segmentsize;
            const T* inputdata = items + beginOffset;
            T* outputdata = items + beginOffset;

            for(int iter = 0; iter < numIterations; iter++){
                const int validItems = min(remainingItems, itemsPerIteration);
                BlockLoad(temp_storage.load).Load(inputdata, myitems, validItems);

                #pragma unroll
                for(int i = 0; i < ITEMS_PER_THREAD; i++){
                    if(threadIdx.x * ITEMS_PER_THREAD + i < validItems && myitems[i] != idToRemove){
                        flags[i] = 1;
                    }else{
                        flags[i] = 0;
                    }
                }

                __syncthreads();

                const int numSelected = MyBlockSelect(temp_storage.select).ForEachFlagged(myitems, flags, validItems,
                    [&](const auto& item, const int& pos){
                        outputdata[pos] = item;
                    }
                );
                assert(numSelected <= validItems);

                numSelectedTotal += numSelected;
                outputdata += numSelected;
                inputdata += validItems;
                remainingItems -= validItems;

                __syncthreads();
            }

            assert(segmentsize >= numSelectedTotal);

            //update segment size
            if(numSelectedTotal != segmentsize){
                if(threadIdx.x == 0){
                    segmentSizes[s] = numSelectedTotal;
                }
            }
        }
    }

    template<class T, int BLOCKSIZE, int ITEMS_PER_THREAD>
    void callFindAndRemoveFromSegmentKernel(
        const T* d_toFind,
        T* d_items,
        int numSegments,
        int* d_segmentSizes,
        const int* d_segmentBeginOffsets,
        cudaStream_t stream
    ){
        if(numSegments <= 0){
            return;
        }

        dim3 block = BLOCKSIZE;
        dim3 grid = numSegments;

        findAndRemoveFromSegmentKernel<T, BLOCKSIZE, ITEMS_PER_THREAD><<<grid, block, 0, stream>>>
            (d_toFind, d_items, numSegments, d_segmentSizes, d_segmentBeginOffsets);
    }
}
}





#endif