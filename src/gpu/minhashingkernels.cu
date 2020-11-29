#include <cstdint>
#include <cstddef>
#include <limits>

#include <config.hpp>
#include <hpc_helpers.cuh>
#include <sequencehelpers.hpp>
#include <gpu/cuda_block_select.cuh>

#include <cub/cub.cuh>

namespace care{
namespace gpu{


__global__
void minhashSignaturesKernel(
    std::uint64_t* __restrict__ signatures,
    std::size_t signaturesRowPitchElements,
    const unsigned int* __restrict__ sequences2Bit,
    std::size_t sequenceRowPitchElements,
    int numSequences,
    const int* __restrict__ sequenceLengths,
    int k,
    int numHashFuncs,
    int firstHashFunc
){
            
    //constexpr int blocksize = 128;
    constexpr int maximum_kmer_length = max_k<std::uint64_t>::value;

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
            using hasher = hashers::MurmurHash<std::uint64_t>;

            const auto smallest = min(fwd, rc);
            const auto hashvalue = hasher::hash(smallest + hashFuncId);
            minHashValue = min(minHashValue, hashvalue);
        };

        if(myLength >= k){
            //const int numKmers = myLength - k + 1;
            const std::uint64_t kmer_mask = std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - k) * 2);
            const int rcshiftamount = (maximum_kmer_length - k) * 2;

            //Compute the first kmer
            std::uint64_t kmer_encoded = mySequence[0];
            if(k <= 16){
                kmer_encoded >>= (16 - k) * 2;
            }else{
                kmer_encoded = (kmer_encoded << 32) | mySequence[1];
                kmer_encoded >>= (32 - k) * 2;
            }

            kmer_encoded >>= 2; //k-1 bases, allows easier loop

            std::uint64_t rc_kmer_encoded = SequenceHelpers::reverseComplementInt2Bit(kmer_encoded);

            auto addBase = [&](std::uint64_t encBase){
                kmer_encoded <<= 2;
                rc_kmer_encoded >>= 2;

                const std::uint64_t revcBase = (~encBase) & 3;
                kmer_encoded |= encBase;
                rc_kmer_encoded |= revcBase << 62;
            };

            constexpr int basesPerInt = SequenceHelpers::basesPerInt2Bit();

            const int itersend1 = min(SDIV(k-1, basesPerInt) * basesPerInt, myLength);

            //process sequence positions one by one
            // until the next encoded sequence data element is reached
            for(int nextSequencePos = k - 1; nextSequencePos < itersend1; nextSequencePos++){
                const int nextIntIndex = nextSequencePos / basesPerInt;
                const int nextPositionInInt = nextSequencePos % basesPerInt;

                const std::uint64_t nextBase = mySequence[nextIntIndex] >> (30 - 2 * nextPositionInInt);

                addBase(nextBase);

                handlekmer(
                    kmer_encoded & kmer_mask, 
                    rc_kmer_encoded >> rcshiftamount
                );
            }

            const int fullIntIters = (myLength - itersend1) / basesPerInt;

            //process all fully occupied encoded sequence data elements
            // improves memory access
            for(int iter = 0; iter < fullIntIters; iter++){
                const int intIndex = (itersend1 + iter * basesPerInt) / basesPerInt;
                const unsigned int data = mySequence[intIndex];

                #pragma unroll
                for(int posInInt = 0; posInInt < basesPerInt; posInInt++){
                    const std::uint64_t nextBase = data >> (30 - 2 * posInInt);

                    addBase(nextBase);

                    handlekmer(
                        kmer_encoded & kmer_mask, 
                        rc_kmer_encoded >> rcshiftamount
                    );
                }
            }

            //process remaining positions one by one
            for(int nextSequencePos = fullIntIters * basesPerInt + itersend1; nextSequencePos < myLength; nextSequencePos++){
                const int nextIntIndex = nextSequencePos / basesPerInt;
                const int nextPositionInInt = nextSequencePos % basesPerInt;

                const std::uint64_t nextBase = mySequence[nextIntIndex] >> (30 - 2 * nextPositionInInt);

                addBase(nextBase);

                handlekmer(
                    kmer_encoded & kmer_mask, 
                    rc_kmer_encoded >> rcshiftamount
                );
            }

            mySignature[myNumHashFunc] = minHashValue & kmer_mask;

        }else{
            mySignature[myNumHashFunc] = std::numeric_limits<std::uint64_t>::max();
        }
    }
} 



void callMinhashSignaturesKernel(
    std::uint64_t* __restrict__ signatures,
    std::size_t signaturesRowPitchElements,
    const unsigned int* __restrict__ sequences2Bit,
    std::size_t sequenceRowPitchElements,
    int numSequences,
    const int* __restrict__ sequenceLengths,
    int k,
    int numHashFuncs,
    int firstHashFunc,
    cudaStream_t stream
){
    dim3 block(128,1,1);
    dim3 grid(SDIV(numHashFuncs * numSequences, block.x),1,1);

    minhashSignaturesKernel<<<grid, block, 0, stream>>>(
        signatures,
        signaturesRowPitchElements,
        sequences2Bit,
        sequenceRowPitchElements,
        numSequences,
        sequenceLengths,
        k,
        numHashFuncs,
        firstHashFunc
    ); CUERR;
}



/*
    If toFind[s] exists in segment s, remove it from this segment s 
    by shifting remaining elements to the left.
    segmentSizes[s] will be updated.
*/
template<class T, int BLOCKSIZE, int ITEMS_PER_THREAD>
__global__ 
void findAndRemoveFromSegmentKernel(
    const T* __restrict__ toFind,
    T* __restrict__ items,
    int numSegments,
    int* __restrict__ segmentSizes,
    const int* __restrict__ segmentBeginOffsets
){
    constexpr int itemsPerIteration = ITEMS_PER_THREAD * BLOCKSIZE;

    assert(BLOCKSIZE == blockDim.x);
    assert(false);

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
        T items[ITEMS_PER_THREAD];
        int flags[ITEMS_PER_THREAD];

        int numSelectedTotal = 0;
        int remainingItems = segmentsize;
        const T* inputdata = items + beginOffset;
        T* outputdata = items + beginOffset;

        for(int iter = 0; iter < numIterations; iter++){
            const int validItems = min(remainingItems, itemsPerIteration);
            BlockLoad(temp_storage.load).Load(inputdata, items, validItems);

            #pragma unroll
            for(int i = 0; i < ITEMS_PER_THREAD; i++){
                if(threadIdx.x * ITEMS_PER_THREAD + i < validItems && items[i] != idToRemove){
                    flags[i] = 1;
                }else{
                    flags[i] = 0;
                }
            }

            __syncthreads();

            const int numSelected = MyBlockSelect(temp_storage.select).ForEachFlagged(items, flags, validItems,
                [&](const auto& item, const int& pos){
                    outputdata[pos] = item;
                }
            );
            assert(numSelected <= validItems);

            numSelectedTotal += numSelected;
            outputdata += numSelected;
            inputdata += validItems;

            __syncthreads();
        }

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

    dim3 block = 128;
    dim3 grid = numSegments;

    findAndRemoveFromSegmentKernel<BLOCKSIZE, ITEMS_PER_THREAD><<<grid, block, 0, stream>>>
        (d_toFind, d_items, numSegments, d_segmentSizes, d_segmentBeginOffsets);
}



} //namespace gpu
} //namespace care