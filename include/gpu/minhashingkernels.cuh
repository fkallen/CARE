#ifndef CARE_MINHASHINGKERNELS_CUH
#define CARE_MINHASHINGKERNELS_CUH

#include <cstdint>
#include <config.hpp>
#include <sequencehelpers.hpp>
#include <gpu/cudaerrorcheck.cuh>
#include <hpc_helpers.cuh>

#include <gpu/cuda_block_select.cuh>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

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
        const int* __restrict__ hashFunctionNumbers,
        std::uint64_t* kmersOfHashes,
        int* primes
    ){
                
        constexpr int maximum_kmer_length = max_k<std::uint64_t>::value;
        const std::uint64_t kmer_mask = std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - k) * 2);

        const int tid = threadIdx.x + blockIdx.x * blockDim.x;

        if(tid < numSequences * numHashFuncs){
            const int mySequenceIndex = tid / numHashFuncs;
            const int myNumHashFunc = tid % numHashFuncs;
            const int hashFuncId = hashFunctionNumbers[myNumHashFunc];
            assert(hashFuncId < 64);

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

    template<class KT>
    __global__
    void countUniqueKmersKernel(
        int numSequences,
        const KT* kmersOfHashes_sorted,
        int numHashFuncs,
        int* totalUnique
    ){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        for(int s = tid; s < numSequences; s += stride){
            const KT* const myKmers = kmersOfHashes_sorted + s * 64;

            int unique = 1;

            for(int i = 0; i < numHashFuncs-1; i++){
                if(myKmers[i] != myKmers[i+1]){
                    unique++;
                }
            }

            // if(numHashFuncs != unique){
            //     printf("s %d, unique %d\n", s, unique);
            // }

            atomicAdd(totalUnique, unique);
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
        std::uint64_t* d_kmersOfHashes;
        CUDACHECK(cudaMallocAsync(&d_kmersOfHashes, sizeof(std::uint64_t) * numSequences * 64, stream));

        helpers::call_fill_kernel_async(d_kmersOfHashes, numSequences * 64, std::numeric_limits<std::uint64_t>::max(), stream);

        constexpr std::array<int, 64> h_primes{2 ,3 ,5 ,7 ,11 ,13 ,17 ,19 ,23 ,29 ,31 ,37 ,41 ,43 ,47 ,53 ,59 ,61 ,67 ,71,
            73 ,79 ,83 ,89 ,97 ,101 ,103 ,107 ,109 ,113 ,127 ,131 ,137 ,139 ,149 ,151 ,157 ,163 ,167 ,173,
            179 ,181 ,191 ,193 ,197 ,199 ,211 ,223 ,227 ,229 ,233 ,239 ,241 ,251 ,257 ,263 ,269 ,271 ,277 ,281,
            283 ,293 ,307 ,311};
        static_assert(h_primes.size() == 64);

        int* d_primes;
        CUDACHECK(cudaMallocAsync(&d_primes, sizeof(int) * h_primes.size(), stream));
        CUDACHECK(cudaMemcpyAsync(d_primes, h_primes.data(), sizeof(int) * h_primes.size(), H2D, stream));

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
            hashFunctionNumbers,
            d_kmersOfHashes,
            d_primes
        ); CUDACHECKASYNC;

        CUDACHECK(cudaFreeAsync(d_primes, stream));

        std::uint64_t* d_kmersOfHashes_sorted;
        CUDACHECK(cudaMallocAsync(&d_kmersOfHashes_sorted, sizeof(std::uint64_t) * numSequences * 64, stream));

        auto offsets = thrust::make_transform_iterator(
            thrust::make_counting_iterator(0),
            [] __host__ __device__ (const int pos){ return pos * 64; }
        );

        std::size_t temp_storage_bytes = 0;

        cub::DeviceSegmentedRadixSort::SortKeys(
            nullptr,
            temp_storage_bytes,
            d_kmersOfHashes,
            d_kmersOfHashes_sorted,
            numSequences * 64,
            numSequences,
            offsets,
            offsets + 1,
            0,
            k * 2,
            stream
        );

        void* d_temp;
        CUDACHECK(cudaMallocAsync(&d_temp, temp_storage_bytes, stream));

        cub::DeviceSegmentedRadixSort::SortKeys(
            d_temp,
            temp_storage_bytes,
            d_kmersOfHashes,
            d_kmersOfHashes_sorted,
            numSequences * 64,
            numSequences,
            offsets,
            offsets + 1,
            0,
            k * 2,
            stream
        );

        CUDACHECK(cudaFreeAsync(d_temp, stream));

        int* d_totalunique;
        CUDACHECK(cudaMallocAsync(&d_totalunique, sizeof(int), stream));

        CUDACHECK(cudaMemsetAsync(d_totalunique, 0, sizeof(int), stream));

        countUniqueKmersKernel<<<SDIV(numSequences, 64), 64, 0, stream>>>(
            numSequences,
            d_kmersOfHashes_sorted,
            numHashFuncs,
            d_totalunique
        );
        CUDACHECKASYNC;

        int h_totalunique = 0;
        CUDACHECK(cudaMemcpyAsync(&h_totalunique, d_totalunique, sizeof(int), D2H, stream));
        CUDACHECK(cudaStreamSynchronize(stream));

        //std::cerr << "totalunique: " << h_totalunique << ", average: " << (h_totalunique /float(numSequences)) << "\n";

        CUDACHECK(cudaFreeAsync(d_totalunique, stream));
        CUDACHECK(cudaFreeAsync(d_kmersOfHashes_sorted, stream));
        CUDACHECK(cudaFreeAsync(d_kmersOfHashes, stream));

        CUDACHECK(cudaStreamSynchronize(stream));
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