#ifndef CARE_GPU_SEQUENCE_HASHER_CUH
#define CARE_GPU_SEQUENCE_HASHER_CUH


#include <config.hpp>
#include <hpc_helpers.cuh>
#include <gpu/cudaerrorcheck.cuh>
#include <gpu/cubwrappers.cuh>

#include <cassert>
#include <cstdint>

#include <cub/cub.cuh>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <gpu/rmm_utilities.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>



namespace care{
namespace gpu{
namespace gpusequencehasher{

    struct GetNumKmers{
        int k;
        __host__ __device__
        constexpr GetNumKmers(int k_) : k(k_){}

        __host__ __device__
        constexpr int operator()(int length) const noexcept{
            return (length >= k) ? (length - k + 1) : 0;
        }
    };

    template<int blocksize, class InputIter, class OutputIter>
    __global__
    void minmaxSingleBlockKernel(InputIter begin, int N, OutputIter minmax){
        using value_type_in = typename std::iterator_traits<InputIter>::value_type;
        using value_type_out = typename std::iterator_traits<OutputIter>::value_type;
        static_assert(std::is_same_v<value_type_in, value_type_out>);

        using value_type = value_type_in;

        using BlockReduce = cub::BlockReduce<value_type, blocksize>;
        __shared__ typename BlockReduce::TempStorage temp1;
        __shared__ typename BlockReduce::TempStorage temp2;

        if(blockIdx.x == 0){

            const int tid = threadIdx.x;
            const int stride = blockDim.x;

            value_type myMin = std::numeric_limits<value_type>::max();
            value_type myMax = 0;

            for(int i = tid; i < N; i += stride){
                const value_type val = *(begin + i);
                myMin = min(myMin, val);
                myMax = max(myMax, val);
            }

            myMin = BlockReduce(temp1).Reduce(myMin, cub::Min{});
            myMax = BlockReduce(temp2).Reduce(myMax, cub::Max{});

            if(tid == 0){
                *(minmax + 0) = myMin;
                *(minmax + 1) = myMax;
            }
        }
    }

    template<class HashValueType>
    __global__
    void minhashSignatures3264Kernel(
        HashValueType* __restrict__ signatures,
        std::size_t signaturesRowPitchElements,
        bool* __restrict__ valid,
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
            assert(hashFuncId < 64);

            const unsigned int* const mySequence = sequences2Bit + mySequenceIndex * sequenceRowPitchElements;
            const int myLength = sequenceLengths[mySequenceIndex];

            HashValueType* const mySignature = signatures + mySequenceIndex * signaturesRowPitchElements;
            bool* const myValid = valid + mySequenceIndex * numHashFuncs;

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
                myValid[myNumHashFunc] = true;
            }else{
                mySignature[myNumHashFunc] = std::numeric_limits<HashValueType>::max();
                myValid[myNumHashFunc] = false;
            }
        }
    }


    template<class ConstBeginOffsetsIter>
    __global__
    void makeKmersKernel(
        kmer_type* __restrict__ kmersoutput,
        ConstBeginOffsetsIter outputBeginOffsets,
        const unsigned int* __restrict__ sequences2Bit,
        std::size_t sequenceRowPitchElements,
        int numSequences,
        const int* __restrict__ sequenceLengths,
        int k
    ){
        assert(sizeof(kmer_type) * 8 / 2 >= k);

        //constexpr int blocksize = 128;
        constexpr int maximum_kmer_length = max_k<std::uint64_t>::value;
        const std::uint64_t kmer_mask = std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - k) * 2);

        for(int s = blockIdx.x; s < numSequences; s += gridDim.x){
            //compute kmers of sequences[s]

            const auto kmerOffset = outputBeginOffsets[s];

            const unsigned int* const mySequence = sequences2Bit + s * sequenceRowPitchElements;
            const int myLength = sequenceLengths[s];

            const int numKmers = (myLength >= k) ? (myLength - k + 1) : 0;
            // if(threadIdx.x == 0){
            //     printf("s %d, length %d, numkmers %d\n", s, myLength, numKmers);
            // }

            for(int i = threadIdx.x; i < numKmers; i += blockDim.x){
                //compute kmer i

                const int firstIntIndex = i / 16;
                const int secondIntIndex = (i + k - 1) / 16;
                std::uint64_t kmer = 0;
                if(firstIntIndex == secondIntIndex){
                    const std::uint64_t firstInt = mySequence[firstIntIndex];
                    kmer = (firstInt >> 2*(16 - (i+k)%16)) & kmer_mask;
                }else{
                    const std::uint64_t firstInt = mySequence[firstIntIndex];
                    const std::uint64_t secondInt = mySequence[secondIntIndex];
                    const int basesInFirst = 16 - (i % 16);
                    const int basesInSecond = k - basesInFirst;
                    kmer = ((firstInt << 2*basesInSecond) | (secondInt >> 2*(16 - (i+k)%16))) & kmer_mask;
                }

                kmersoutput[kmerOffset + i] = kmer_type(kmer);

                // if(s == 0){
                //     printf("i %d, kmer %lu\n", i, kmer_type(kmer));
                // }
            }
        }
    }


    template<class HashValueType>
    __global__
    void hashKmersKernel(
        HashValueType* __restrict__ signatures,
        std::size_t signaturesRowPitchElements,
        bool* __restrict__ isValid,
        const kmer_type* __restrict__ kmers,
        const int* __restrict__ kmerBeginOffsets,
        const int* __restrict__ kmerEndOffsets,
        int k,
        int numSequences,
        int numHashFuncs,
        const int* __restrict__ hashFunctionNumbers
    ){

        using hasher = hashers::MurmurHash<std::uint64_t>;
                
        constexpr int maximum_kmer_length = max_k<std::uint64_t>::value;
        const std::uint64_t kmer_mask = std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - k) * 2);
        const int rcshiftamount = (maximum_kmer_length - k) * 2;

        for(int s = blockIdx.x; s < numSequences; s += gridDim.x){
            const int kmersBegin = kmerBeginOffsets[s];
            const int kmersEnd = kmerEndOffsets[s];
            const int numKmers = kmersEnd - kmersBegin;

            HashValueType* const mySignature = signatures + s * signaturesRowPitchElements;
            bool* const myIsValid = isValid + s * numHashFuncs;

            if(numKmers > 0){
                for(int i = threadIdx.x; i < numHashFuncs; i += blockDim.x){

                    const int hashFuncId = hashFunctionNumbers[i];

                    std::uint64_t minHashValue = std::numeric_limits<std::uint64_t>::max();

                    for(int x = 0; x < numKmers; x++){
                        const std::uint64_t kmer = kmers[kmersBegin + x];
                        const std::uint64_t rc_kmer = SequenceHelpers::reverseComplementInt2Bit(kmer) >> rcshiftamount;
                        const auto smallest = min(kmer, rc_kmer);
                        const auto hashvalue = hasher::hash(smallest + hashFuncId);
                        minHashValue = min(minHashValue, hashvalue);
                    }

                    mySignature[i] = minHashValue;
                    myIsValid[i] = true;
                }
            }else{
                for(int i = threadIdx.x; i < numHashFuncs; i += blockDim.x){
                    myIsValid[i] = false;
                }
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
            const KT* const myKmers = kmersOfHashes_sorted + s * numHashFuncs;

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
} //namespace gpusequencehasher

template<class HashValueType>
struct GPUSequenceHasher{

    struct Result{
        Result(
            int numSequences,
            int numHashFuncs,
            cudaStream_t stream, 
            rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
        ) : d_hashvalues(numSequences * numHashFuncs, stream, mr),
            d_isValid(numSequences * numHashFuncs, stream, mr){

            //CUDACHECK(cudaMemsetAsync(d_isValid.data(), 0, sizeof(bool) * d_isValid.size(), stream));
        }

        rmm::device_uvector<HashValueType> d_hashvalues;
        rmm::device_uvector<bool> d_isValid;
    };

    struct ComputedKmers{
        ComputedKmers(
            int numSequences,
            cudaStream_t stream, 
            rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
        ) : d_offsets(numSequences + 1, stream, mr),
            d_kmers(0, stream, mr){

        }

        rmm::device_uvector<int> d_offsets;
        rmm::device_uvector<kmer_type> d_kmers;
    };

    ComputedKmers computeKmers(
        const unsigned int* __restrict__ d_sequences2Bit,
        std::size_t sequenceRowPitchElements,
        int numSequences,
        const int* __restrict__ d_sequenceLengths,
        int k,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
    ){
        assert(sizeof(kmer_type) * 8 / 2 >= k);
        assert(k > 0);

        ComputedKmers result(numSequences, stream, mr);

        CUDACHECK(cudaMemsetAsync(result.d_offsets.data(), 0, sizeof(int), stream));

        auto d_numKmersPerSequence = thrust::make_transform_iterator(
            d_sequenceLengths,
            gpusequencehasher::GetNumKmers{k}
        );

        CubCallWrapper cub(mr);
        cub.cubInclusiveSum(d_numKmersPerSequence, result.d_offsets.data() + 1, numSequences, stream);

        const int totalNumKmers = result.d_offsets.back_element(stream);
        CUDACHECK(cudaStreamSynchronize(stream));

        result.d_kmers.resize(totalNumKmers, stream);

        gpusequencehasher::makeKmersKernel<<<numSequences, 128, 0, stream>>>(
            result.d_kmers.data(),
            result.d_offsets.data(),
            d_sequences2Bit,
            sequenceRowPitchElements,
            numSequences,
            d_sequenceLengths,
            k
        );
        CUDACHECKASYNC;

        return result;
    }


    ComputedKmers computeUniqueKmers(
        const unsigned int* __restrict__ d_sequences2Bit,
        std::size_t sequenceRowPitchElements,
        int numSequences,
        const int* __restrict__ d_sequenceLengths,
        int k,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
    ){
        assert(sizeof(kmer_type) * 8 / 2 >= k);
        assert(k > 0);

        ComputedKmers computedKmers = computeKmers(
            d_sequences2Bit,
            sequenceRowPitchElements,
            numSequences,
            d_sequenceLengths,
            k,
            stream,
            mr
        );

        // helpers::lambda_kernel<<<1,1,0,stream>>>(
        //     [offsets = computedKmers.d_offsets.data(), size = int(computedKmers.d_offsets.size())] __device__ (){
        //         printf("before: ");
        //         for(int i = 0; i < min(size, 20); i++){
        //             printf("%d ", offsets[i]);
        //         }
        //         printf("\n");
        //     }
        // );

        auto d_numKmersPerSequence = thrust::make_transform_iterator(
            d_sequenceLengths,
            gpusequencehasher::GetNumKmers{k}
        );

        int h_minmaxNumKmersPerSequence[2];
        rmm::device_uvector<int> d_minmaxNumKmersPerSequence(2, stream, mr);

        gpusequencehasher::minmaxSingleBlockKernel<512><<<1, 512, 0, stream>>>(
            d_numKmersPerSequence,
            numSequences,
            d_minmaxNumKmersPerSequence.data()
        ); CUDACHECKASYNC; 

        CUDACHECK(cudaMemcpyAsync(&h_minmaxNumKmersPerSequence[0], d_minmaxNumKmersPerSequence.data(), sizeof(int) * 2, D2H, stream));
        CUDACHECK(cudaStreamSynchronize(stream));

        constexpr int begin_bit = 0;
        const int end_bit = 2 * k;

        rmm::device_uvector<kmer_type> d_uniqueKmers(computedKmers.d_kmers.size(), stream, mr);
        rmm::device_uvector<int> d_numUniquePerSequence(numSequences, stream, mr);

        GpuSegmentedUnique::unique(
            computedKmers.d_kmers.data(),
            computedKmers.d_kmers.size(),
            d_uniqueKmers.data(),
            d_numUniquePerSequence.data(),
            numSequences,
            h_minmaxNumKmersPerSequence[1],
            computedKmers.d_offsets.data(),
            computedKmers.d_offsets.data() + 1,
            begin_bit,
            end_bit,
            stream,
            mr
        );

        CubCallWrapper cub(mr);
        cub.cubInclusiveSum(d_numUniquePerSequence.data(), computedKmers.d_offsets.data() + 1, numSequences, stream);

        std::swap(computedKmers.d_kmers, d_uniqueKmers);

        // helpers::lambda_kernel<<<1,1,0,stream>>>(
        //     [offsets = computedKmers.d_offsets.data(), size = int(computedKmers.d_offsets.size())] __device__ (){
        //         printf("after: ");
        //         for(int i = 0; i < min(size, 20); i++){
        //             printf("%d ", offsets[i]);
        //         }
        //         printf("\n");
        //     }
        // );

        return computedKmers;
    }

    Result hashUniqueKmers(
        const unsigned int* __restrict__ d_sequences2Bit,
        std::size_t sequenceRowPitchElements,
        int numSequences,
        const int* __restrict__ d_sequenceLengths,
        int k,
        int numHashFuncs,
        const int* __restrict__ d_hashFunctionNumbers,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
    ){
        assert(sizeof(kmer_type) * 8 / 2 >= k);
        assert(k > 0);

        if(numSequences == 0){
            return Result{0,0,stream, mr};
        }

        ComputedKmers uniqueKmers = computeUniqueKmers(
            d_sequences2Bit,
            sequenceRowPitchElements,
            numSequences,
            d_sequenceLengths,
            k,
            stream,
            mr
        );

        Result result(numSequences, numHashFuncs, stream, mr);

        gpusequencehasher::hashKmersKernel<<<numSequences, 128, 0, stream>>>(
            result.d_hashvalues.data(),
            numHashFuncs,
            result.d_isValid.data(),
            uniqueKmers.d_kmers.data(),
            uniqueKmers.d_offsets.data(),
            uniqueKmers.d_offsets.data() + 1,
            k,
            numSequences,
            numHashFuncs,
            d_hashFunctionNumbers
        ); 
        CUDACHECKASYNC;

        // helpers::lambda_kernel<<<1,1,0,stream>>>(
        //     [vec = result.d_hashvalues.data()] __device__ (){
        //         for(int i = 0; i < 48; i++){
        //             printf("%lu ", vec[i]);
        //         }
        //         printf("\n");
        //     }
        // );

        CUDACHECKASYNC;

        return result;
    }

    Result hash(
        const unsigned int* __restrict__ sequences2Bit,
        std::size_t sequenceRowPitchElements,
        int numSequences,
        const int* __restrict__ sequenceLengths,
        int k,
        int numHashFuncs,
        const int* __restrict__ hashFunctionNumbers,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
    ){
        if(numSequences == 0){
            return Result{0,0,stream, mr};
        }
        Result result(numSequences, numHashFuncs, stream, mr);

        dim3 block(128,1,1);
        dim3 grid(SDIV(numHashFuncs * numSequences, block.x),1,1);

        gpusequencehasher::minhashSignatures3264Kernel<<<grid, block, 0, stream>>>(
            result.d_hashvalues.data(),
            numHashFuncs,
            result.d_isValid.data(),
            sequences2Bit,
            sequenceRowPitchElements,
            numSequences,
            sequenceLengths,
            k,
            numHashFuncs,
            hashFunctionNumbers
        ); CUDACHECKASYNC;

        return result;

    }


};


} //namespace gpu
} //namespace care



#endif