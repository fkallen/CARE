#ifndef CARE_MINHASHKERNELS_HPP
#define CARE_MINHASHKERNELS_HPP

#include <config.hpp>
#include <hpc_helpers.cuh>

#include <array>
#include <vector>

namespace care{


template<class T>
struct MergeRangesGpuHandle{
    helpers::SimpleAllocationDevice<T> d_data;
    helpers::SimpleAllocationPinnedHost<T> h_data;
    
    helpers::SimpleAllocationPinnedHost<int> h_rangesBeginPerSequence;
    helpers::SimpleAllocationDevice<int> d_rangesBeginPerSequence;

    helpers::SimpleAllocationDevice<T> d_results;
    helpers::SimpleAllocationPinnedHost<T> h_results;

    helpers::SimpleAllocationPinnedHost<int> h_numresults;

    helpers::SimpleAllocationPinnedHost<int> h_num_runs;
    helpers::SimpleAllocationDevice<int> d_num_runs;

    helpers::SimpleAllocationPinnedHost<int> h_uniqueRangeLengths;
    helpers::SimpleAllocationDevice<int> d_uniqueRangeLengths;

    helpers::SimpleAllocationDevice<int> d_uniqueRangeLengthsPrefixsum;

    helpers::SimpleAllocationDevice<char> cubTempStorage;

    int initialDataSize = 0;

    std::array<cudaStream_t, 1> streams{};
    std::array<cudaEvent_t, 2> events{};    
};

template<class T>
MergeRangesGpuHandle<T> makeMergeRangesGpuHandle(){
    MergeRangesGpuHandle<T> handle;
    for(auto& pipelinestream : handle.streams){
        cudaStreamCreate(&pipelinestream); CUERR;
    }

    for(auto& event : handle.events){
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming); CUERR;
    }  

    return handle;
}


template<class T>
void destroyMergeRangesGpuHandle(MergeRangesGpuHandle<T>& handle){

    handle.d_data.destroy();
    handle.h_data.destroy();
    handle.d_rangesBeginPerSequence.destroy();
    handle.h_rangesBeginPerSequence.destroy();
    handle.d_results.destroy();
    handle.h_results.destroy();
    handle.h_numresults.destroy();
    handle.d_num_runs.destroy();
    handle.h_num_runs.destroy();
    handle.d_uniqueRangeLengths.destroy();
    handle.h_uniqueRangeLengths.destroy();
    handle.d_uniqueRangeLengthsPrefixsum.destroy();
    handle.cubTempStorage.destroy();

    for(auto& pipelinestream : handle.streams){
        cudaStreamDestroy(pipelinestream); CUERR;
    }

    for(auto& event : handle.events){
        cudaEventDestroy(event); CUERR;
    }  
}




struct OperationResult{
    std::vector<read_number> candidateIds;
    std::vector<int> candidateIdsPerSequence;
};

enum class MergeRangesKernelType{
    devicewide,
    allcub,
    popcmultiwarp,
    popcsinglewarp,
    popcsinglewarpchunked,
};

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
        MergeRangesKernelType kernelType);


OperationResult mergeRangesGpu(
        MergeRangesGpuHandle<read_number>& handle, 
        const std::pair<const read_number*, const read_number*>* h_ranges, 
        int numRanges, 
        const read_number* d_anchorIds, 
        int rangesPerSequence, 
        cudaStream_t stream,
        MergeRangesKernelType kernelType);


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
        cudaStream_t stream);

void callMinhashSignaturesKernel_async(
        std::uint64_t* d_signatures,
        size_t signaturesRowPitchElements,
        const unsigned int* d_sequences2Bit,
        size_t sequenceRowPitchElements,
        int numSequences,
        const int* d_sequenceLengths,
        int k,
        int numHashFuncs,
        cudaStream_t stream);



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
);


void callMinhashSignaturesOfUniqueKmersKernel128_async(
    std::uint64_t* d_signatures,
    size_t signaturesRowPitchElements,
    const unsigned int* d_sequences2Bit,
    size_t sequenceRowPitchElements,
    int numSequences,
    const int* d_sequenceLengths,
    int k,
    int numHashFuncs,
    cudaStream_t stream
);











} //namespace care


#endif
