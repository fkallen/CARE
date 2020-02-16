#ifndef CARE_MINHASHKERNELS_HPP
#define CARE_MINHASHKERNELS_HPP

#include <config.hpp>
#include <hpc_helpers.cuh>

#include <array>
#include <vector>

namespace care{


template<class T>
struct MergeRangesGpuHandle{
    T* d_data = nullptr;
    T* h_data = nullptr;
    size_t datacapacity = 0;
    
    int* h_rangesPerSequence = nullptr;
    int* d_rangesPerSequence = nullptr;
    size_t rangesPerSequenceBeginscapacity = 0;

    T* d_results = nullptr;
    T* h_results = nullptr;
    size_t resultscapacity = 0;

    int* h_numresults = nullptr;
    size_t numresultscapacity = 0;

    int* h_num_runs = nullptr;
    int* d_num_runs = nullptr;
    size_t num_runscapacity = 0;

    int* h_uniqueRangeLengths = nullptr;
    int* d_uniqueRangeLengths = nullptr;
    size_t uniqueRangeLengthscapacity = 0;

    int* d_uniqueRangeLengthsPrefixsum = nullptr;
    size_t uniqueRangeLengthsPrefixsumcapacity = 0;

    void* cubTempStorage = nullptr;
    size_t tempStorageCapacity = 0;

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

    cudaFree(handle.d_data); CUERR;
    cudaFreeHost(handle.h_data); CUERR;
    cudaFree(handle.d_rangesPerSequence); CUERR;
    cudaFreeHost(handle.h_rangesPerSequence); CUERR;
    cudaFree(handle.d_results); CUERR;
    cudaFreeHost(handle.h_results); CUERR;
    cudaFreeHost(handle.h_numresults); CUERR;
    cudaFree(handle.d_num_runs); CUERR;
    cudaFreeHost(handle.h_num_runs); CUERR;
    cudaFree(handle.d_uniqueRangeLengths); CUERR;
    cudaFreeHost(handle.h_uniqueRangeLengths); CUERR;
    cudaFree(handle.d_uniqueRangeLengthsPrefixsum); CUERR;
    cudaFree(handle.cubTempStorage); CUERR;


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


OperationResult mergeRangesGpu(
        MergeRangesGpuHandle<read_number>& handle, 
        const std::pair<const read_number*, const read_number*>* ranges, 
        int numRanges, 
        int rangesPerSequence, 
        cudaStream_t stream,
        MergeRangesKernelType kernelType);














} //namespace care


#endif
