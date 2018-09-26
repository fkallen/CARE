#ifndef CARE_GPU_READ_STORAGE_HPP
#define CARE_GPU_READ_STORAGE_HPP

#include "../hpc_helpers.cuh"

#include <iostream>
#include <limits>

namespace care{

struct GPUReadStorage{

    char* d_sequence_data = nullptr;
    int max_sequence_bytes = 0;
    int deviceId = -1;

#ifdef __NVCC__
    template<class CPUReadStorage>
    static bool isEnoughMemoryAvailable(const CPUReadStorage& cpurs, int max_sequence_bytes, float maxPercentOfTotalMem, int deviceId){
        int oldId;
        cudaGetDevice(&oldId); CUERR;
        cudaSetDevice(deviceId); CUERR;

        std::uint64_t requiredSequenceMem = max_sequence_bytes * cpurs.sequences.size();

        std::size_t freeMem;
        std::size_t totalMem;
        cudaMemGetInfo(&freeMem, &totalMem); CUERR;

        std::cout << "GPUReadStorage: required " << requiredSequenceMem << ", free " << freeMem << ", total " << totalMem << std::endl;
        bool result = false;
        if(maxPercentOfTotalMem * totalMem < requiredSequenceMem && requiredSequenceMem < freeMem){
            result = true;
        }

        cudaSetDevice(oldId); CUERR;

        return result;
    }

    template<class CPUReadStorage>
    static GPUReadStorage createFrom(const CPUReadStorage& cpurs, int max_sequence_bytes, int deviceId){
        using ReadId_t = typename CPUReadStorage::ReadId_t;
        using Sequence_t = typename CPUReadStorage::Sequence_t;

        static_assert(std::numeric_limits<ReadId_t>::max() <= std::numeric_limits<std::uint64_t>::max());

        constexpr std::uint64_t maxcopybatchsequences = std::size_t(10000000);

        int oldId;
        cudaGetDevice(&oldId); CUERR;
        cudaSetDevice(deviceId); CUERR;

        std::uint64_t nSequences = cpurs.sequences.size();
        std::uint64_t requiredSequenceMem = max_sequence_bytes * nSequences;

        GPUReadStorage gpurs;
        gpurs.deviceId = deviceId;
        gpurs.max_sequence_bytes = max_sequence_bytes;
        cudaMalloc(&gpurs.d_sequence_data, requiredSequenceMem); CUERR;

        const std::uint64_t copybatchsequences = std::min(nSequences, maxcopybatchsequences);

        std::uint64_t tmpstoragesize = copybatchsequences * max_sequence_bytes;
        char* h_tmp;
        //h_tmp = new char[tmpstoragesize];
        cudaMallocHost(&h_tmp, tmpstoragesize); CUERR;

        int iters = SDIV(nSequences, copybatchsequences);
        for(int iter = 0; iter < iters; ++iter){
            std::memset(h_tmp, 0, tmpstoragesize);

            std::uint64_t sequencesToCopy = std::min((iter + 1) * copybatchsequences, nSequences);

            for(ReadId_t readId = iter * copybatchsequences, count = 0; readId < std::min((iter + 1) * copybatchsequences, nSequences); ++readId, ++count){
                const Sequence_t* sequence = cpurs.fetchSequence_ptr(readId);
                std::memcpy(h_tmp + count * max_sequence_bytes,
                    sequence->begin(),
                    sequence->length());
            }

            cudaMemcpyAsync(gpurs.d_sequence_data + iter * copybatchsequences + max_sequence_bytes,
                            h_tmp,
                            sequencesToCopy * max_sequence_bytes,
                            H2D,
                            nullptr);
        }

        //delete [] h_tmp;
        cudaFreeHost(h_tmp); CUERR;

        cudaSetDevice(oldId); CUERR;

        return gpurs;
    }

    static void destroy(GPUReadStorage& gpurs){
        int oldId;
        cudaGetDevice(&oldId); CUERR;
        cudaSetDevice(gpurs.deviceId); CUERR;

        cudaFree(gpurs.d_sequence_data); CUERR;

        cudaSetDevice(oldId); CUERR;
    }

#endif
};

}

#endif
