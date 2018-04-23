#include "../inc/shifted_hamming_distance.hpp"
#include "../inc/shifted_hamming_distance_impl.hpp"

#include <algorithm>
#include <stdexcept>

namespace care{
/*
    SHDdata implementation
*/

void SHDdata::resize(int n_sub, int n_quer){
#ifdef __NVCC__
    cudaSetDevice(deviceId); CUERR;

    bool resizeResult = false;

    if(n_sub > max_n_subjects){
        size_t oldpitch = sequencepitch;
        cudaFree(d_subjectsdata); CUERR;
        cudaMallocPitch(&d_subjectsdata, &sequencepitch, max_sequence_bytes, n_sub); CUERR;
        assert(!oldpitch || oldpitch == sequencepitch);

        cudaFreeHost(h_subjectsdata); CUERR;
        cudaMallocHost(&h_subjectsdata, sequencepitch * n_sub); CUERR;

        cudaFree(d_subjectlengths); CUERR;
        cudaMalloc(&d_subjectlengths, sizeof(int) * n_sub); CUERR;

        cudaFreeHost(h_subjectlengths); CUERR;
        cudaMallocHost(&h_subjectlengths, sizeof(int) * n_sub); CUERR;

        cudaFree(d_NqueriesPrefixSum); CUERR;
        cudaMalloc(&d_NqueriesPrefixSum, sizeof(int) * (n_sub+1)); CUERR;

        cudaFreeHost(h_NqueriesPrefixSum); CUERR;
        cudaMallocHost(&h_NqueriesPrefixSum, sizeof(int) * (n_sub+1)); CUERR;

        max_n_subjects = n_sub;

        resizeResult = true;
    }


    if(n_quer > max_n_queries){
        size_t oldpitch = sequencepitch;
        cudaFree(d_queriesdata); CUERR;
        cudaMallocPitch(&d_queriesdata, &sequencepitch, max_sequence_bytes, n_quer); CUERR;
        assert(!oldpitch || oldpitch == sequencepitch);

        cudaFreeHost(h_queriesdata); CUERR;
        cudaMallocHost(&h_queriesdata, sequencepitch * n_quer); CUERR;

        cudaFree(d_querylengths); CUERR;
        cudaMalloc(&d_querylengths, sizeof(int) * n_quer); CUERR;

        cudaFreeHost(h_querylengths); CUERR;
        cudaMallocHost(&h_querylengths, sizeof(int) * n_quer); CUERR;

        max_n_queries = n_quer;

        resizeResult = true;
    }

    if(resizeResult){
        cudaFree(d_results); CUERR;
        cudaMalloc(&d_results, sizeof(AlignResultCompact) * max_n_subjects * max_n_queries); CUERR;

        cudaFreeHost(h_results); CUERR;
        cudaMallocHost(&h_results, sizeof(AlignResultCompact) * max_n_subjects * max_n_queries); CUERR;
    }
#endif
    n_subjects = n_sub;
    n_queries = n_quer;
}

void cuda_init_SHDdata(SHDdata& data, int deviceId,
                        int max_sequence_length,
                        int max_sequence_bytes,
                        int gpuThreshold){
    data.deviceId = deviceId;
    data.max_sequence_length = max_sequence_length;
    data.max_sequence_bytes = max_sequence_bytes;
    data.gpuThreshold = gpuThreshold;
#ifdef __NVCC__

    cudaSetDevice(deviceId); CUERR;

    for(int i = 0; i < SHDdata::n_streams; i++)
        cudaStreamCreate(&(data.streams[i])); CUERR;
#endif
}

void cuda_cleanup_SHDdata(SHDdata& data){
#ifdef __NVCC__
    cudaSetDevice(data.deviceId); CUERR;

    cudaFree(data.d_results); CUERR;
    cudaFree(data.d_subjectsdata); CUERR;
    cudaFree(data.d_queriesdata); CUERR;
    cudaFree(data.d_subjectlengths); CUERR;
    cudaFree(data.d_querylengths); CUERR;
    cudaFree(data.d_NqueriesPrefixSum); CUERR;

    cudaFreeHost(data.h_results); CUERR;
    cudaFreeHost(data.h_subjectsdata); CUERR;
    cudaFreeHost(data.h_queriesdata); CUERR;
    cudaFreeHost(data.h_subjectlengths); CUERR;
    cudaFreeHost(data.h_querylengths); CUERR;
    cudaFreeHost(data.h_NqueriesPrefixSum); CUERR;

    for(int i = 0; i < SHDdata::n_streams; i++)
        cudaStreamDestroy(data.streams[i]); CUERR;
#endif
}


}
