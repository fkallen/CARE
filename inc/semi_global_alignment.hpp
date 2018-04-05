#ifndef CARE_SEMI_GLOBAL_ALIGNMENT_HPP
#define CARE_SEMI_GLOBAL_ALIGNMENT_HPP

#include "alignment.hpp"
#include "batchelem.hpp"
#include "options.hpp"

#include <cstdint>
#include <memory>

namespace care{

    //Buffers for both GPU alignment and CPU alignment
    struct SGAdata{
        AlignOp* d_ops = nullptr;
        AlignOp* h_ops = nullptr;

        AlignResultCompact* d_results = nullptr;
        char* d_subjectsdata = nullptr;
        char* d_queriesdata = nullptr;
        int* d_subjectlengths = nullptr;
        int* d_querylengths = nullptr;

        AlignResultCompact* h_results = nullptr;
        char* h_subjectsdata = nullptr;
        char* h_queriesdata = nullptr;
        int* h_subjectlengths = nullptr;
        int* h_querylengths = nullptr;

    #ifdef __NVCC__
        static constexpr int n_streams = 1;
        cudaStream_t streams[n_streams];
    #endif

        int deviceId;

        int max_sequence_length = 0;
        int max_sequence_bytes = 0;
        int max_ops_per_alignment = 0;
        int n_subjects = 0;
        int n_queries = 0;
        int max_n_subjects = 0;
        int max_n_queries = 0;

        std::size_t sequencepitch = 0;
        int gpuThreshold = 0; // if number of alignments to calculate is >= gpuThreshold, use GPU.

        std::chrono::duration<double> resizetime{0};
        std::chrono::duration<double> preprocessingtime{0};
        std::chrono::duration<double> h2dtime{0};
        std::chrono::duration<double> alignmenttime{0};
        std::chrono::duration<double> d2htime{0};
        std::chrono::duration<double> postprocessingtime{0};

        SGAdata(){}
        SGAdata(const SGAdata& other) = default;
        SGAdata(SGAdata&& other) = default;
        SGAdata& operator=(const SGAdata& other) = default;
        SGAdata& operator=(SGAdata&& other) = default;

        void resize(int n_sub, int n_quer);
    };

    void cuda_init_SGAdata(SGAdata& data,
                           int deviceId,
                           int max_sequence_length,
                           int max_sequence_bytes,
                           int gpuThreshold);

    void cuda_cleanup_SGAdata(SGAdata& data);

    int find_semi_global_alignment_gpu_threshold(int deviceId, int minsequencelength, int minsequencebytes);

    //In BatchElem b, calculate alignments[firstIndex] to alignments[firstIndex + N - 1]
    AlignmentDevice semi_global_alignment_async(SGAdata& mybuffers, BatchElem& b,
                                    int firstIndex, int N,
                                    const AlignmentOptions& alignmentOptions,
                                    bool canUseGpu);

    void get_semi_global_alignment_results(SGAdata& mybuffers, BatchElem& b,
                                    int firstIndex, int N,
                                    const AlignmentOptions& alignmentOptions,
                                    bool canUseGpu);
}

#endif
