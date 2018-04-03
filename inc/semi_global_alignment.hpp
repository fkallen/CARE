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
        static constexpr int max_batch_size = 16;
        cudaStream_t streams[max_batch_size];
    #endif

        int deviceId;
        int batchsize;

        int max_sequence_length = 0;
        int max_sequence_bytes = 0;
        int max_ops_per_alignment = 0;
        int n_subjects = 0;
        int n_queries = 0;
        int max_n_subjects = 0;
        int max_n_queries = 0;

        std::size_t sequencepitch = 0;

        std::chrono::duration<double> resizetime{0};
        std::chrono::duration<double> preprocessingtime{0};
        std::chrono::duration<double> h2dtime{0};
        std::chrono::duration<double> alignmenttime{0};
        std::chrono::duration<double> d2htime{0};
        std::chrono::duration<double> postprocessingtime{0};

        void resize(int n_sub, int n_quer);

        SGAdata(int deviceId, int maxseqlength, int maxseqbytes, int batchsize);
    };

    void cuda_cleanup_SGAdata(SGAdata& data);

	void semi_global_alignment(SGAdata& mybuffers, const AlignmentOptions& alignmentOptions,
                                std::vector<BatchElem>& batch, bool useGpu);
}

#endif
