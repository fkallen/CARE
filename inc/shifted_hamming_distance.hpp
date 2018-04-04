#ifndef CARE_SHIFTED_HAMMING_DISTANCE_HPP
#define CARE_SHIFTED_HAMMING_DISTANCE_HPP

#include "alignment.hpp"
#include "batchelem.hpp"
#include "options.hpp"

#include <vector>
#include <chrono>
#include <memory>

namespace care{

    // Buffers for both GPU alignment and CPU alignment
    struct SHDdata{

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

    	int deviceId = -1;
    	size_t sequencepitch = 0;
    	int max_sequence_length = 0;
    	int max_sequence_bytes = 0;
        int min_sequence_length = 0;
        int min_sequence_bytes = 0;
    	int n_subjects = 0;
    	int n_queries = 0;
    	int max_n_subjects = 0;
    	int max_n_queries = 0;
    	int batchsize = 1;

        int gpuThreshold = 0; // if number of alignments to calculate is >= gpuThreshold, use GPU.

    	std::chrono::duration<double> resizetime{0};
    	std::chrono::duration<double> preprocessingtime{0};
    	std::chrono::duration<double> h2dtime{0};
    	std::chrono::duration<double> alignmenttime{0};
    	std::chrono::duration<double> d2htime{0};
    	std::chrono::duration<double> postprocessingtime{0};

    	SHDdata(int deviceId, int maxseqlength, int maxseqbytes, int batchsize, int gpuThreshold = 0);
    	void resize(int n_sub, int n_quer);
    };

    //free buffers
    void cuda_cleanup_SHDdata(SHDdata& data);

    int find_shifted_hamming_distance_gpu_threshold(int deviceId,
                                                    int minsequencelength,
                                                    int minsequencebytes);

    //For each BatchElem, align all candidate reads to the respective read
    AlignmentDevice shifted_hamming_distance(SHDdata& mybuffers, std::vector<BatchElem>& batch,
                                const GoodAlignmentProperties& props, bool canUseGpu);

    //In BatchElem b, calculate alignments[firstIndex] to alignments[firstIndex + N - 1]
    AlignmentDevice shifted_hamming_distance(SHDdata& mybuffers, BatchElem& b,
                                    int firstIndex, int N,
                                const GoodAlignmentProperties& props, bool canUseGpu);



}

#endif
