#ifndef CARE_SHIFTED_HAMMING_DISTANCE_HPP
#define CARE_SHIFTED_HAMMING_DISTANCE_HPP

#include "alignment.hpp"
#include "options.hpp"

#include <vector>
#include <chrono>
#include <memory>
#include <cstring>
#include <cassert>

namespace care{

    // Buffers for both GPU alignment and CPU alignment
    struct SHDdata{

    	AlignResultCompact* d_results = nullptr;
    	char* d_subjectsdata = nullptr;
    	char* d_queriesdata = nullptr;
    	int* d_subjectlengths = nullptr;
    	int* d_querylengths = nullptr;
        int* d_NqueriesPrefixSum = nullptr;

    	AlignResultCompact* h_results = nullptr;
    	char* h_subjectsdata = nullptr;
    	char* h_queriesdata = nullptr;
    	int* h_subjectlengths = nullptr;
    	int* h_querylengths = nullptr;
        int* h_NqueriesPrefixSum = nullptr;

    #ifdef __NVCC__
        static constexpr int n_streams = 1;
        cudaStream_t streams[n_streams];
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

        int gpuThreshold = 0; // if number of alignments to calculate is >= gpuThreshold, use GPU.

    	std::chrono::duration<double> resizetime{0};
    	std::chrono::duration<double> preprocessingtime{0};
    	std::chrono::duration<double> h2dtime{0};
    	std::chrono::duration<double> alignmenttime{0};
    	std::chrono::duration<double> d2htime{0};
    	std::chrono::duration<double> postprocessingtime{0};

    	void resize(int n_sub, int n_quer);
    };

	struct shdparams{
		GoodAlignmentProperties props;
		int max_sequence_bytes;
		int sequencepitch;
		int n_queries;
		int subjectlength;
		const int* __restrict__ querylengths;
		const char* __restrict__ subjectdata;
		const char* __restrict__ queriesdata;
		AlignResultCompact* __restrict__ results;
	};

    //init buffers
    void cuda_init_SHDdata(SHDdata& data, int deviceId,
                            int max_sequence_length,
                            int max_sequence_bytes,
                            int gpuThreshold);

    //free buffers
    void cuda_cleanup_SHDdata(SHDdata& data);




template<class Sequence_t>
AlignResultCompact cpu_shifted_hamming_distance(const char* subject,
												const char* query,
												int ns,
												int nq,
                                                int min_overlap,
                                                double maxErrorRate,
                                                double min_overlap_ratio);



#ifdef __NVCC__

//wrapper functions to call kernels

template<class Sequence_t>
void call_shd_kernel(const shdparams& buffer, int maxQueryLength, cudaStream_t stream);

template<class Sequence_t>
void call_shd_kernel_async(const shdparams& buffer, int maxQueryLength, cudaStream_t stream);

template<class Sequence_t>
void call_shd_kernel(AlignResultCompact* results,
                      const char* subjectsdata,
                      const int* subjectlengths,
                      const char* queriesdata,
                      const int* querylengths,
                      const int* NqueriesPrefixSum,
                      int Nsubjects,
                      int Nqueries,
                      int max_sequence_bytes,
                      size_t sequencepitch,
                      const GoodAlignmentProperties& props,
                      int maxSubjectLength,
                      int maxQueryLength,
                      cudaStream_t stream);

template<class Sequence_t>
void call_shd_kernel_async(AlignResultCompact* results,
                      const char* subjectsdata,
                      const int* subjectlengths,
                      const char* queriesdata,
                      const int* querylengths,
                      const int* NqueriesPrefixSum,
                      int Nsubjects,
                      int Nqueries,
                      int max_sequence_bytes,
                      size_t sequencepitch,
                      const GoodAlignmentProperties& props,
                      int maxSubjectLength,
                      int maxQueryLength,
                      cudaStream_t stream);

template<int BLOCKSIZE, class Accessor>
__global__
void
cuda_shifted_hamming_distance(const shdparams buffers, Accessor getChar);

template<int BLOCKSIZE, class Accessor>
__global__
void
cuda_shifted_hamming_distance(AlignResultCompact* results,
                              const char* subjectsdata,
                              const int* subjectlengths,
                              const char* queriesdata,
                              const int* querylengths,
                              const int* NqueriesPrefixSum,
                              int Nsubjects,
                              int max_sequence_bytes,
                              size_t sequencepitch,
                              GoodAlignmentProperties props,
                              Accessor getChar);

#endif

template<class Sequence_t>
int find_shifted_hamming_distance_gpu_threshold(int deviceId, int minsequencelength, int minsequencebytes){
    int threshold = std::numeric_limits<int>::max();
#ifdef __NVCC__
    SHDdata shddata;
    cuda_init_SHDdata(shddata, deviceId, minsequencelength, minsequencebytes, 0);

    const int increment = 20;
    int nalignments = 0;

    std::chrono::time_point<std::chrono::system_clock> gpustart;
    std::chrono::time_point<std::chrono::system_clock> gpuend;
    std::chrono::time_point<std::chrono::system_clock> cpustart;
    std::chrono::time_point<std::chrono::system_clock> cpuend;

    std::string seqstring = "";
    for(int i = 0; i < minsequencelength; i++)
        seqstring += "C";

    Sequence_t sequence(seqstring);

    GoodAlignmentProperties alignProps;

    do{
        nalignments += increment;
        std::vector<Sequence_t> sequences(nalignments, sequence);
        std::vector<AlignResultCompact> gpuresults(nalignments);
        std::vector<AlignResultCompact> cpuresults(nalignments);

        gpustart = std::chrono::system_clock::now();

        shddata.resize(1, nalignments);
        shdparams params;

        params.props = alignProps;
        params.max_sequence_bytes = shddata.max_sequence_bytes;
        params.sequencepitch = shddata.sequencepitch;
        params.subjectlength = minsequencelength;
        params.n_queries = nalignments;
        params.querylengths = shddata.d_querylengths;
        params.subjectdata = shddata.d_subjectsdata;
        params.queriesdata = shddata.d_queriesdata;
        params.results = shddata.d_results;

        int* querylengths = shddata.h_querylengths;
        char* subjectdata = shddata.h_subjectsdata;
        char* queriesdata = shddata.h_queriesdata;

        std::memcpy(subjectdata, sequences[0].begin(), sequences[0].getNumBytes());
        for(int count = 0; count < nalignments; count++){
            const auto& seq = sequences[count];

            std::memcpy(queriesdata + count * shddata.sequencepitch,
                    seq.begin(),
                    seq.getNumBytes());

            querylengths[count] = seq.length();
        }
        cudaMemcpyAsync(const_cast<char*>(params.subjectdata),
                subjectdata,
                shddata.sequencepitch,
                H2D,
                shddata.streams[0]); CUERR;
        cudaMemcpyAsync(const_cast<char*>(params.queriesdata),
                queriesdata,
                shddata.sequencepitch * params.n_queries,
                H2D,
                shddata.streams[0]); CUERR;
        cudaMemcpyAsync(const_cast<int*>(params.querylengths),
                querylengths,
                sizeof(int) * params.n_queries,
                H2D,
                shddata.streams[0]); CUERR;

        // start kernel
        call_shd_kernel_async<Sequence_t>(params, minsequencelength, shddata.streams[0]);

        AlignResultCompact* results = shddata.h_results;

        cudaMemcpyAsync(results,
            params.results,
            sizeof(AlignResultCompact) * params.n_queries,
            D2H,
            shddata.streams[0]); CUERR;

        cudaStreamSynchronize(shddata.streams[0]); CUERR;

        for(int count = 0; count < nalignments; count++){
            gpuresults[count] = results[count];
        }

        gpuend = std::chrono::system_clock::now();


        cpustart = std::chrono::system_clock::now();

        const char* const subject = (const char*)sequences[0].begin();
        const int subjectLength = sequences[0].length();

        for(int i = 0; i < nalignments; i++){
            const char* query =  (const char*)sequences[i].begin();
            const int queryLength = sequences[i].length();
            cpuresults[i] = cpu_shifted_hamming_distance<Sequence_t>(alignProps, subject, query, subjectLength, queryLength);
        }

        cpuend = std::chrono::system_clock::now();


    }while(gpuend - gpustart > cpuend - cpustart || nalignments == 1000);

    if(gpuend - gpustart <= cpuend - cpustart){
        threshold = nalignments;
    }

    cuda_cleanup_SHDdata(shddata);
#endif
    return threshold;
}

/*
    Calculate
    alignments[firstIndex]
    to
    alignments[min(firstIndex + N-1, number of alignments - 1)]
    Both forward alignments and reverse complement alignments are calculated.

    If GPU is used, transfers data to gpu and launches kernel, but does not wait for kernel completion
    If CPU is used, no alignment is performed.
*/
template<class BatchElem_t>
AlignmentDevice shifted_hamming_distance_async(SHDdata& mybuffers, BatchElem_t& b,
                                int firstIndex, int N,
                            const GoodAlignmentProperties& props, bool canUseGpu){



    AlignmentDevice device = AlignmentDevice::None;

    const int lastIndex_excl = std::min(size_t(firstIndex + N), b.fwdSequences.size());
    const int numberOfCandidates = firstIndex >= lastIndex_excl ? 0 : lastIndex_excl - firstIndex;
    const int numberOfAlignments = 2 * numberOfCandidates;

    //nothing to do here
    if(!b.active || numberOfAlignments == 0)
        return device;

#ifdef __NVCC__
    std::chrono::time_point<std::chrono::system_clock> tpa = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> tpb = std::chrono::system_clock::now();

    if(canUseGpu && numberOfAlignments >= mybuffers.gpuThreshold){ // use gpu for alignment
        device = AlignmentDevice::GPU;
        tpa = std::chrono::system_clock::now();

        cudaSetDevice(mybuffers.deviceId); CUERR;

        mybuffers.resize(1, numberOfAlignments);

        tpb = std::chrono::system_clock::now();

        mybuffers.resizetime += tpb - tpa;

        shdparams params;

        tpa = std::chrono::system_clock::now();

        params.props = props;
        params.max_sequence_bytes = mybuffers.max_sequence_bytes;
        params.sequencepitch = mybuffers.sequencepitch;
        params.subjectlength = b.fwdSequence->length();
        params.n_queries = numberOfAlignments;
        params.querylengths = mybuffers.d_querylengths;
        params.subjectdata = mybuffers.d_subjectsdata;
        params.queriesdata = mybuffers.d_queriesdata;
        params.results = mybuffers.d_results;

        int* querylengths = mybuffers.h_querylengths;
        char* subjectdata = mybuffers.h_subjectsdata;
        char* queriesdata = mybuffers.h_queriesdata;

        //copy subject to transfer buffer
        std::memcpy(subjectdata, b.fwdSequence->begin(), b.fwdSequence->getNumBytes());

        //copy candidate forward sequences to transfer buffer
        int count = 0;
        for(int index = firstIndex; index < lastIndex_excl; index++){
            const auto& seq = b.fwdSequences[index];

            assert(seq->length() <= mybuffers.max_sequence_length);
            assert(seq->getNumBytes() <= mybuffers.max_sequence_bytes);

            std::memcpy(queriesdata + count * mybuffers.sequencepitch,
                    seq->begin(),
                    seq->getNumBytes());

            querylengths[count] = seq->length();
            count++;
        }
        //copy candidate reverse complement sequences to transfer buffer
        for(int index = firstIndex; index < lastIndex_excl; index++){
            const auto& seq = b.revcomplSequences[index];

            assert(seq->length() <= mybuffers.max_sequence_length);
            assert(seq->getNumBytes() <= mybuffers.max_sequence_bytes);

            std::memcpy(queriesdata + count * mybuffers.sequencepitch,
                    seq->begin(),
                    seq->getNumBytes());

            querylengths[count] = seq->length();
            count++;
        }

        tpb = std::chrono::system_clock::now();
        mybuffers.preprocessingtime += tpb - tpa;

        int maxQueryLength = 0;
        for(int k = 0; k < params.n_queries; k++)
            if(maxQueryLength < querylengths[k])
                maxQueryLength = querylengths[k];

        // copy data to gpu
        cudaMemcpyAsync(const_cast<char*>(params.subjectdata),
                subjectdata,
                mybuffers.sequencepitch,
                H2D,
                mybuffers.streams[0]); CUERR;
        cudaMemcpyAsync(const_cast<char*>(params.queriesdata),
                queriesdata,
                mybuffers.sequencepitch * params.n_queries,
                H2D,
                mybuffers.streams[0]); CUERR;
        cudaMemcpyAsync(const_cast<int*>(params.querylengths),
                querylengths,
                sizeof(int) * params.n_queries,
                H2D,
                mybuffers.streams[0]); CUERR;

        // start kernel
		using Sequence_t = typename BatchElem_t::Sequence_t;
        call_shd_kernel_async<Sequence_t>(params, maxQueryLength, mybuffers.streams[0]);
    }else{ // use cpu for alignment

#endif // __NVCC__
        device = AlignmentDevice::CPU;
#ifdef __NVCC__
    }
#endif // __NVCC__

    return device;
}


/*
    If GPU is used, copies results from gpu to cpu and stores alignments in BatchElem_t.
    If CPU is used, perform alignment on cpu and store alignments in BatchElem_t
*/
template<class BatchElem_t>
void get_shifted_hamming_distance_results(SHDdata& mybuffers, BatchElem_t& b,
                                int firstIndex, int N, const GoodAlignmentProperties& props,
                                bool canUseGpu){
	using Sequence_t = typename BatchElem_t::Sequence_t;

    std::chrono::time_point<std::chrono::system_clock> tpa = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> tpb = std::chrono::system_clock::now();

    const int lastIndex_excl = std::min(size_t(firstIndex + N), b.fwdSequences.size());
    const int numberOfCandidates = firstIndex >= lastIndex_excl ? 0 : lastIndex_excl - firstIndex;
    const int numberOfAlignments = 2 * numberOfCandidates;

    //nothing to do here
    if(!b.active || numberOfAlignments == 0)
        return;

#ifdef __NVCC__

    if(canUseGpu && numberOfAlignments >= mybuffers.gpuThreshold){ // use gpu for alignment
        cudaSetDevice(mybuffers.deviceId); CUERR;

        AlignResultCompact* results = mybuffers.h_results;
        AlignResultCompact* d_results = mybuffers.d_results;

        cudaMemcpyAsync(results,
            d_results,
            sizeof(AlignResultCompact) * numberOfAlignments,
            D2H,
            mybuffers.streams[0]); CUERR;

        cudaStreamSynchronize(mybuffers.streams[0]); CUERR;

        tpa = std::chrono::system_clock::now();

        int count = 0;
        for(int index = firstIndex; index < lastIndex_excl; index++){
            b.fwdAlignments[index] = results[count];
            count++;
        }
        for(int index = firstIndex; index < lastIndex_excl; index++){
            b.revcomplAlignments[index] = results[count];
            count++;
        }

        tpb = std::chrono::system_clock::now();
        mybuffers.postprocessingtime += tpb - tpa;

    }else{ // use cpu for alignment

#endif // __NVCC__
        tpa = std::chrono::system_clock::now();

        const char* const subject = (const char*)b.fwdSequence->begin();
        const int subjectLength = b.fwdSequence->length();

        for(int index = firstIndex; index < lastIndex_excl; index++){
            const char* query =  (const char*)b.fwdSequences[index]->begin();
            const int queryLength = b.fwdSequences[index]->length();
            b.fwdAlignments[index] = cpu_shifted_hamming_distance<Sequence_t>(props, subject, query, subjectLength, queryLength);
        }
        for(int index = firstIndex; index < lastIndex_excl; index++){
            const char* query =  (const char*)b.revcomplSequences[index]->begin();
            const int queryLength = b.revcomplSequences[index]->length();
            b.revcomplAlignments[index] = cpu_shifted_hamming_distance<Sequence_t>(props, subject, query, subjectLength, queryLength);
        }

        tpb = std::chrono::system_clock::now();

        mybuffers.alignmenttime += tpb - tpa;
#ifdef __NVCC__
    }
#endif
}

template<class BatchElem_t>
AlignmentDevice shifted_hamming_distance(SHDdata& mybuffers,
                                        std::vector<BatchElem_t>& batch,
                                        const GoodAlignmentProperties& props,
                                        bool canUseGpu){

	using Sequence_t = typename BatchElem_t::Sequence_t;

	std::chrono::time_point<std::chrono::system_clock> tpa = std::chrono::system_clock::now();
	std::chrono::time_point<std::chrono::system_clock> tpb = std::chrono::system_clock::now();
	AlignmentDevice device = AlignmentDevice::None;

	int numberOfSubjects = 0;
	int totalNumberOfAlignments = 0;

	for(auto& b : batch){
		if(b.active){
			numberOfSubjects++;
			totalNumberOfAlignments += b.fwdSequences.size();
			totalNumberOfAlignments += b.revcomplSequences.size();
		}
	}

	// check for empty input
	if(totalNumberOfAlignments == 0){
		return device;
	}

#ifdef __NVCC__

	if(canUseGpu){ // use gpu for alignment
		device = AlignmentDevice::GPU;
		tpa = std::chrono::system_clock::now();

		cudaSetDevice(mybuffers.deviceId); CUERR;

		mybuffers.resize(numberOfSubjects, totalNumberOfAlignments);

		tpb = std::chrono::system_clock::now();

		mybuffers.resizetime += tpb - tpa;

		int querysum = 0;
		int subjectindex = 0;
		mybuffers.h_NqueriesPrefixSum[0] = 0;

		//copy everything to transfer buffers
		tpa = std::chrono::system_clock::now();

		for(auto& b : batch){
			if(b.active){

				char* h_subjectsdata = mybuffers.h_subjectsdata + mybuffers.sequencepitch * subjectindex;
				char* h_queriesdata = mybuffers.h_queriesdata + mybuffers.sequencepitch * querysum;
				int* h_subjectlengths = mybuffers.h_subjectlengths;
				int* h_querylengths = mybuffers.h_querylengths + querysum;
				int* h_NqueriesPrefixSum = mybuffers.h_NqueriesPrefixSum;

				assert(b.fwdSequence->length() <= mybuffers.max_sequence_length);
				assert(b.fwdSequence->getNumBytes() <= mybuffers.max_sequence_bytes);

				std::memcpy(h_subjectsdata, b.fwdSequence->begin(), b.fwdSequence->getNumBytes());
				h_subjectlengths[subjectindex] = b.fwdSequence->length();

				int count = 0;
				for(const auto& seqptr : b.fwdSequences){
					assert(seqptr->length() <= mybuffers.max_sequence_length);
					assert(seqptr->getNumBytes() <= mybuffers.max_sequence_bytes);

					std::memcpy(h_queriesdata + count * mybuffers.sequencepitch,
							seqptr->begin(),
							seqptr->getNumBytes());

					h_querylengths[count] = seqptr->length();
					count++;
				}
				for(const auto& seqptr : b.revcomplSequences){
					assert(seqptr->length() <= mybuffers.max_sequence_length);
					assert(seqptr->getNumBytes() <= mybuffers.max_sequence_bytes);

					std::memcpy(h_queriesdata + count * mybuffers.sequencepitch,
							seqptr->begin(),
							seqptr->getNumBytes());

					h_querylengths[count] = seqptr->length();
					count++;
				}

				h_NqueriesPrefixSum[subjectindex + 1] = h_NqueriesPrefixSum[subjectindex] + count;

				querysum += count;
				subjectindex++;
			}
		}

		tpb = std::chrono::system_clock::now();
		mybuffers.preprocessingtime += tpb - tpa;

		// copy data to gpu
		cudaMemcpyAsync(mybuffers.d_subjectsdata,
				mybuffers.h_subjectsdata,
				numberOfSubjects * mybuffers.sequencepitch,
				H2D,
				mybuffers.streams[0]); CUERR;
		cudaMemcpyAsync(mybuffers.d_queriesdata,
				mybuffers.h_queriesdata,
				totalNumberOfAlignments * mybuffers.sequencepitch,
				H2D,
				mybuffers.streams[0]); CUERR;
		cudaMemcpyAsync(mybuffers.d_subjectlengths,
				mybuffers.h_subjectlengths,
				sizeof(int) * numberOfSubjects,
				H2D,
				mybuffers.streams[0]); CUERR;
		cudaMemcpyAsync(mybuffers.d_querylengths,
				mybuffers.h_querylengths,
				sizeof(int) * totalNumberOfAlignments,
				H2D,
				mybuffers.streams[0]); CUERR;
		cudaMemcpyAsync(mybuffers.d_NqueriesPrefixSum,
				mybuffers.h_NqueriesPrefixSum,
				sizeof(int) * (numberOfSubjects + 1),
				H2D,
				mybuffers.streams[0]); CUERR;

		const int maxSubjectLength = *std::max_element(mybuffers.h_subjectlengths, mybuffers.h_subjectlengths + numberOfSubjects);
		const int maxQueryLength = *std::max_element(mybuffers.h_querylengths, mybuffers.h_querylengths + totalNumberOfAlignments);

		// run kernel
		call_shd_kernel_async<Sequence_t>(mybuffers.d_results,
								mybuffers.d_subjectsdata,
								mybuffers.d_subjectlengths,
								mybuffers.d_queriesdata,
								mybuffers.d_querylengths,
								mybuffers.d_NqueriesPrefixSum,
								numberOfSubjects,
								totalNumberOfAlignments,
								mybuffers.max_sequence_bytes,
								mybuffers.sequencepitch,
								props,
								maxSubjectLength,
								maxQueryLength,
								mybuffers.streams[0]);

		cudaMemcpyAsync(mybuffers.h_results,
						mybuffers.d_results,
						sizeof(AlignResultCompact) * totalNumberOfAlignments,
						D2H,
						mybuffers.streams[0]); CUERR;

		cudaStreamSynchronize(mybuffers.streams[0]); CUERR;

		subjectindex = 0;
		querysum = 0;

		//wait for d2h transfer to complete and fetch results
		for(auto& b : batch){
			if(b.active){
				const AlignResultCompact* results = mybuffers.h_results + querysum;

				tpa = std::chrono::system_clock::now();

				int count = 0;
				for(auto& alignment : b.fwdAlignments){
					alignment = results[count++];
				}
				for(auto& alignment : b.revcomplAlignments){
					alignment = results[count++];
				}

				tpb = std::chrono::system_clock::now();
				mybuffers.postprocessingtime += tpb - tpa;

				querysum += count;
			}
		}

	}else{ // use cpu for alignment

#endif // __NVCC__
		device = AlignmentDevice::CPU;
		tpa = std::chrono::system_clock::now();

		for(auto& b : batch){
			if(b.active){
				const char* const subject = (const char*)b.fwdSequence->begin();
				const int subjectLength = b.fwdSequence->length();

				for(size_t i = 0; i < b.fwdSequences.size(); i++){
					const char* query =  (const char*)b.fwdSequences[i]->begin();
					const int queryLength = b.fwdSequences[i]->length();
					b.fwdAlignments[i] = cpu_shifted_hamming_distance<Sequence_t>(subject, query, subjectLength, queryLength, props.min_overlap, props.maxErrorRate, props.min_overlap_ratio);
				}

				for(size_t i = 0; i < b.revcomplSequences.size(); i++){
					const char* query =  (const char*)b.revcomplSequences[i]->begin();
					const int queryLength = b.revcomplSequences[i]->length();
					b.revcomplAlignments[i] = cpu_shifted_hamming_distance<Sequence_t>(subject, query, subjectLength, queryLength, props.min_overlap, props.maxErrorRate, props.min_overlap_ratio);
				}
			}
		}

		tpb = std::chrono::system_clock::now();

		mybuffers.alignmenttime += tpb - tpa;
#ifdef __NVCC__
	}
#endif // __NVCC__

	return device;
}












/*
    does not wait for gpu kernel to finish

    SubjectIter,QueryIter: Iterator to const Sequence_t*
    AlignmentIter: Iterator to AlignResultCompact
*/
template<class Sequence_t, class SubjectIter, class QueryIter, class AlignmentIter>
AlignmentDevice shifted_hamming_distance_async(SHDdata& mybuffers,
                                SubjectIter subjectsbegin,
                                SubjectIter subjectsend,
                                QueryIter queriesbegin,
                                QueryIter queriesend,
                                AlignmentIter alignmentsbegin,
                                AlignmentIter alignmentsend,
                                const std::vector<int>& queriesPerSubject,
                                int min_overlap,
                                double maxErrorRate,
                                double min_overlap_ratio,
                                bool canUseGpu){

    static_assert(std::is_same<typename AlignmentIter::value_type, AlignResultCompact>::value, "shifted hamming distance unexpected Alignement type");

    AlignmentDevice device = AlignmentDevice::None;

    const int numberOfSubjects = queriesPerSubject.size();
    const int numberOfAlignments = std::distance(alignmentsbegin, alignmentsend);
    assert(numberOfAlignments == std::distance(queriesbegin, queriesend));

    //nothing to do here
    if(numberOfAlignments == 0)
        return device;
#ifdef __NVCC__

    std::chrono::time_point<std::chrono::system_clock> tpa = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> tpb = std::chrono::system_clock::now();

    if(canUseGpu && numberOfAlignments >= mybuffers.gpuThreshold){ // use gpu for alignment
        device = AlignmentDevice::GPU;
        tpa = std::chrono::system_clock::now();

        cudaSetDevice(mybuffers.deviceId); CUERR;

        mybuffers.resize(numberOfSubjects, numberOfAlignments);

        mybuffers.n_subjects = numberOfSubjects;
        mybuffers.n_queries = numberOfAlignments;

        tpb = std::chrono::system_clock::now();

        mybuffers.resizetime += tpb - tpa;

        tpa = std::chrono::system_clock::now();

        int maxSubjectLength = 0;

        for(auto t = std::make_pair(0, subjectsbegin); t.second != subjectsend; t.first++, t.second++){
            auto& count = t.first;
            auto& it = t.second;

            assert((*it)->length() <= mybuffers.max_sequence_length);
            assert((*it)->getNumBytes() <= mybuffers.max_sequence_bytes);

            std::memcpy(mybuffers.h_subjectsdata + count * mybuffers.sequencepitch,
                        (*it)->begin(),
                        (*it)->getNumBytes());

            mybuffers.h_subjectlengths[count] = (*it)->length();
            maxSubjectLength = std::max(int((*it)->length()), maxSubjectLength);
        }

        int maxQueryLength = 0;

        for(auto t = std::make_pair(0, queriesbegin); t.second != queriesend; t.first++, t.second++){
            auto& count = t.first;
            auto& it = t.second;

            assert((*it)->length() <= mybuffers.max_sequence_length);
            assert((*it)->getNumBytes() <= mybuffers.max_sequence_bytes);

            std::memcpy(mybuffers.h_queriesdata + count * mybuffers.sequencepitch,
                        (*it)->begin(),
                        (*it)->getNumBytes());

            mybuffers.h_querylengths[count] = (*it)->length();
            maxQueryLength = std::max(int((*it)->length()), maxQueryLength);
        }

        mybuffers.h_NqueriesPrefixSum[0] = 0;
        for(std::size_t i = 0; i < queriesPerSubject.size(); i++)
            mybuffers.h_NqueriesPrefixSum[i+1] = mybuffers.h_NqueriesPrefixSum[i] + queriesPerSubject[i];

        assert(numberOfAlignments == mybuffers.h_NqueriesPrefixSum[queriesPerSubject.size()]);

        tpb = std::chrono::system_clock::now();

        mybuffers.preprocessingtime += tpb - tpa;


        // copy data to gpu
        cudaMemcpyAsync(mybuffers.d_subjectsdata,
                mybuffers.h_subjectsdata,
                numberOfSubjects * mybuffers.sequencepitch,
                H2D,
                mybuffers.streams[0]); CUERR;
        cudaMemcpyAsync(mybuffers.d_queriesdata,
                mybuffers.h_queriesdata,
                numberOfAlignments * mybuffers.sequencepitch,
                H2D,
                mybuffers.streams[0]); CUERR;
        cudaMemcpyAsync(mybuffers.d_subjectlengths,
                mybuffers.h_subjectlengths,
                numberOfSubjects * sizeof(int),
                H2D,
                mybuffers.streams[0]); CUERR;
        cudaMemcpyAsync(mybuffers.d_querylengths,
                mybuffers.h_querylengths,
                numberOfAlignments * sizeof(int),
                H2D,
                mybuffers.streams[0]); CUERR;
        cudaMemcpyAsync(mybuffers.d_NqueriesPrefixSum,
                mybuffers.h_NqueriesPrefixSum,
                (numberOfSubjects+1) * sizeof(int),
                H2D,
                mybuffers.streams[0]); CUERR;

        call_shd_kernel_async<Sequence_t>(mybuffers.d_results,
                                    mybuffers.d_subjectsdata,
                                    mybuffers.d_subjectlengths,
                                    mybuffers.d_queriesdata,
                                    mybuffers.d_querylengths,
                                    mybuffers.d_NqueriesPrefixSum,
                                    numberOfSubjects,
                                    numberOfAlignments,
                                    mybuffers.max_sequence_bytes,
                                    mybuffers.sequencepitch,
                                    min_overlap,
                                    maxErrorRate,
                                    min_overlap_ratio,
                                    maxSubjectLength,
                                    maxQueryLength,
                                    mybuffers.streams[0]);

        AlignResultCompact* results = mybuffers.h_results;
        AlignResultCompact* d_results = mybuffers.d_results;

        cudaMemcpyAsync(results,
            d_results,
            sizeof(AlignResultCompact) * numberOfAlignments,
            D2H,
            mybuffers.streams[0]); CUERR;

    }else{ // use cpu for alignment

#endif
        device = AlignmentDevice::CPU;

        tpa = std::chrono::system_clock::now();

        auto queryIt = queriesbegin;
        auto alignmentsIt = alignmentsbegin;

        for(auto t = std::make_pair(0, subjectsbegin); t.second != subjectsend; t.first++, t.second++){
            auto& subjectcount = t.first;
            auto& subjectIt = t.second;

            assert((*subjectIt)->length() <= mybuffers.max_sequence_length);
            assert((*subjectIt)->getNumBytes() <= mybuffers.max_sequence_bytes);

            const char* const subject = (const char*)(*subjectIt)->begin();
            const int subjectLength = (*subjectIt)->length();

            const int nQueries = queriesPerSubject[subjectcount];

            for(int i = 0; i < nQueries; i++){
                const char* query =  (const char*)(*queryIt)->begin();
                const int queryLength = (*queryIt)->length();

                *alignmentsIt = cpu_shifted_hamming_distance<Sequence_t>(subject, query, subjectLength, queryLength,
                                                                        min_overlap,
                                                                        maxErrorRate,
                                                                        min_overlap_ratio);

                queryIt++;
                alignmentsIt++;
            }
        }

        tpb = std::chrono::system_clock::now();

        mybuffers.alignmenttime += tpb - tpa;

#ifdef __NVCC__
    }
#endif

    return device;
}



template<class AlignmentIter>
void semi_global_alignment_get_results(SHDdata& mybuffers,
                                AlignmentIter alignmentsbegin,
                                AlignmentIter alignmentsend,
                                bool canUseGpu){

    static_assert(std::is_same<typename AlignmentIter::value_type, AlignResultCompact>::value, "shifted hamming distance unexpected Alignement type");

    const int numberOfAlignments = std::distance(alignmentsbegin, alignmentsend);

    //nothing to do here
    if(numberOfAlignments == 0)
        return;
#ifdef __NVCC__

    std::chrono::time_point<std::chrono::system_clock> tpa = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> tpb = std::chrono::system_clock::now();

    if(canUseGpu && numberOfAlignments >= mybuffers.gpuThreshold){ // use gpu for alignment
        cudaSetDevice(mybuffers.deviceId); CUERR;

        AlignResultCompact* results = mybuffers.h_results;

        cudaStreamSynchronize(mybuffers.streams[0]); CUERR;

        tpa = std::chrono::system_clock::now();

        for(auto t = std::make_pair(0, alignmentsbegin); t.second != alignmentsend; t.first++, t.second++){
            auto& count = t.first;
            auto& it = t.second;

            *it = results[count];
        }

        tpb = std::chrono::system_clock::now();
        mybuffers.postprocessingtime += tpb - tpa;


    }else{ // cpu already done

#endif

#ifdef __NVCC__
    }
#endif

    return;
}






















/*
    Alignment functions implementations
*/

template<class Sequence_t>
AlignResultCompact cpu_shifted_hamming_distance(const char* subject,
												const char* query,
												int ns,
												int nq,
                                                int min_overlap,
                                                double maxErrorRate,
                                                double min_overlap_ratio){

    auto accessor = [] (const char* data, int length, int index){
        return Sequence_t::get(data, length, index);
    };

    return cpu_shifted_hamming_distance_impl(subject, query, ns, nq, min_overlap, maxErrorRate, min_overlap_ratio, accessor);
}


#ifdef __NVCC__

//wrapper functions to call kernels

template<class Sequence_t>
void call_shd_kernel(const shdparams& buffer, int maxQueryLength, cudaStream_t stream){
    call_shd_kernel_async<Sequence_t>(buffer, maxQueryLength, stream);

    cudaStreamSynchronize(stream); CUERR;
}

template<class Sequence_t>
void call_shd_kernel_async(const shdparams& buffer, int maxQueryLength, cudaStream_t stream){
    const int minoverlap = max(buffer.props.min_overlap, int(double(buffer.subjectlength) * buffer.props.min_overlap_ratio));
    const int maxShiftsToCheck = buffer.subjectlength+1 + maxQueryLength - 2*minoverlap;
    dim3 block(std::min(256, 32 * SDIV(maxShiftsToCheck, 32)), 1, 1);
    dim3 grid(buffer.n_queries, 1, 1);

    size_t smem = 0;
    smem += sizeof(char) * 2 * buffer.max_sequence_bytes;

    auto accessor = [] __device__ (const char* data, int length, int index){
        return Sequence_t::get(data, length, index);
    };

    switch(block.x){
    case 32: cuda_shifted_hamming_distance<32><<<grid, block, smem, stream>>>(buffer, accessor); CUERR; break;
    case 64: cuda_shifted_hamming_distance<64><<<grid, block, smem, stream>>>(buffer, accessor); CUERR; break;
    case 96: cuda_shifted_hamming_distance<96><<<grid, block, smem, stream>>>(buffer, accessor); CUERR; break;
    case 128: cuda_shifted_hamming_distance<128><<<grid, block, smem, stream>>>(buffer, accessor); CUERR; break;
    case 160: cuda_shifted_hamming_distance<160><<<grid, block, smem, stream>>>(buffer, accessor); CUERR; break;
    case 192: cuda_shifted_hamming_distance<192><<<grid, block, smem, stream>>>(buffer, accessor); CUERR; break;
    case 224: cuda_shifted_hamming_distance<224><<<grid, block, smem, stream>>>(buffer, accessor); CUERR; break;
    case 256: cuda_shifted_hamming_distance<256><<<grid, block, smem, stream>>>(buffer, accessor); CUERR; break;
    default: throw std::runtime_error("Want to call shd kernel with 0 threads due to a bug.");
    }
}

template<class Sequence_t>
void call_shd_kernel(AlignResultCompact* results,
                      const char* subjectsdata,
                      const int* subjectlengths,
                      const char* queriesdata,
                      const int* querylengths,
                      const int* NqueriesPrefixSum,
                      int Nsubjects,
                      int Nqueries,
                      int max_sequence_bytes,
                      size_t sequencepitch,
                      const GoodAlignmentProperties& props,
                      int maxSubjectLength,
                      int maxQueryLength,
                      cudaStream_t stream){

    call_shd_kernel_async<Sequence_t>(results,
                        subjectsdata,
                        subjectlengths,
                        queriesdata,
                        querylengths,
                        NqueriesPrefixSum,
                        Nsubjects,
                        Nqueries,
                        max_sequence_bytes,
                        sequencepitch,
                        props,
                        maxSubjectLength,
                        maxQueryLength,
                        stream);
    cudaStreamSynchronize(stream); CUERR;
}

template<class Sequence_t>
void call_shd_kernel_async(AlignResultCompact* results,
                      const char* subjectsdata,
                      const int* subjectlengths,
                      const char* queriesdata,
                      const int* querylengths,
                      const int* NqueriesPrefixSum,
                      int Nsubjects,
                      int Nqueries,
                      int max_sequence_bytes,
                      size_t sequencepitch,
                      const GoodAlignmentProperties& props,
                      int maxSubjectLength,
                      int maxQueryLength,
                      cudaStream_t stream){

    const int minoverlap = max(props.min_overlap, int(double(maxSubjectLength) * props.min_overlap_ratio));
    const int maxShiftsToCheck = maxSubjectLength+1 + maxQueryLength - 2*minoverlap;
    dim3 block(std::min(256, 32 * SDIV(maxShiftsToCheck, 32)), 1, 1);
    dim3 grid(Nqueries, 1, 1);

    size_t smem = 0;
    smem += sizeof(char) * 2 * max_sequence_bytes;

    auto accessor = [] __device__ (const char* data, int length, int index){
      return Sequence_t::get(data, length, index);
    };

    switch(block.x){
    case 32: cuda_shifted_hamming_distance<32><<<grid, block, smem, stream>>>(results,
                                                                            subjectsdata, subjectlengths,
                                                                            queriesdata, querylengths,
                                                                            NqueriesPrefixSum, Nsubjects,
                                                                            max_sequence_bytes, sequencepitch, props, accessor); CUERR; break;
    case 64: cuda_shifted_hamming_distance<64><<<grid, block, smem, stream>>>(results,
                                                                            subjectsdata, subjectlengths,
                                                                            queriesdata, querylengths,
                                                                            NqueriesPrefixSum, Nsubjects,
                                                                            max_sequence_bytes, sequencepitch, props, accessor); CUERR; break;
    case 96: cuda_shifted_hamming_distance<96><<<grid, block, smem, stream>>>(results,
                                                                            subjectsdata, subjectlengths,
                                                                            queriesdata, querylengths,
                                                                            NqueriesPrefixSum, Nsubjects,
                                                                            max_sequence_bytes, sequencepitch, props, accessor); CUERR; break;
    case 128: cuda_shifted_hamming_distance<128><<<grid, block, smem, stream>>>(results,
                                                                            subjectsdata, subjectlengths,
                                                                            queriesdata, querylengths,
                                                                            NqueriesPrefixSum, Nsubjects,
                                                                            max_sequence_bytes, sequencepitch, props, accessor); CUERR; break;
    case 160: cuda_shifted_hamming_distance<160><<<grid, block, smem, stream>>>(results,
                                                                            subjectsdata, subjectlengths,
                                                                            queriesdata, querylengths,
                                                                            NqueriesPrefixSum, Nsubjects,
                                                                            max_sequence_bytes, sequencepitch, props, accessor); CUERR; break;
    case 192: cuda_shifted_hamming_distance<192><<<grid, block, smem, stream>>>(results,
                                                                            subjectsdata, subjectlengths,
                                                                            queriesdata, querylengths,
                                                                            NqueriesPrefixSum, Nsubjects,
                                                                            max_sequence_bytes, sequencepitch, props, accessor); CUERR; break;
    case 224: cuda_shifted_hamming_distance<224><<<grid, block, smem, stream>>>(results,
                                                                            subjectsdata, subjectlengths,
                                                                            queriesdata, querylengths,
                                                                            NqueriesPrefixSum, Nsubjects,
                                                                            max_sequence_bytes, sequencepitch, props, accessor); CUERR; break;
    case 256: cuda_shifted_hamming_distance<256><<<grid, block, smem, stream>>>(results,
                                                                            subjectsdata, subjectlengths,
                                                                            queriesdata, querylengths,
                                                                            NqueriesPrefixSum, Nsubjects,
                                                                            max_sequence_bytes, sequencepitch, props, accessor); CUERR; break;
    default: throw std::runtime_error("Want to call shd kernel with 0 threads due to a bug.");
    }
}








template<class Sequence_t>
void call_shd_kernel(AlignResultCompact* results,
                      const char* subjectsdata,
                      const int* subjectlengths,
                      const char* queriesdata,
                      const int* querylengths,
                      const int* NqueriesPrefixSum,
                      int Nsubjects,
                      int Nqueries,
                      int max_sequence_bytes,
                      size_t sequencepitch,
                      int min_overlap,
                      double maxErrorRate,
                      double min_overlap_ratio,
                      int maxSubjectLength,
                      int maxQueryLength,
                      cudaStream_t stream){

    call_shd_kernel_async<Sequence_t>(results,
                        subjectsdata,
                        subjectlengths,
                        queriesdata,
                        querylengths,
                        NqueriesPrefixSum,
                        Nsubjects,
                        Nqueries,
                        max_sequence_bytes,
                        sequencepitch,
                        min_overlap,
                        maxErrorRate,
                        min_overlap_ratio,
                        maxSubjectLength,
                        maxQueryLength,
                        stream);
    cudaStreamSynchronize(stream); CUERR;
}

template<class Sequence_t>
void call_shd_kernel_async(AlignResultCompact* results,
                      const char* subjectsdata,
                      const int* subjectlengths,
                      const char* queriesdata,
                      const int* querylengths,
                      const int* NqueriesPrefixSum,
                      int Nsubjects,
                      int Nqueries,
                      int max_sequence_bytes,
                      size_t sequencepitch,
                      int min_overlap,
                      double maxErrorRate,
                      double min_overlap_ratio,
                      int maxSubjectLength,
                      int maxQueryLength,
                      cudaStream_t stream){

    const int minoverlap = max(min_overlap, int(double(maxSubjectLength) * min_overlap_ratio));
    const int maxShiftsToCheck = maxSubjectLength+1 + maxQueryLength - 2*minoverlap;
    dim3 block(std::min(256, 32 * SDIV(maxShiftsToCheck, 32)), 1, 1);
    dim3 grid(Nqueries, 1, 1);

    size_t smem = 0;
    smem += sizeof(char) * 2 * max_sequence_bytes;

    auto accessor = [] __device__ (const char* data, int length, int index){
      return Sequence_t::get(data, length, index);
    };


    #define mycall(blocksize) cuda_shifted_hamming_distance<(blocksize)><<<grid, block, smem, stream>>>(results, \
                                                                            subjectsdata, subjectlengths, \
                                                                            queriesdata, querylengths, \
                                                                            NqueriesPrefixSum, Nsubjects, \
                                                                            max_sequence_bytes, \
                                                                            sequencepitch, \
                                                                            min_overlap, \
                                                                            maxErrorRate, \
                                                                            min_overlap_ratio, \
                                                                            accessor); CUERR;

    switch(block.x){
    case 32: mycall(32); break;
    case 64: mycall(64); break;
    case 96: mycall(96); break;
    case 128: mycall(128); break;
    case 160: mycall(160); break;
    case 192: mycall(192); break;
    case 224: mycall(224); break;
    case 256: mycall(256); break;
    default: throw std::runtime_error("Want to call shd kernel with 0 threads due to a bug.");
    }

    #undef mycall
}












#endif

}

#include "shifted_hamming_distance_impl.hpp"

#endif
