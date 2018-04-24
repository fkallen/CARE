#ifndef CARE_SEMI_GLOBAL_ALIGNMENT_HPP
#define CARE_SEMI_GLOBAL_ALIGNMENT_HPP

#include "alignment.hpp"
#include "batchelem.hpp"
#include "options.hpp"

#include <cstdint>
#include <cstring>
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

	struct sgaparams{
		int max_sequence_length;
		int max_ops_per_alignment;
		int sequencepitch;
		int n_queries;
		int subjectlength;
		int alignmentscore_match = 1;
		int alignmentscore_sub = -1;
		int alignmentscore_ins = -1;
		int alignmentscore_del = -1;
		const int* __restrict__ querylengths;
		const char* __restrict__ subjectdata;
		const char* __restrict__ queriesdata;
		AlignResultCompact* __restrict__ results;
		AlignOp* __restrict__ ops;
	};

    void cuda_init_SGAdata(SGAdata& data,
                           int deviceId,
                           int max_sequence_length,
                           int max_sequence_bytes,
                           int gpuThreshold);

    void cuda_cleanup_SGAdata(SGAdata& data);





template<class Sequence_t>
AlignResult cpu_semi_global_alignment(const SGAdata* buffers, const AlignmentOptions& alignmentOptions,
                                      const char* r1, const char* r2, int r1length, int r2bases);
template<class Accessor>
AlignResult
cpu_semi_global_alignment_impl(const SGAdata* buffers,
                               const AlignmentOptions& alignmentOptions,
                               const char* subject,
                               const char* query,
                               int subjectbases,
                               int querybases,
                               Accessor getChar);

#ifdef __NVCC__

template<class Sequence_t>
void call_cuda_semi_global_alignment_kernel_async(const sgaparams& buffers, cudaStream_t stream);

template<class Sequence_t>
void call_cuda_semi_global_alignment_kernel(const sgaparams& buffers, cudaStream_t stream);

template<int MAX_SEQUENCE_LENGTH, class Accessor>
__global__
void cuda_semi_global_alignment_kernel(const sgaparams buffers, Accessor getChar);

#endif



template<class Sequence_t>
int find_semi_global_alignment_gpu_threshold(int deviceId, int minsequencelength, int minsequencebytes){
    int threshold = std::numeric_limits<int>::max();

#ifdef __NVCC__
    SGAdata sgadata;
    cuda_init_SGAdata(sgadata, deviceId, minsequencelength, minsequencebytes, 0);

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

    //GoodAlignmentProperties alignProps;
    AlignmentOptions alignmentOptions;

    do{
        nalignments += increment;
        std::vector<Sequence_t> sequences(nalignments, sequence);
        std::vector<AlignResultCompact> gpuresults(nalignments);
        std::vector<AlignResultCompact> cpuresults(nalignments);
        std::vector<std::vector<AlignOp>> gpuops(nalignments);
        std::vector<std::vector<AlignOp>> cpuops(nalignments);

        gpustart = std::chrono::system_clock::now();

        sgadata.resize(1, nalignments);
        sgaparams params;

        params.max_sequence_length = sgadata.max_sequence_length;
        params.max_ops_per_alignment = sgadata.max_ops_per_alignment;
        params.sequencepitch = sgadata.sequencepitch;
        params.subjectlength = minsequencelength;
        params.n_queries = nalignments;
        params.querylengths = sgadata.d_querylengths;
        params.subjectdata = sgadata.d_subjectsdata;
        params.queriesdata = sgadata.d_queriesdata;
        params.results = sgadata.d_results;
        params.ops = sgadata.d_ops;
        params.alignmentscore_match = alignmentOptions.alignmentscore_match;
        params.alignmentscore_sub = alignmentOptions.alignmentscore_sub;
        params.alignmentscore_ins = alignmentOptions.alignmentscore_ins;
        params.alignmentscore_del = alignmentOptions.alignmentscore_del;

        int* querylengths = sgadata.h_querylengths;
        char* subjectdata = sgadata.h_subjectsdata;
        char* queriesdata = sgadata.h_queriesdata;

        std::memcpy(subjectdata, sequences[0].begin(), sequences[0].getNumBytes());
        for(int count = 0; count < nalignments; count++){
            const auto& seq = sequences[count];

            std::memcpy(queriesdata + count * sgadata.sequencepitch,
                    seq.begin(),
                    seq.getNumBytes());

            querylengths[count] = seq.length();
        }
        cudaMemcpyAsync(const_cast<char*>(params.subjectdata),
                subjectdata,
                sgadata.sequencepitch,
                H2D,
                sgadata.streams[0]); CUERR;
        cudaMemcpyAsync(const_cast<char*>(params.queriesdata),
                queriesdata,
                sgadata.sequencepitch * params.n_queries,
                H2D,
                sgadata.streams[0]); CUERR;
        cudaMemcpyAsync(const_cast<int*>(params.querylengths),
                querylengths,
                sizeof(int) * params.n_queries,
                H2D,
                sgadata.streams[0]); CUERR;

        // start kernel
        call_cuda_semi_global_alignment_kernel_async<Sequence_t>(params, sgadata.streams[0]);

        AlignResultCompact* results = sgadata.h_results;
        AlignOp* ops = sgadata.h_ops;

        cudaMemcpyAsync(results,
            params.results,
            sizeof(AlignResultCompact) * params.n_queries,
            D2H,
            sgadata.streams[0]); CUERR;

        cudaMemcpyAsync(ops,
            params.ops,
            sizeof(AlignOp) * params.n_queries * sgadata.max_ops_per_alignment,
            D2H,
            sgadata.streams[0]); CUERR;

        cudaStreamSynchronize(sgadata.streams[0]); CUERR;

        for(int count = 0; count < nalignments; count++){
            gpuresults[count] = results[count];
            gpuops[count].resize(gpuresults[count].nOps);
            std::reverse_copy(ops + count * sgadata.max_ops_per_alignment,
                      ops + count * sgadata.max_ops_per_alignment + gpuresults[count].nOps,
                      gpuops[count].begin());
        }

        gpuend = std::chrono::system_clock::now();


        cpustart = std::chrono::system_clock::now();

        const char* const subject = (const char*)sequences[0].begin();
        const int subjectLength = sequences[0].length();

        for(int i = 0; i < nalignments; i++){
            const char* query =  (const char*)sequences[i].begin();
            const int queryLength = sequences[i].length();
            auto res = cpu_semi_global_alignment<Sequence_t>(&sgadata, alignmentOptions, subject, query, subjectLength, queryLength);
            gpuresults[i] = res.arc;
            gpuops[i] = std::move(res.operations);
        }

        cpuend = std::chrono::system_clock::now();


    }while(gpuend - gpustart > cpuend - cpustart || nalignments == 1000);

    if(gpuend - gpustart <= cpuend - cpustart){
        threshold = nalignments;
    }

    cuda_cleanup_SGAdata(sgadata);
#endif

    return threshold;
}


//In BatchElem b, calculate alignments[firstIndex] to alignments[firstIndex + N - 1]
template<class BatchElem_t>
AlignmentDevice semi_global_alignment_async(SGAdata& mybuffers, BatchElem_t& b,
                                int firstIndex, int N,
                                const AlignmentOptions& alignmentOptions,
                                bool canUseGpu){

    AlignmentDevice device = AlignmentDevice::None;

    const int lastIndex_excl = std::min(size_t(firstIndex + N), b.fwdSequences.size());
    const int numberOfCandidates = firstIndex >= lastIndex_excl ? 0 : lastIndex_excl - firstIndex;
    const int numberOfAlignments = 2 * numberOfCandidates; //fwd and rev compl

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

        mybuffers.n_subjects = 1;
        mybuffers.n_queries = numberOfAlignments;

        tpb = std::chrono::system_clock::now();

        mybuffers.resizetime += tpb - tpa;

        tpa = std::chrono::system_clock::now();

        sgaparams params;

        tpa = std::chrono::system_clock::now();

        params.max_sequence_length = mybuffers.max_sequence_length;
        params.max_ops_per_alignment = mybuffers.max_ops_per_alignment;
        params.sequencepitch = mybuffers.sequencepitch;
        params.subjectlength = b.fwdSequence->length();
        params.n_queries = mybuffers.n_queries;
        params.querylengths = mybuffers.d_querylengths;
        params.subjectdata = mybuffers.d_subjectsdata;
        params.queriesdata = mybuffers.d_queriesdata;
        params.results = mybuffers.d_results;
        params.ops = mybuffers.d_ops;
        params.alignmentscore_match = alignmentOptions.alignmentscore_match;
        params.alignmentscore_sub = alignmentOptions.alignmentscore_sub;
        params.alignmentscore_ins = alignmentOptions.alignmentscore_ins;
        params.alignmentscore_del = alignmentOptions.alignmentscore_del;

        int* querylengths = mybuffers.h_querylengths;
        char* subjectdata = mybuffers.h_subjectsdata;
        char* queriesdata = mybuffers.h_queriesdata;

        assert(b.fwdSequence->length() <= mybuffers.max_sequence_length);

        std::memcpy(subjectdata, b.fwdSequence->begin(), b.fwdSequence->getNumBytes());

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

		using Sequence_t = typename BatchElem_t::Sequence_t;
        call_cuda_semi_global_alignment_kernel_async<Sequence_t>(params, mybuffers.streams[0]);

    }else{ // use cpu for alignment

#endif
        device = AlignmentDevice::CPU;
#ifdef __NVCC__
    }
#endif

    return device;
}


template<class BatchElem_t>
void get_semi_global_alignment_results(SGAdata& mybuffers, BatchElem_t& b,
                                int firstIndex, int N,
                                const AlignmentOptions& alignmentOptions,
                                bool canUseGpu){

	using Sequence_t = typename BatchElem_t::Sequence_t;

    std::chrono::time_point<std::chrono::system_clock> tpa = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> tpb = std::chrono::system_clock::now();

    const int lastIndex_excl = std::min(size_t(firstIndex + N), b.fwdSequences.size());
    const int numberOfCandidates = firstIndex >= lastIndex_excl ? 0 : lastIndex_excl - firstIndex;
    const int numberOfAlignments = 2 * numberOfCandidates; //fwd and rev compl

    //nothing to do here
    if(!b.active || numberOfAlignments == 0)
        return;
#ifdef __NVCC__

    if(canUseGpu && numberOfAlignments >= mybuffers.gpuThreshold){ // use gpu for alignment

        AlignResultCompact* results = mybuffers.h_results;
        AlignResultCompact* d_results = mybuffers.d_results;
        AlignOp* ops = mybuffers.h_ops;
        AlignOp* d_ops = mybuffers.d_ops;

        cudaMemcpyAsync(results,
            d_results,
            sizeof(AlignResultCompact) * numberOfAlignments,
            D2H,
            mybuffers.streams[0]); CUERR;

        cudaMemcpyAsync(ops,
            d_ops,
            sizeof(AlignOp) * numberOfAlignments * mybuffers.max_ops_per_alignment,
            D2H,
            mybuffers.streams[0]); CUERR;

        cudaStreamSynchronize(mybuffers.streams[0]); CUERR;

        tpa = std::chrono::system_clock::now();

        int count = 0;
        for(int index = firstIndex; index < lastIndex_excl; index++){
            b.fwdAlignments[index].arc = results[count];
            std::vector<AlignOp>& opvector = b.fwdAlignments[index].operations;
            opvector.resize(b.fwdAlignments[index].get_nOps());
            std::reverse_copy(ops + count * mybuffers.max_ops_per_alignment,
                      ops + count * mybuffers.max_ops_per_alignment + b.fwdAlignments[index].get_nOps(),
                      opvector.begin());
            count++;
        }

        for(int index = firstIndex; index < lastIndex_excl; index++){
            b.revcomplAlignments[index].arc = results[count];
            std::vector<AlignOp>& opvector = b.revcomplAlignments[index].operations;
            opvector.resize(b.revcomplAlignments[index].get_nOps());
            std::reverse_copy(ops + count * mybuffers.max_ops_per_alignment,
                      ops + count * mybuffers.max_ops_per_alignment + b.revcomplAlignments[index].get_nOps(),
                      opvector.begin());
            count++;
        }

        tpb = std::chrono::system_clock::now();
        mybuffers.postprocessingtime += tpb - tpa;

    }else{ // use cpu for alignment

#endif
        tpa = std::chrono::system_clock::now();

        const char* const subject = (const char*)b.fwdSequence->begin();
        const int subjectLength = b.fwdSequence->length();

        for(int index = firstIndex; index < lastIndex_excl; index++){
            const char* query =  (const char*)b.fwdSequences[index]->begin();
            const int queryLength = b.fwdSequences[index]->length();
            b.fwdAlignments[index] = cpu_semi_global_alignment<Sequence_t>(&mybuffers, alignmentOptions, subject, query, subjectLength, queryLength);
        }

        for(int index = firstIndex; index < lastIndex_excl; index++){
            const char* query =  (const char*)b.revcomplSequences[index]->begin();
            const int queryLength = b.revcomplSequences[index]->length();
            b.revcomplAlignments[index] = cpu_semi_global_alignment<Sequence_t>(&mybuffers, alignmentOptions, subject, query, subjectLength, queryLength);
        }

        tpb = std::chrono::system_clock::now();

        mybuffers.alignmenttime += tpb - tpa;

#ifdef __NVCC__
    }
#endif

}


/*
    Alignment functions implementations
*/

/*
    CPU alignment
*/
template<class Sequence_t>
AlignResult cpu_semi_global_alignment(const SGAdata* buffers, const AlignmentOptions& alignmentOptions,
                                      const char* r1, const char* r2, int r1length, int r2bases){

    auto accessor = [] (const char* data, int length, int index){
        return Sequence_t::get(data, length, index);
    };

    return cpu_semi_global_alignment_impl(buffers, alignmentOptions, r1, r2, r1length, r2bases, accessor);
}

/*
    GPU alignment
*/

#ifdef __NVCC__

template<class Sequence_t>
void call_cuda_semi_global_alignment_kernel_async(const sgaparams& buffers, cudaStream_t stream){

        dim3 block(buffers.max_sequence_length, 1, 1);
        dim3 grid(buffers.n_queries, 1, 1);

        auto accessor = [] __device__ (const char* data, int length, int index){
            return Sequence_t::get(data, length, index);
        };

        // start kernel
        switch(buffers.max_sequence_length){
        case 32: cuda_semi_global_alignment_kernel<32><<<grid, block, 0, stream>>>(buffers, accessor); break;
        case 64: cuda_semi_global_alignment_kernel<64><<<grid, block, 0, stream>>>(buffers, accessor); break;
        case 96: cuda_semi_global_alignment_kernel<96><<<grid, block, 0, stream>>>(buffers, accessor); break;
        case 128: cuda_semi_global_alignment_kernel<128><<<grid, block, 0, stream>>>(buffers, accessor); break;
        case 160: cuda_semi_global_alignment_kernel<160><<<grid, block, 0, stream>>>(buffers, accessor); break;
        case 192: cuda_semi_global_alignment_kernel<192><<<grid, block, 0, stream>>>(buffers, accessor); break;
        case 224: cuda_semi_global_alignment_kernel<224><<<grid, block, 0, stream>>>(buffers, accessor); break;
        case 256: cuda_semi_global_alignment_kernel<256><<<grid, block, 0, stream>>>(buffers, accessor); break;
        case 288: cuda_semi_global_alignment_kernel<288><<<grid, block, 0, stream>>>(buffers, accessor); break;
        case 320: cuda_semi_global_alignment_kernel<320><<<grid, block, 0, stream>>>(buffers, accessor); break;
        default: throw std::runtime_error("cannot use cuda semi global alignment for sequences longer than 320.");
        }

        CUERR;
}

template<class Sequence_t>
void call_cuda_semi_global_alignment_kernel(const sgaparams& buffers, cudaStream_t stream){

        call_cuda_semi_global_alignment_kernel_async<Sequence_t>(buffers, stream);

        cudaStreamSynchronize(stream); CUERR;
}

#endif

}

#include "semi_global_alignment_impl.hpp"

#endif
