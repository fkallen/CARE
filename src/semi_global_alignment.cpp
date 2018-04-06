#include "../inc/semi_global_alignment.hpp"
#include "../inc/semi_global_alignment_impl.hpp"

namespace care{

/*
    SGAdata implementation
*/

void SGAdata::resize(int n_sub, int n_quer){
#ifdef __NVCC__
	cudaSetDevice(deviceId); CUERR;

	bool resizeResult = false;

	if(n_sub > max_n_subjects){
		const int newmax = 1.5 * n_sub;
		std::size_t oldpitch = sequencepitch;
		cudaFree(d_subjectsdata); CUERR;
		cudaMallocPitch(&d_subjectsdata, &sequencepitch, max_sequence_bytes, newmax); CUERR;
		assert(!oldpitch || oldpitch == sequencepitch);

		cudaFreeHost(h_subjectsdata); CUERR;
		cudaMallocHost(&h_subjectsdata, sequencepitch * newmax); CUERR;

		cudaFree(d_subjectlengths); CUERR;
		cudaMalloc(&d_subjectlengths, sizeof(int) * newmax); CUERR;

		cudaFreeHost(h_subjectlengths); CUERR;
		cudaMallocHost(&h_subjectlengths, sizeof(int) * newmax); CUERR;

		max_n_subjects = newmax;
		resizeResult = true;
	}


	if(n_quer > max_n_queries){
		const int newmax = 1.5 * n_quer;
		size_t oldpitch = sequencepitch;
		cudaFree(d_queriesdata); CUERR;
		cudaMallocPitch(&d_queriesdata, &sequencepitch, max_sequence_bytes, newmax); CUERR;
		assert(!oldpitch || oldpitch == sequencepitch);

		cudaFreeHost(h_queriesdata); CUERR;
		cudaMallocHost(&h_queriesdata, sequencepitch * newmax); CUERR;

		cudaFree(d_querylengths); CUERR;
		cudaMalloc(&d_querylengths, sizeof(int) * newmax); CUERR;

		cudaFreeHost(h_querylengths); CUERR;
		cudaMallocHost(&h_querylengths, sizeof(int) * newmax); CUERR;

		max_n_queries = newmax;
		resizeResult = true;
	}

	if(resizeResult){
		cudaFree(d_results); CUERR;
		cudaMalloc(&d_results, sizeof(AlignResultCompact) * max_n_subjects * max_n_queries); CUERR;

		cudaFreeHost(h_results); CUERR;
		cudaMallocHost(&h_results, sizeof(AlignResultCompact) * max_n_subjects * max_n_queries); CUERR;

		cudaFree(d_ops); CUERR;
		cudaFreeHost(h_ops); CUERR;

		cudaMalloc(&d_ops, sizeof(AlignOp) * max_n_queries * max_ops_per_alignment); CUERR;
		cudaMallocHost(&h_ops, sizeof(AlignOp) * max_n_queries * max_ops_per_alignment); CUERR;
	}
#endif

	n_subjects = n_sub;
	n_queries = n_quer;
}


void cuda_init_SGAdata(SGAdata& data,
                       int deviceId,
                       int max_sequence_length,
                       int max_sequence_bytes,
                       int gpuThreshold){

    data.deviceId = deviceId;
    data.max_sequence_length = 32 * SDIV(max_sequence_length, 32); //round up to multiple of 32;
    data.max_sequence_bytes = max_sequence_bytes;
    data.max_ops_per_alignment = 2 * (data.max_sequence_length + 1);
    data.gpuThreshold = gpuThreshold;

#ifdef __NVCC__
    cudaSetDevice(deviceId); CUERR;

    for(int i = 0; i < SGAdata::n_streams; i++)
        cudaStreamCreate(&(data.streams[i])); CUERR;
#endif
}

void cuda_cleanup_SGAdata(SGAdata& data){
	#ifdef __NVCC__
		cudaSetDevice(data.deviceId); CUERR;

		cudaFree(data.d_results); CUERR;
		cudaFree(data.d_ops); CUERR;
		cudaFree(data.d_subjectsdata); CUERR;
		cudaFree(data.d_queriesdata); CUERR;
		cudaFree(data.d_subjectlengths); CUERR;
		cudaFree(data.d_querylengths); CUERR;

		cudaFreeHost(data.h_results); CUERR;
		cudaFreeHost(data.h_ops); CUERR;
		cudaFreeHost(data.h_subjectsdata); CUERR;
		cudaFreeHost(data.h_queriesdata); CUERR;
		cudaFreeHost(data.h_subjectlengths); CUERR;
		cudaFreeHost(data.h_querylengths); CUERR;

		for(int i = 0; i < SGAdata::n_streams; i++)
			cudaStreamDestroy(data.streams[i]); CUERR;
	#endif
}


/*
    Alignment functions definitions
*/

AlignResult cpu_semi_global_alignment(const SGAdata* buffers, const AlignmentOptions& alignmentOptions,
                                      const char* r1, const char* r2, int r1length, int r2bases);

#ifdef __NVCC__

void call_cuda_semi_global_alignment_kernel_async(const sgaparams& buffers, cudaStream_t stream);
void call_cuda_semi_global_alignment_kernel(const sgaparams& buffers, cudaStream_t stream);

#endif


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

    Sequence sequence(seqstring);

    //GoodAlignmentProperties alignProps;
    AlignmentOptions alignmentOptions;

    do{
        nalignments += increment;
        std::vector<Sequence> sequences(nalignments, sequence);
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
        call_cuda_semi_global_alignment_kernel_async(params, sgadata.streams[0]);

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
            auto res = cpu_semi_global_alignment(&sgadata, alignmentOptions, subject, query, subjectLength, queryLength);
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
AlignmentDevice semi_global_alignment_async(SGAdata& mybuffers, BatchElem& b,
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
        call_cuda_semi_global_alignment_kernel_async(params, mybuffers.streams[0]);

    }else{ // use cpu for alignment

#endif
        device = AlignmentDevice::CPU;
#ifdef __NVCC__
    }
#endif

    return device;
}


void get_semi_global_alignment_results(SGAdata& mybuffers, BatchElem& b,
                                int firstIndex, int N,
                                const AlignmentOptions& alignmentOptions,
                                bool canUseGpu){

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

        b.fwdAlignOps.resize(b.fwdSequences.size());
        b.revcomplAlignOps.resize(b.revcomplSequences.size());

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
            b.fwdAlignments[index] = results[count];
            b.fwdAlignOps[index].resize(b.fwdAlignments[index].nOps);
            std::reverse_copy(ops + count * mybuffers.max_ops_per_alignment,
                      ops + count * mybuffers.max_ops_per_alignment + b.fwdAlignments[index].nOps,
                      b.fwdAlignOps[index].begin());
        }

        for(int index = firstIndex; index < lastIndex_excl; index++){
            b.revcomplAlignments[index] = results[(numberOfCandidates + count)];
            b.revcomplAlignOps[index].resize(b.revcomplAlignments[index].nOps);
            std::reverse_copy(ops + (numberOfCandidates + count) * mybuffers.max_ops_per_alignment,
                      ops + (numberOfCandidates + count) * mybuffers.max_ops_per_alignment + b.revcomplAlignments[index].nOps,
                      b.revcomplAlignOps[index].begin());
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
            auto al = cpu_semi_global_alignment(&mybuffers, alignmentOptions, subject, query, subjectLength, queryLength);
            b.fwdAlignments[index] = al.arc;
            b.fwdAlignOps[index] = std::move(al.operations);
        }

        for(int index = firstIndex; index < lastIndex_excl; index++){
            const char* query =  (const char*)b.revcomplSequences[index]->begin();
            const int queryLength = b.revcomplSequences[index]->length();
            auto al = cpu_semi_global_alignment(&mybuffers, alignmentOptions, subject, query, subjectLength, queryLength);
            b.revcomplAlignments[index] = al.arc;
            b.revcomplAlignOps[index] = std::move(al.operations);
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

namespace sgadetail{
    #include "../inc/sequenceaccessor.hpp"
}

/*
    CPU alignment
*/

AlignResult cpu_semi_global_alignment(const SGAdata* buffers, const AlignmentOptions& alignmentOptions,
                                      const char* r1, const char* r2, int r1length, int r2bases){

    auto accessor = [] (const char* data, int length, int index){
        return sgadetail::encoded_accessor(data, length, index);
    };

    return cpu_semi_global_alignment_impl(buffers, alignmentOptions, r1, r2, r1length, r2bases, accessor);
}

/*
    GPU alignment
*/

#ifdef __NVCC__

void call_cuda_semi_global_alignment_kernel_async(const sgaparams& buffers, cudaStream_t stream){

        dim3 block(buffers.max_sequence_length, 1, 1);
        dim3 grid(buffers.n_queries, 1, 1);

        auto accessor = [] __device__ (const char* data, int length, int index){
            return sgadetail::encoded_accessor(data, length, index);
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

void call_cuda_semi_global_alignment_kernel(const sgaparams& buffers, cudaStream_t stream){

        call_cuda_semi_global_alignment_kernel_async(buffers, stream);

        cudaStreamSynchronize(stream); CUERR;
}







#endif

}
