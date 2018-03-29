#include "../inc/semi_global_alignment.hpp"
#include "../inc/semi_global_alignment_impl.hpp"

namespace care{

/*
    SGAdata implementation
*/

SGAdata::SGAdata(int deviceId_, int maxseqlength, int maxseqbytes, int batchsize)
		: deviceId(deviceId_), batchsize(batchsize),
					max_sequence_length(32 * SDIV(maxseqlength, 32)), //round up to multiple of 32
					max_sequence_bytes(maxseqbytes),
					max_ops_per_alignment(2 * (max_sequence_length + 1)){
	#ifdef __NVCC__

	cudaSetDevice(deviceId); CUERR;

    streams = std::make_unique<cudaStream_t[]>(batchsize);
    for(int i = 0; i < batchsize; i++)
        cudaStreamCreate(&(streams.get()[i])); CUERR;

	#endif
};

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

		for(int i = 0; i < data.batchsize; i++)
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

/*
    Batch alignment implementation
*/

void semi_global_alignment(SGAdata& mybuffers, const AlignmentOptions& alignmentOptions,
                            std::vector<BatchElem>& batch, bool useGpu){

    std::chrono::time_point<std::chrono::system_clock> tpa;
    std::chrono::time_point<std::chrono::system_clock> tpb;

    int numberOfRealSubjects = 0;
    int totalNumberOfAlignments = 0;

    for(auto& b : batch){
        if(b.active){
            numberOfRealSubjects++;
            totalNumberOfAlignments += b.fwdSequences.size();
            totalNumberOfAlignments += b.revcomplSequences.size();
        }
    }

    // check for empty input
    if(totalNumberOfAlignments == 0){
        return;
    }

#ifdef __NVCC__

    if(useGpu){ // use gpu for alignment

        tpa = std::chrono::system_clock::now();

        cudaSetDevice(mybuffers.deviceId); CUERR;

        mybuffers.resize(numberOfRealSubjects, totalNumberOfAlignments);

        mybuffers.n_subjects = numberOfRealSubjects;
        mybuffers.n_queries = totalNumberOfAlignments;

        tpb = std::chrono::system_clock::now();

        mybuffers.resizetime += tpb - tpa;

        tpa = std::chrono::system_clock::now();

        int subjectindex = 0;
        int querysum = 0;
        int batchid = 0;
        std::vector<sgaparams> params(batch.size());

        for(auto& b : batch){
            if(b.active){
                tpa = std::chrono::system_clock::now();
                batchid = subjectindex;

                params[batchid].max_sequence_length = mybuffers.max_sequence_length;
                params[batchid].max_ops_per_alignment = mybuffers.max_ops_per_alignment;
                params[batchid].sequencepitch = mybuffers.sequencepitch;
                params[batchid].subjectlength = b.fwdSequence->length();
                params[batchid].n_queries = b.fwdSequences.size() + b.revcomplSequences.size();
                params[batchid].querylengths = mybuffers.d_querylengths + querysum;
                params[batchid].subjectdata = mybuffers.d_subjectsdata + mybuffers.sequencepitch * subjectindex;
                params[batchid].queriesdata = mybuffers.d_queriesdata + mybuffers.sequencepitch * querysum;
                params[batchid].results = mybuffers.d_results + querysum;
                params[batchid].ops = mybuffers.d_ops + querysum * mybuffers.max_ops_per_alignment;
                params[batchid].alignmentscore_match = alignmentOptions.alignmentscore_match;
                params[batchid].alignmentscore_sub = alignmentOptions.alignmentscore_sub;
                params[batchid].alignmentscore_ins = alignmentOptions.alignmentscore_ins;
                params[batchid].alignmentscore_del = alignmentOptions.alignmentscore_del;

                int* querylengths = mybuffers.h_querylengths + querysum;
                char* subjectdata = mybuffers.h_subjectsdata + mybuffers.sequencepitch * subjectindex;
                char* queriesdata = mybuffers.h_queriesdata + mybuffers.sequencepitch * querysum;

                assert(b.fwdSequence->length() <= mybuffers.max_sequence_length);

                std::memcpy(subjectdata, b.fwdSequence->begin(), b.fwdSequence->getNumBytes());

                int count = 0;
                for(const auto seq : b.fwdSequences){
                    assert(seq->length() <= mybuffers.max_sequence_length);
                    assert(seq->getNumBytes() <= mybuffers.max_sequence_bytes);

                    std::memcpy(queriesdata + count * mybuffers.sequencepitch,
                            seq->begin(),
                            seq->getNumBytes());

                    querylengths[count] = seq->length();
                    count++;
                }
                for(const auto seq : b.revcomplSequences){
                    assert(seq->length() <= mybuffers.max_sequence_length);
                    assert(seq->getNumBytes() <= mybuffers.max_sequence_bytes);

                    std::memcpy(queriesdata + count * mybuffers.sequencepitch,
                            seq->begin(),
                            seq->getNumBytes());

                    querylengths[count] = seq->length();
                    count++;
                }
                assert(params[batchid].n_queries == count);

                tpb = std::chrono::system_clock::now();

                mybuffers.preprocessingtime += tpb - tpa;
                // copy data to gpu
                cudaMemcpyAsync(const_cast<char*>(params[batchid].subjectdata),
                        subjectdata,
                        mybuffers.sequencepitch,
                        H2D,
                        mybuffers.streams[batchid]); CUERR;
                cudaMemcpyAsync(const_cast<char*>(params[batchid].queriesdata),
                        queriesdata,
                        mybuffers.sequencepitch * params[batchid].n_queries,
                        H2D,
                        mybuffers.streams[batchid]); CUERR;
                cudaMemcpyAsync(const_cast<int*>(params[batchid].querylengths),
                        querylengths,
                        sizeof(int) * params[batchid].n_queries,
                        H2D,
                        mybuffers.streams[batchid]); CUERR;
                call_cuda_semi_global_alignment_kernel_async(params[batchid], mybuffers.streams[batchid]);

                querysum += count;
                subjectindex++;

                b.fwdAlignOps.resize(b.fwdSequences.size());
                b.revcomplAlignOps.resize(b.revcomplSequences.size());
            }
        }


        subjectindex = 0;
        querysum = 0;
        //initialize transfer d2h
        for(auto& b : batch){
            if(b.active){
                batchid = subjectindex;
                AlignResultCompact* results = mybuffers.h_results + querysum;
                AlignOp* ops = mybuffers.h_ops + querysum * mybuffers.max_ops_per_alignment;

                cudaMemcpyAsync(results,
                    params[batchid].results,
                    sizeof(AlignResultCompact) * params[batchid].n_queries,
                    D2H,
                    mybuffers.streams[batchid]); CUERR;

                cudaMemcpyAsync(ops,
                    params[batchid].ops,
                    sizeof(AlignOp) * params[batchid].n_queries * mybuffers.max_ops_per_alignment,
                    D2H,
                    mybuffers.streams[batchid]); CUERR;

                subjectindex++;
                querysum += params[batchid].n_queries;
            }
        }

        subjectindex = 0;
        querysum = 0;

        //wait for d2h transfer to complete and fetch results
        for(auto& b : batch){
            if(b.active){
                batchid = subjectindex;
                AlignResultCompact* results = mybuffers.h_results + querysum;
                AlignOp* ops = mybuffers.h_ops + querysum * mybuffers.max_ops_per_alignment;

                cudaStreamSynchronize(mybuffers.streams[batchid]); CUERR;

                tpa = std::chrono::system_clock::now();

                int count = 0;
                int localcount = 0;
                for(auto& alignment : b.fwdAlignments){

                    alignment = results[count];
                    b.fwdAlignOps[localcount].resize(alignment.nOps);
                    std::reverse_copy(ops + count * mybuffers.max_ops_per_alignment,
                              ops + count * mybuffers.max_ops_per_alignment + alignment.nOps,
                              b.fwdAlignOps[localcount].begin());
                    count++;
                    localcount++;
                }
                localcount = 0;
                for(auto& alignment : b.revcomplAlignments){

                    alignment = results[count];
                    b.revcomplAlignOps[localcount].resize(alignment.nOps);
                    std::copy(ops + count * mybuffers.max_ops_per_alignment,
                              ops + count * mybuffers.max_ops_per_alignment + alignment.nOps,
                              b.revcomplAlignOps[localcount].begin());
                    count++;
                    localcount++;
                }


                tpb = std::chrono::system_clock::now();
                mybuffers.postprocessingtime += tpb - tpa;

                subjectindex++;
                querysum += params[batchid].n_queries;
            }
        }

    }else{ // use cpu for alignment

#endif

        tpa = std::chrono::system_clock::now();

        for(auto& b : batch){
            if(b.active){
                const char* const subject = (const char*)b.fwdSequence->begin();
                const int subjectLength = b.fwdSequence->length();

                for(size_t i = 0; i < b.fwdSequences.size(); i++){
                    const char* query =  (const char*)b.fwdSequences[i]->begin();
                    const int queryLength = b.fwdSequences[i]->length();
                    auto al = cpu_semi_global_alignment(&mybuffers, alignmentOptions, subject, query, subjectLength, queryLength);
                    b.fwdAlignments[i] = al.arc;
                    b.fwdAlignOps[i] = std::move(al.operations);
                }

                for(size_t i = 0; i < b.revcomplSequences.size(); i++){
                    const char* query =  (const char*)b.revcomplSequences[i]->begin();
                    const int queryLength = b.revcomplSequences[i]->length();
                    auto al = cpu_semi_global_alignment(&mybuffers, alignmentOptions, subject, query, subjectLength, queryLength);
                    b.revcomplAlignments[i] = al.arc;
                    b.revcomplAlignOps[i] = std::move(al.operations);
                }
            }
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
