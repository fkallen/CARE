#include "../inc/shifted_hamming_distance.hpp"
#include "../inc/shifted_hamming_distance_impl.hpp"

namespace care{
/*
    SHDdata implementation
*/
SHDdata::SHDdata(int deviceId_, int maxseqlength, int maxseqbytes, int batchsize)
      : deviceId(deviceId_), max_sequence_length(maxseqlength),
        max_sequence_bytes(maxseqbytes), batchsize(batchsize){
#ifdef __NVCC__
    cudaSetDevice(deviceId); CUERR;

    streams = std::make_unique<cudaStream_t[]>(batchsize);
    for(int i = 0; i < batchsize; i++)
        cudaStreamCreate(&(streams.get()[i])); CUERR;
#endif
};

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

void cuda_cleanup_SHDdata(SHDdata& data){
#ifdef __NVCC__
    cudaSetDevice(data.deviceId); CUERR;

    cudaFree(data.d_results); CUERR;
    cudaFree(data.d_subjectsdata); CUERR;
    cudaFree(data.d_queriesdata); CUERR;
    cudaFree(data.d_subjectlengths); CUERR;
    cudaFree(data.d_querylengths); CUERR;

    cudaFreeHost(data.h_results); CUERR;
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
AlignResultCompact cpu_shifted_hamming_distance(const GoodAlignmentProperties& prop,
    const char* subject, const char* query, int ns, int nq);

#ifdef __NVCC__

void call_shd_kernel(const shdparams& buffer, int maxQueryLength, cudaStream_t stream);
void call_shd_kernel_async(const shdparams& buffer, int maxQueryLength, cudaStream_t stream);

#endif

/*
    Batch alignment implementation
*/
void shifted_hamming_distance(SHDdata& mybuffers, std::vector<BatchElem>& batch,
                            const GoodAlignmentProperties& props, bool useGpu){

    std::chrono::time_point<std::chrono::system_clock> tpa = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> tpb = std::chrono::system_clock::now();

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

        tpb = std::chrono::system_clock::now();

        mybuffers.resizetime += tpb - tpa;

        int querysum = 0;
        int subjectindex = 0;
        int batchid = 0;
        std::vector<shdparams> params(batch.size());

        for(auto& b : batch){
            if(b.active){
                tpa = std::chrono::system_clock::now();
                batchid = subjectindex;

                params[batchid].props = props;
                params[batchid].max_sequence_bytes = mybuffers.max_sequence_bytes;
                params[batchid].sequencepitch = mybuffers.sequencepitch;
                params[batchid].subjectlength = b.fwdSequence->length();
                params[batchid].n_queries = b.fwdSequences.size() + b.revcomplSequences.size();
                params[batchid].querylengths = mybuffers.d_querylengths + querysum;
                params[batchid].subjectdata = mybuffers.d_subjectsdata + mybuffers.sequencepitch * subjectindex;
                params[batchid].queriesdata = mybuffers.d_queriesdata + mybuffers.sequencepitch * querysum;
                params[batchid].results = mybuffers.d_results + querysum;

                int* querylengths = mybuffers.h_querylengths + querysum;
                char* subjectdata = mybuffers.h_subjectsdata + mybuffers.sequencepitch * subjectindex;
                char* queriesdata = mybuffers.h_queriesdata + mybuffers.sequencepitch * querysum;

                assert(b.fwdSequence->length() <= mybuffers.max_sequence_length);
                assert(b.fwdSequence->getNumBytes() <= mybuffers.max_sequence_bytes);

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

                int maxQueryLength = 0;
                for(int k = 0; k < params[batchid].n_queries; k++)
                    if(maxQueryLength < querylengths[k])
                        maxQueryLength = querylengths[k];

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

                // start kernel
                call_shd_kernel_async(params[batchid], maxQueryLength, mybuffers.streams[batchid]);

                querysum += count;
                subjectindex++;
            }
        }

        subjectindex = 0;
        querysum = 0;
        //initialize transfer d2h
        for(auto& b : batch){
            if(b.active){
                batchid = subjectindex;
                AlignResultCompact* results = mybuffers.h_results + querysum;

                cudaMemcpyAsync(results,
                    params[batchid].results,
                    sizeof(AlignResultCompact) * params[batchid].n_queries,
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

                cudaStreamSynchronize(mybuffers.streams[batchid]); CUERR;

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

                subjectindex++;
                querysum += params[batchid].n_queries;
            }
        }

    }else{ // use cpu for alignment

#endif // __NVCC__

        tpa = std::chrono::system_clock::now();

        for(auto& b : batch){
            if(b.active){
                const char* const subject = (const char*)b.fwdSequence->begin();
                const int subjectLength = b.fwdSequence->length();

                for(size_t i = 0; i < b.fwdSequences.size(); i++){
                    const char* query =  (const char*)b.fwdSequences[i]->begin();
                    const int queryLength = b.fwdSequences[i]->length();
                    b.fwdAlignments[i] = cpu_shifted_hamming_distance(props, subject, query, subjectLength, queryLength);
                }

                for(size_t i = 0; i < b.revcomplSequences.size(); i++){
                    const char* query =  (const char*)b.revcomplSequences[i]->begin();
                    const int queryLength = b.revcomplSequences[i]->length();
                    b.revcomplAlignments[i] = cpu_shifted_hamming_distance(props, subject, query, subjectLength, queryLength);
                }
            }
        }

        tpb = std::chrono::system_clock::now();

        mybuffers.alignmenttime += tpb - tpa;
#ifdef __NVCC__
    }
#endif // __NVCC__
}

/*
    Alignment functions implementations
*/
namespace shddetail{
    #include "../inc/sequenceaccessor.hpp"
}

AlignResultCompact
cpu_shifted_hamming_distance(const GoodAlignmentProperties& prop,
                             const char* subject,
                             const char* query,
                             int ns,
                             int nq){

    auto accessor = [] (const char* data, int length, int index){
        return shddetail::encoded_accessor(data, length, index);
    };

    return cpu_shifted_hamming_distance_impl(prop, subject, query, ns, nq, accessor);
}


#ifdef __NVCC__

size_t shd_kernel_getSharedMemSize(const shdparams& buffer){

    size_t smem = 0;
    smem += sizeof(char) * 2 * buffer.max_sequence_bytes;

    return smem;
}

//wrapper functions to call kernels

void call_shd_kernel(const shdparams& buffer, int maxQueryLength, cudaStream_t stream){
    call_shd_kernel_async(buffer, maxQueryLength, stream);

    cudaStreamSynchronize(stream); CUERR;
}

void call_shd_kernel_async(const shdparams& buffer, int maxQueryLength, cudaStream_t stream){
    const int minoverlap = max(buffer.props.min_overlap, int(double(buffer.subjectlength) * buffer.props.min_overlap_ratio));
    const int maxShiftsToCheck = buffer.subjectlength+1 + maxQueryLength - 2*minoverlap; //FIXME
    dim3 block(std::min(256, 32 * SDIV(maxShiftsToCheck, 32)), 1, 1);
    dim3 grid(buffer.n_queries, 1, 1);

    size_t smem = shd_kernel_getSharedMemSize(buffer);

    auto accessor = [] __device__ (const char* data, int length, int index){
        return shddetail::encoded_accessor(data, length, index);
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

#endif

}
