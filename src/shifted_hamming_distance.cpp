#include "../inc/shifted_hamming_distance.hpp"

#include "../inc/cudareduce.cuh"

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

HOSTDEVICEQUALIFIER
char encoded_accessor(const char* data, int bases, int index){
    const int byte = index / 4;
    const int basepos = index % 4;
    return (data[byte] >> (3-basepos) * 2) & 0x03;
}

HOSTDEVICEQUALIFIER
char twobitToChar(char bits){
    if(bits == 0x00) return 'A';
    if(bits == 0x01) return 'C';
    if(bits == 0x02) return 'G';
    if(bits == 0x03) return 'T';
    return '_';
}

AlignResultCompact cpu_shifted_hamming_distance(const GoodAlignmentProperties& prop, const char* subject, const char* query, int ns, int nq){
    const int totalbases = ns + nq;
    const int minoverlap = std::max(prop.min_overlap, int(double(ns) * prop.min_overlap_ratio));
    int bestScore = totalbases; // score is number of mismatches
    int bestShift = -nq; // shift of query relative to subject. shift < 0 if query begins before subject

    for(int shift = -nq + minoverlap; shift < ns - minoverlap; shift++){
        const int overlapsize = std::min(nq, ns - shift) - std::max(-shift, 0);

        int score = 0;

        for(int j = std::max(-shift, 0); j < std::min(nq, ns - shift); j++){
            score += encoded_accessor(subject, ns, j + shift) != encoded_accessor(query, nq, j);
        }

        score += totalbases - overlapsize;

        if(score < bestScore){
            bestScore = score;
            bestShift = shift;
        }
    }

    AlignResultCompact result;
    result.isValid = (bestShift != -nq);

    const int queryoverlapbegin_incl = std::max(-bestShift, 0);
    const int queryoverlapend_excl = std::min(nq, ns - bestShift);
    const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
    const int opnr = bestScore - totalbases + overlapsize;

    result.score = bestScore;
    result.subject_begin_incl = std::max(0, bestShift);
    result.query_begin_incl = queryoverlapbegin_incl;
    result.overlap = overlapsize;
    result.shift = bestShift;
    result.nOps = opnr;
    result.isNormalized = false;

    return result;
}

#ifdef __NVCC__



template<int BLOCKSIZE>
__global__
void cuda_shifted_hamming_distance(const shdparams buffers){
    constexpr int WARPSIZE = 32;
    constexpr int NWARPS = (BLOCKSIZE + WARPSIZE - 1) / WARPSIZE;

    static_assert(sizeof(int2) == sizeof(unsigned long long), "sizeof(int2) != sizeof(unsigned long long)");
    static_assert(BLOCKSIZE % WARPSIZE == 0,
        "BLOCKSIZE must be multiple of WARPSIZE");

    extern __shared__ char smem[];

    /* set up shared memory */
    char* sr1 = (char*)(smem);
    char* sr2 = (char*)(sr1 + buffers.max_sequence_bytes);

    const int subjectbases = buffers.subjectlength;
    const char* subject = buffers.subjectdata;
    for(int threadid = threadIdx.x; threadid < buffers.max_sequence_bytes; threadid += BLOCKSIZE){
        sr1[threadid] = subject[threadid];
    }

    const int minoverlap = max(buffers.props.min_overlap, int(double(subjectbases) * buffers.props.min_overlap_ratio));

    for(int queryId = blockIdx.x; queryId < buffers.n_queries; queryId += gridDim.x){

        const char* query = buffers.queriesdata + queryId * buffers.sequencepitch;

        for(int threadid = threadIdx.x; threadid < buffers.max_sequence_bytes; threadid += BLOCKSIZE){
            sr2[threadid] = query[threadid];
        }

        __syncthreads();

        //begin SHD algorithm

        const int querybases = buffers.querylengths[queryId];

        const int totalbases = subjectbases + querybases;

        int bestScore = totalbases; // score is number of mismatches
        int bestShift = -querybases; // shift of query relative to subject. shift < 0 if query begins before subject

        for(int shift = -querybases + minoverlap + threadIdx.x; shift < subjectbases - minoverlap; shift += BLOCKSIZE){
            const int overlapsize = min(querybases, subjectbases - shift) - max(-shift, 0);
            int score = 0;

            for(int j = max(-shift, 0); j < min(querybases, subjectbases - shift); j++){
                score += encoded_accessor(sr1, subjectbases, j + shift) != encoded_accessor(sr2, querybases, j);
            }
            score += totalbases - overlapsize;

            if(score < bestScore){
                bestScore = score;
                bestShift = shift;
            }
        }



        // perform reduction to find smallest score in block. the corresponding shift is required, too
        // pack both score and shift into int2 and perform int2-reduction by only comparing the score

        static_assert(sizeof(int2) == sizeof(unsigned long long), "sizeof(int2) != sizeof(unsigned long long)");

        int2 myval = make_int2(bestScore, bestShift);

        __shared__ unsigned long long blockreducetmp[NWARPS];

        auto func = [](unsigned long long a, unsigned long long b){
            return (*((int2*)&a)).x < (*((int2*)&b)).x ? a : b;
        };

#if __CUDACC_VER_MAJOR__ < 9
        unsigned long long tilereduced = reduceTile<32>(*((unsigned long long*)&myval), func);
        int warp = threadIdx.x / WARPSIZE;
        int lane = threadIdx.x % WARPSIZE;
        if(lane == 0)
            blockreducetmp[warp] = tilereduced;

#else
        auto tile = tiled_partition<32>(this_thread_block());
        unsigned long long tilereduced = reduceTile(tile,
                                    *((unsigned long long*)&myval),
                                    func);
        int warp = threadIdx.x / WARPSIZE;
        if(tile.thread_rank() == 0)
            blockreducetmp[warp] = tilereduced;
#endif

        __syncthreads();

        //make result
        if(threadIdx.x == 0){
            //reduce warp results
            unsigned long long reduced = blockreducetmp[0];
            for(int i = 0; i < NWARPS; i++){
                reduced = func(reduced, blockreducetmp[i]);
            }

            bestScore = ((int2*)&reduced)->x;
            bestShift = ((int2*)&reduced)->y;

            AlignResultCompact result;

            result.isValid = (bestShift != -querybases);
            const int queryoverlapbegin_incl = max(-bestShift, 0);
            const int queryoverlapend_excl = min(querybases, subjectbases - bestShift);
            const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
            const int opnr = bestScore - totalbases + overlapsize;

            result.score = bestScore;
            result.subject_begin_incl = max(0, bestShift);
            result.query_begin_incl = queryoverlapbegin_incl;
            result.overlap = overlapsize;
            result.shift = bestShift;
            result.nOps = opnr;
            result.isNormalized = false;

            buffers.results[queryId] = result;
        }
    }
}

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

    switch(block.x){
    case 32: cuda_shifted_hamming_distance<32><<<grid, block, smem, stream>>>(buffer); CUERR; break;
    case 64: cuda_shifted_hamming_distance<64><<<grid, block, smem, stream>>>(buffer); CUERR; break;
    case 96: cuda_shifted_hamming_distance<96><<<grid, block, smem, stream>>>(buffer); CUERR; break;
    case 128: cuda_shifted_hamming_distance<128><<<grid, block, smem, stream>>>(buffer); CUERR; break;
    case 160: cuda_shifted_hamming_distance<160><<<grid, block, smem, stream>>>(buffer); CUERR; break;
    case 192: cuda_shifted_hamming_distance<192><<<grid, block, smem, stream>>>(buffer); CUERR; break;
    case 224: cuda_shifted_hamming_distance<224><<<grid, block, smem, stream>>>(buffer); CUERR; break;
    case 256: cuda_shifted_hamming_distance<256><<<grid, block, smem, stream>>>(buffer); CUERR; break;
    default: std::cout << "error call_shd_kernel_async\n"; break;
    }
}

#endif

}
