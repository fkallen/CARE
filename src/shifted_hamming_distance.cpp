#include "../inc/shifted_hamming_distance.hpp"

#include <cassert>

namespace shd{

    bool AlignmentResult::operator==(const AlignmentResult& rhs) const noexcept{
        return score == rhs.score
            && subject_begin_incl == rhs.subject_begin_incl
            && query_begin_incl == rhs.query_begin_incl
            && overlap == rhs.overlap
            && shift == rhs.shift
            && nOps == rhs.nOps
            && isNormalized == rhs.isNormalized
            && isValid == rhs.isValid;
    }
    bool AlignmentResult::operator!=(const AlignmentResult& rhs) const noexcept{
        return !(*this == rhs);
    }
    int AlignmentResult::get_score() const noexcept{
        return score;
    }
    int AlignmentResult::get_subject_begin_incl() const noexcept{
        return subject_begin_incl;
    }
    int AlignmentResult::get_query_begin_incl() const noexcept{
        return query_begin_incl;
    }
    int AlignmentResult::get_overlap() const noexcept{
        return overlap;
    }
    int AlignmentResult::get_shift() const noexcept{
        return shift;
    }
    int AlignmentResult::get_nOps() const noexcept{
        return nOps;
    }
    bool AlignmentResult::get_isNormalized() const noexcept{
        return isNormalized;
    }
    bool AlignmentResult::get_isValid() const noexcept{
        return isValid;
    }
    int& AlignmentResult::get_score() noexcept{
        return score;
    }
    int& AlignmentResult::get_subject_begin_incl() noexcept{
        return subject_begin_incl;
    }
    int& AlignmentResult::get_query_begin_incl() noexcept{
        return query_begin_incl;
    }
    int& AlignmentResult::get_overlap() noexcept{
        return overlap;
    }
    int& AlignmentResult::get_shift() noexcept{
        return shift;
    }
    int& AlignmentResult::get_nOps() noexcept{
        return nOps;
    }
    bool& AlignmentResult::get_isNormalized() noexcept{
        return isNormalized;
    }
    bool& AlignmentResult::get_isValid() noexcept{
        return isValid;
    }

#if 0
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
            cudaMalloc(&d_results, sizeof(AlignmentResult) * max_n_subjects * max_n_queries); CUERR;

            cudaFreeHost(h_results); CUERR;
            cudaMallocHost(&h_results, sizeof(AlignmentResult) * max_n_subjects * max_n_queries); CUERR;
        }
    #endif
        n_subjects = n_sub;
        n_queries = n_quer;
    }

#else


    void SHDdata::resize(int n_sub, int n_quer){
        resize(n_sub, n_quer, n_quer);
    }

    void SHDdata::resize(int n_sub, int n_quer, int n_res){

    #ifdef __NVCC__
        constexpr double factor = 1.2; //overprovisioning
        
        cudaSetDevice(deviceId); CUERR;

        n_subjects = n_sub;
        n_queries = n_quer;
        n_results = n_res;

        const std::size_t memSubjects = n_sub * sequencepitch;
        const std::size_t memSubjectLengths = SDIV(n_sub * sizeof(int), sequencepitch) * sequencepitch;
        const std::size_t memNqueriesPrefixSum = SDIV((n_sub+1) * sizeof(int), sequencepitch) * sequencepitch;
        const std::size_t memQueries = n_quer * sequencepitch;
        const std::size_t memQueryLengths = SDIV(n_quer * sizeof(int), sequencepitch) * sequencepitch;
        const std::size_t memResults = SDIV(sizeof(AlignmentResult) * n_results, sequencepitch) * sequencepitch;
        const std::size_t memBestAlignmentFlags = SDIV(sizeof(BestAlignment_t) * n_results, sequencepitch) * sequencepitch;

        const std::size_t requiredMem = memSubjects + memSubjectLengths + memNqueriesPrefixSum
                                        + memQueries + memQueryLengths + memResults
                                        + memBestAlignmentFlags;

        if(requiredMem > allocatedMem){
            cudaFree(deviceptr); CUERR;
            cudaFreeHost(hostptr); CUERR;
            cudaMalloc(&deviceptr, std::size_t(requiredMem * factor)); CUERR;
            cudaMallocHost(&hostptr, std::size_t(requiredMem * factor)); CUERR;

            allocatedMem = requiredMem * factor;
        }

        transfersizeH2D = memSubjects; // d_subjectsdata
        transfersizeH2D += memSubjectLengths; // d_subjectlengths
        transfersizeH2D += memNqueriesPrefixSum; // d_NqueriesPrefixSum
        transfersizeH2D += memQueries; // d_queriesdata
        transfersizeH2D += memQueryLengths; // d_querylengths

        transfersizeD2H = memResults; //d_results
        transfersizeD2H += sizeof(BestAlignment_t) * n_results; // d_bestAlignmentFlags

        d_subjectsdata = (char*)deviceptr;
        d_subjectlengths = (int*)(((char*)d_subjectsdata) + memSubjects);
        d_NqueriesPrefixSum = (int*)(((char*)d_subjectlengths) + memSubjectLengths);
        d_queriesdata = (char*)(((char*)d_NqueriesPrefixSum) + memNqueriesPrefixSum);
        d_querylengths = (int*)(((char*)d_queriesdata) + memQueries);
        d_results = (AlignmentResult*)(((char*)d_querylengths) + memQueryLengths);
        d_bestAlignmentFlags = (BestAlignment_t*)(((char*)d_results) + memResults);

        h_subjectsdata = (char*)hostptr;
        h_subjectlengths = (int*)(((char*)h_subjectsdata) + memSubjects);
        h_NqueriesPrefixSum = (int*)(((char*)h_subjectlengths) + memSubjectLengths);
        h_queriesdata = (char*)(((char*)h_NqueriesPrefixSum) + memNqueriesPrefixSum);
        h_querylengths = (int*)(((char*)h_queriesdata) + memQueries);
        h_results = (AlignmentResult*)(((char*)h_querylengths) + memQueryLengths);
        h_bestAlignmentFlags = (BestAlignment_t*)(((char*)h_results) + memResults);

        #endif
    }

#endif

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

        void* ptr;
        std::size_t pitch;
        cudaMallocPitch(&ptr, &pitch, max_sequence_bytes, 1); CUERR;
        cudaFree(ptr); CUERR;
        data.sequencepitch = pitch;
    #endif
    }

    void cuda_cleanup_SHDdata(SHDdata& data){
    #ifdef __NVCC__
        cudaSetDevice(data.deviceId); CUERR;
#if 0
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
#else
        cudaFree(data.deviceptr); CUERR;
        cudaFreeHost(data.hostptr); CUERR;
#endif
        for(int i = 0; i < SHDdata::n_streams; i++)
            cudaStreamDestroy(data.streams[i]); CUERR;
    #endif
    }

}//namespace shd
