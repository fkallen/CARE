#include "../inc/shifted_hamming_distance.hpp"

namespace shd{

    bool AlignmentResult::operator==(const AlignmentResult& rhs) const{
        return score == rhs.score
            && subject_begin_incl == rhs.subject_begin_incl
            && query_begin_incl == rhs.query_begin_incl
            && overlap == rhs.overlap
            && shift == rhs.shift
            && nOps == rhs.nOps
            && isNormalized == rhs.isNormalized
            && isValid == rhs.isValid;
    }
    bool AlignmentResult::operator!=(const AlignmentResult& rhs) const{
        return !(*this == rhs);
    }
    int AlignmentResult::get_score() const{
        return score;
    }
    int AlignmentResult::get_subject_begin_incl() const{
        return subject_begin_incl;
    }
    int AlignmentResult::get_query_begin_incl() const{
        return query_begin_incl;
    }
    int AlignmentResult::get_overlap() const{
        return overlap;
    }
    int AlignmentResult::get_shift() const{
        return shift;
    }
    int AlignmentResult::get_nOps() const{
        return nOps;
    }
    bool AlignmentResult::get_isNormalized() const{
        return isNormalized;
    }
    bool AlignmentResult::get_isValid() const{
        return isValid;
    }
    int& AlignmentResult::get_score(){
        return score;
    }
    int& AlignmentResult::get_subject_begin_incl(){
        return subject_begin_incl;
    }
    int& AlignmentResult::get_query_begin_incl(){
        return query_begin_incl;
    }
    int& AlignmentResult::get_overlap(){
        return overlap;
    }
    int& AlignmentResult::get_shift(){
        return shift;
    }
    int& AlignmentResult::get_nOps(){
        return nOps;
    }
    bool& AlignmentResult::get_isNormalized(){
        return isNormalized;
    }
    bool& AlignmentResult::get_isValid(){
        return isValid;
    }


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
    #endif
    }

    void cuda_cleanup_SHDdata(SHDdata& data){
    #ifdef __NVCC__
        cudaSetDevice(data.deviceId); CUERR;

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

        for(int i = 0; i < SHDdata::n_streams; i++)
            cudaStreamDestroy(data.streams[i]); CUERR;
    #endif
    }

}//namespace shd
