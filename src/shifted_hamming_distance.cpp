#include "../inc/shifted_hamming_distance.hpp"
#include "../inc/msa_kernels.hpp"

#include <cassert>

namespace shd{

    bool AlignmentResult::operator==(const AlignmentResult& rhs) const noexcept{
        return score == rhs.score
            && overlap == rhs.overlap
            && shift == rhs.shift
            && nOps == rhs.nOps
            && isValid == rhs.isValid;
    }
    bool AlignmentResult::operator!=(const AlignmentResult& rhs) const noexcept{
        return !(*this == rhs);
    }
    int AlignmentResult::get_score() const noexcept{
        return score;
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
    bool AlignmentResult::get_isValid() const noexcept{
        return isValid;
    }
    int& AlignmentResult::get_score() noexcept{
        return score;
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
    bool& AlignmentResult::get_isValid() noexcept{
        return isValid;
    }

    void SHDdata::resize(int n_sub, int n_quer){
        resize(n_sub, n_quer, n_quer);
    }

    void SHDdata::resize(int n_sub, int n_quer, int n_res, double factor){

    #ifdef __NVCC__
        static_assert(sizeof(float) == sizeof(int), "sizeof(float) != sizeof(int)");

        cudaSetDevice(deviceId); CUERR;

        n_subjects = n_sub;
        n_queries = n_quer;
        n_results = n_res;

        msa_row_pitch = SDIV((max_sequence_length * 3 - 2), sequencepitch) * sequencepitch;
        msa_weights_row_pitch = SDIV(sizeof(float) * (max_sequence_length * 3 - 2), sequencepitch) * sequencepitch;

        memSubjects = sizeof(char) * n_sub * sequencepitch;
        memSubjectLengths = SDIV(n_sub * sizeof(int), sequencepitch) * sequencepitch;
        memNqueriesPrefixSum = SDIV((n_sub+1) * sizeof(int), sequencepitch) * sequencepitch;
        memQueries = sizeof(char) * n_quer * sequencepitch;
        memQueryLengths = SDIV(n_quer * sizeof(int), sequencepitch) * sequencepitch;
        memResults = SDIV(sizeof(AlignmentResult) * n_results, sequencepitch) * sequencepitch;
        memBestAlignmentFlags = SDIV(sizeof(BestAlignment_t) * n_results, sequencepitch) * sequencepitch;
        memUnpackedSubjects = SDIV(sizeof(char) * n_sub * max_sequence_length, sequencepitch) * sequencepitch;
        memUnpackedQueries = SDIV(sizeof(char) * n_quer * max_sequence_length, sequencepitch) * sequencepitch;
        memMultipleSequenceAlignment = msa_row_pitch * (n_quer + n_sub);
        memMultipleSequenceAlignmentWeights = msa_weights_row_pitch * (n_quer + n_sub);
        memConsensus = msa_row_pitch * n_sub;
        memSupport = msa_weights_row_pitch * n_sub;
        memCoverage = msa_weights_row_pitch * n_sub;
        memOrigWeights = msa_weights_row_pitch * n_sub;
        memOrigCoverage = msa_weights_row_pitch * n_sub;
        memQualityScores = msa_row_pitch * (n_quer + n_sub);
        memMSAColumnProperties = SDIV(sizeof(care::msa::MSAColumnProperties) * n_sub, sequencepitch) * sequencepitch;

        const std::size_t requiredMem = memSubjects + memSubjectLengths + memNqueriesPrefixSum
                                        + memQueries + memQueryLengths + memResults
                                        + memBestAlignmentFlags
                                        + memUnpackedSubjects
                                        + memUnpackedQueries;
                                        /*+ memMultipleSequenceAlignment
                                        + memConsensus
                                        + memSupport
                                        + memCoverage
                                        + memOrigWeights
                                        + memOrigCoverage
                                        + memQualityScores
                                        + memMSAColumnProperties;*/

        if(requiredMem > allocatedMem){
            cudaFree(deviceptr); CUERR;
            cudaFreeHost(hostptr); CUERR;
            cudaMalloc(&deviceptr, std::size_t(requiredMem * factor)); CUERR;
            cudaMallocHost(&hostptr, std::size_t(requiredMem * factor)); CUERR;

            allocatedMem = requiredMem * factor;
        }

        d_subjectsdata = (char*)deviceptr;
        d_queriesdata = (char*)(((char*)d_subjectsdata) + memSubjects);
        d_NqueriesPrefixSum = (int*)(((char*)d_queriesdata) + memQueries);
        d_subjectlengths = (int*)(((char*)d_NqueriesPrefixSum) + memNqueriesPrefixSum);
        d_querylengths = (int*)(((char*)d_subjectlengths) + memSubjectLengths);

        d_results = (AlignmentResult*)(((char*)d_querylengths) + memQueryLengths);
        d_bestAlignmentFlags = (BestAlignment_t*)(((char*)d_results) + memResults);
        d_unpacked_subjects = (char*)(((char*)d_bestAlignmentFlags) + memBestAlignmentFlags);
        d_unpacked_queries = (char*)(((char*)d_unpacked_subjects) + memUnpackedSubjects);

        d_multiple_sequence_alignment = (char*)(((char*)d_unpacked_queries) + memUnpackedQueries);
        d_multiple_sequence_alignment_weights = (float*)(((char*)d_multiple_sequence_alignment) + memMultipleSequenceAlignment);
        d_consensus = (char*)(((char*)d_multiple_sequence_alignment_weights) + memMultipleSequenceAlignmentWeights);
        d_support = (float*)(((char*)d_consensus) + memConsensus);
        d_coverage = (int*)(((char*)d_support) + memSupport);
        d_origWeights = (float*)(((char*)d_coverage) + memCoverage);
        d_origCoverages = (int*)(((char*)d_origWeights) + memOrigWeights);
        d_qualityscores = (char*)(((char*)d_origCoverages) + memOrigCoverage);
        d_msa_column_properties = (care::msa::MSAColumnProperties*)(((char*)d_qualityscores) + memQualityScores);


        h_subjectsdata = (char*)hostptr;
        h_queriesdata = (char*)(((char*)h_subjectsdata) + memSubjects);
        h_NqueriesPrefixSum = (int*)(((char*)h_queriesdata) + memQueries);
        h_subjectlengths = (int*)(((char*)h_NqueriesPrefixSum) + memNqueriesPrefixSum);
        h_querylengths = (int*)(((char*)h_subjectlengths) + memSubjectLengths);

        h_results = (AlignmentResult*)(((char*)h_querylengths) + memQueryLengths);
        h_bestAlignmentFlags = (BestAlignment_t*)(((char*)h_results) + memResults);
        h_unpacked_subjects = (char*)(((char*)h_bestAlignmentFlags) + memBestAlignmentFlags);
        h_unpacked_queries = (char*)(((char*)h_unpacked_subjects) + memUnpackedSubjects);

        h_multiple_sequence_alignment_weights = (float*)(((char*)h_multiple_sequence_alignment) + memMultipleSequenceAlignment);
        h_consensus = (char*)(((char*)h_multiple_sequence_alignment_weights) + memMultipleSequenceAlignmentWeights);
        h_consensus = (char*)(((char*)h_multiple_sequence_alignment) + memMultipleSequenceAlignment);
        h_support = (float*)(((char*)h_consensus) + memConsensus);
        h_coverage = (int*)(((char*)h_support) + memSupport);
        h_origWeights = (float*)(((char*)h_coverage) + memCoverage);
        h_origCoverages = (int*)(((char*)h_origWeights) + memOrigWeights);
        h_qualityscores = (char*)(((char*)h_origCoverages) + memOrigCoverage);
        h_msa_column_properties = (care::msa::MSAColumnProperties*)(((char*)d_qualityscores) + memQualityScores);

        #endif
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
