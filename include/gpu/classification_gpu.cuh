#ifndef CARE_CLASSIFICATION_GPU_CUH
#define CARE_CLASSIFICATION_GPU_CUH

#include <gpu/gpumsa.cuh>

namespace care{

namespace gpu{

    struct ExtractAnchorInputData{
        char origBase;
        char consensusBase;
        float estimatedCoverage;
        int msaPos;
        int subjectColumnsBegin_incl;
        int subjectColumnsEnd_excl;
        GpuMSAProperties msaProperties;
        GpuSingleMSA msa;
    };

    namespace detail{

        struct extract_anchor_transformed{
            static constexpr int numFeatures() noexcept{
                return 37;
            }

            template<class OutIter>
            HOSTDEVICEQUALIFIER
            void operator()(OutIter features, const ExtractAnchorInputData& input){
                const int msaPos = input.msaPos;
                const int pitch = input.msa.columnPitchInElements;
                const float countsACGT = input.msa.coverages[msaPos];
                const int* const countsA = &input.msa.counts[0 * pitch];
                const int* const countsC = &input.msa.counts[1 * pitch];
                const int* const countsG = &input.msa.counts[2 * pitch];
                const int* const countsT = &input.msa.counts[3 * pitch];

                const float* const weightsA = &input.msa.weights[0 * pitch];
                const float* const weightsC = &input.msa.weights[1 * pitch];
                const float* const weightsG = &input.msa.weights[2 * pitch];
                const float* const weightsT = &input.msa.weights[3 * pitch];

                *(features + 0) = float(input.origBase == 'A');
                *(features + 1) = float(input.origBase == 'C');
                *(features + 2) = float(input.origBase == 'G');
                *(features + 3) = float(input.origBase == 'T');
                *(features + 4) = float(input.consensusBase == 'A');
                *(features + 5) = float(input.consensusBase == 'C');
                *(features + 6) = float(input.consensusBase == 'G');
                *(features + 7) = float(input.consensusBase == 'T');
                *(features + 8) = input.origBase == 'A' ? countsA[msaPos] / countsACGT : 0;
                *(features + 9) = input.origBase == 'C' ? countsC[msaPos] / countsACGT : 0;
                *(features + 10) = input.origBase == 'G' ? countsG[msaPos] / countsACGT : 0;
                *(features + 11) = input.origBase == 'T' ? countsT[msaPos] / countsACGT : 0;
                *(features + 12) = input.origBase == 'A' ? weightsA[msaPos]:0;
                *(features + 13) = input.origBase == 'C' ? weightsC[msaPos]:0;
                *(features + 14) = input.origBase == 'G' ? weightsG[msaPos]:0;
                *(features + 15) = input.origBase == 'T' ? weightsT[msaPos]:0;
                *(features + 16) = input.consensusBase == 'A' ? countsA[msaPos] / countsACGT : 0;
                *(features + 17) = input.consensusBase == 'C' ? countsC[msaPos] / countsACGT : 0;
                *(features + 18) = input.consensusBase == 'G' ? countsG[msaPos] / countsACGT : 0;
                *(features + 19) = input.consensusBase == 'T' ? countsT[msaPos] / countsACGT : 0;
                *(features + 20) = input.consensusBase == 'A' ? weightsA[msaPos]:0;
                *(features + 21) = input.consensusBase == 'C' ? weightsC[msaPos]:0;
                *(features + 22) = input.consensusBase == 'G' ? weightsG[msaPos]:0;
                *(features + 23) = input.consensusBase == 'T' ? weightsT[msaPos]:0;
                *(features + 24) = weightsA[msaPos];
                *(features + 25) = weightsC[msaPos];
                *(features + 26) = weightsG[msaPos];
                *(features + 27) = weightsT[msaPos];
                *(features + 28) = countsA[msaPos] / countsACGT;
                *(features + 29) = countsC[msaPos] / countsACGT;
                *(features + 30) = countsG[msaPos] / countsACGT;
                *(features + 31) = countsT[msaPos] / countsACGT;
                *(features + 32) = input.msaProperties.avg_support;
                *(features + 33) = input.msaProperties.min_support;
                *(features + 34) = float(input.msaProperties.max_coverage) / input.estimatedCoverage;
                *(features + 35) = float(input.msaProperties.min_coverage) / input.estimatedCoverage;
                *(features + 36) = float(std::max(input.subjectColumnsBegin_incl - msaPos, msaPos - input.subjectColumnsEnd_excl)) / (input.subjectColumnsEnd_excl-input.subjectColumnsBegin_incl);
            }
        };

        struct ExtractCandidateInputData{
            char origBase;
            char consensusBase;
            float estimatedCoverage;
            int msaPos;
            int subjectColumnsBegin_incl;
            int subjectColumnsEnd_excl;
            int queryColumnsBegin_incl;
            int queryColumnsEnd_excl;
            GpuMSAProperties msaProperties;
            GpuSingleMSA msa;
        };

        struct extract_cands_transformed{
            static constexpr int numFeatures() noexcept{
                return 42;
            }

            template<class OutIter>
            HOSTDEVICEQUALIFIER
            void operator()(OutIter features, const ExtractCandidateInputData& input){
                const int a_begin = input.subjectColumnsBegin_incl;
                const int a_end = input.subjectColumnsEnd_excl;
                const int c_begin = input.queryColumnsBegin_incl;
                const int c_end = input.queryColumnsEnd_excl;

                const int msaPos = input.msaPos;
                const int pitch = input.msa.columnPitchInElements;
                const float countsACGT = input.msa.coverages[msaPos];
                const int* const countsA = &input.msa.counts[0 * pitch];
                const int* const countsC = &input.msa.counts[1 * pitch];
                const int* const countsG = &input.msa.counts[2 * pitch];
                const int* const countsT = &input.msa.counts[3 * pitch];

                const float* const weightsA = &input.msa.weights[0 * pitch];
                const float* const weightsC = &input.msa.weights[1 * pitch];
                const float* const weightsG = &input.msa.weights[2 * pitch];
                const float* const weightsT = &input.msa.weights[3 * pitch];

                *(features + 0) = float(origBase == 'A');
                *(features + 1) = float(origBase == 'C');
                *(features + 2) = float(origBase == 'G');
                *(features + 3) = float(origBase == 'T');
                *(features + 4) = float(consensusBase == 'A');
                *(features + 5) = float(consensusBase == 'C');
                *(features + 6) = float(consensusBase == 'G');
                *(features + 7) = float(consensusBase == 'T');
                *(features + 8) = origBase == 'A'? countsA[msaPos] / countsACGT : 0;
                *(features + 9) = origBase == 'C'? countsC[msaPos] / countsACGT : 0;
                *(features + 10) = origBase == 'G'? countsG[msaPos] / countsACGT : 0;
                *(features + 11) = origBase == 'T'? countsT[msaPos] / countsACGT : 0;
                *(features + 12) = origBase == 'A'? weightsA[msaPos]:0;
                *(features + 13) = origBase == 'C'? weightsC[msaPos]:0;
                *(features + 14) = origBase == 'G'? weightsG[msaPos]:0;
                *(features + 15) = origBase == 'T'? weightsT[msaPos]:0;
                *(features + 16) = consensusBase == 'A'? countsA[msaPos] / countsACGT : 0;
                *(features + 17) = consensusBase == 'C'? countsC[msaPos] / countsACGT : 0;
                *(features + 18) = consensusBase == 'G'? countsG[msaPos] / countsACGT : 0;
                *(features + 19) = consensusBase == 'T'? countsT[msaPos] / countsACGT : 0;
                *(features + 20) = consensusBase == 'A'? weightsA[msaPos]:0;
                *(features + 21) = consensusBase == 'C'? weightsC[msaPos]:0;
                *(features + 22) = consensusBase == 'G'? weightsG[msaPos]:0;
                *(features + 23) = consensusBase == 'T'? weightsT[msaPos]:0;
                *(features + 24) = weightsA[msaPos];
                *(features + 25) = weightsC[msaPos];
                *(features + 26) = weightsG[msaPos];
                *(features + 27) = weightsT[msaPos];
                *(features + 28) = countsA[msaPos] / countsACGT;
                *(features + 29) = countsC[msaPos] / countsACGT;
                *(features + 30) = countsG[msaPos] / countsACGT;
                *(features + 31) = countsT[msaPos] / countsACGT;
                *(features + 32) = msaProperties.avg_support;
                *(features + 33) = msaProperties.min_support;
                *(features + 34) = float(msaProperties.max_coverage)/estimatedCoverage;
                *(features + 35) = float(msaProperties.min_coverage)/estimatedCoverage;
                *(features + 36) = float(std::max(std::abs(c_begin-a_begin), std::abs(a_end-c_end)))/(c_end-c_begin); // absolute shift (compatible with differing read lengths)
                *(features + 37) = float(std::max(std::abs(c_begin-a_begin), std::abs(a_end-c_end)))/(a_end-a_begin);
                *(features + 38) = float(std::min(a_end, c_end)-std::max(a_begin, c_begin))/(a_end-a_begin); // relative overlap (ratio of a or c length in case of diff. read len)
                *(features + 39) = float(std::min(a_end, c_end)-std::max(a_begin, c_begin))/(c_end-c_begin);
                *(features + 40) = float(std::max(a_begin-msaPos, msaPos-a_end))/(a_end-a_begin);
                *(features + 41) = float(std::max(a_begin-msaPos, msaPos-a_end))/(c_end-c_begin);
            }
        };

    }

    using anchor_extractor = detail::extract_anchor_transformed;

}



}



#endif