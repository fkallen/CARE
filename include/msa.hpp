#ifndef CARE_CPU_MSA_HPP
#define CARE_CPU_MSA_HPP

#include "bestalignment.hpp"
#include "cpu_alignment.hpp"

#include <string>
#include <cassert>
#include <vector>
#include <numeric>
#include <functional>
#include <algorithm>
#include <array>
#include <iostream>

namespace care{
namespace cpu{

struct MultipleSequenceAlignment{

public:
    struct ColumnProperties{
        int startindex;
        int endindex;
        int columnsToCheck;
        int subjectColumnsBegin_incl;
        int subjectColumnsEnd_excl;
    };

    struct MSAProperties{
        float avg_support;
        float min_support;
        int max_coverage;
        int min_coverage;
        bool isHQ;
        bool failedAvgSupport;
        bool failedMinSupport;
        bool failedMinCoverage;
    };

    struct CorrectedCandidate{
        int index;
        std::string sequence;
        CorrectedCandidate() noexcept{}
        CorrectedCandidate(int index, const std::string& sequence) noexcept
            : index(index), sequence(sequence){}
    };

    struct CorrectionResult{
        std::string correctedSequence;
        bool isCorrected;
        MSAProperties msaProperties;
    };

    std::vector<char> multiple_sequence_alignment;
    std::vector<float> multiple_sequence_alignment_weights;
    std::vector<char> consensus;
    std::vector<float> support;
    std::vector<int> coverage;
    std::vector<float> origWeights;
    std::vector<int> origCoverages;

    ColumnProperties columnProperties;

    int nRows;
    int nColumns;
    int insertedCandidates;
    bool canUseWeights;

    float m_coverage;
    int kmerlength;
    float estimatedCoverage;
    float estimatedErrorrate;

    MultipleSequenceAlignment(bool canUseWeights, float m_coverage, int kmerlength, float estimatedCoverage, float estimatedErrorrate)
        : MultipleSequenceAlignment(canUseWeights, m_coverage, kmerlength, estimatedCoverage, estimatedErrorrate, 0, 0){}

    MultipleSequenceAlignment(bool canUseWeights, float m_coverage, int kmerlength, float estimatedCoverage, float estimatedErrorrate, int rows, int cols)
        : insertedCandidates(0), canUseWeights(canUseWeights), m_coverage(m_coverage),
            kmerlength(kmerlength), estimatedCoverage(estimatedCoverage), estimatedErrorrate(estimatedErrorrate){

        resize(rows, cols);
    }

    void resize(int rows, int cols){
        const std::size_t twodimsize = std::size_t(rows) * std::size_t(cols);

        multiple_sequence_alignment.resize(twodimsize);
        multiple_sequence_alignment_weights.resize(twodimsize);
        consensus.resize(cols);
        support.resize(cols);
        coverage.resize(cols);
        origWeights.resize(cols);
        origCoverages.resize(cols);

        nRows = rows;
        nColumns = cols;
        insertedCandidates = 0;
    }

    void fillzero(){
        auto zero = [](auto& vec){
            std::fill(vec.begin(), vec.end(), 0);
        };

        zero(multiple_sequence_alignment);
        zero(multiple_sequence_alignment_weights);
        zero(consensus);
        zero(support);
        zero(coverage);
        zero(origWeights);
        zero(origCoverages);

        insertedCandidates = 0;
    }

    void init(int subjectlength,
        const std::vector<int>& candidate_sequences_lengths,
        const std::vector<BestAlignment_t>& bestAlignmentFlags,
        const std::vector<SHDResult>& bestAlignments){

        //determine number of columns in pileup image
        columnProperties.startindex = 0;
        columnProperties.endindex = subjectlength;

        assert(candidate_sequences_lengths.size() == bestAlignmentFlags.size());
        assert(candidate_sequences_lengths.size() == bestAlignments.size());

        for(std::size_t i = 0; i < bestAlignments.size(); ++i){
            const int shift = bestAlignments[i].shift;
            const int candidateEndsAt = candidate_sequences_lengths[i] + shift;
            columnProperties.startindex = std::min(shift, columnProperties.startindex);
            columnProperties.endindex = std::max(candidateEndsAt, columnProperties.endindex);
        }

        columnProperties.columnsToCheck = columnProperties.endindex - columnProperties.startindex;
        columnProperties.subjectColumnsBegin_incl = std::max(-columnProperties.startindex,0);
        columnProperties.subjectColumnsEnd_excl = columnProperties.subjectColumnsBegin_incl + subjectlength;

        resize(1 + bestAlignments.size(), columnProperties.columnsToCheck);

        fillzero();
    }

    template<class GetQualityWeight>
    void insert(int row, int column, const std::string& sequence, GetQualityWeight getQualityWeight){
        assert(row < nRows);
        assert(column < nColumns);
        assert(column + int(sequence.length()) <= nColumns);

        std::copy(sequence.begin(), sequence.end(), multiple_sequence_alignment.begin() + row * nColumns + column);

        if(canUseWeights){
            for(std::size_t i = 0; i < sequence.length(); ++i){
                multiple_sequence_alignment_weights[row * nColumns + column + i] = getQualityWeight(i);
            }
        }
    }

    template<class GetQualityWeight>
    void insertSubject(const std::string& subject, GetQualityWeight getQualityWeight){
        insert(0, columnProperties.subjectColumnsBegin_incl, subject, getQualityWeight);
    }

    template<class GetQualityWeight>
    void insertCandidate(const std::string& candidate, int alignment_shift, GetQualityWeight getQualityWeight){
        assert(insertedCandidates < nRows-1);

        insert(1 + insertedCandidates, columnProperties.subjectColumnsBegin_incl + alignment_shift, candidate, getQualityWeight);

        ++insertedCandidates;
    }

    void find_consensus(){
        for(int column = 0; column < nColumns; ++column){
            std::array<int, 5> counts{{0,0,0,0,0}}; //A,C,G,T, other
            std::array<float, 5> weights{{0,0,0,0,0}}; //A,C,G,T, other
            for(int row = 0; row < nRows; ++row){
                const char base = multiple_sequence_alignment[row * nColumns + column];
                switch(base){
                    case 'A': counts[0]++; break;
                    case 'C': counts[1]++; break;
                    case 'G': counts[2]++; break;
                    case 'T': counts[3]++; break;
                    default: counts[4]++; break;
                }

                const float weight = canUseWeights ? multiple_sequence_alignment_weights[row * nColumns + column] : 1.0f;
                switch(base){
                    case 'A': weights[0] += weight; break;
                    case 'C': weights[1] += weight; break;
                    case 'G': weights[2] += weight; break;
                    case 'T': weights[3] += weight; break;
                    default: weights[4] += weight; break;
                }
            }

            coverage[column] = counts[0] + counts[1] + counts[2] + counts[3];

            char cons = 'A';
            float consWeight = weights[0];
            if(weights[1] > consWeight){
                cons = 'C';
                consWeight = weights[1];
            }
            if(weights[2] > consWeight){
                cons = 'G';
                consWeight = weights[2];
            }
            if(weights[3] > consWeight){
                cons = 'T';
                consWeight = weights[3];
            }
            consensus[column] = cons;

            const float columnWeight = weights[0] + weights[1] + weights[2] + weights[3];
            support[column] = consWeight / columnWeight;

            if(columnProperties.subjectColumnsBegin_incl <= column && column < columnProperties.subjectColumnsEnd_excl){
                const char subjectBase = multiple_sequence_alignment[0 * nColumns + column];
                switch(subjectBase){
                    case 'A':origWeights[column] = weights[0]; origCoverages[column] = counts[0]; break;
                    case 'C':origWeights[column] = weights[1]; origCoverages[column] = counts[1]; break;
                    case 'G':origWeights[column] = weights[2]; origCoverages[column] = counts[2]; break;
                    case 'T':origWeights[column] = weights[3]; origCoverages[column] = counts[3]; break;
                    default: assert(false && "This should not happen in find_consensus"); break;
                }
            }
        }
    }

    MSAProperties getMSAProperties() const{
        const float avg_support_threshold = 1.0f-1.0f*estimatedErrorrate;
        const float min_support_threshold = 1.0f-3.0f*estimatedErrorrate;
        const float min_coverage_threshold = m_coverage / 6.0f * estimatedCoverage;

        const int subjectlength = columnProperties.subjectColumnsEnd_excl - columnProperties.subjectColumnsBegin_incl;

        MSAProperties msaProperties;

        msaProperties.min_support = *std::min_element(support.begin() + columnProperties.subjectColumnsBegin_incl,
                                                    support.begin() + columnProperties.subjectColumnsEnd_excl);

        const float supportsum = std::accumulate(support.begin() + columnProperties.subjectColumnsBegin_incl,
                                                support.begin() + columnProperties.subjectColumnsEnd_excl,
                                                0.0f,
                                                std::plus<float>{});
        msaProperties.avg_support = supportsum / subjectlength;

        auto minmax = std::minmax_element(coverage.begin() + columnProperties.subjectColumnsBegin_incl,
                                                    coverage.begin() + columnProperties.subjectColumnsEnd_excl);

        msaProperties.min_coverage = *minmax.second;
        msaProperties.max_coverage = *minmax.first;

        auto isGoodAvgSupport = [=](float avgsupport){
            return avgsupport >= avg_support_threshold;
        };
        auto isGoodMinSupport = [=](float minsupport){
            return minsupport >= min_support_threshold;
        };
        auto isGoodMinCoverage = [=](float mincoverage){
            return mincoverage >= min_coverage_threshold;
        };

        msaProperties.isHQ = isGoodAvgSupport(msaProperties.avg_support)
                            && isGoodMinSupport(msaProperties.min_support)
                            && isGoodMinCoverage(msaProperties.min_coverage);

        msaProperties.failedAvgSupport = !isGoodAvgSupport(msaProperties.avg_support);
        msaProperties.failedMinSupport = !isGoodMinSupport(msaProperties.min_support);
        msaProperties.failedMinCoverage = !isGoodMinCoverage(msaProperties.min_coverage);

        return msaProperties;
    }

    CorrectionResult getCorrectedSubject(){

        //const float avg_support_threshold = 1.0f-1.0f*estimatedErrorrate;
        //const float min_support_threshold = 1.0f-3.0f*estimatedErrorrate;
        const float min_coverage_threshold = m_coverage / 6.0f * estimatedCoverage;

        const MSAProperties msaProperties = getMSAProperties();

        const int subjectlength = columnProperties.subjectColumnsEnd_excl - columnProperties.subjectColumnsBegin_incl;

        CorrectionResult result;
        result.isCorrected = false;
        result.correctedSequence.resize(subjectlength);
        result.msaProperties = msaProperties;

        if(msaProperties.isHQ){
            //corrected sequence = consensus;

            std::copy(consensus.begin() + columnProperties.subjectColumnsBegin_incl,
                      consensus.begin() + columnProperties.subjectColumnsEnd_excl,
                      result.correctedSequence.begin());
            result.isCorrected = true;
        }else{
            //set corrected sequence to original subject. then search for positions with good properties. correct these positions
            std::copy(multiple_sequence_alignment.begin() + columnProperties.subjectColumnsBegin_incl,
                      multiple_sequence_alignment.begin() + columnProperties.subjectColumnsEnd_excl,
                      result.correctedSequence.begin());

            bool foundAColumn = false;
            for(int i = 0; i < subjectlength; i++){
                const int globalIndex = columnProperties.subjectColumnsBegin_incl + i;

                if(support[globalIndex] > 0.5f && origCoverages[globalIndex] < min_coverage_threshold){
                    float avgsupportkregion = 0;
                    int c = 0;
                    bool kregioncoverageisgood = true;

                    for(int j = i - kmerlength/2; j <= i + kmerlength/2 && kregioncoverageisgood; j++){
                        if(j != i && j >= 0 && j < subjectlength){
                            avgsupportkregion += support[columnProperties.subjectColumnsBegin_incl + j];
                            kregioncoverageisgood &= (coverage[columnProperties.subjectColumnsBegin_incl + j] >= min_coverage_threshold);
                            c++;
                        }
                    }

                    avgsupportkregion /= c;
                    if(kregioncoverageisgood && avgsupportkregion >= 1.0f-estimatedErrorrate){
                        result.correctedSequence[i] = consensus[globalIndex];
                        foundAColumn = true;
                    }
                }
            }

            result.isCorrected = foundAColumn;
        }

        return result;
    }


    std::vector<CorrectedCandidate> getCorrectedCandidates(const std::vector<int>& candidate_sequences_lengths,
                                                        const std::vector<SHDResult>& bestAlignments,
                                                        int new_columns_to_correct) const{

        //const float avg_support_threshold = 1.0f-1.0f*estimatedErrorrate;
        const float min_support_threshold = 1.0f-3.0f*estimatedErrorrate;
        const float min_coverage_threshold = m_coverage / 6.0f * estimatedCoverage;

        std::vector<CorrectedCandidate> result;
        result.reserve(nRows - 1);

        for(int row = 1; row < nRows; ++row){
            const int candidate_index = row - 1;
            const int queryColumnsBegin_incl = bestAlignments[candidate_index].shift - columnProperties.startindex;
            const int queryLength = candidate_sequences_lengths[candidate_index];
            const int queryColumnsEnd_excl = queryColumnsBegin_incl + queryLength;

            //check range condition and length condition
            if(columnProperties.subjectColumnsBegin_incl - new_columns_to_correct <= queryColumnsBegin_incl
                && queryColumnsBegin_incl <= columnProperties.subjectColumnsBegin_incl + new_columns_to_correct
                && queryColumnsEnd_excl <= columnProperties.subjectColumnsEnd_excl + new_columns_to_correct){

                float newColMinSupport = 1.0f;
                int newColMinCov = std::numeric_limits<int>::max();

                //check new columns left of subject
                for(int columnindex = columnProperties.subjectColumnsBegin_incl - new_columns_to_correct;
                    columnindex < columnProperties.subjectColumnsBegin_incl;
                    columnindex++){

                    assert(columnindex < columnProperties.columnsToCheck);
                    if(queryColumnsBegin_incl <= columnindex){
                        newColMinSupport = support[columnindex] < newColMinSupport ? support[columnindex] : newColMinSupport;
                        newColMinCov = coverage[columnindex] < newColMinCov ? coverage[columnindex] : newColMinCov;
                    }
                }
                //check new columns right of subject
                for(int columnindex = columnProperties.subjectColumnsEnd_excl;
                    columnindex < columnProperties.subjectColumnsEnd_excl + new_columns_to_correct
                    && columnindex < columnProperties.columnsToCheck;
                    columnindex++){

                    newColMinSupport = support[columnindex] < newColMinSupport ? support[columnindex] : newColMinSupport;
                    newColMinCov = coverage[columnindex] < newColMinCov ? coverage[columnindex] : newColMinCov;
                }

                if(newColMinSupport >= min_support_threshold
                    && newColMinCov >= min_coverage_threshold){

                    std::string correctedString(&consensus[queryColumnsBegin_incl], &consensus[queryColumnsEnd_excl]);

                    result.emplace_back(candidate_index, std::move(correctedString));
                }
            }
        }

        return result;
    }
};

}
}





#endif
