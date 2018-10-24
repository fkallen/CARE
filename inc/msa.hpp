#ifndef CARE_CPU_MSA_HPP
#define CARE_CPU_MSA_HPP

#include "bestalignment.hpp"
#include "cpu_alignment.hpp"

#include <vector>

namespace care{
namespace cpu{

struct MultipleSequenceAlignment{

    struct ColumnProperties{
        int startindex;
        int endindex;
        int columnsToCheck;
        int subjectColumnsBegin_incl;
        int subjectColumnsEnd_excl;
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

    MultipleSequenceAlignment(bool canUseWeights, float m_coverage, int kmerlength, float estimatedCoverage)
        : MultipleSequenceAlignment(canUseWeights, m_coverage, kmerlength, estimatedCoverage, 0, 0){}

    MultipleSequenceAlignment(bool canUseWeights, float m_coverage, int kmerlength, float estimatedCoverage, int rows, int cols)
        : insertedCandidates(0), canUseWeights(canUseWeights), m_coverage(m_coverage), kmerlength(kmerlength), estimatedCoverage(estimatedCoverage){

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
        origCoverage.resize(cols);

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
        zero(origCoverage);

        insertedCandidates = 0;
    }

    void init(int subjectlength,
        const std::vector<int>& candidate_sequences_lengths,
        const std::vector<BestAlignment_t>& bestAlignmentFlags,
        const std::vector<SHDResult>& bestAlignments){

        //determine number of columns in pileup image
        columnProperties.startindex = 0;
        columnProperties.endindex = subjectlength;

        assert(candidate_sequences_lengths.size() == bestAlignmentFlag.size());
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
        assert(column + int(sequence.length()) < nColumns);

        std::copy(sequence.begin(), sequence.end(), multiple_sequence_alignment + row * nColumns + column);

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
};

}
}





#endif
