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

    struct MinimizationResult{
        bool performedMinimization = false;
        std::vector<int> remaining_candidates;
        int num_discarded_candidates = 0;

        int column = 0;
        char significantBase = 'F';
        char consensusBase = 'F';
        char originalBase = 'F';
        int significantCount = 0;
        int consensuscount = 0;
    };

    std::vector<char> multiple_sequence_alignment;
    std::vector<float> multiple_sequence_alignment_weights;
    std::vector<char> consensus;
    std::vector<float> support;
    std::vector<int> coverage;
    std::vector<float> origWeights;
    std::vector<int> origCoverages;

    std::vector<int> countsA;
    std::vector<int> countsC;
    std::vector<int> countsG;
    std::vector<int> countsT;

    std::vector<float> weightsA;
    std::vector<float> weightsC;
    std::vector<float> weightsG;
    std::vector<float> weightsT;

    std::vector<std::array<int, 4>> countsPerBase;
    std::vector<std::array<float, 4>> weightsPerBase;

    std::vector<int> sequenceLengths;
    std::vector<int> shifts;

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
        countsA.resize(cols);
        countsC.resize(cols);
        countsG.resize(cols);
        countsT.resize(cols);
        weightsA.resize(cols);
        weightsC.resize(cols);
        weightsG.resize(cols);
        weightsT.resize(cols);

        countsPerBase.resize(cols);
        weightsPerBase.resize(cols);

        sequenceLengths.resize(rows);
        shifts.resize(rows);

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
        zero(countsA);
        zero(countsC);
        zero(countsG);
        zero(countsT);
        zero(weightsA);
        zero(weightsC);
        zero(weightsG);
        zero(weightsT);
        zero(sequenceLengths);
        zero(shifts);

        std::fill(countsPerBase.begin(), countsPerBase.end(), std::array<int, 4>{});
        std::fill(weightsPerBase.begin(), weightsPerBase.end(), std::array<float, 4>{});

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
        sequenceLengths[row] = int(sequence.length());
        shifts[row] = column - columnProperties.subjectColumnsBegin_incl;

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
        find_consensus1();
    }

    void find_consensus1(){
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
#if 0
            countsA[column] = counts[0];
            countsC[column] = counts[1];
            countsG[column] = counts[2];
            countsT[column] = counts[3];

            weightsA[column] = weights[0];
            weightsC[column] = weights[1];
            weightsG[column] = weights[2];
            weightsT[column] = weights[3];
#else
            std::copy(counts.begin(), counts.begin() + 4, countsPerBase[column].begin());
            std::copy(weights.begin(), weights.begin() + 4, weightsPerBase[column].begin());
#endif
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

    void find_consensus_2(){
        constexpr int lanesPerBatch = 4;
        std::array<std::array<int, 5>, lanesPerBatch> batchcounts;
        std::array<std::array<float, 5>, lanesPerBatch> batchweights;

        int batches = SDIV(nColumns, lanesPerBatch);

        for(int batch = 0; batch < batches; batch++){
            for(auto& arr : batchcounts){
                std::fill(arr.begin(), arr.end(), 0);
            }
            for(auto& arr : batchweights){
                std::fill(arr.begin(), arr.end(), 0);
            }

            for(int row = 0; row < nRows; row++){
                for(int lane = 0; lane < lanesPerBatch; lane++){
                    const int column = batch * lanesPerBatch + lane;
                    if(column < nColumns){
                        const char base = multiple_sequence_alignment[row * nColumns + column];
                        switch(base){
                            case 'A': batchcounts[lane][0]++; break;
                            case 'C': batchcounts[lane][1]++; break;
                            case 'G': batchcounts[lane][2]++; break;
                            case 'T': batchcounts[lane][3]++; break;
                            default: batchcounts[lane][4]++; break;
                        }
                        const float weight = canUseWeights ? multiple_sequence_alignment_weights[row * nColumns + column] : 1.0f;
                        switch(base){
                            case 'A': batchweights[lane][0] += weight; break;
                            case 'C': batchweights[lane][1] += weight; break;
                            case 'G': batchweights[lane][2] += weight; break;
                            case 'T': batchweights[lane][3] += weight; break;
                            default: batchweights[lane][4] += weight; break;
                        }
                    }
                }
            }

            for(int lane = 0; lane < lanesPerBatch; lane++){
                const int column = batch * lanesPerBatch + lane;
                if(column < nColumns){
                    std::copy(batchcounts[lane].begin(), batchcounts[lane].begin() + 4, countsPerBase[column].begin());
                    std::copy(batchweights[lane].begin(), batchweights[lane].begin() + 4, weightsPerBase[column].begin());
                    coverage[column] = batchcounts[lane][0] + batchcounts[lane][1] + batchcounts[lane][2] + batchcounts[lane][3];

                    char cons = 'A';
                    float consWeight = batchweights[lane][0];
                    if(batchweights[lane][1] > consWeight){
                        cons = 'C';
                        consWeight = batchweights[lane][1];
                    }
                    if(batchweights[lane][2] > consWeight){
                        cons = 'G';
                        consWeight = batchweights[lane][2];
                    }
                    if(batchweights[lane][3] > consWeight){
                        cons = 'T';
                        consWeight = batchweights[lane][3];
                    }
                    consensus[column] = cons;
                    const float columnWeight = batchweights[lane][0] + batchweights[lane][1] + batchweights[lane][2] + batchweights[lane][3];
                    support[column] = consWeight / columnWeight;

                    if(columnProperties.subjectColumnsBegin_incl <= column && column < columnProperties.subjectColumnsEnd_excl){
                        const char subjectBase = multiple_sequence_alignment[0 * nColumns + column];
                        switch(subjectBase){
                            case 'A':origWeights[column] = batchweights[lane][0]; origCoverages[column] = batchcounts[lane][0]; break;
                            case 'C':origWeights[column] = batchweights[lane][1]; origCoverages[column] = batchcounts[lane][1]; break;
                            case 'G':origWeights[column] = batchweights[lane][2]; origCoverages[column] = batchcounts[lane][2]; break;
                            case 'T':origWeights[column] = batchweights[lane][3]; origCoverages[column] = batchcounts[lane][3]; break;
                            default: assert(false && "This should not happen in find_consensus"); break;
                        }
                    }
                }
            }
        }
    }


    //remove all candidate reads from alignment which are assumed to originate from a different genomic region
    //returns local candidate indices of remaining candidates
    MinimizationResult minimize(int dataset_coverage){
        auto is_significant_count = [&](int count, int consensuscount, int columncoverage, int dataset_coverage)->bool{
            if(int(dataset_coverage * 0.3f) <= count)
                return true;
            return false;
        };

        constexpr std::array<char, 4> index_to_base{'A','C','G','T'};

        //find column with a non-consensus base with significant coverage
        int col = 0;
        bool foundColumn = false;
        char foundBase = 'F';
        int foundBaseIndex = 0;
        int consindex = 0;

        //if anchor has no mismatch to consensus, don't minimize
        auto pair = std::mismatch(&multiple_sequence_alignment[columnProperties.subjectColumnsBegin_incl],
                                    &multiple_sequence_alignment[columnProperties.subjectColumnsEnd_excl],
                                    &consensus[columnProperties.subjectColumnsBegin_incl]);

        if(pair.first == &multiple_sequence_alignment[columnProperties.subjectColumnsEnd_excl]){
            MinimizationResult result;
            result.performedMinimization = false;
            return result;
        }

        for(int columnindex = columnProperties.subjectColumnsBegin_incl; columnindex < columnProperties.subjectColumnsEnd_excl && !foundColumn; columnindex++){
            std::array<int,4> counts;
            //std::array<float,4> weights;
#if 0
            counts[0] = countsA[columnindex];
            counts[1] = countsC[columnindex];
            counts[2] = countsG[columnindex];
            counts[3] = countsT[columnindex];

            weights[0] = weightsA[columnindex];
            weights[1] = weightsC[columnindex];
            weights[2] = weightsG[columnindex];
            weights[3] = weightsT[columnindex];
#else

            counts = countsPerBase[columnindex];
            //weights = weightsPerBase[columnindex];
#endif
            char cons = consensus[columnindex];
            int consensuscount = 0;
            consindex = -1;

            switch(cons){
                case 'A': consensuscount = counts[0]; consindex = 0;break;
                case 'C': consensuscount = counts[1]; consindex = 1;break;
                case 'G': consensuscount = counts[2]; consindex = 2;break;
                case 'T': consensuscount = counts[3]; consindex = 3;break;
            }

            const char originalbase = multiple_sequence_alignment[0 * nColumns + col];

            //find out if there is a non-consensus base with significant coverage
            int significantBaseIndex = -1;
            int maxcount = 0;
            for(int i = 0; i < 4; i++){
                if(i != consindex){
                    bool significant = is_significant_count(counts[i], consensuscount, coverage[columnindex], dataset_coverage);

                    bool process = significant; //maxcount < counts[i] && significant && (cons == originalbase || index_to_base[i] == originalbase);

                    significantBaseIndex = process ? i : significantBaseIndex;

                    //maxcount = process ? std::max(maxcount, counts[i]) : maxcount;
                }
            }

            if(significantBaseIndex != -1){
                foundColumn = true;
                col = columnindex;
                foundBase = index_to_base[significantBaseIndex];
                foundBaseIndex = significantBaseIndex;
            }
        }



        MinimizationResult result;
        result.performedMinimization = foundColumn;
        result.column = col;

        if(foundColumn){

            result.remaining_candidates.reserve(nRows-1);

            auto discard_rows = [&](bool keepMatching){
                char* insertionposition1 = &multiple_sequence_alignment[1 * nColumns + 0];
                float* insertionposition2 = &multiple_sequence_alignment_weights[1 * nColumns + 0];
                int insertionrow = 1;

                for(int row = 1; row < nRows; row++){
                    //check if row is affected by column col
                    const int row_begin_incl = columnProperties.subjectColumnsBegin_incl + shifts[row];
                    const int row_end_excl = row_begin_incl + sequenceLengths[row];
                    const bool notAffected = (col < row_begin_incl || row_end_excl <= col);

                    const char base = multiple_sequence_alignment[row * nColumns + col];

                    if(notAffected || (!(keepMatching ^ (base == foundBase)))){
                        insertionposition1 = std::copy(&multiple_sequence_alignment[row * nColumns],
                                                        &multiple_sequence_alignment[(row+1) * nColumns],
                                                        insertionposition1);
                        insertionposition2 = std::copy(&multiple_sequence_alignment_weights[row * nColumns],
                                                        &multiple_sequence_alignment_weights[(row+1) * nColumns],
                                                        insertionposition2);
                        shifts[insertionrow] = shifts[row];
                        sequenceLengths[insertionrow] = sequenceLengths[row];

                        insertionrow++;
                        result.remaining_candidates.emplace_back(row-1);
                    }else{
                        result.num_discarded_candidates++;
                    }
                }
                nRows = insertionrow;
            };

            //compare found base to original base
            const char originalbase = multiple_sequence_alignment[0 * nColumns + col];

            result.significantBase = foundBase;
            result.originalBase = originalbase;
            result.consensusBase = consensus[col];

            const std::array<int,4> counts = countsPerBase[col];
            //std::array<int,4> weights = weightsPerBase[columnindex];

            result.significantCount = counts[foundBaseIndex];
            result.consensuscount = counts[consindex];


            if(originalbase == foundBase){
                //discard all candidates whose base in column col differs from foundBase
                discard_rows(true);
            }else{
                //discard all candidates whose base in column col matches foundBase
                discard_rows(false);
            }

            if(result.num_discarded_candidates > 0){
                find_consensus();
            }

            return result;
        }else{

            return result;
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
