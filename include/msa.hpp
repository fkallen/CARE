#ifndef CARE_CPU_MSA_HPP
#define CARE_CPU_MSA_HPP

#include "bestalignment.hpp"
#include "cpu_alignment.hpp"
#include "util.hpp"

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

    template<int dummy=0>
    std::pair<int,int> find_good_consensus_region_of_subject(const View<char>& subject,
                                                        const View<char>& consensus,
                                                        const View<int>& shifts){
        constexpr int max_clip = 20;
        constexpr int mismatches_required_for_clipping = 5;
        constexpr float badShiftRatio = 0.10;

        const int subjectLength = subject.size();

        //const auto minmaxshift = std::minmax_element(shifts.begin(), shifts.end());
        const int negativeShifts = std::count_if(shifts.begin(), shifts.end(), [](int s){return s < 0;});
        const int positiveShifts = std::count_if(shifts.begin(), shifts.end(), [](int s){return s > 0;});

        //const int nonZeroShifts = negativeShifts + positiveShifts;
        //const float negativeShiftRatio = negativeShifts / float(nonZeroShifts);
        //const float positiveShiftRatio = positiveShifts / float(nonZeroShifts);

        const int smallerShifts = std::min(negativeShifts, positiveShifts);
        const int greaterShifts = std::max(negativeShifts, positiveShifts);

        int remainingRegionBegin = 0;
        int remainingRegionEnd = subjectLength; //exclusive

        auto getRemainingRegionBegin = [&](){
            //look for mismatches on the left end
            int nMismatches = 0;
            int lastMismatchPos = -1;
            for(int localIndex = 0; localIndex < max_clip && localIndex < subjectLength; localIndex++){
                if(consensus[localIndex] != subject[localIndex]){
                    nMismatches++;
                    lastMismatchPos = localIndex;
                }
            }
            if(nMismatches >= mismatches_required_for_clipping){
                //clip after position of last mismatch in max_clip region
                return std::min(subjectLength, lastMismatchPos+1);
            }else{
                //everything is fine
                return 0;
            }
        };

        auto getRemainingRegionEnd = [&](){
            //look for mismatches on the right end
            int nMismatches = 0;
            int firstMismatchPos = subjectLength;
            const int begin = std::max(subjectLength - max_clip, 0);

            for(int localIndex = begin; localIndex < max_clip && localIndex < subjectLength; localIndex++){
                if(consensus[localIndex] != subject[localIndex]){
                    nMismatches++;
                    firstMismatchPos = localIndex;
                }
            }
            if(nMismatches >= mismatches_required_for_clipping){
                //clip after position of last mismatch in max_clip region
                return firstMismatchPos;
            }else{
                //everything is fine
                return subjectLength;
            }
        };

        //every shift is zero
        if(greaterShifts == 0){
            //check both ends
            remainingRegionBegin = getRemainingRegionBegin();
            remainingRegionEnd = getRemainingRegionEnd();
        }else{

            if(smallerShifts / greaterShifts < badShiftRatio){
                // look for consensus mismatches of subject

                if(smallerShifts == negativeShifts){
                    remainingRegionBegin = getRemainingRegionBegin();
                }else{
                    remainingRegionEnd = getRemainingRegionEnd();
                }
            }else{
                ; //everything is fine
            }
        }

        return {remainingRegionBegin, remainingRegionEnd};
    }








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

    template<class GetQualityWeight>
    void insert_implicit(int row, const std::string& sequence, int alignment_shift, GetQualityWeight getQualityWeight){
        sequenceLengths[row] = int(sequence.length());
        shifts[row] = alignment_shift;

        for(int i = 0; i < int(sequence.length()); i++){
            const int globalIndex = columnProperties.subjectColumnsBegin_incl + alignment_shift + i;
            const char base = sequence[i];
            const float weight = canUseWeights ? getQualityWeight(i) : 1.0f;
            switch(base){
                case 'A': countsA[globalIndex]++; weightsA[globalIndex] += weight;break;
                case 'C': countsC[globalIndex]++; weightsC[globalIndex] += weight;break;
                case 'G': countsG[globalIndex]++; weightsG[globalIndex] += weight;break;
                case 'T': countsT[globalIndex]++; weightsT[globalIndex] += weight;break;
                default: assert(false); break;
            }
            coverage[globalIndex]++;
        }
    }

    template<class GetQualityWeight>
    void insertSubject_implicit(const std::string& subject, GetQualityWeight getQualityWeight){
        insert_implicit(0,subject, 0, getQualityWeight);
    }

    template<class GetQualityWeight>
    void insertCandidate_implicit(const std::string& candidate, int alignment_shift, GetQualityWeight getQualityWeight){
        assert(insertedCandidates < nRows-1);


        insert_implicit(1+insertedCandidates, candidate, alignment_shift, getQualityWeight);

        ++insertedCandidates;


    }

    void find_consensus_implicit(const std::string& subject){
        for(int column = 0; column < nColumns; ++column){
            char cons = 'A';
            float consWeight = weightsA[column];
            if(weightsC[column] > consWeight){
                cons = 'C';
                consWeight = weightsC[column];
            }
            if(weightsG[column] > consWeight){
                cons = 'G';
                consWeight = weightsG[column];
            }
            if(weightsT[column] > consWeight){
                cons = 'T';
                consWeight = weightsT[column];
            }
            consensus[column] = cons;

            const float columnWeight = weightsA[column] + weightsC[column] + weightsG[column] + weightsT[column];
            support[column] = consWeight / columnWeight;

            if(columnProperties.subjectColumnsBegin_incl <= column && column < columnProperties.subjectColumnsEnd_excl){
                const int localIndex = column - columnProperties.subjectColumnsBegin_incl;
                const char subjectBase = subject[localIndex];
                switch(subjectBase){
                    case 'A':origWeights[column] = weightsA[column]; origCoverages[column] = countsA[column]; break;
                    case 'C':origWeights[column] = weightsG[column]; origCoverages[column] = countsC[column]; break;
                    case 'G':origWeights[column] = weightsC[column]; origCoverages[column] = countsG[column]; break;
                    case 'T':origWeights[column] = weightsT[column]; origCoverages[column] = countsT[column]; break;
                    default: assert(false && "This should not happen in find_consensus_implicit"); break;
                }
            }
        }
    }

    std::pair<int,int> findGoodConsensusRegionOfSubject() const{
        View<char> subjectview{&multiple_sequence_alignment[columnProperties.subjectColumnsBegin_incl],
                                columnProperties.subjectColumnsEnd_excl - columnProperties.subjectColumnsBegin_incl};
        View<char> consensusview{&consensus[columnProperties.subjectColumnsBegin_incl],
                                    columnProperties.subjectColumnsEnd_excl - columnProperties.subjectColumnsBegin_incl};
        View<int> shiftview{&shifts[0], nRows-1};

        auto result = find_good_consensus_region_of_subject(subjectview, consensusview, shiftview);

        return result;
    }

    std::pair<int,int> findGoodConsensusRegionOfSubject_implicit(const std::string& subject) const{
        View<char> subjectview{&subject[0], int(subject.size())};
        View<char> consensusview{&consensus[columnProperties.subjectColumnsBegin_incl],
                                    columnProperties.subjectColumnsEnd_excl - columnProperties.subjectColumnsBegin_incl};
        View<int> shiftview{&shifts[0], nRows-1};

        auto result = find_good_consensus_region_of_subject(subjectview, consensusview, shiftview);

        return result;
    }




    CorrectionResult getCorrectedSubject_implicit(const std::string& subject){

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
            std::copy(subject.begin(),
                      subject.end(),
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

    std::vector<CorrectedCandidate> getCorrectedCandidates_implicit(const std::vector<int>& candidate_sequences_lengths,
                                                        const std::vector<SHDResult>& bestAlignments,
                                                        int new_columns_to_correct) const{

        return getCorrectedCandidates(candidate_sequences_lengths, bestAlignments, new_columns_to_correct);
    }

    //remove all candidate reads from alignment which are assumed to originate from a different genomic region
    //the indices of remaining candidates are returned in MinimizationResult::remaining_candidates
    //candidates in vector must be in the same order as they were inserted into the msa!!!

    template<class T>
    MinimizationResult minimize_implicit(const std::string& subject,
                                        const std::vector<std::string>& candidates,
                                        int dataset_coverage,
                                        const std::vector<T>& getQualityWeightFunctions){
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
        auto pair = std::mismatch(subject.begin(),
                                    subject.end(),
                                    &consensus[columnProperties.subjectColumnsBegin_incl]);

        if(pair.first == subject.end()){
            MinimizationResult result;
            result.performedMinimization = false;
            return result;
        }

        for(int columnindex = columnProperties.subjectColumnsBegin_incl; columnindex < columnProperties.subjectColumnsEnd_excl && !foundColumn; columnindex++){
            std::array<int,4> counts;
            //std::array<float,4> weights;

            counts[0] = countsA[columnindex];
            counts[1] = countsC[columnindex];
            counts[2] = countsG[columnindex];
            counts[3] = countsT[columnindex];

            /*weights[0] = weightsA[columnindex];
            weights[1] = weightsC[columnindex];
            weights[2] = weightsG[columnindex];
            weights[3] = weightsT[columnindex];*/

            char cons = consensus[columnindex];
            int consensuscount = 0;
            consindex = -1;

            switch(cons){
                case 'A': consensuscount = counts[0]; consindex = 0;break;
                case 'C': consensuscount = counts[1]; consindex = 1;break;
                case 'G': consensuscount = counts[2]; consindex = 2;break;
                case 'T': consensuscount = counts[3]; consindex = 3;break;
            }

            const char originalbase = subject[columnindex - columnProperties.subjectColumnsBegin_incl];

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
                int insertionrow = 1;

                for(int row = 1; row < nRows; row++){
                    //check if row is affected by column col
                    const int row_begin_incl = columnProperties.subjectColumnsBegin_incl + shifts[row];
                    const int row_end_excl = row_begin_incl + sequenceLengths[row];
                    const bool notAffected = (col < row_begin_incl || row_end_excl <= col);
                    const char base = notAffected ? 'F' : candidates[row-1][col - row_begin_incl];

                    if(notAffected || (!(keepMatching ^ (base == foundBase)))){

                        shifts[insertionrow] = shifts[row];
                        sequenceLengths[insertionrow] = sequenceLengths[row];

                        insertionrow++;
                        result.remaining_candidates.emplace_back(row-1);
                    }else{
                        for(int i = 0; i < sequenceLengths[row]; i++){
                            const int globalIndex = columnProperties.subjectColumnsBegin_incl + shifts[row] + i;
                            const char base = candidates[row-1][i];
                            const float weight = canUseWeights ? getQualityWeightFunctions[row-1](i) : 1.0f;
                            switch(base){
                                case 'A': countsA[globalIndex]--; weightsA[globalIndex] -= weight;break;
                                case 'C': countsC[globalIndex]--; weightsC[globalIndex] -= weight;break;
                                case 'G': countsG[globalIndex]--; weightsG[globalIndex] -= weight;break;
                                case 'T': countsT[globalIndex]--; weightsT[globalIndex] -= weight;break;
                                default: assert(false); break;
                            }
                            coverage[globalIndex]--;
                        }

                        result.num_discarded_candidates++;
                    }
                }
                nRows = insertionrow;
            };

            //compare found base to original base
            const char originalbase = subject[col - columnProperties.subjectColumnsBegin_incl];

            result.significantBase = foundBase;
            result.originalBase = originalbase;
            result.consensusBase = consensus[col];

            std::array<int,4> counts;

            counts[0] = countsA[col];
            counts[1] = countsC[col];
            counts[2] = countsG[col];
            counts[3] = countsT[col];

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
                find_consensus_implicit(subject);
            }

            return result;
        }else{

            return result;
        }
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

            countsA[column] = counts[0];
            countsC[column] = counts[1];
            countsG[column] = counts[2];
            countsT[column] = counts[3];

            weightsA[column] = weights[0];
            weightsC[column] = weights[1];
            weightsG[column] = weights[2];
            weightsT[column] = weights[3];

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

            counts[0] = countsA[columnindex];
            counts[1] = countsC[columnindex];
            counts[2] = countsG[columnindex];
            counts[3] = countsT[columnindex];

            /*weights[0] = weightsA[columnindex];
            weights[1] = weightsC[columnindex];
            weights[2] = weightsG[columnindex];
            weights[3] = weightsT[columnindex];*/

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

            switch(foundBaseIndex){
                case 0: result.significantCount = countsA[col];
                case 1: result.significantCount = countsC[col];
                case 2: result.significantCount = countsG[col];
                case 3: result.significantCount = countsT[col];
            }

            switch(consindex){
                case 0: result.consensuscount = countsA[col];
                case 1: result.consensuscount = countsC[col];
                case 2: result.consensuscount = countsG[col];
                case 3: result.consensuscount = countsT[col];
            }

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
