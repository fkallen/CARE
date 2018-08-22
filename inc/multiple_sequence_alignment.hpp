#ifndef CARE_MSA_HPP
#define CARE_MSA_HPP

#include "qualityscoreweights.hpp"
#include "featureextractor.hpp"

#include <vector>
#include <string>
#include <cassert>
#include <climits>
#include <cmath>
#include <memory>
#include <tuple>
#include <limits>
#include <iostream>

namespace care{

    namespace pileup{

        struct PileupProperties{
            double avg_support;
            double min_support;
            int max_coverage;
            int min_coverage;
            bool isHQ;
            bool failedAvgSupport;
            bool failedMinSupport;
            bool failedMinCoverage;
        };



// TEST

struct PileupImage{

    /*struct Feature{
        double min_support;
        double min_coverage;
        double max_support;
        double max_coverage;
        double mean_support;
        double mean_coverage;
        double median_support;
        double median_coverage;
        int position;
        int original_base_coverage;
        int original_base_support;
    };*/

    struct CorrectedCandidate{
        std::uint64_t index;
        std::string sequence;
        CorrectedCandidate() noexcept{}
        CorrectedCandidate(std::uint64_t index, const std::string& sequence) noexcept
            : index(index), sequence(sequence){}
    };

    struct CorrectionResult{
        std::string correctedSequence;
        std::vector<CorrectedCandidate> correctedCandidates;
        PileupProperties stats;
        bool isCorrected;
    };

    struct PileupCorrectionSettings{
        double m;
        int k;
        int dataset_coverage;
    };

    struct PileupColumnProperties{
        int startindex;
        int endindex;
        int columnsToCheck;
        int subjectColumnsBegin_incl;
        int subjectColumnsEnd_excl;
    };

    //buffers
    std::vector<int> h_As;
    std::vector<int> h_Cs;
    std::vector<int> h_Gs;
    std::vector<int> h_Ts;
    std::vector<double> h_Aweights;
    std::vector<double> h_Cweights;
    std::vector<double> h_Gweights;
    std::vector<double> h_Tweights;
    std::vector<char> h_consensus;
    std::vector<double> h_support;
    std::vector<int> h_coverage;
    std::vector<double> h_origWeights;
    std::vector<int> h_origCoverage;

    int max_n_columns = 0; //number of elements per buffer
    int n_columns = 0; //number of used elements per buffer

    PileupColumnProperties columnProperties;
    PileupCorrectionSettings correctionSettings;

    PileupImage(double m_coverage,
                int kmerlength,
                int dataset_coverage);

    PileupImage(const PileupImage& other);

    PileupImage(PileupImage&& other);

    PileupImage& operator=(const PileupImage& other);

    PileupImage& operator=(PileupImage&& other);

    void resize(int cols);

    void clear();

    void destroy();

    void cpu_find_consensus_internal();

    CorrectionResult cpu_correct_sequence_internal(const std::string& sequence_to_correct,
                                        double estimatedErrorrate,
                                        double avg_support_threshold,
                                        double min_support_threshold,
                                        double min_coverage_threshold);

    bool shouldCorrect(double min_support,
                double min_coverage,
                double max_support,
                double max_coverage,
                double mean_support,
                double mean_coverage,
                double median_support,
                double median_coverage) const;

    std::vector<MSAFeature> getFeaturesOfNonConsensusPositions(
                                    const std::string& sequence,
                                    int k,
                                    double support_threshold) const;

    CorrectionResult cpu_correct_sequence_internal_RF(const std::string& sequence_to_correct);



/*
    AlignmentIter: Iterator to Alignment pointer
    SequenceIter: Iterator to std::string
    QualityIter: Iter to pointer to std::string
*/
    template<class AlignmentIter, class SequenceIter, class QualityIter>
    void init(const std::string& sequence_to_correct,
              const std::string* quality_of_sequence_to_correct,
                AlignmentIter alignmentsBegin,
                AlignmentIter alignmentsEnd,
                SequenceIter candidateSequencesBegin,
                SequenceIter candidateSequencesEnd,
                QualityIter candidateQualitiesBegin,
                QualityIter candidateQualitiesEnd){

        const int subjectlength = sequence_to_correct.length();

        //determine number of columns in pileup image
        columnProperties.startindex = 0;
        columnProperties.endindex = sequence_to_correct.length();

        for(auto p = std::make_pair(alignmentsBegin, candidateSequencesBegin);
            p.first != alignmentsEnd;
            p.first++, p.second++){

            auto& alignmentiter = p.first;
            auto& sequenceiter = p.second;

            const int shift = (*alignmentiter)->get_shift();
            columnProperties.startindex = std::min(shift, columnProperties.startindex);
            const int queryEndsAt = sequenceiter->length() + shift;
            columnProperties.endindex = std::max(queryEndsAt, columnProperties.endindex);
        }

        columnProperties.columnsToCheck = columnProperties.endindex - columnProperties.startindex;
        columnProperties.subjectColumnsBegin_incl = std::max(-columnProperties.startindex,0);
        columnProperties.subjectColumnsEnd_excl = columnProperties.subjectColumnsBegin_incl + sequence_to_correct.length();

        resize(columnProperties.columnsToCheck);

        clear();

        //add subject weights
        for(int i = 0; i < subjectlength; i++){
            const int globalIndex = columnProperties.subjectColumnsBegin_incl + i;
            assert(globalIndex < max_n_columns);
            double qw = 1.0;
            if(quality_of_sequence_to_correct != nullptr)
                qw *= qscore_to_weight[(unsigned char)(*quality_of_sequence_to_correct)[i]];

            const char base = sequence_to_correct[i];
            switch(base){
                case 'A': h_Aweights[globalIndex] += qw; h_As[globalIndex] += 1; break;
                case 'C': h_Cweights[globalIndex] += qw; h_Cs[globalIndex] += 1; break;
                case 'G': h_Gweights[globalIndex] += qw; h_Gs[globalIndex] += 1; break;
                case 'T': h_Tweights[globalIndex] += qw; h_Ts[globalIndex] += 1; break;
                default: std::cout << "Pileup: Found invalid base in sequence to be corrected\n"; break;
            }
            h_coverage[globalIndex]++;
        }
    }

    /*
        AlignmentIter: Iterator to Alignment pointer
        SequenceIter: Iterator to std::string
        CountIter: Iterator to int
        QualityIter: Iter to pointer to std::string
    */
    template<class AlignmentIter, class SequenceIter, /*class CountIter,*/ class QualityIter>
    void cpu_add_candidates(const std::string& sequence_to_correct,
                AlignmentIter alignmentsBegin,
                AlignmentIter alignmentsEnd,
                double desiredAlignmentMaxErrorRate,
                SequenceIter candidateSequencesBegin,
                SequenceIter candidateSequencesEnd,
                //CountIter candidateCountsBegin,
                //CountIter candidateCountsEnd,
                QualityIter candidateQualitiesBegin,
                QualityIter candidateQualitiesEnd){

        // add weights for each base in every candidate sequences
	auto alignmentiter = alignmentsBegin;
	auto sequenceiter = candidateSequencesBegin;
	//auto countiter = candidateCountsBegin;
	auto candidateQualityiter = candidateQualitiesBegin;
#if 0
        for(auto t = std::make_tuple(alignmentsBegin, candidateSequencesBegin, candidateCountsBegin, candidateQualitiesBegin);
            std::get<0>(t) != alignmentsEnd;
            std::get<0>(t)++, std::get<1>(t)++, std::get<2>(t)++/*quality iter is incremented in loop body*/){

            auto& alignmentiter = std::get<0>(t);
            auto& sequenceiter = std::get<1>(t);
            auto& countiter = std::get<2>(t);
            auto& candidateQualityiter = std::get<3>(t);
#else
	//for(; alignmentiter != alignmentsEnd; alignmentiter++, sequenceiter++, countiter++){
    for(; alignmentiter != alignmentsEnd; alignmentiter++, sequenceiter++){
#endif
            const double defaultweight = 1.0 - std::sqrt((*alignmentiter)->get_nOps()
                                                        / ((*alignmentiter)->get_overlap()
                                                            * desiredAlignmentMaxErrorRate));
            const int len = sequenceiter->length();
            const int freq = 1;//*countiter;
            const int defaultcolumnoffset = columnProperties.subjectColumnsBegin_incl + (*alignmentiter)->get_shift();

            //use h_support as temporary storage to accumulate the quality factors for position j
            for(int f = 0; f < freq; f++){
                if(*candidateQualityiter != nullptr){
                    for(int j = 0; j < len; j++){
                        h_support[j] += qscore_to_weight[(unsigned char)(*(*candidateQualityiter))[j]];
                    }
                }else{
                    for(int j = 0; j < len; j++){
                        h_support[j] += 1;
                    }
                }
                candidateQualityiter++;
            }

            for(int j = 0; j < len; j++){
                const int globalIndex = defaultcolumnoffset + j;
                assert(globalIndex < max_n_columns);
                assert(j < max_n_columns);

                const double qw = h_support[j] * defaultweight;
                const char base = (*sequenceiter)[j];
                switch(base){
                    case 'A': h_Aweights[globalIndex] += qw; h_As[globalIndex] += freq;
                    break;
                    case 'C': h_Cweights[globalIndex] += qw; h_Cs[globalIndex] += freq;
                    break;
                    case 'G': h_Gweights[globalIndex] += qw; h_Gs[globalIndex] += freq;
                    break;
                    case 'T': h_Tweights[globalIndex] += qw; h_Ts[globalIndex] += freq;
                    break;
                    default: std::cout << "Pileup: Found invalid base in candidate sequence\n"; break;
                }
                h_coverage[globalIndex] += freq;
                h_support[j] = 0;
            }
        }

        // after adding all candidate sequences, find weight and coverage of bases in sequence_to_correct
        for(std::size_t i = 0; i < sequence_to_correct.length(); i++){
            const std::size_t globalIndex = columnProperties.subjectColumnsBegin_incl + i;
            const char base = sequence_to_correct[i];
            switch(base){
                case 'A':   h_origCoverage[globalIndex] = h_As[globalIndex];
                            h_origWeights[globalIndex] = h_Aweights[globalIndex];
                            break;
                case 'C':   h_origCoverage[globalIndex] = h_Cs[globalIndex];
                            h_origWeights[globalIndex] = h_Cweights[globalIndex];
                            break;
                case 'G':   h_origCoverage[globalIndex] = h_Gs[globalIndex];
                            h_origWeights[globalIndex] = h_Gweights[globalIndex];
                            break;
                case 'T':   h_origCoverage[globalIndex] = h_Ts[globalIndex];
                            h_origWeights[globalIndex] = h_Tweights[globalIndex];
                            break;
                default: std::cout << "Pileup: Found invalid base in sequence to be corrected\n"; break;
            }
        }
    }










    /*
        AlignmentIter: Iterator to Alignment pointer
        SequenceIter: Iterator to std::string
    */
    template<class AlignmentIter, class SequenceIter>
    std::vector<CorrectedCandidate> cpu_correct_candidates_internal(
                AlignmentIter alignmentsBegin,
                AlignmentIter alignmentsEnd,
                SequenceIter candidateSequencesBegin,
                SequenceIter candidateSequencesEnd,
                double avg_support_threshold,
                double min_support_threshold,
                double min_coverage_threshold,
                int new_columns_to_correct) const{

        std::vector<CorrectedCandidate> result;
        result.reserve(std::distance(candidateSequencesBegin, candidateSequencesEnd));

        /*
            Correct candidates which begin in column range
            [subjectColumnsBegin_incl - candidate_correction_new_cols, subjectColumnsBegin_incl + candidate_correction_new_cols],
            and are not longer than subjectlength + candidate_correction_new_cols
        */

	auto alignmentiter = alignmentsBegin;
	auto sequenceiter = candidateSequencesBegin;
	int i = 0;
#if 0
        for(auto t = std::make_tuple(alignmentsBegin, candidateSequencesBegin, 0);
            std::get<0>(t) != alignmentsEnd;
            std::get<0>(t)++, std::get<1>(t)++, std::get<2>(t)++){

            auto& alignmentiter = std::get<0>(t);
            auto& sequenceiter = std::get<1>(t);
            auto i = std::get<2>(t);
#endif
	for(; alignmentiter != alignmentsEnd; alignmentiter++, sequenceiter++, i++){

            const int queryColumnsBegin_incl = (*alignmentiter)->get_shift() - columnProperties.startindex;
            const int queryLength = sequenceiter->length();
            const int queryColumnsEnd_excl = queryColumnsBegin_incl + queryLength;

            //check range condition and length condition
            if(columnProperties.subjectColumnsBegin_incl - new_columns_to_correct <= queryColumnsBegin_incl
                && queryColumnsBegin_incl <= columnProperties.subjectColumnsBegin_incl + new_columns_to_correct
                && queryColumnsEnd_excl <= columnProperties.subjectColumnsEnd_excl + new_columns_to_correct){

                double newColMinSupport = 1.0;
                int newColMinCov = std::numeric_limits<int>::max();
                //check new columns left of subject
                for(int columnindex = columnProperties.subjectColumnsBegin_incl - new_columns_to_correct;
                    columnindex < columnProperties.subjectColumnsBegin_incl;
                    columnindex++){

                    assert(columnindex < columnProperties.columnsToCheck);
                    if(queryColumnsBegin_incl <= columnindex){
                        newColMinSupport = h_support[columnindex] < newColMinSupport ? h_support[columnindex] : newColMinSupport;
                        newColMinCov = h_coverage[columnindex] < newColMinCov ? h_coverage[columnindex] : newColMinCov;
                    }
                }
                //check new columns right of subject
                for(int columnindex = columnProperties.subjectColumnsEnd_excl;
                    columnindex < columnProperties.subjectColumnsEnd_excl + new_columns_to_correct
                    && columnindex < columnProperties.columnsToCheck;
                    columnindex++){

                    newColMinSupport = h_support[columnindex] < newColMinSupport ? h_support[columnindex] : newColMinSupport;
                    newColMinCov = h_coverage[columnindex] < newColMinCov ? h_coverage[columnindex] : newColMinCov;
                }

                if(newColMinSupport >= min_support_threshold
                    && newColMinCov >= min_coverage_threshold){

                    std::string correctedString(&h_consensus[queryColumnsBegin_incl], &h_consensus[queryColumnsEnd_excl]);

                    result.emplace_back(i, std::move(correctedString));
                }
            }
        }

        return result;
    }

    /*
        AlignmentIter: Iterator to Alignment pointer
        SequenceIter: Iterator to std::string
    */
    template<class AlignmentIter, class SequenceIter>
    CorrectionResult cpu_correct(const std::string& sequence_to_correct,
                AlignmentIter alignmentsBegin,
                AlignmentIter alignmentsEnd,
                SequenceIter candidateSequencesBegin,
                SequenceIter candidateSequencesEnd,
                double estimatedErrorrate,
                double estimatedCoverage,
                bool correctCandidates,
                int new_columns_to_correct){

        const double avg_support_threshold = 1.0-1.0*estimatedErrorrate;
        const double min_support_threshold = 1.0-3.0*estimatedErrorrate;
        const double min_coverage_threshold = correctionSettings.m / 6.0 * estimatedCoverage;

#if 1
        CorrectionResult result = cpu_correct_sequence_internal(sequence_to_correct,
                                                        estimatedErrorrate,
                                                        avg_support_threshold,
                                                        min_support_threshold,
                                                        min_coverage_threshold);
#else

        CorrectionResult result = cpu_correct_sequence_internal_RF(sequence_to_correct);

#endif
        if(correctCandidates && result.stats.isHQ){
            result.correctedCandidates = cpu_correct_candidates_internal(alignmentsBegin,
                                                            alignmentsEnd,
                                                            candidateSequencesBegin,
                                                            candidateSequencesEnd,
                                                            avg_support_threshold,
                                                            min_support_threshold,
                                                            min_coverage_threshold,
                                                            new_columns_to_correct);
        }

        return result;
    }

};

}

}
#endif
