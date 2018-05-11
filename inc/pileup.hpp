#ifndef PILEUP_HPP
#define PILEUP_HPP

#include "qualityscoreweights.hpp"

#include <vector>
#include <string>
#include <cassert>
#include <climits>
#include <cmath>
#include <memory>
#include <tuple>

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
        double k;
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
                int kmerlength){

        correctionSettings.m = m_coverage;
        correctionSettings.k = kmerlength;
    }
/*
    PileupImage(const PileupImage& other){
        *this = other;
    }

    PileupImage(PileupImage&& other){
        *this = std::move(other);
    }

    PileupImage& operator=(const PileupImage& other){
        resize(other.max_n_columns);
        std::memcpy(h_As.data(), other.h_As.data(), sizeof(int) * other.max_n_columns);
        std::memcpy(h_Cs.data(), other.h_Cs.data(), sizeof(int) * other.max_n_columns);
        std::memcpy(h_Gs.data(), other.h_Gs.data(), sizeof(int) * other.max_n_columns);
        std::memcpy(h_Ts.data(), other.h_Ts.data(), sizeof(int) * other.max_n_columns);
        std::memcpy(h_Aweights.data(), other.h_Aweights.data(), sizeof(double) * other.max_n_columns);
        std::memcpy(h_Cweights.data(), other.h_Cweights.data(), sizeof(double) * other.max_n_columns);
        std::memcpy(h_Gweights.data(), other.h_Gweights.data(), sizeof(double) * other.max_n_columns);
        std::memcpy(h_Tweights.data(), other.h_Tweights.data(), sizeof(double) * other.max_n_columns);
        std::memcpy(h_consensus.data(), other.h_consensus.data(), sizeof(char) * other.max_n_columns);
        std::memcpy(h_support.data(), other.h_support.data(), sizeof(double) * other.max_n_columns);
        std::memcpy(h_coverage.data(), other.h_coverage.data(), sizeof(int) * other.max_n_columns);
        std::memcpy(h_origWeights.data(), other.h_origWeights.data(), sizeof(double) * other.max_n_columns);
        std::memcpy(h_origCoverage.data(), other.h_origCoverage.data(), sizeof(int) * other.max_n_columns);

        n_columns = other.n_columns;
        columnProperties = other.columnProperties;
        correctionSettings = other.correctionSettings;

        return *this;
    }
*/

/*
    PileupImage& operator=(PileupImage&& other){
        h_As = std::move(other.h_As);
        h_Cs = std::move(other.h_Cs);
        h_Gs = std::move(other.h_Gs);
        h_Ts = std::move(other.h_Ts);
        h_Aweights = std::move(other.h_Aweights);
        h_Cweights = std::move(other.h_Cweights);
        h_Gweights = std::move(other.h_Gweights);
        h_Tweights = std::move(other.h_Tweights);
        h_consensus = std::move(other.h_consensus);
        h_support = std::move(other.h_support);
        h_coverage = std::move(other.h_coverage);
        h_origWeights = std::move(other.h_origWeights);
        h_origCoverage = std::move(other.h_origCoverage);

        n_columns = other.n_columns;
        columnProperties = other.columnProperties;
        correctionSettings = other.correctionSettings;

        return *this;
    }
*/

    void resize(int cols){

        if(cols > max_n_columns){
            const int newmaxcols = 1.5 * cols;

            h_consensus.resize(newmaxcols);
            h_support.resize(newmaxcols);
            h_coverage.resize(newmaxcols);
            h_origWeights.resize(newmaxcols);
            h_origCoverage.resize(newmaxcols);
            h_As.resize(newmaxcols);
            h_Cs.resize(newmaxcols);
            h_Gs.resize(newmaxcols);
            h_Ts.resize(newmaxcols);
            h_Aweights.resize(newmaxcols);
            h_Cweights.resize(newmaxcols);
            h_Gweights.resize(newmaxcols);
            h_Tweights.resize(newmaxcols);

            max_n_columns = newmaxcols;
        }

        n_columns = cols;
    }

    void clear(){
			auto zero = [](auto& vec){
				std::fill(vec.begin(), vec.end(), 0);
			};
			
			zero(h_support);
			zero(h_coverage);
			zero(h_origWeights);
			zero(h_origCoverage);
			zero(h_As);
			zero(h_Cs);
			zero(h_Gs);
			zero(h_Ts);
			zero(h_Aweights);
			zero(h_Cweights);
			zero(h_Gweights);
			zero(h_Tweights);

			std::fill(h_consensus.begin(), h_consensus.end(), '\0');
    }

    void destroy(){
			auto destroyvec = [](auto& vec){
				vec.clear();
				vec.shrink_to_fit();
			};
		
			destroyvec(h_support);
			destroyvec(h_coverage);
			destroyvec(h_origWeights);
			destroyvec(h_origCoverage);
			destroyvec(h_As);
			destroyvec(h_Cs);
			destroyvec(h_Gs);
			destroyvec(h_Ts);
			destroyvec(h_Aweights);
			destroyvec(h_Cweights);
			destroyvec(h_Gweights);
			destroyvec(h_Tweights);

			destroyvec(h_consensus);
    }

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
    template<class AlignmentIter, class SequenceIter, class CountIter, class QualityIter>
    void cpu_add_candidates(const std::string& sequence_to_correct,
                AlignmentIter alignmentsBegin,
                AlignmentIter alignmentsEnd,
                double desiredAlignmentMaxErrorRate,
                SequenceIter candidateSequencesBegin,
                SequenceIter candidateSequencesEnd,
                CountIter candidateCountsBegin,
                CountIter candidateCountsEnd,
                QualityIter candidateQualitiesBegin,
                QualityIter candidateQualitiesEnd){

        // add weights for each base in every candidate sequences
	auto alignmentiter = alignmentsBegin;
	auto sequenceiter = candidateSequencesBegin;
	auto countiter = candidateCountsBegin;
	auto candidateQualityiter = candidateQualitiesBegin;
#if 0
        for(auto t = std::make_tuple(alignmentsBegin, candidateSequencesBegin, candidateCountsBegin, candidateQualitiesBegin);
            std::get<0>(t) != alignmentsEnd;
            std::get<0>(t)++, std::get<1>(t)++, std::get<2>(t)++/*quality iter is incremented in loop body*/){

            auto& alignmentiter = std::get<0>(t);
            auto& sequenceiter = std::get<1>(t);
            auto& countiter = std::get<2>(t);
            auto& candidateQualityiter = std::get<3>(t);
#endif
	for(; alignmentiter != alignmentsEnd; alignmentiter++, sequenceiter++, countiter++){
            const double defaultweight = 1.0 - std::sqrt((*alignmentiter)->get_nOps()
                                                        / ((*alignmentiter)->get_overlap()
                                                            * desiredAlignmentMaxErrorRate));
            const int len = sequenceiter->length();
            const int freq = *countiter;
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

    void cpu_find_consensus_internal(){
        for(int i = 0; i < columnProperties.columnsToCheck; i++){
            char cons = 'A';
            double consWeight = h_Aweights[i];
            if(h_Cweights[i] > consWeight){
                cons = 'C';
                consWeight = h_Cweights[i];
            }
            if(h_Gweights[i] > consWeight){
                cons = 'G';
                consWeight = h_Gweights[i];
            }
            if(h_Tweights[i] > consWeight){
                cons = 'T';
                consWeight = h_Tweights[i];
            }
            h_consensus[i] = cons;

            const double columnWeight = h_Aweights[i] + h_Cweights[i] + h_Gweights[i] + h_Tweights[i];
            h_support[i] = consWeight / columnWeight;
        }
    }

    CorrectionResult cpu_correct_sequence_internal(const std::string& sequence_to_correct,
                                        double estimatedErrorrate,
                                        double avg_support_threshold,
                                        double min_support_threshold,
                                        double min_coverage_threshold){

        cpu_find_consensus_internal();

        const int subjectlength = sequence_to_correct.length();

        CorrectionResult result;
        result.isCorrected = false;
        result.correctedSequence.resize(subjectlength);
        result.stats.avg_support = 0;
        result.stats.min_support = 1.0;
        result.stats.max_coverage = 0;
        result.stats.min_coverage = std::numeric_limits<int>::max();
        //get stats for subject columns
        for(int i = columnProperties.subjectColumnsBegin_incl; i < columnProperties.subjectColumnsEnd_excl; i++){
            assert(i < columnProperties.columnsToCheck);

            result.stats.avg_support += h_support[i];
            result.stats.min_support = std::min(h_support[i], result.stats.min_support);
            result.stats.max_coverage = std::max(h_coverage[i], result.stats.max_coverage);
            result.stats.min_coverage = std::min(h_coverage[i], result.stats.min_coverage);
        }

        result.stats.avg_support /= subjectlength;

        auto isGoodAvgSupport = [&](double avgsupport){
            return avgsupport >= avg_support_threshold;
        };
        auto isGoodMinSupport = [&](double minsupport){
            return minsupport >= min_support_threshold;
        };
        auto isGoodMinCoverage = [&](double mincoverage){
            return mincoverage >= min_coverage_threshold;
        };

        //TODO vary parameters
        result.stats.isHQ = isGoodAvgSupport(result.stats.avg_support)
                            && isGoodMinSupport(result.stats.min_support)
                            && isGoodMinCoverage(result.stats.min_coverage);

        result.stats.failedAvgSupport = !isGoodAvgSupport(result.stats.avg_support);
        result.stats.failedMinSupport = !isGoodMinSupport(result.stats.min_support);
        result.stats.failedMinCoverage = !isGoodMinCoverage(result.stats.min_coverage);

        if(result.stats.isHQ){
    #if 1
            //correct anchor
            for(int i = 0; i < subjectlength; i++){
                const int globalIndex = columnProperties.subjectColumnsBegin_incl + i;
                result.correctedSequence[i] = h_consensus[globalIndex];
            }
            result.isCorrected = true;
    #endif
        }else{
    #if 1
            //correct anchor


#if 1
            result.correctedSequence = sequence_to_correct;
            bool foundAColumn = false;
            for(int i = 0; i < subjectlength; i++){
                const int globalIndex = columnProperties.subjectColumnsBegin_incl + i;

                if(h_support[globalIndex] > 0.5 && h_origCoverage[globalIndex] < min_coverage_threshold){
                    double avgsupportkregion = 0;
                    int c = 0;
                    bool kregioncoverageisgood = true;

                    for(int j = i - correctionSettings.k/2; j <= i + correctionSettings.k/2 && kregioncoverageisgood; j++){
                        if(j != i && j >= 0 && j < subjectlength){
                            avgsupportkregion += h_support[columnProperties.subjectColumnsBegin_incl + j];
                            kregioncoverageisgood &= (h_coverage[columnProperties.subjectColumnsBegin_incl + j] >= min_coverage_threshold);
                            c++;
                        }
                    }

                    avgsupportkregion /= c;
                    if(kregioncoverageisgood && avgsupportkregion >= 1.0-estimatedErrorrate){
                        result.correctedSequence[i] = h_consensus[globalIndex];
                        foundAColumn = true;
                    }
                }
            }

            result.isCorrected = foundAColumn;
#else
            result.correctedSequence = sequence_to_correct;
            const int regionsize = correctionSettings.k;
            bool foundAColumn = false;

            for(int columnindex = columnProperties.subjectColumnsBegin_incl - regionsize/2;
                columnindex < columnProperties.subjectColumnsEnd_excl;
                columnindex += regionsize){

                double supportsum = 0;
                double minsupport = std::numeric_limits<double>::max();
                double maxsupport = std::numeric_limits<double>::min();

                int origcoveragesum = 0;
                int minorigcoverage = std::numeric_limits<int>::max();
                int maxorigcoverage = std::numeric_limits<int>::min();

                int coveragesum = 0;
                int mincoverage = std::numeric_limits<int>::max();
                int maxcoverage = std::numeric_limits<int>::min();

                int c = 0;
                for(int i = 0; i < regionsize; i++){
                    const int index = columnindex + i;
                    if(0 <= index && index < columnProperties.columnsToCheck){
                        supportsum += h_support[index];
                        minsupport = std::min(minsupport, h_support[index]);
                        maxsupport = std::max(maxsupport, h_support[index]);

                        origcoveragesum += h_origCoverage[index];
                        minorigcoverage = std::min(minorigcoverage, h_origCoverage[index]);
                        maxorigcoverage = std::max(maxorigcoverage, h_origCoverage[index]);

                        coveragesum += h_coverage[index];
                        mincoverage = std::min(mincoverage, h_coverage[index]);
                        maxcoverage = std::max(maxcoverage, h_coverage[index]);

                        c++;
                    }
                }
                const double avgsupport = supportsum / c;

                bool isHQregion = isGoodAvgSupport(avgsupport)
                                   && isGoodMinSupport(minsupport)
                                   && isGoodMinCoverage(mincoverage);

               if(isHQregion){
                   //correct anchor
                   for(int i = 0; i < regionsize; i++){
                       const int index = columnindex + i;
                       if(columnProperties.subjectColumnsBegin_incl <= index && index < columnProperties.subjectColumnsEnd_excl){
                           const int localindex = index - columnProperties.subjectColumnsBegin_incl;
                           result.correctedSequence[localindex] = h_consensus[index];
                       }
                   }
                   result.isCorrected = true;
               }else{
                   for(int i = 0; i < regionsize; i++){
                       const int index = columnindex + i;
                       if(columnProperties.subjectColumnsBegin_incl <= index
                           && index < columnProperties.subjectColumnsEnd_excl){

                           if(h_support[index] > 0.5
                               && h_origCoverage[index] < min_coverage_threshold
                               && isGoodAvgSupport(avgsupport)
                               && mincoverage >= min_coverage_threshold){

                                   const int localindex = index - columnProperties.subjectColumnsBegin_incl;
                                   result.correctedSequence[localindex] = h_consensus[index];
                           }
                       }
                   }
                   result.isCorrected = true;
               }
            }

            result.isCorrected = foundAColumn;

#endif


    #endif
        }

        return result;
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

        CorrectionResult result = cpu_correct_sequence_internal(sequence_to_correct,
                                                        estimatedErrorrate,
                                                        avg_support_threshold,
                                                        min_support_threshold,
                                                        min_coverage_threshold);

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
