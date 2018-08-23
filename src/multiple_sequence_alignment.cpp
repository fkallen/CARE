#include "../inc/multiple_sequence_alignment.hpp"

#include "../inc/celeganssrx218989.hpp"
#include "../inc/ecolisrr490124.hpp"
#include "../inc/dmelanogastersrr82337.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>

namespace care{

namespace pileup{

    PileupImage::PileupImage(double m_coverage,
                int kmerlength,
                int dataset_coverage){

        correctionSettings.m = m_coverage;
        correctionSettings.k = kmerlength;
        correctionSettings.dataset_coverage = dataset_coverage;
    }

    PileupImage::PileupImage(const PileupImage& other){
        *this = other;
    }

    PileupImage::PileupImage(PileupImage&& other){
        *this = std::move(other);
    }

    PileupImage& PileupImage::operator=(const PileupImage& other){
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



    PileupImage& PileupImage::operator=(PileupImage&& other){
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


    void PileupImage::resize(int cols){

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

    void PileupImage::clear(){
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

    void PileupImage::destroy(){
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


void PileupImage::cpu_find_consensus_internal(){
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

PileupImage::CorrectionResult PileupImage::cpu_correct_sequence_internal(const std::string& sequence_to_correct,
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


std::vector<MSAFeature> PileupImage::getFeaturesOfNonConsensusPositions(
                                const std::string& sequence,
                                int k,
                                double support_threshold) const{

    assert(k <= 30);
    auto result = extractFeatures(*this, sequence,
                                    k, support_threshold,
                                    correctionSettings.dataset_coverage);

    return result;
}

PileupImage::CorrectionResult PileupImage::cpu_correct_sequence_internal_RF(const std::string& sequence_to_correct){

    cpu_find_consensus_internal();

    const int subjectlength = sequence_to_correct.length();

    CorrectionResult result;
    result.isCorrected = false;
    result.correctedSequence.resize(subjectlength);
    result.stats.avg_support = 0;
    result.stats.min_support = 1.0;
    result.stats.max_coverage = 0;
    result.stats.min_coverage = std::numeric_limits<int>::max();
    result.correctedSequence = sequence_to_correct;

    auto features = getFeaturesOfNonConsensusPositions(sequence_to_correct,
                                                        correctionSettings.k,
                                                        0.0);

    for(const auto& feature : features){

        constexpr double maxgini = 0.05;

        //namespace speciestype = ecoli_srr490124;
        namespace speciestype = celegans_srx218989;
        //namespace speciestype = dmelanogaster_srr82337;
#if 1

#if 0
        const bool doCorrect = speciestype::shouldCorrect(feature.min_support,
                                            feature.min_coverage,
                                            feature.max_support,
                                            feature.max_coverage,
                                            feature.mean_support,
                                            feature.mean_coverage,
                                            feature.median_support,
                                            feature.median_coverage,
                                            maxgini);
#else


        const bool doCorrect = speciestype::shouldCorrect(feature.position_support,
                                            feature.position_coverage,
                                            feature.alignment_coverage,
                                            feature.dataset_coverage,
                                            feature.min_support,
                                            feature.min_coverage / feature.alignment_coverage,
                                            feature.max_support,
                                            feature.max_coverage / feature.alignment_coverage,
                                            feature.mean_support,
                                            feature.mean_coverage / feature.alignment_coverage,
                                            feature.median_support,
                                            feature.median_coverage / feature.alignment_coverage,
                                            maxgini);
#endif

#else
        std::pair<int, int> p = speciestype::shouldCorrect_forest(feature.min_support,
                                            feature.min_coverage,
                                            feature.max_support,
                                            feature.max_coverage,
                                            feature.mean_support,
                                            feature.mean_coverage,
                                            feature.median_support,
                                            feature.median_coverage,
                                            maxgini);

        const bool doCorrect = p.second / (p.first + p.second) > 0.5;

#endif

        if(doCorrect){
            const int globalIndex = columnProperties.subjectColumnsBegin_incl + feature.position;
            result.correctedSequence[feature.position] = h_consensus[globalIndex];
        }
    }

    result.isCorrected = true;

    return result;
}


}
}
