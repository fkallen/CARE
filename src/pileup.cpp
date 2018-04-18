#include "../inc/pileup.hpp"
#include "../inc/alignment.hpp"
#include "../inc/hpc_helpers.cuh"
#include "../inc/cudareduce.cuh"
#include "../inc/batchelem.hpp"

#include "../inc/qualityscoreweights.hpp"

#include <vector>
#include <string>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>

#ifdef __NVCC__
#include <cooperative_groups.h>

#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_vector.h>

using namespace cooperative_groups;
#endif

namespace care{

        PileupImage::PileupImage(const CorrectionOptions& CO, const GoodAlignmentProperties GAP){
            correctionSettings.useQScores = CO.useQualityScores;
            correctionSettings.correctCandidates = CO.correctCandidates;
            correctionSettings.estimatedCoverage = CO.estimatedCoverage;
            correctionSettings.maxErrorRate = GAP.maxErrorRate;
            correctionSettings.errorrate = CO.estimatedErrorrate;
            correctionSettings.m = CO.m_coverage;
            correctionSettings.k = CO.kmerlength;
        }

        PileupImage::PileupImage(const PileupImage& other){
            *this = other;
        }

        PileupImage::PileupImage(PileupImage&& other){
            *this = std::move(other);
        }

        PileupImage& PileupImage::operator=(const PileupImage& other){
            resize(other.max_n_columns);
            std::memcpy(h_As.get(), other.h_As.get(), sizeof(int) * other.max_n_columns);
            std::memcpy(h_Cs.get(), other.h_Cs.get(), sizeof(int) * other.max_n_columns);
            std::memcpy(h_Gs.get(), other.h_Gs.get(), sizeof(int) * other.max_n_columns);
            std::memcpy(h_Ts.get(), other.h_Ts.get(), sizeof(int) * other.max_n_columns);
            std::memcpy(h_Aweights.get(), other.h_Aweights.get(), sizeof(double) * other.max_n_columns);
            std::memcpy(h_Cweights.get(), other.h_Cweights.get(), sizeof(double) * other.max_n_columns);
            std::memcpy(h_Gweights.get(), other.h_Gweights.get(), sizeof(double) * other.max_n_columns);
            std::memcpy(h_Tweights.get(), other.h_Tweights.get(), sizeof(double) * other.max_n_columns);
            std::memcpy(h_consensus.get(), other.h_consensus.get(), sizeof(char) * other.max_n_columns);
            std::memcpy(h_support.get(), other.h_support.get(), sizeof(double) * other.max_n_columns);
            std::memcpy(h_coverage.get(), other.h_coverage.get(), sizeof(int) * other.max_n_columns);
            std::memcpy(h_origWeights.get(), other.h_origWeights.get(), sizeof(double) * other.max_n_columns);
            std::memcpy(h_origCoverage.get(), other.h_origCoverage.get(), sizeof(int) * other.max_n_columns);

            n_columns = other.n_columns;
            properties = other.properties;
            columnProperties = other.columnProperties;
            correctionSettings = other.correctionSettings;
            timings = other.timings;
            taskTimings = other.taskTimings;

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
            properties = other.properties;
            columnProperties = other.columnProperties;
            correctionSettings = other.correctionSettings;
            timings = other.timings;
            taskTimings = other.taskTimings;

            return *this;
        }


        void PileupImage::resize(int cols){

    		if(cols > max_n_columns){
    			const int newmaxcols = 1.5 * cols;

    			h_consensus.reset();
    			h_support.reset();
    			h_coverage.reset();
    			h_origWeights.reset();
    			h_origCoverage.reset();
    			h_As.reset();
    			h_Cs.reset();
    			h_Gs.reset();
    			h_Ts.reset();
    			h_Aweights.reset();
    			h_Cweights.reset();
    			h_Gweights.reset();
    			h_Tweights.reset();

    			h_consensus.reset(new char[newmaxcols]);
    			h_support.reset(new double[newmaxcols]);
    			h_coverage.reset(new int[newmaxcols]);
    			h_origWeights.reset(new double[newmaxcols]);
    			h_origCoverage.reset(new int[newmaxcols]);
    			h_As.reset(new int[newmaxcols]);
    			h_Cs.reset(new int[newmaxcols]);
    			h_Gs.reset(new int[newmaxcols]);
    			h_Ts.reset(new int[newmaxcols]);
    			h_Aweights.reset(new double[newmaxcols]);
    			h_Cweights.reset(new double[newmaxcols]);
    			h_Gweights.reset(new double[newmaxcols]);
    			h_Tweights.reset(new double[newmaxcols]);

                assert(h_consensus.get() != nullptr);
                assert(h_support.get() != nullptr);
                assert(h_coverage.get() != nullptr);
                assert(h_origWeights.get() != nullptr);
                assert(h_origCoverage.get() != nullptr);
                assert(h_As.get() != nullptr);
                assert(h_Cs.get() != nullptr);
                assert(h_Gs.get() != nullptr);
                assert(h_Ts.get() != nullptr);
                assert(h_Aweights.get() != nullptr);
                assert(h_Cweights.get() != nullptr);
                assert(h_Gweights.get() != nullptr);
                assert(h_Tweights.get() != nullptr);

    			max_n_columns = newmaxcols;
    		}

    		n_columns = cols;
    	}

        void PileupImage::clear(){
    			std::memset(h_consensus.get(), 0, sizeof(char) * n_columns);
    			std::memset(h_support.get(), 0, sizeof(double) * n_columns);
    			std::memset(h_coverage.get(), 0, sizeof(int) * n_columns);
    			std::memset(h_origWeights.get(), 0, sizeof(double) * n_columns);
    			std::memset(h_origCoverage.get(), 0, sizeof(int) * n_columns);
    			std::memset(h_As.get(), 0, sizeof(int) * n_columns);
    			std::memset(h_Cs.get(), 0, sizeof(int) * n_columns);
    			std::memset(h_Gs.get(), 0, sizeof(int) * n_columns);
    			std::memset(h_Ts.get(), 0, sizeof(int) * n_columns);
    			std::memset(h_Aweights.get(), 0, sizeof(double) * n_columns);
    			std::memset(h_Cweights.get(), 0, sizeof(double) * n_columns);
    			std::memset(h_Gweights.get(), 0, sizeof(double) * n_columns);
    			std::memset(h_Tweights.get(), 0, sizeof(double) * n_columns);

                properties.avg_support = 0;
                properties.min_support = 0;
                properties.max_coverage = 0;
                properties.min_coverage = 0;
                properties.isHQ = false;
                properties.failedAvgSupport = false;
                properties.failedMinSupport = false;
                properties.failedMinCoverage = false;
    	}

        TaskTimings PileupImage::correct_batch_elem(BatchElem& batchElem){
            TaskTimings tt;
            std::chrono::time_point<std::chrono::system_clock> tpc, tpd;

            tpc = std::chrono::system_clock::now();
            init_from_batch_elem(batchElem);
            tpd = std::chrono::system_clock::now();
            taskTimings.preprocessingtime += tpd - tpc;
            tt.preprocessingtime += tpd - tpc;

            tpc = std::chrono::system_clock::now();
            cpu_add_weights(batchElem);
            tpd = std::chrono::system_clock::now();
            taskTimings.executiontime += tpd - tpc;
            tt.executiontime += tpd - tpc;
            timings.findconsensustime += tpd - tpc;

            tpc = std::chrono::system_clock::now();
            cpu_find_consensus(batchElem);
            tpd = std::chrono::system_clock::now();
            taskTimings.executiontime += tpd - tpc;
            tt.executiontime += tpd - tpc;
            timings.findconsensustime += tpd - tpc;

            tpc = std::chrono::system_clock::now();
            cpu_correct(batchElem);
            tpd = std::chrono::system_clock::now();
            taskTimings.executiontime += tpd - tpc;
            tt.executiontime += tpd - tpc;
            timings.correctiontime += tpd - tpc;

            return tt;
        }

        void PileupImage::init_from_batch_elem(const BatchElem& batchElem){

            //determine number of columns in pileup image
            columnProperties.startindex = 0;
            columnProperties.endindex = batchElem.fwdSequenceString.length();

            for(size_t i = 0; i < batchElem.n_unique_candidates; i++){
                const int shift = batchElem.bestAlignments[i].shift;
                columnProperties.startindex = std::min(shift, columnProperties.startindex);
                const int queryEndsAt = batchElem.bestSequences[i]->length() + shift;
                columnProperties.endindex = std::max(queryEndsAt, columnProperties.endindex);
            }

            columnProperties.columnsToCheck = columnProperties.endindex - columnProperties.startindex;
            columnProperties.subjectColumnsBegin_incl = std::max(-columnProperties.startindex,0);
            columnProperties.subjectColumnsEnd_excl = columnProperties.subjectColumnsBegin_incl + batchElem.fwdSequenceString.length();

            resize(columnProperties.columnsToCheck);

            clear();
        }

        void PileupImage::cpu_add_weights(const BatchElem& batchElem){
            const int subjectlength = batchElem.fwdSequenceString.length();

            //add subject weights
            for(int i = 0; i < subjectlength; i++){

                const int globalIndex = columnProperties.subjectColumnsBegin_incl + i;
                assert(globalIndex < max_n_columns);
                double qw = 1.0;
                if(correctionSettings.useQScores)
                    qw *= qscore_to_weight[(unsigned char)(*batchElem.fwdQuality)[i]];

                const char base = batchElem.fwdSequenceString[i];
                switch(base){
                    case 'A': h_Aweights[globalIndex] += qw; h_As[globalIndex] += 1; break;
                    case 'C': h_Cweights[globalIndex] += qw; h_Cs[globalIndex] += 1; break;
                    case 'G': h_Gweights[globalIndex] += qw; h_Gs[globalIndex] += 1; break;
                    case 'T': h_Tweights[globalIndex] += qw; h_Ts[globalIndex] += 1; break;
                    default: std::cout << "this A should not happen in pileup\n"; break;
                }
                h_coverage[globalIndex]++;
            }

            //add candidate weights

        //TIMERSTARTCPU(addcandidates);

        #define WEIGHTMODE 3


        #if WEIGHTMODE == 2

            for(size_t i = 0; i < batchElem.n_unique_candidates; i++){
                const double defaultweight = 1.0 - std::sqrt(batchElem.bestAlignments[i].nOps / (batchElem.bestAlignments[i].overlap * correctionSettings.maxErrorRate));
                const int len = batchElem.bestSequenceStrings[i].length();
                const int freq = batchElem.candidateCountsPrefixSum[i+1] - batchElem.candidateCountsPrefixSum[i];

                for(int j = 0; j < len; j++){
                    const int globalIndex = columnProperties.subjectColumnsBegin_incl + batchElem.bestAlignments[i].shift + j;
                    assert(globalIndex < max_n_columns);
                    double qw = 0.0;
                    for(int f = 0; f < freq; f++){
                        const std::string* scores = batchElem.bestQualities[batchElem.candidateCountsPrefixSum[i] + f];
                        if(correctionSettings.useQScores)
                            qw += qscore_to_weight[(unsigned char)(*scores)[j]];
                        else
                            qw += 1.0;
                    }
                    qw *= defaultweight;
                    const char base = batchElem.bestSequenceStrings[i][j];
                    //const char base = (*batchElem.bestSequences[i])[j];
                    switch(base){
                        case 'A': h_Aweights[globalIndex] += qw; h_As[globalIndex] += freq; break;
                        case 'C': h_Cweights[globalIndex] += qw; h_Cs[globalIndex] += freq; break;
                        case 'G': h_Gweights[globalIndex] += qw; h_Gs[globalIndex] += freq; break;
                        case 'T': h_Tweights[globalIndex] += qw; h_Ts[globalIndex] += freq; break;
                        default: std::cout << "this B should not happen in pileup\n"; break;
                    }
                    h_coverage[globalIndex] += freq;
                }
            }


        #elif WEIGHTMODE == 3

            for(size_t i = 0; i < batchElem.n_unique_candidates; i++){
        //TIMERSTARTCPU(prepare);

                const double defaultweight = 1.0 - std::sqrt(batchElem.bestAlignments[i].nOps / (batchElem.bestAlignments[i].overlap * correctionSettings.maxErrorRate));
                const int len = batchElem.bestSequenceStrings[i].length();
                const int freq = batchElem.candidateCountsPrefixSum[i+1] - batchElem.candidateCountsPrefixSum[i];
                const int defaultcolumnoffset = columnProperties.subjectColumnsBegin_incl + batchElem.bestAlignments[i].shift;
        //TIMERSTOPCPU(prepare);
        //TIMERSTARTCPU(addquality);
                //use h_support as temporary storage to store sum of quality weights
                if(correctionSettings.useQScores){
                    for(int f = 0; f < freq; f++){
                        const std::string* scores = batchElem.bestQualities[batchElem.candidateCountsPrefixSum[i] + f];
                        for(int j = 0; j < len; j++){
                            h_support[j] += qscore_to_weight[(unsigned char)(*scores)[j]];
                        }
                    }
                }else{
                        for(int j = 0; j < len; j++){
                            h_support[j] += freq;
                        }
                }
        //TIMERSTOPCPU(addquality);
        //TIMERSTARTCPU(addbase);
                for(int j = 0; j < len; j++){
                    const int globalIndex = defaultcolumnoffset + j;
                    assert(globalIndex < max_n_columns);
                    assert(j < max_n_columns);
                    assert(i < batchElem.bestSequenceStrings.size());
                    const double qw = h_support[j] * defaultweight;
                    const char base = batchElem.bestSequenceStrings[i][j];
                    //const char base = (*batchElem.bestSequences[i])[j];
                    switch(base){
                        case 'A': h_Aweights[globalIndex] += qw; h_As[globalIndex] += freq;
                        break;
                        case 'C': h_Cweights[globalIndex] += qw; h_Cs[globalIndex] += freq;
                        break;
                        case 'G': h_Gweights[globalIndex] += qw; h_Gs[globalIndex] += freq;
                        break;
                        case 'T': h_Tweights[globalIndex] += qw; h_Ts[globalIndex] += freq;
                        break;
                        default: std::cout << "this C should not happen in pileup\n"; break;
                    }
                    h_coverage[globalIndex] += freq;
                    h_support[j] = 0;
                }
        //TIMERSTOPCPU(addbase);
            }

        #elif
        static_assert(false, "invalid WEIGHTMODE");
        #endif

        //TIMERSTOPCPU(addcandidates);
        }


        void PileupImage::cpu_find_consensus(const BatchElem& batchElem){
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

			const int subjectlength = batchElem.fwdSequenceString.length();

			for(int i = 0; i < subjectlength; i++){
				const int globalIndex = columnProperties.subjectColumnsBegin_incl + i;
				const char base = batchElem.fwdSequenceString[i];
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
					default: std::cout << "this D should not happen in pileup\n"; break;
				}
			}
        }

        void PileupImage::cpu_correct(BatchElem& batchElem){
			properties.avg_support = 0;
			properties.min_support = 1.0;
			properties.max_coverage = 0;
		    properties.min_coverage = std::numeric_limits<int>::max();
			//get stats for subject columns
			for(int i = columnProperties.subjectColumnsBegin_incl; i < columnProperties.subjectColumnsEnd_excl; i++){
				assert(i < columnProperties.columnsToCheck);

				properties.avg_support += h_support[i];
				properties.min_support = std::min(h_support[i], properties.min_support);
				properties.max_coverage = std::max(h_coverage[i], properties.max_coverage);
			    properties.min_coverage = std::min(h_coverage[i], properties.min_coverage);
			}
			const int subjectlength = batchElem.fwdSequenceString.length();
			properties.avg_support /= subjectlength;


			batchElem.correctedSequence.resize(subjectlength);

            const double avg_support_threshold = 1.0-1.0*correctionSettings.errorrate;
            const double min_support_threshold = 1.0-3.0*correctionSettings.errorrate;
            const double min_coverage_threshold = correctionSettings.m / 2.0 * correctionSettings.estimatedCoverage;

            auto isGoodAvgSupport = [&](){
                return properties.avg_support >= avg_support_threshold;
            };
            auto isGoodMinSupport = [&](){
                return properties.min_support >= min_support_threshold;
            };
            auto isGoodMinCoverage = [&](){
                return properties.min_coverage >= min_coverage_threshold;
            };

			//TODO vary parameters
			properties.isHQ = isGoodAvgSupport() && isGoodMinSupport() && isGoodMinCoverage();

            properties.failedAvgSupport = !isGoodAvgSupport();
            properties.failedMinSupport = !isGoodMinSupport();
            properties.failedMinCoverage = !isGoodMinCoverage();

			if(properties.isHQ){
		#if 1
				//correct anchor
				for(int i = 0; i < subjectlength; i++){
					const int globalIndex = columnProperties.subjectColumnsBegin_incl + i;
					batchElem.correctedSequence[i] = h_consensus[globalIndex];
				}
				batchElem.corrected = true;
		#endif
		#if 1
				//correct candidates
				if(correctionSettings.correctCandidates){
                    /*
                        Correct candidates which begin in column range
                        [subjectColumnsBegin_incl - candidate_correction_new_cols, subjectColumnsBegin_incl + candidate_correction_new_cols],
                        and are not longer than subjectlength + candidate_correction_new_cols
                    */

					for(size_t i = 0; i < batchElem.n_unique_candidates; i++){
						const int queryColumnsBegin_incl = batchElem.bestAlignments[i].shift - columnProperties.startindex;
                        const int queryLength = batchElem.bestSequences[i]->length();
                        const int queryColumnsEnd_excl = queryColumnsBegin_incl + queryLength;

						//check range condition and length condition
						if(columnProperties.subjectColumnsBegin_incl - candidate_correction_new_cols <= queryColumnsBegin_incl
                            && queryColumnsBegin_incl <= columnProperties.subjectColumnsBegin_incl + candidate_correction_new_cols
							&& queryColumnsEnd_excl <= columnProperties.subjectColumnsEnd_excl + candidate_correction_new_cols){

							double newColMinSupport = 1.0;
							int newColMinCov = std::numeric_limits<int>::max();
							//check new columns left of subject
							for(int columnindex = columnProperties.subjectColumnsBegin_incl - candidate_correction_new_cols;
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
								columnindex < columnProperties.subjectColumnsEnd_excl + candidate_correction_new_cols
								&& columnindex < columnProperties.columnsToCheck;
								columnindex++){

								newColMinSupport = h_support[columnindex] < newColMinSupport ? h_support[columnindex] : newColMinSupport;
								newColMinCov = h_coverage[columnindex] < newColMinCov ? h_coverage[columnindex] : newColMinCov;
							}

							if(newColMinSupport >= min_support_threshold
								&& newColMinCov >= min_coverage_threshold){

                                std::string correctedString(&h_consensus[queryColumnsBegin_incl], &h_consensus[queryColumnsEnd_excl]);

                                batchElem.correctedCandidates
                                    .emplace_back(
                                        i,
                                        std::move(correctedString)
                                    );
							}
						}
					}
				}
		#endif
			}else{
		#if 1
				//correct anchor
		//TODO vary parameters

				batchElem.correctedSequence = batchElem.fwdSequenceString;

				bool foundAColumn = false;
				for(int i = 0; i < subjectlength; i++){
					const int globalIndex = columnProperties.subjectColumnsBegin_incl + i;

		#if 1
					if(h_support[globalIndex] >= min_support_threshold){
						batchElem.correctedSequence[i] = h_consensus[globalIndex];
						foundAColumn = true;
					}else{
		//#else
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
						if(kregioncoverageisgood && avgsupportkregion >= 1.0-correctionSettings.errorrate){
							batchElem.correctedSequence[i] = h_consensus[globalIndex];
							foundAColumn = true;
						}
					}

				}
		#endif
				}

				batchElem.corrected = foundAColumn;

		#endif
			}
        }

}
