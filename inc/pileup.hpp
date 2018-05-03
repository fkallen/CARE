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

        struct PileupColumnProperties{
            int startindex;
            int endindex;
            int columnsToCheck;
            int subjectColumnsBegin_incl;
            int subjectColumnsEnd_excl;
        };

        struct PileupCorrectionSettings{
            bool useQScores;
            bool correctCandidates;
			int candidate_correction_new_cols;
            int estimatedCoverage;
            double maxErrorRate;
            double errorrate;
            double m;
            double k;
        };

        struct PileupTimings{
            std::chrono::duration<double> findconsensustime{0};
            std::chrono::duration<double> correctiontime{0};
        };
#if 0
		template<class BatchElem_t>
        struct PileupImage{
			using BatchElem = BatchElem_t;

            //buffers
            std::unique_ptr<int[]> h_As;
            std::unique_ptr<int[]> h_Cs;
            std::unique_ptr<int[]> h_Gs;
            std::unique_ptr<int[]> h_Ts;
            std::unique_ptr<double[]> h_Aweights;
            std::unique_ptr<double[]> h_Cweights;
            std::unique_ptr<double[]> h_Gweights;
            std::unique_ptr<double[]> h_Tweights;
            std::unique_ptr<char[]> h_consensus;
            std::unique_ptr<double[]> h_support;
            std::unique_ptr<int[]> h_coverage;
            std::unique_ptr<double[]> h_origWeights;
            std::unique_ptr<int[]> h_origCoverage;

            int max_n_columns = 0; //number of elements per buffer
            int n_columns = 0; //number of used elements per buffer

            PileupProperties properties;
            PileupColumnProperties columnProperties;
            PileupCorrectionSettings correctionSettings;
            PileupTimings timings;
            TaskTimings taskTimings;

			PileupImage(const CorrectionOptions& CO, const GoodAlignmentProperties GAP){
				correctionSettings.useQScores = CO.useQualityScores;
				correctionSettings.correctCandidates = CO.correctCandidates;
				correctionSettings.candidate_correction_new_cols = 0;
				correctionSettings.estimatedCoverage = CO.estimatedCoverage;
				correctionSettings.maxErrorRate = GAP.maxErrorRate;
				correctionSettings.errorrate = CO.estimatedErrorrate;
				correctionSettings.m = CO.m_coverage;
				correctionSettings.k = CO.kmerlength;
			}

			PileupImage(const PileupImage& other){
				*this = other;
			}

			PileupImage(PileupImage&& other){
				*this = std::move(other);
			}

			PileupImage& operator=(const PileupImage& other){
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
				properties = other.properties;
				columnProperties = other.columnProperties;
				correctionSettings = other.correctionSettings;
				timings = other.timings;
				taskTimings = other.taskTimings;

				return *this;
			}


			void resize(int cols){

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

			void clear(){
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

			void destroy(){
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

					properties.avg_support = 0;
					properties.min_support = 0;
					properties.max_coverage = 0;
					properties.min_coverage = 0;
					properties.isHQ = false;
					properties.failedAvgSupport = false;
					properties.failedMinSupport = false;
					properties.failedMinCoverage = false;
			}

			TaskTimings correct_batch_elem(BatchElem& batchElem){
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

			void init_from_batch_elem(const BatchElem& batchElem){

				//determine number of columns in pileup image
				columnProperties.startindex = 0;
				columnProperties.endindex = batchElem.fwdSequenceString.length();

				for(size_t i = 0; i < batchElem.n_unique_candidates; i++){
					const int shift = batchElem.bestAlignments[i]->get_shift();
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

			void cpu_add_weights(const BatchElem& batchElem){
				const int subjectlength = batchElem.fwdSequenceString.length();

				//add subject weights
				for(int i = 0; i < subjectlength; i++){

					const int globalIndex = columnProperties.subjectColumnsBegin_incl + i;
					assert(globalIndex < max_n_columns);
					double qw = 1.0;
					if(BatchElem::canUseQualityScores)
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
					const double defaultweight = 1.0 - std::sqrt(batchElem.bestAlignments[i]->get_nOps()
                                                                / (batchElem.bestAlignments[i]->get_overlap()
                                                                    * correctionSettings.maxErrorRate));
					const int len = batchElem.bestSequenceStrings[i].length();
					const int freq = batchElem.candidateCountsPrefixSum[i+1] - batchElem.candidateCountsPrefixSum[i];

					for(int j = 0; j < len; j++){
						const int globalIndex = columnProperties.subjectColumnsBegin_incl + batchElem.bestAlignments[i]->get_shift() + j;
						assert(globalIndex < max_n_columns);
						double qw = 0.0;
						for(int f = 0; f < freq; f++){
							if(BatchElem::canUseQualityScores){
								const std::string* scores = batchElem.bestQualities[batchElem.candidateCountsPrefixSum[i] + f];
								qw += qscore_to_weight[(unsigned char)(*scores)[j]];
							}
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

					const double defaultweight = 1.0 - std::sqrt(batchElem.bestAlignments[i]->get_nOps()
                                                                / (batchElem.bestAlignments[i]->get_overlap()
                                                                    * correctionSettings.maxErrorRate));
					const int len = batchElem.bestSequenceStrings[i].length();
					const int freq = batchElem.candidateCountsPrefixSum[i+1] - batchElem.candidateCountsPrefixSum[i];
					const int defaultcolumnoffset = columnProperties.subjectColumnsBegin_incl + batchElem.bestAlignments[i]->get_shift();
			//TIMERSTOPCPU(prepare);
			//TIMERSTARTCPU(addquality);
					//use h_support as temporary storage to store sum of quality weights
					if(BatchElem::canUseQualityScores){
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


			void cpu_find_consensus(const BatchElem& batchElem){
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

			void cpu_correct(BatchElem& batchElem){
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
				const double min_coverage_threshold = correctionSettings.m / 6.0 * correctionSettings.estimatedCoverage;

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
							const int queryColumnsBegin_incl = batchElem.bestAlignments[i]->get_shift() - columnProperties.startindex;
							const int queryLength = batchElem.bestSequences[i]->length();
							const int queryColumnsEnd_excl = queryColumnsBegin_incl + queryLength;

							//check range condition and length condition
							if(columnProperties.subjectColumnsBegin_incl - correctionSettings.candidate_correction_new_cols <= queryColumnsBegin_incl
								&& queryColumnsBegin_incl <= columnProperties.subjectColumnsBegin_incl + correctionSettings.candidate_correction_new_cols
								&& queryColumnsEnd_excl <= columnProperties.subjectColumnsEnd_excl + correctionSettings.candidate_correction_new_cols){

								double newColMinSupport = 1.0;
								int newColMinCov = std::numeric_limits<int>::max();
								//check new columns left of subject
								for(int columnindex = columnProperties.subjectColumnsBegin_incl - correctionSettings.candidate_correction_new_cols;
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
									columnindex < columnProperties.subjectColumnsEnd_excl + correctionSettings.candidate_correction_new_cols
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

			#if 0
						if(h_support[globalIndex] >= min_support_threshold){
							batchElem.correctedSequence[i] = h_consensus[globalIndex];
							foundAColumn = true;
						}else{
			#else
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

					//}
			#endif
					}

					batchElem.corrected = foundAColumn;

			#endif
				}
			}
        };

#endif

#if 1

struct CorrectedCandidate2{
    std::uint64_t index;
    std::string sequence;
    CorrectedCandidate2() noexcept{}
    CorrectedCandidate2(std::uint64_t index, const std::string& sequence) noexcept
        : index(index), sequence(sequence){}
};

struct CorrectionResult{
    std::string correctedSequence;
    std::vector<CorrectedCandidate2> correctedCandidates;
    PileupProperties stats;
    bool isCorrected;
};

struct PileupCorrectionSettings2{
    double m;
    double k;
};

// TEST

struct PileupImage2{

    //buffers
    std::unique_ptr<int[]> h_As;
    std::unique_ptr<int[]> h_Cs;
    std::unique_ptr<int[]> h_Gs;
    std::unique_ptr<int[]> h_Ts;
    std::unique_ptr<double[]> h_Aweights;
    std::unique_ptr<double[]> h_Cweights;
    std::unique_ptr<double[]> h_Gweights;
    std::unique_ptr<double[]> h_Tweights;
    std::unique_ptr<char[]> h_consensus;
    std::unique_ptr<double[]> h_support;
    std::unique_ptr<int[]> h_coverage;
    std::unique_ptr<double[]> h_origWeights;
    std::unique_ptr<int[]> h_origCoverage;

    int max_n_columns = 0; //number of elements per buffer
    int n_columns = 0; //number of used elements per buffer

    PileupColumnProperties columnProperties;
    PileupCorrectionSettings2 correctionSettings;
    PileupTimings timings;
    TaskTimings taskTimings;

    PileupImage2(double m_coverage,
                int kmerlength){

        correctionSettings.m = m_coverage;
        correctionSettings.k = kmerlength;
    }

    PileupImage2(const PileupImage2& other){
        *this = other;
    }

    PileupImage2(PileupImage2&& other){
        *this = std::move(other);
    }

    PileupImage2& operator=(const PileupImage2& other){
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
        columnProperties = other.columnProperties;
        correctionSettings = other.correctionSettings;
        timings = other.timings;
        taskTimings = other.taskTimings;

        return *this;
    }

    PileupImage2& operator=(PileupImage2&& other){
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
        timings = other.timings;
        taskTimings = other.taskTimings;

        return *this;
    }


    void resize(int cols){

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

    void clear(){
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
    }

    void destroy(){
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
    }

#if 0
    TaskTimings correct_batch_elem(BatchElem& batchElem){
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
#endif

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
                QualityIter candidateQualitiesEnd) const{

        // add weights for each base in every candidate sequences
        for(auto t = std::make_tuple(alignmentsBegin, candidateSequencesBegin, candidateCountsBegin, candidateQualitiesBegin);
            std::get<0>(t) != alignmentsEnd;
            std::get<0>(t)++, std::get<1>(t)++, std::get<2>(t)++/*quality iter is incremented in loop body*/){

            auto& alignmentiter = std::get<0>(t);
            auto& sequenceiter = std::get<1>(t);
            auto& countiter = std::get<2>(t);
            auto& candidateQualityiter = std::get<3>(t);

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

    void cpu_find_consensus_internal() const{
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
                                        double min_coverage_threshold) const{

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

        auto isGoodAvgSupport = [&](){
            return result.stats.avg_support >= avg_support_threshold;
        };
        auto isGoodMinSupport = [&](){
            return result.stats.min_support >= min_support_threshold;
        };
        auto isGoodMinCoverage = [&](){
            return result.stats.min_coverage >= min_coverage_threshold;
        };

        //TODO vary parameters
        result.stats.isHQ = isGoodAvgSupport() && isGoodMinSupport() && isGoodMinCoverage();

        result.stats.failedAvgSupport = !isGoodAvgSupport();
        result.stats.failedMinSupport = !isGoodMinSupport();
        result.stats.failedMinCoverage = !isGoodMinCoverage();

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

    #endif
        }

        return result;
    }

    /*
        AlignmentIter: Iterator to Alignment pointer
        SequenceIter: Iterator to std::string
    */
    template<class AlignmentIter, class SequenceIter>
    std::vector<CorrectedCandidate2> cpu_correct_candidates_internal(
                AlignmentIter alignmentsBegin,
                AlignmentIter alignmentsEnd,
                SequenceIter candidateSequencesBegin,
                SequenceIter candidateSequencesEnd,
                double avg_support_threshold,
                double min_support_threshold,
                double min_coverage_threshold,
                int new_columns_to_correct) const{

        std::vector<CorrectedCandidate2> result;
        result.reserve(std::distance(candidateSequencesBegin, candidateSequencesEnd));

        /*
            Correct candidates which begin in column range
            [subjectColumnsBegin_incl - candidate_correction_new_cols, subjectColumnsBegin_incl + candidate_correction_new_cols],
            and are not longer than subjectlength + candidate_correction_new_cols
        */

        for(auto t = std::make_tuple(alignmentsBegin, candidateSequencesBegin, 0);
            std::get<0>(t) != alignmentsEnd;
            std::get<0>(t)++, std::get<1>(t)++, std::get<2>(t)++){

            auto& alignmentiter = std::get<0>(t);
            auto& sequenceiter = std::get<1>(t);
            auto i = std::get<2>(t);

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
                int new_columns_to_correct) const{

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

#endif

}

}
#endif
