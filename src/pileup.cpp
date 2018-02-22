#include "../inc/pileup.hpp"
#include "../inc/alignment.hpp"
#include "../inc/hammingtools.hpp"
#include "../inc/ganja/hpc_helpers.cuh"
#include "../inc/cudareduce.cuh"

#include <vector>
#include <string>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>

#ifdef __NVCC__
#include <cooperative_groups.h>
#include <cublas_v2.h>

#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/device_vector.h>

using namespace cooperative_groups;
#endif

namespace hammingtools{

	namespace correction{

		constexpr int candidate_correction_new_cols = 0;

		double	qscore_to_error_prob[256];
		double	qscore_to_weight[256];

		#ifdef __NVCC__
		__device__ double d_qscore_to_weight[256];
		#endif

		void init_once(){

			constexpr int ASCII_BASE = 33;
			constexpr double MIN_WEIGHT = 0.001;

			for(int i = 0; i < 256; i++){
				if(i < ASCII_BASE)
					qscore_to_error_prob[i] = 1.0;
				else
					qscore_to_error_prob[i] = std::pow(10.0, -(i-ASCII_BASE)/10.0);
			}

			for(int i = 0; i < 256; i++){
				qscore_to_weight[i] = std::max(MIN_WEIGHT, 1.0 - qscore_to_error_prob[i]);
			}

			#ifdef __NVCC__
				int devices;
				cudaGetDeviceCount(&devices); CUERR;
				for(int i = 0; i < devices; i++){
					cudaSetDevice(i);
					cudaMemcpyToSymbol(d_qscore_to_weight, qscore_to_weight, 256*sizeof(double)); CUERR;
				}
			#endif
		}

		std::tuple<int,std::chrono::duration<double>,std::chrono::duration<double>>
		cpu_pileup_all_in_one(const CorrectionBuffers* buffers, std::string& subject,
						int nQueries,
						std::vector<std::string>& queries,
						const std::vector<AlignResultCompact>& alignments,
						const std::string& subjectqualityScores,
						const std::vector<const std::string*>& queryqualityScores,
						const std::vector<int>& frequenciesPrefixSum,
						const int startindex, const int endindex,
						const int columnsToCheck, const int subjectColumnsBegin_incl, const int subjectColumnsEnd_excl,
						double maxErrorRate,
						bool useQScores,
						std::vector<bool>& correctedQueries,
						const bool correctQueries,
						int estimatedCoverage,
						double errorrate,
						double m,
						int k){

			std::chrono::duration<double> majorityvotetime(0);
			std::chrono::duration<double> basecorrectiontime(0);
			std::chrono::time_point<std::chrono::system_clock> tpa, tpb;

			tpa = std::chrono::system_clock::now();

			int status = 0;

			//add subject weights
			for(int i = 0; i < int(subject.length()); i++){
				const int globalIndex = subjectColumnsBegin_incl + i;
				double qw = 1.0;
				if(useQScores)
					qw *= qscore_to_weight[(unsigned char)subjectqualityScores[i]];
				switch(subject[i]){
					case 'A': buffers->h_Aweights[globalIndex] += qw; buffers->h_As[globalIndex] += 1; break;
					case 'C': buffers->h_Cweights[globalIndex] += qw; buffers->h_Cs[globalIndex] += 1; break;
					case 'G': buffers->h_Gweights[globalIndex] += qw; buffers->h_Gs[globalIndex] += 1; break;
					case 'T': buffers->h_Tweights[globalIndex] += qw; buffers->h_Ts[globalIndex] += 1; break;
					default: std::cout << "this should not happen in pileup\n"; break;
				}
				buffers->h_coverage[globalIndex]++;
			}

			//add candidate weights

#define WEIGHTMODE 3

#if WEIGHTMODE == 0
			for(int i = 0; i < nQueries; i++){
				const double defaultweight = 1.0 - std::sqrt(alignments[i].nOps / (alignments[i].overlap * maxErrorRate));

				for(int j = 0; j < int(queries[i].length()); j++){
					const int globalIndex = subjectColumnsBegin_incl + alignments[i].shift + j;

					for(int f = 0; f < frequenciesPrefixSum[i+1] - frequenciesPrefixSum[i]; f++){
						const int qualityindex = frequenciesPrefixSum[i] + f;
						double qw = defaultweight;
						if(useQScores)
							qw *= qscore_to_weight[(unsigned char)(*queryqualityScores[qualityindex])[j]];

						switch(queries[i][j]){
							case 'A': buffers->h_Aweights[globalIndex] += qw; buffers->h_As[globalIndex] += 1; break;
							case 'C': buffers->h_Cweights[globalIndex] += qw; buffers->h_Cs[globalIndex] += 1; break;
							case 'G': buffers->h_Gweights[globalIndex] += qw; buffers->h_Gs[globalIndex] += 1; break;
							case 'T': buffers->h_Tweights[globalIndex] += qw; buffers->h_Ts[globalIndex] += 1; break;
							default: std::cout << "this should not happen in pileup\n"; break;
						}
						buffers->h_coverage[globalIndex]++;
					}
				}
			}
#elif WEIGHTMODE == 1

			for(int i = 0; i < nQueries; i++){
				const double defaultweight = 1.0 - std::sqrt(alignments[i].nOps / (alignments[i].overlap * maxErrorRate));
				const int prefix = frequenciesPrefixSum[i];
				const int freq = frequenciesPrefixSum[i+1] - prefix;
				const int len = queries[i].length();
				const int defaultcolumnoffset = subjectColumnsBegin_incl + alignments[i].shift;
				for(int f = 0; f < freq; f++){
					const int qualityindex = prefix + f;
					if(useQScores){
						for(int j = 0; j < len; j++){
							//use h_support as temporary storage
							buffers->h_support[j] = defaultweight * qscore_to_weight[(unsigned char)(*queryqualityScores[qualityindex])[j]];
						}
					}else{
						for(int j = 0; j < len; j++){
							buffers->h_support[j] = defaultweight;
						}
					}

					for(int j = 0; j <len; j++){
						const double qw = buffers->h_support[j];
						const char base = queries[i][j];
						const int globalIndex = defaultcolumnoffset + j;
						switch(base){
							case 'A': buffers->h_Aweights[globalIndex] += qw; buffers->h_As[globalIndex] += 1; break;
							case 'C': buffers->h_Cweights[globalIndex] += qw; buffers->h_Cs[globalIndex] += 1; break;
							case 'G': buffers->h_Gweights[globalIndex] += qw; buffers->h_Gs[globalIndex] += 1; break;
							case 'T': buffers->h_Tweights[globalIndex] += qw; buffers->h_Ts[globalIndex] += 1; break;
							default: std::cout << "this should not happen in pileup\n"; break;
						}
						//buffers->h_coverage[globalIndex]++;
					}
				}
				for(int j = 0; j < len; j++){
					const int globalIndex = defaultcolumnoffset + j;
					buffers->h_coverage[globalIndex] += freq;
				}
			}

#elif WEIGHTMODE == 2

			for(int i = 0; i < nQueries; i++){
				const double defaultweight = 1.0 - std::sqrt(alignments[i].nOps / (alignments[i].overlap * maxErrorRate));
				const int prefix = frequenciesPrefixSum[i];
				const int freq = frequenciesPrefixSum[i+1] - prefix;
				const int len = queries[i].length();

				for(int j = 0; j < len; j++){
					const int globalIndex = subjectColumnsBegin_incl + alignments[i].shift + j;
					double qw = 0.0;
					for(int f = 0; f < freq; f++){
						const int qualityindex = prefix + f;
						if(useQScores)
							qw += qscore_to_weight[(unsigned char)(*queryqualityScores[qualityindex])[j]];
						else
							qw += 1.0;
					}
					qw *= defaultweight;

					switch(queries[i][j]){
						case 'A': buffers->h_Aweights[globalIndex] += qw; buffers->h_As[globalIndex] += freq; break;
						case 'C': buffers->h_Cweights[globalIndex] += qw; buffers->h_Cs[globalIndex] += freq; break;
						case 'G': buffers->h_Gweights[globalIndex] += qw; buffers->h_Gs[globalIndex] += freq; break;
						case 'T': buffers->h_Tweights[globalIndex] += qw; buffers->h_Ts[globalIndex] += freq; break;
						default: std::cout << "this should not happen in pileup\n"; break;
					}
					buffers->h_coverage[globalIndex] += freq;
				}
			}
#elif WEIGHTMODE == 3

			for(int i = 0; i < nQueries; i++){
				const double defaultweight = 1.0 - std::sqrt(alignments[i].nOps / (alignments[i].overlap * maxErrorRate));
				const int prefix = frequenciesPrefixSum[i];
				const int freq = frequenciesPrefixSum[i+1] - prefix;
				const int len = queries[i].length();
				const int defaultcolumnoffset = subjectColumnsBegin_incl + alignments[i].shift;
				//use h_support as temporary storage to store sum of quality weights
				for(int f = 0; f < freq; f++){
					const int qualityindex = prefix + f;
					if(useQScores){
						for(int j = 0; j < len; j++){
							buffers->h_support[j] += qscore_to_weight[(unsigned char)(*queryqualityScores[qualityindex])[j]];
						}
					}else{
						for(int j = 0; j < len; j++){
							buffers->h_support[j] += 1.0;
						}
					}
				}

				for(int j = 0; j < len; j++){
					const double qw = buffers->h_support[j] * defaultweight;
					const char base = queries[i][j];
					const int globalIndex = defaultcolumnoffset + j;
					switch(base){
						case 'A': buffers->h_Aweights[globalIndex] += qw; buffers->h_As[globalIndex] += freq; break;
						case 'C': buffers->h_Cweights[globalIndex] += qw; buffers->h_Cs[globalIndex] += freq; break;
						case 'G': buffers->h_Gweights[globalIndex] += qw; buffers->h_Gs[globalIndex] += freq; break;
						case 'T': buffers->h_Tweights[globalIndex] += qw; buffers->h_Ts[globalIndex] += freq; break;
						default: std::cout << "this should not happen in pileup\n"; break;
					}
					buffers->h_coverage[globalIndex] += freq;
					buffers->h_support[j] = 0;
				}
			}
#elif
	static_assert(false, "invalid WEIGHTMODE");
#endif


			//find consensus and support in each column
			for(int i = 0; i < columnsToCheck; i++){
				char cons = 'A';
				double consWeight = buffers->h_Aweights[i];
				if(buffers->h_Cweights[i] > consWeight){
					cons = 'C';
					consWeight = buffers->h_Cweights[i];
				}
				if(buffers->h_Gweights[i] > consWeight){
					cons = 'G';
					consWeight = buffers->h_Gweights[i];
				}
				if(buffers->h_Tweights[i] > consWeight){
					cons = 'T';
					consWeight = buffers->h_Tweights[i];
				}
				buffers->h_consensus[i] = cons;

				const double columnWeight = buffers->h_Aweights[i] + buffers->h_Cweights[i] + buffers->h_Gweights[i] + buffers->h_Tweights[i];
				buffers->h_support[i] = consWeight / columnWeight;

			}

			    for(int i = 0; i < int(subject.length()); i++){
				const int globalIndex = subjectColumnsBegin_incl + i;
				switch(subject[i]){
							case 'A':   buffers->h_origCoverage[globalIndex] = buffers->h_As[globalIndex];
				                buffers->h_origWeights[globalIndex] = buffers->h_Aweights[globalIndex];
				                break;
							case 'C':   buffers->h_origCoverage[globalIndex] = buffers->h_Cs[globalIndex];
				                buffers->h_origWeights[globalIndex] = buffers->h_Cweights[globalIndex];
				                break;
							case 'G':   buffers->h_origCoverage[globalIndex] = buffers->h_Gs[globalIndex];
				                buffers->h_origWeights[globalIndex] = buffers->h_Gweights[globalIndex];
				                break;
							case 'T':   buffers->h_origCoverage[globalIndex] = buffers->h_Ts[globalIndex];
				                buffers->h_origWeights[globalIndex] = buffers->h_Tweights[globalIndex];
				                break;
							default: std::cout << "this D should not happen in pileup\n"; break;
						}
			    }

            /*std::cout << "cons: ";
            for(int i = 0; i < columnsToCheck; i++)
                std::cout << buffers->h_consensus[i];
            std::cout << std::endl;
            std::cout << "sup: ";
            for(int i = 0; i < columnsToCheck; i++)
                std::cout << buffers->h_support[i];
            std::cout << std::endl;*/

			tpb = std::chrono::system_clock::now();

			majorityvotetime += tpb - tpa;

			tpa = std::chrono::system_clock::now();

			double avg_support = 0;
			double min_support = 1.0;
			int max_coverage = 0;
			int min_coverage = std::numeric_limits<int>::max();
			//get stats for subject columns
			for(int i = subjectColumnsBegin_incl; i < subjectColumnsEnd_excl; i++){
				assert(i < columnsToCheck);

				avg_support += buffers->h_support[i];
				min_support = buffers->h_support[i] < min_support? buffers->h_support[i] : min_support;
				max_coverage = buffers->h_coverage[i] > max_coverage ? buffers->h_coverage[i] : max_coverage;
				min_coverage = buffers->h_coverage[i] < min_coverage ? buffers->h_coverage[i] : min_coverage;
				//printf("i %d sup %f cov %d\n", i, support[i], coverage[i]);
			}
			avg_support /= subject.length();

			/*for(int i = 0; i < columnsToCheck; i++){
				printf("%2c ", consensus[i]);
			}
			printf("\n");
			for(int i = 0; i < columnsToCheck; i++){
				printf("%2d ", coverage[i]);
			}
			printf("\n");
			printf("%f %f %d %d\n", avg_support, min_support, max_coverage, min_coverage);*/


		#if 0
			std::cout << "avgsup " << avg_support << " >= " << (1-errorrate) << '\n';
			std::cout << "minsup " << min_support << " >= " << (1-2*errorrate) << '\n';
			std::cout << "mincov " << min_coverage << " >= " << (m) << '\n';
			std::cout << "maxcov " << max_coverage << " <= " << (3*m) << '\n';
		#endif

			//TODO vary parameters
			bool isHQ = avg_support >= 1.0-errorrate
				 && min_support >= 1.0-3.0*errorrate
				 && min_coverage >= m / 2.0 * estimatedCoverage;

			if(isHQ){
		#if 1
				//correct anchor
				for(int i = 0; i < int(subject.length()); i++){
					const int globalIndex = subjectColumnsBegin_incl + i;
					subject[i] = buffers->h_consensus[globalIndex];
				}
		#endif
		#if 0
				//correct candidates
				if(correctQueries){

					for(int i = 0; i < nQueries; i++){
						int queryColumnsBegin_incl = alignments[i].shift - startindex;
						bool queryWasCorrected = false;
						//correct candidates which are shifted by at most candidate_correction_new_cols columns relative to subject
						if(queryColumnsBegin_incl >= subjectColumnsBegin_incl - candidate_correction_new_cols
							&& subjectColumnsEnd_excl + candidate_correction_new_cols >= queryColumnsBegin_incl + int(queries[i].length())){

							double newColMinSupport = 1.0;
							int newColMinCov = std::numeric_limits<int>::max();
							//check new columns left of subject
							for(int columnindex = subjectColumnsBegin_incl - candidate_correction_new_cols;
								columnindex < subjectColumnsBegin_incl;
								columnindex++){

								assert(columnindex < columnsToCheck);
								if(queryColumnsBegin_incl <= columnindex){
									newColMinSupport = buffers->h_support[columnindex] < newColMinSupport ? buffers->h_support[columnindex] : newColMinSupport;
									newColMinCov = buffers->h_coverage[columnindex] < newColMinCov ? buffers->h_coverage[columnindex] : newColMinCov;
								}
							}
							//check new columns right of subject
							for(int columnindex = subjectColumnsEnd_excl;
								columnindex < subjectColumnsEnd_excl + candidate_correction_new_cols
								&& columnindex < columnsToCheck;
								columnindex++){

								newColMinSupport = buffers->h_support[columnindex] < newColMinSupport ? buffers->h_support[columnindex] : newColMinSupport;
								newColMinCov = buffers->h_coverage[columnindex] < newColMinCov ? buffers->h_coverage[columnindex] : newColMinCov;
							}

							if(newColMinSupport >= 1-3*errorrate
								&& newColMinCov >= m / 2.0 * estimatedCoverage){
								//assert(subjectColumnsBegin_incl == queryColumnsBegin_incl && subject.length() == queries[i].length());

								for(int j = 0; j < int(queries[i].length()); j++){
									int columnindex = queryColumnsBegin_incl + j;
									queries[i][j] = buffers->h_consensus[columnindex];
									queryWasCorrected = true;
								}
							}
						}
						if(queryWasCorrected){
							correctedQueries[i] = true;
						}
					}
				}
		#endif
			}else{
				if(avg_support < 1.0-errorrate)
					status |= (1 << 0);
				if(min_support < 1.0-3.0*errorrate)
					status |= (1 << 1);
				if(min_coverage < m / 2.0 * estimatedCoverage)
					status |= (1 << 2);
		#if 0
				//correct anchor
//TODO vary parameters
				bool foundAColumn = false;
				for(int i = 0; i < int(subject.length()); i++){
					const int globalIndex = subjectColumnsBegin_incl + i;

#if 1
					if(buffers->h_support[globalIndex] >= 1.0-3.0*errorrate){
						subject[i] = buffers->h_consensus[globalIndex];
						foundAColumn = true;
					}else{
						if(buffers->h_support[globalIndex] > 0.5 && buffers->h_origCoverage[globalIndex] < m / 2.0 * estimatedCoverage){
							double avgsupportkregion = 0;
							int c = 0;
							bool kregioncoverageisgood = true;
							for(int j = i - k/2; j <= i + k/2 && kregioncoverageisgood; j++){
								if(j != i && j >= 0 && j < int(subject.length())){
									avgsupportkregion += buffers->h_support[subjectColumnsBegin_incl + j];
									kregioncoverageisgood &= (buffers->h_coverage[subjectColumnsBegin_incl + j] >= m / 2.0 * estimatedCoverage);
									c++;
								}
							}
							if(kregioncoverageisgood && avgsupportkregion / c >= 1.0-errorrate){
								subject[i] = buffers->h_consensus[globalIndex];
								foundAColumn = true;
							}
						}
					}
#else


#if 1
					if(buffers->h_support[globalIndex] >= 1.0-3.0*errorrate){
						subject[i] = buffers->h_consensus[globalIndex];
						foundAColumn = true;
					}
#else
					if(buffers->h_support[globalIndex] > 0.5 && buffers->h_origCoverage[globalIndex] < m / 2.0 * estimatedCoverage){
						double avgsupportkregion = 0;
						int c = 0;
						bool kregioncoverageisgood = true;
						for(int j = i - k/2; j <= i + k/2 && kregioncoverageisgood; j++){
							if(j != i && j >= 0 && j < int(subject.length())){
								avgsupportkregion += buffers->h_support[subjectColumnsBegin_incl + j];
								kregioncoverageisgood &= (buffers->h_coverage[subjectColumnsBegin_incl + j] >= m / 2.0 * estimatedCoverage);
								c++;
							}
						}
						if(kregioncoverageisgood && avgsupportkregion / c >= 1.0-errorrate){
							subject[i] = buffers->h_consensus[globalIndex];
							foundAColumn = true;
						}
					}
#endif

#endif
				}

				if(!foundAColumn)
					status |= (1 << 3);
		#endif
			}

			tpb = std::chrono::system_clock::now();

			basecorrectiontime += tpb - tpa;

			return std::tie(status, majorityvotetime, basecorrectiontime);
		}

		void cpu_pileup_create(const CorrectionBuffers* buffers, std::string& subject,
						std::vector<std::string>& queries,
						const std::string& subjectqualityScores,
						const std::vector<const std::string*>& queryqualityScores,
						const std::vector<AlignResultCompact>& alignments,
						const std::vector<int>& frequenciesPrefixSum,
						const int subjectColumnsBegin_incl){
			const int nQueries = queries.size();

			//clear sequences
			std::memset(buffers->h_pileup, 'F', sizeof(char) * buffers->max_n_columns *  buffers->max_n_sequences);
			//clear qualityscores
			std::memset(buffers->h_qual_pileup, 0, sizeof(char) * buffers->max_n_columns *  buffers->max_n_qualityscores);

			//copy subject to buffer
			std::memcpy(buffers->h_pileup + 0 * buffers->max_n_columns + subjectColumnsBegin_incl, subject.data(), sizeof(char) * subject.length());
			//copy subject quality to buffer
			std::memcpy(buffers->h_qual_pileup + 0 * buffers->max_n_columns + subjectColumnsBegin_incl,
					subjectqualityScores.data(),
					sizeof(char) * subjectqualityScores.length());

			for(int i = 0; i < nQueries; i++){
				const int freqSum = frequenciesPrefixSum[i];
				const int freqSumNext = frequenciesPrefixSum[i+1];
				const int freq = freqSumNext - freqSum;
				const int colOffset = subjectColumnsBegin_incl + alignments[i].shift;
				//copy queries[i] to buffer
				for(int f = 0; f < freq; f++){
					const int row = 1 + freqSum + f;
					std::memcpy(buffers->h_pileup + row * buffers->max_n_columns
									+ colOffset,
							queries[i].data(),
							sizeof(char) * queries[i].length());
					std::memcpy(buffers->h_qual_pileup + row * buffers->max_n_columns
									+ colOffset,
							queryqualityScores[i]->data(),
							sizeof(char) * queryqualityScores[i]->length());
				}
			}
		}

		void cpu_pileup_vote(CorrectionBuffers* buffers, const std::vector<AlignResultCompact>& alignments,
						const std::vector<int>& frequenciesPrefixSum,
						const double maxErrorRate,
						const bool useQScores,
						const int subjectColumnsBegin_incl, const int subjectColumnsEnd_excl){


			int unique_sequence_index = 0;
			//row in original pileup
			for(int row = 0; row < buffers->n_sequences; row += 1){
				if(row-1 >= frequenciesPrefixSum[unique_sequence_index+1]){
					unique_sequence_index++;
				}
				//col in original pileup
				for(int column = 0; column < buffers->n_columns; column += 1){
					const int cellindex = row * buffers->max_n_columns + column;
					const char base = buffers->h_pileup[cellindex];
					const char qual = buffers->h_qual_pileup[cellindex];
					const double qualweight = qscore_to_weight[(unsigned char)qual];

					if(base != 'F'){
						if(row == 0){
							double qw = 1.0;
							if(useQScores){
								qw *= qualweight;
							}

							switch(base){
								case 'A': buffers->h_Aweights[column] += qw; buffers->h_As[column] += 1; break;
								case 'C': buffers->h_Cweights[column] += qw; buffers->h_Cs[column] += 1; break;
								case 'G': buffers->h_Gweights[column] += qw; buffers->h_Gs[column] += 1; break;
								case 'T': buffers->h_Tweights[column] += qw; buffers->h_Ts[column] += 1; break;
								default:  break; //all other chars are invalid and not counted
							}
						}else{
							const AlignResultCompact& arc = alignments[unique_sequence_index];
							const double defaultweight = 1.0 - sqrt(arc.nOps / (arc.overlap * maxErrorRate));

							double qw = defaultweight;
							if(useQScores){
								qw *= qualweight;
							}

							switch(base){
								case 'A': buffers->h_Aweights[column] += qw; buffers->h_As[column] += 1; break;
								case 'C': buffers->h_Cweights[column] += qw; buffers->h_Cs[column] += 1; break;
								case 'G': buffers->h_Gweights[column] += qw; buffers->h_Gs[column] += 1; break;
								case 'T': buffers->h_Tweights[column] += qw; buffers->h_Ts[column] += 1; break;
								default:  break; //all other chars are invalid and not counted
							}
						}
					}
				}

			}

			//find consensus
			for(int column = 0; column < buffers->n_columns; column += 1){
				char cons = 'F';
				double consWeight = 0;
				if(buffers->h_Aweights[column] > consWeight){
					cons = 'A';
					consWeight = buffers->h_Aweights[column];
				}
				if(buffers->h_Cweights[column] > consWeight){
					cons = 'C';
					consWeight = buffers->h_Cweights[column];
				}
				if(buffers->h_Gweights[column] > consWeight){
					cons = 'G';
					consWeight = buffers->h_Gweights[column];
				}
				if(buffers->h_Tweights[column] > consWeight){
					cons = 'T';
					consWeight = buffers->h_Tweights[column];
				}

				const double columnWeight = buffers->h_Aweights[column] + buffers->h_Cweights[column]
								+ buffers->h_Gweights[column] + buffers->h_Tweights[column];
				const int coverage = buffers->h_As[column] + buffers->h_Cs[column]
								+ buffers->h_Gs[column] + buffers->h_Ts[column];

				buffers->h_coverage[column] = coverage;
				buffers->h_consensus[column] = cons;
				buffers->h_support[column] = consWeight / columnWeight;

				if(column >= subjectColumnsBegin_incl && column < subjectColumnsEnd_excl){
					switch(cons){
						case 'A': 	buffers->h_origCoverage[column] = buffers->h_As[column];
								buffers->h_origWeights[column] = buffers->h_Aweights[column]; break;
						case 'C': 	buffers->h_origCoverage[column] = buffers->h_Cs[column];
								buffers->h_origWeights[column] = buffers->h_Cweights[column]; break;
						case 'G': 	buffers->h_origCoverage[column] = buffers->h_Gs[column];
								buffers->h_origWeights[column] = buffers->h_Gweights[column]; break;
						case 'T': 	buffers->h_origCoverage[column] = buffers->h_Ts[column];
								buffers->h_origWeights[column] = buffers->h_Tweights[column]; break;
						default:  printf("this should never happen in a subject column. w = %f\n", columnWeight); break;
					}
				}
			}

			double avg_support = 0;
			double min_support = 1.0;
			int max_coverage = 0;
			int min_coverage = std::numeric_limits<int>::max();
			//get stats for subject columns
			for(int i = subjectColumnsBegin_incl; i < subjectColumnsEnd_excl; i++){
				const double sup = buffers->h_support[i];
				const int cov = buffers->h_coverage[i];

				avg_support += sup;
				min_support = sup < min_support? sup : min_support;
				max_coverage = cov > max_coverage ? cov : max_coverage;
				min_coverage = cov < min_coverage ? cov : min_coverage;
				//printf("i %d sup %f cov %d\n", i, support[i], coverage[i]);
			}
			avg_support /= buffers->h_lengths[0];

			buffers->avg_support = avg_support;
			buffers->min_support = min_support;
			buffers->max_coverage = max_coverage;
			buffers->min_coverage = min_coverage;
		}

		int cpu_pileup_correct(CorrectionBuffers* buffers, const std::vector<AlignResultCompact>& alignments, const std::vector<int>& frequenciesPrefixSum,
						const int columnsToCheck,
						const int subjectColumnsBegin_incl, const int subjectColumnsEnd_excl,
						const int startindex, const int endindex,
						const double errorrate, const int estimatedCoverage, const double m,
						const bool correctQueries, int k, std::vector<bool>& correctedQueries){

			const bool isHQ = buffers->avg_support >= 1.0-errorrate
				 && buffers->min_support >= 1.0-3.0*errorrate
				 && buffers->min_coverage >= m / 2.0 * estimatedCoverage;

			int status = 0;

			if(isHQ){
		#if 1
				//correct anchor
				std::memcpy(	buffers->h_pileup + subjectColumnsBegin_incl,
						buffers->h_consensus + subjectColumnsBegin_incl,
						sizeof(char) * buffers->h_lengths[0]);

		#endif
		#if 0
				//correct candidates
				if(correctQueries){

					for(int i = 0; i < buffers->n_sequences - 1; i++){
						const int row = 1 + frequenciesPrefixSum[i];
						int queryColumnsBegin_incl = alignments[i].shift - startindex;
						bool queryWasCorrected = false;
						//correct candidates which are shifted by at most candidate_correction_new_cols columns relative to subject
						if(queryColumnsBegin_incl >= subjectColumnsBegin_incl - candidate_correction_new_cols
							&& subjectColumnsEnd_excl + candidate_correction_new_cols >= queryColumnsBegin_incl + buffers->h_lengths[1+i]){

							double newColMinSupport = 1.0;
							int newColMinCov = std::numeric_limits<int>::max();
							//check new columns left of subject
							for(int columnindex = subjectColumnsBegin_incl - candidate_correction_new_cols;
								columnindex < subjectColumnsBegin_incl;
								columnindex++){

								assert(columnindex < columnsToCheck);
								if(queryColumnsBegin_incl <= columnindex){
									newColMinSupport = buffers->h_support[columnindex] < newColMinSupport ? buffers->h_support[columnindex] : newColMinSupport;
									newColMinCov = buffers->h_coverage[columnindex] < newColMinCov ? buffers->h_coverage[columnindex] : newColMinCov;
								}
							}
							//check new columns right of subject
							for(int columnindex = subjectColumnsEnd_excl;
								columnindex < subjectColumnsEnd_excl + candidate_correction_new_cols
								&& columnindex < columnsToCheck;
								columnindex++){

								newColMinSupport = buffers->h_support[columnindex] < newColMinSupport ? buffers->h_support[columnindex] : newColMinSupport;
								newColMinCov = buffers->h_coverage[columnindex] < newColMinCov ? buffers->h_coverage[columnindex] : newColMinCov;
							}

							if(newColMinSupport >= 1-3*errorrate
								&& newColMinCov >= m / 2.0 * estimatedCoverage){

								std::memcpy(	buffers->h_pileup + row * buffers->max_n_columns + queryColumnsBegin_incl,
										buffers->h_consensus + queryColumnsBegin_incl,
										sizeof(char) * buffers->h_lengths[row]);
							}
						}
						if(queryWasCorrected){
							correctedQueries[i] = true;
						}
					}
				}
		#endif
			}else{
				if(buffers->avg_support < 1.0-errorrate)
					status |= (1 << 0);
				if(buffers->min_support < 1.0-3.0*errorrate)
					status |= (1 << 1);
				if(buffers->min_coverage < m / 2.0 * estimatedCoverage)
					status |= (1 << 2);
		#if 1
				//correct anchor
				bool foundAColumn = false;
				for(int i = 0; i < buffers->h_lengths[0]; i++){
					const int globalIndex = subjectColumnsBegin_incl + i;

					if(buffers->h_support[globalIndex] >= 1.0-3.0*errorrate){
						buffers->h_pileup[globalIndex] = buffers->h_consensus[globalIndex]; //assign to subject[i]
						foundAColumn = true;
					}else{
						if(buffers->h_support[globalIndex] > 0.5 && buffers->h_origCoverage[globalIndex] < m / 2.0 * estimatedCoverage){
							double avgsupportkregion = 0;
							int c = 0;
							bool kregioncoverageisgood = true;
							for(int j = i - k/2; j <= i + k/2 && kregioncoverageisgood; j++){
								if(j != i && j >= 0 && j < buffers->h_lengths[0]){
									avgsupportkregion += buffers->h_support[subjectColumnsBegin_incl + j];
									kregioncoverageisgood &= (buffers->h_coverage[subjectColumnsBegin_incl + j] >= m / 2.0 * estimatedCoverage);
									c++;
								}
							}
							if(kregioncoverageisgood && avgsupportkregion / c >= 1.0-errorrate){
								buffers->h_pileup[globalIndex] = buffers->h_consensus[globalIndex]; //assign to subject[i]
								foundAColumn = true;
							}
						}
					}
				}

				if(!foundAColumn)
					status |= (1 << 3);
		#endif
			}

			return status;
		}



#ifdef __NVCC__



		// convert a linear index to a linear index in the transpose
		struct transpose_index : public thrust::unary_function<size_t,size_t>
		{
			size_t m, n;

			__host__ __device__
			transpose_index(size_t _m, size_t _n) : m(_m), n(_n) {}

			__host__ __device__
			size_t operator()(size_t linear_index)
			{
				size_t i = linear_index / n;
				size_t j = linear_index % n;

				return m * j + i;
			}
		};

		void gpu_pileup_transpose(const CorrectionBuffers* buffers){
			//copy h_pileup to d_pileup, then store transposed in d_pileup_transposed

			const size_t rows = buffers->max_n_sequences;
			const size_t cols = buffers->max_n_columns;

			cudaSetDevice(buffers->deviceId);
			cudaMemcpyAsync(buffers->d_pileup, buffers->h_pileup, sizeof(char) * rows * cols, H2D, buffers->stream); CUERR;
			cudaStreamSynchronize(buffers->stream);

			thrust::device_ptr<char> d_pileup_ptr = thrust::device_pointer_cast(buffers->d_pileup);
			thrust::device_ptr<char> d_pileup_transposed_ptr = thrust::device_pointer_cast(buffers->d_pileup_transposed);

			thrust::counting_iterator<size_t> indices(0);

			thrust::gather(thrust::cuda::par.on(buffers->stream),
					thrust::make_transform_iterator(indices, transpose_index(cols, rows)),
					thrust::make_transform_iterator(indices, transpose_index(cols, rows)) + rows * cols,
					d_pileup_ptr, d_pileup_transposed_ptr);

			/*printf("before transpose\n");
			for(int i = 0; i < buffers->n_sequences; i++){
				for(int j = 0; j < buffers->n_columns; j++){
					printf("%c", buffers->h_pileup[i * cols + j]);
				}
				printf("\n");
			}
			printf("\n");*/

			/*cudaMemcpyAsync(buffers->h_pileup, buffers->d_pileup_transposed, sizeof(char) * rows * cols, D2H, buffers->stream); CUERR;
			cudaStreamSynchronize(buffers->stream);

			printf("after transpose\n");
			for(int j = 0; j < cols; j++){
				for(int i = 0; i < rows; i++){
					printf("%c", buffers->h_pileup[j * rows + i]);
				}
				printf("\n");
			}*/
		}

		void gpu_qual_pileup_transpose(const CorrectionBuffers* buffers){
			//copy h_pileup to d_pileup, then store transposed in d_pileup_transposed

			const size_t rows = buffers->max_n_qualityscores;
			const size_t cols = buffers->max_n_columns;

			cudaSetDevice(buffers->deviceId);
			cudaMemcpyAsync(buffers->d_qual_pileup, buffers->h_qual_pileup, sizeof(char) * rows * cols, H2D, buffers->stream); CUERR;
			cudaStreamSynchronize(buffers->stream);

			thrust::device_ptr<char> d_qual_pileup_ptr = thrust::device_pointer_cast(buffers->d_qual_pileup);
			thrust::device_ptr<char> d_qual_pileup_transposed_ptr = thrust::device_pointer_cast(buffers->d_qual_pileup_transposed);

			thrust::counting_iterator<size_t> indices(0);

			thrust::gather(thrust::cuda::par.on(buffers->stream),
					thrust::make_transform_iterator(indices, transpose_index(cols, rows)),
					thrust::make_transform_iterator(indices, transpose_index(cols, rows)) + rows * cols,
					d_qual_pileup_ptr, d_qual_pileup_transposed_ptr);
		}

		__global__
		void cuda_pileup_vote_transposed_kernel(CorrectionBuffers* buffers, const int nSequences, const int nQualityScores, const int columnsToCheck,
						const int subjectColumnsBegin_incl, const int subjectColumnsEnd_excl,
						const double maxErrorRate,
						const bool useQScores){

			__shared__ double smem[128];
			// multiple blocks, one block per column in pileup -> one block per row in transposed pileup

			//column in original pileup
			for(int column = blockIdx.x; column < columnsToCheck; column += gridDim.x){
				double Aweight = 0;
				double Cweight = 0;
				double Gweight = 0;
				double Tweight = 0;
				int As = 0;
				int Cs = 0;
				int Gs = 0;
				int Ts = 0;

				//row in original pileup
				for(int row = threadIdx.x; row < nSequences; row += blockDim.x){
					const int cellindex = column * buffers->max_n_sequences + row;
					const char base = buffers->d_pileup_transposed[cellindex];
					const char qual = buffers->d_qual_pileup_transposed[cellindex];
					const double qualweight = d_qscore_to_weight[(unsigned char)qual];

					if(base != 'F'){
						if(row == 0){
							double qw = 1.0;
							if(useQScores){
								qw *= qualweight;
							}

							switch(base){
								case 'A': Aweight += qw; As += 1; break;
								case 'C': Cweight += qw; Cs += 1; break;
								case 'G': Gweight += qw; Gs += 1; break;
								case 'T': Tweight += qw; Ts += 1; break;
								default:  break; //all other chars are invalid and not counted
							}
						}else{
							const AlignResultCompact arc = buffers->d_alignments[row-1];
							const double defaultweight = 1.0 - sqrt(arc.nOps / (arc.overlap * maxErrorRate));

							double qw = defaultweight;
							if(useQScores){
								qw *= qualweight;
							}

							switch(base){
								case 'A': Aweight += qw; As += 1; break;
								case 'C': Cweight += qw; Cs += 1; break;
								case 'G': Gweight += qw; Gs += 1; break;
								case 'T': Tweight += qw; Ts += 1; break;
								default:  break; //all other chars are invalid and not counted
							}
						}
					}
				}

				//wait before performing reduction
				__syncthreads();

				double Aweight_red = 0;
				double Cweight_red = 0;
				double Gweight_red = 0;
				double Tweight_red = 0;
				int As_red = 0;
				int Cs_red = 0;
				int Gs_red = 0;
				int Ts_red = 0;

				// block reduction
				auto sum = [](auto a, auto b){return a+b;};
				blockreduce(smem, &Aweight_red, Aweight, sum);
				blockreduce(smem, &Cweight_red, Cweight, sum);
				blockreduce(smem, &Gweight_red, Gweight, sum);
				blockreduce(smem, &Tweight_red, Tweight, sum);
				blockreduce((int*)smem, &As_red, As, sum);
				blockreduce((int*)smem, &Cs_red, Gs, sum);
				blockreduce((int*)smem, &Gs_red, Cs, sum);
				blockreduce((int*)smem, &Ts_red, Ts, sum);

				if(threadIdx.x == 0){
					//find consensus
					char cons = 'A';
					double consWeight = Aweight_red;
					if(Cweight_red > consWeight){
						cons = 'C';
						consWeight = Cweight_red;
					}
					if(Gweight_red > consWeight){
						cons = 'G';
						consWeight = Gweight_red;
					}
					if(Tweight_red > consWeight){
						cons = 'T';
						consWeight = Tweight_red;
					}

					const int coverage = As_red + Cs_red + Gs_red + Ts_red;
					const double columnWeight = Aweight_red + Cweight_red + Gweight_red + Tweight_red;

					//save results
					buffers->d_Aweights[column] = Aweight_red;
					buffers->d_Cweights[column] = Cweight_red;
					buffers->d_Gweights[column] = Gweight_red;
					buffers->d_Tweights[column] = Tweight_red;

					buffers->d_As[column] = As_red;
					buffers->d_Cs[column] = Cs_red;
					buffers->d_Gs[column] = Gs_red;
					buffers->d_Ts[column] = Ts_red;

					buffers->d_coverage[column] = coverage;
					buffers->d_consensus[column] = cons;
					buffers->d_support[column] = consWeight / columnWeight;

					switch(cons){
						case 'A': buffers->d_origCoverage[column] = As_red; buffers->d_origWeights[column] = Aweight_red; break;
						case 'C': buffers->d_origCoverage[column] = Cs_red; buffers->d_origWeights[column] = Cweight_red; break;
						case 'G': buffers->d_origCoverage[column] = Gs_red; buffers->d_origWeights[column] = Gweight_red; break;
						case 'T': buffers->d_origCoverage[column] = Ts_red; buffers->d_origWeights[column] = Tweight_red; break;
						default:  printf("this should not happen in pileup\n"); break;
					}
				}
			}
		}

		__global__
		void cuda_pileup_reduce_kernel(CorrectionBuffers* buffers, const int subjectColumnsBegin_incl, const int subjectColumnsEnd_excl){

			__shared__ double smem[128];

			double avg_support = 0;
			double min_support = 1.0;
			int max_coverage = 0;
			int min_coverage = INT_MAX;

			for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += blockDim.x){
				const double sup = buffers->d_support[i];
				const int cov = buffers->d_coverage[i];
				avg_support += sup;
				min_support = sup < min_support? sup : min_support;
				max_coverage = cov > max_coverage ? cov : max_coverage;
				min_coverage = cov < min_coverage ? cov : min_coverage;
			}

			// block reduction
			double avg_support_red = 0;
			double min_support_red = 1.0;
			int max_coverage_red = 0;
			int min_coverage_red = INT_MAX;

			blockreduce(smem, &avg_support_red, avg_support, [](auto a, auto b){return a+b;});
			blockreduce(smem, &min_support_red, min_support, [](auto a, auto b){return min(a,b);});
			blockreduce((int*)smem, &max_coverage_red, max_coverage, [](auto a, auto b){return max(a,b);});
			blockreduce((int*)smem, &min_coverage_red, min_coverage, [](auto a, auto b){return min(a,b);});

			if(threadIdx.x == 0){
				const int subjectlength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
				avg_support_red /= subjectlength;

				buffers->avg_support = avg_support_red;
				buffers->min_support = min_support_red;
				buffers->max_coverage = max_coverage_red;
				buffers->min_coverage = min_coverage_red;

				/*for(int i = 0; i < buffers->n_columns; i++){
					printf("%d", buffers->d_As[i]);
				}
				printf("\nCs\n");
				for(int i = 0; i < buffers->n_columns; i++){
					printf("%d", buffers->d_Cs[i]);
				}
				printf("\n");
				for(int i = 0; i < buffers->n_columns; i++){
					printf("%d", buffers->d_Gs[i]);
				}
				printf("\n");
				for(int i = 0; i < buffers->n_columns; i++){
					printf("%d", buffers->d_Ts[i]);
				}
				printf("\n");

				for(int i = 0; i < buffers->n_columns; i++){
					printf("%2c ", buffers->d_consensus[i]);
				}
				printf("\n");
				for(int i = 0; i < buffers->n_columns; i++){
					printf("%2d ", buffers->d_coverage[i]);
				}
				printf("\n");
				printf("%d %f %f %d %d\n", buffers->n_columns, avg_support_red, min_support_red, max_coverage_red, min_coverage_red);*/
			}
		}

		void call_cuda_pileup_vote_transposed_kernel(CorrectionBuffers* buffers, const int nSequences, const int nQualityScores, const int columnsToCheck,
						const int subjectColumnsBegin_incl, const int subjectColumnsEnd_excl,
						const double maxErrorRate,
						const bool useQScores){

			cudaMemcpyAsync(buffers->d_this,
					buffers,
					sizeof(CorrectionBuffers),
					H2D, buffers->stream); CUERR;

			dim3 block(64);
			dim3 grid(nSequences);
			cuda_pileup_vote_transposed_kernel<<<grid, block, 0, buffers->stream>>>(buffers->d_this, nSequences, nQualityScores, columnsToCheck,
												subjectColumnsBegin_incl, subjectColumnsEnd_excl,
												maxErrorRate,
												useQScores); CUERR;

			//calculate stats of subject. min max avg coverage
			cuda_pileup_reduce_kernel<<<1,128, 0, buffers->stream>>>(buffers->d_this, subjectColumnsBegin_incl, subjectColumnsEnd_excl); CUERR;

			cudaStreamSynchronize(buffers->stream); CUERR;
		}

#endif



	}
}
