#include "../inc/pileup.hpp"
#include "../inc/alignment.hpp"
#include "../inc/hammingtools.hpp"
#include "../inc/ganja/hpc_helpers.cuh"

#include <vector>
#include <string>
#include <cassert>
#include <chrono>
#include <climits>

#ifdef __NVCC__
#include <cooperative_groups.h>

using namespace cooperative_groups;
#endif

namespace hammingtools{

	namespace correction{

		constexpr int candidate_correction_new_cols = 2;

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
		correct_cpu(std::string& subject,
						int nQueries, 
						std::vector<std::string>& queries,
						const std::vector<AlignResultCompact>& alignments,
						const std::string& subjectqualityScores, 
						const std::vector<const std::string*>& queryqualityScores,
						const std::vector<int>& frequenciesPrefixSum,
						double maxErrorRate,
						bool useQScores,
						std::vector<bool>& correctedQueries,
						bool correctQueries_,
						int estimatedCoverage,
						double errorrate,
						double m,
						int k){

			std::chrono::duration<double> majorityvotetime(0);
			std::chrono::duration<double> basecorrectiontime(0);
			std::chrono::time_point<std::chrono::system_clock> tpa, tpb;

			
	
			tpa = std::chrono::system_clock::now();

			const bool correctQueries = correctQueries_;
			int status = 0;

			int startindex = 0;
			int endindex = subject.length();
			for(int i = 0; i < nQueries; i++){
				startindex = alignments[i].shift < startindex ? alignments[i].shift : startindex;
				const int queryEndsAt = queryqualityScores[i]->length() + alignments[i].shift;
				endindex = queryEndsAt > endindex ? queryEndsAt : endindex;
				correctedQueries[i] = false;
			}

			const int columnsToCheck = endindex - startindex;

			// the column index range for the subject begins at max(-leftOfSubjectBegin,0);
			std::vector<char> consensus(columnsToCheck,0);
			std::vector<double> support(columnsToCheck,0);
			std::vector<int> coverage(columnsToCheck,0);
			std::vector<double> origWeights(columnsToCheck,0);
			std::vector<int> origCoverage(columnsToCheck,0);
			std::vector<int> As(columnsToCheck,0);
			std::vector<int> Cs(columnsToCheck,0);
			std::vector<int> Gs(columnsToCheck,0);
			std::vector<int> Ts(columnsToCheck,0);
			std::vector<double> Aweights(columnsToCheck,0);
			std::vector<double> Cweights(columnsToCheck,0);
			std::vector<double> Gweights(columnsToCheck,0);
			std::vector<double> Tweights(columnsToCheck,0);
			const int subjectColumnsBegin_incl = std::max(-startindex,0);
			const int subjectColumnsEnd_excl = subjectColumnsBegin_incl + subject.length();

			//add subject weights
			for(int i = 0; i < int(subject.length()); i++){
				const int globalIndex = subjectColumnsBegin_incl + i;
				double qw = 1.0;
				if(useQScores)
					qw *= qscore_to_weight[(unsigned char)subjectqualityScores[i]];
				switch(subject[i]){
					case 'A': Aweights[globalIndex] += qw; As[globalIndex] += 1; break;
					case 'C': Cweights[globalIndex] += qw; Cs[globalIndex] += 1; break;
					case 'G': Gweights[globalIndex] += qw; Gs[globalIndex] += 1; break;
					case 'T': Tweights[globalIndex] += qw; Ts[globalIndex] += 1; break;
					default: std::cout << "this should not happen in pileup\n"; break;
				}
				coverage[globalIndex]++;				
			}

			//add candidate weights
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
							case 'A': Aweights[globalIndex] += qw; As[globalIndex] += 1; break;
							case 'C': Cweights[globalIndex] += qw; Cs[globalIndex] += 1; break;
							case 'G': Gweights[globalIndex] += qw; Gs[globalIndex] += 1; break;
							case 'T': Tweights[globalIndex] += qw; Ts[globalIndex] += 1; break;
							default: std::cout << "this should not happen in pileup\n"; break;
						}
						coverage[globalIndex]++;
					}
				}

			}

			//find consensus and support in each column
			for(int i = 0; i < columnsToCheck; i++){
				char cons = 'A';
				double consWeight = Aweights[i];
				if(Cweights[i] > consWeight){
					cons = 'C';
					consWeight = Cweights[i];
				}
				if(Gweights[i] > consWeight){
					cons = 'G';
					consWeight = Gweights[i];
				}
				if(Tweights[i] > consWeight){
					cons = 'T';
					consWeight = Tweights[i];
				}
				consensus[i] = cons;

				const double columnWeight = Aweights[i] + Cweights[i] + Gweights[i] + Tweights[i];
				support[i] = consWeight / columnWeight;

				//printf("i %i, %f %f %f %f\n",i, Aweights[i], Cweights[i], Gweights[i], Tweights[i]);

				switch(cons){
					case 'A': origCoverage[i] = As[i]; origWeights[i] = Aweights[i]; break;
					case 'C': origCoverage[i] = Cs[i]; origWeights[i] = Cweights[i]; break;
					case 'G': origCoverage[i] = Gs[i]; origWeights[i] = Gweights[i]; break;
					case 'T': origCoverage[i] = Ts[i]; origWeights[i] = Tweights[i]; break;
					default: std::cout << "this should not happen in pileup\n"; break;
				}	
			}

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
				
				avg_support += support[i];
				min_support = support[i] < min_support? support[i] : min_support;
				max_coverage = coverage[i] > max_coverage ? coverage[i] : max_coverage;
				min_coverage = coverage[i] < min_coverage ? coverage[i] : min_coverage;
				//printf("i %d sup %f cov %d\n", i, support[i], coverage[i]);
			}
			avg_support /= subject.length();


		#if 0
			std::cout << "avgsup " << avg_support << " >= " << (1-errorrate) << '\n';
			std::cout << "minsup " << min_support << " >= " << (1-2*errorrate) << '\n';
			std::cout << "mincov " << min_coverage << " >= " << (m) << '\n';
			std::cout << "maxcov " << max_coverage << " <= " << (3*m) << '\n';
		#endif

			bool isHQ = avg_support >= 1.0-errorrate
				 && min_support >= 1.0-3.0*errorrate
				 && min_coverage >= m / 2.0 * estimatedCoverage;

			if(isHQ){
		#if 1
				//correct anchor
				for(int i = 0; i < int(subject.length()); i++){
					const int globalIndex = subjectColumnsBegin_incl + i;
					subject[i] = consensus[globalIndex];
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
									newColMinSupport = support[columnindex] < newColMinSupport ? support[columnindex] : newColMinSupport;		
									newColMinCov = coverage[columnindex] < newColMinCov ? coverage[columnindex] : newColMinCov;
								}
							}
							//check new columns right of subject
							for(int columnindex = subjectColumnsEnd_excl; 
								columnindex < subjectColumnsEnd_excl + candidate_correction_new_cols 
								&& columnindex < columnsToCheck;
								columnindex++){

								newColMinSupport = support[columnindex] < newColMinSupport ? support[columnindex] : newColMinSupport;		
								newColMinCov = coverage[columnindex] < newColMinCov ? coverage[columnindex] : newColMinCov;
							}

							if(newColMinSupport >= 1-3*errorrate 
								&& newColMinCov >= m / 2.0 * estimatedCoverage){
								//assert(subjectColumnsBegin_incl == queryColumnsBegin_incl && subject.length() == queries[i].length());

								for(int j = 0; j < int(queries[i].length()); j++){
									int columnindex = queryColumnsBegin_incl + j;
									queries[i][j] = consensus[columnindex];
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
		#if 1
				//correct anchor
				bool foundAColumn = false;
				for(int i = 0; i < int(subject.length()); i++){
					const int globalIndex = subjectColumnsBegin_incl + i;

					if(support[globalIndex] >= 1.0-3.0*errorrate){
						subject[i] = consensus[globalIndex];
						foundAColumn = true;
					}else{
						if(support[globalIndex] > 0.5 && origCoverage[globalIndex] < m / 2.0 * estimatedCoverage){
							double avgsupportkregion = 0;
							int c = 0;
							bool kregioncoverageisgood = true;
							for(int j = i - k/2; j <= i + k/2 && kregioncoverageisgood; j++){
								if(j != i && j >= 0 && j < int(subject.length())){
									avgsupportkregion += support[subjectColumnsBegin_incl + j];
									kregioncoverageisgood &= (coverage[subjectColumnsBegin_incl + j] >= m / 2.0 * estimatedCoverage);
									c++;
								}
							}
							if(kregioncoverageisgood && avgsupportkregion / c >= 1.0-errorrate){
								subject[i] = consensus[globalIndex];
								foundAColumn = true;
							}
						}
					}
				}

				if(!foundAColumn)
					status |= (1 << 3);
		#endif
			}

			tpb = std::chrono::system_clock::now();

			basecorrectiontime += tpb - tpa;

			return std::tie(status, majorityvotetime, basecorrectiontime);
		}





		int cpu_pileup_correct(CorrectionBuffers* buffers, const int nQueries, const int columnsToCheck, 
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
				for(int i = 0; i < buffers->h_lengths[0]; i++){
					const int globalIndex = subjectColumnsBegin_incl + i;
					buffers->h_sequences[0 * buffers->max_seq_length + i] = buffers->h_consensus[globalIndex]; //assign to subject[i]
				}
		#endif
		#if 0
				//correct candidates
				if(correctQueries){
			
					for(int i = 0; i < nQueries; i++){
						int queryColumnsBegin_incl = buffers->h_alignments[i].shift - startindex;
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
								//assert(subjectColumnsBegin_incl == queryColumnsBegin_incl && subject.length() == queries[i].length());

								for(int j = 0; j < buffers->h_lengths[1+i]; j++){
									const int columnindex = queryColumnsBegin_incl + j;
									//assign to query[i][j]
									buffers->h_sequences[(1+i) * buffers->max_seq_length + j] = buffers->h_consensus[columnindex];
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
						buffers->h_sequences[0 * buffers->max_seq_length + i] = buffers->h_consensus[globalIndex]; //assign to subject[i]
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
								buffers->h_sequences[0 * buffers->max_seq_length + i] = buffers->h_consensus[globalIndex]; //assign to subject[i]
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



#if 1		
		__global__
		void cuda_pileup_vote_kernel(CorrectionBuffers* buffers, const int nQueries, const int columnsToCheck, 
						const int subjectColumnsBegin_incl, const int subjectColumnsEnd_excl,
						const double maxErrorRate, 
						const bool useQScores){

//using only 1 block, 
#if 1
			thread_block block = this_thread_block();
	
			//add subject weights
			const int subjectlength = buffers->d_lengths[0];
			for(int i = block.thread_rank(); i < subjectlength; i += block.size()){
				const char base = buffers->d_sequences[0 * buffers->max_seq_length + i];
				const int globalIndex = subjectColumnsBegin_incl + i;
				double qw = 1.0;
				if(useQScores)
					qw *= d_qscore_to_weight[(unsigned char)(buffers->d_qualityscores[0 * buffers->max_seq_length + i])];
				switch(base){
					case 'A': buffers->d_Aweights[globalIndex] += qw; buffers->d_As[globalIndex] += 1; break;
					case 'C': buffers->d_Cweights[globalIndex] += qw; buffers->d_Cs[globalIndex] += 1; break;
					case 'G': buffers->d_Gweights[globalIndex] += qw; buffers->d_Gs[globalIndex] += 1; break;
					case 'T': buffers->d_Tweights[globalIndex] += qw; buffers->d_Ts[globalIndex] += 1; break;
					default:  printf("this should not happen in pileup\n"); break;
				}
				assert(buffers->d_coverage[globalIndex] == 0);
				buffers->d_coverage[globalIndex]++;				
			}

			block.sync();

			//add candidate weights
			for(int i = 0; i < nQueries; i++){
				const AlignResultCompact arc = buffers->d_alignments[i];
				const int querylength = buffers->d_lengths[1+i];
				const int freqSum = buffers->d_frequencies_prefix_sum[i];
				const int freqSumNext = buffers->d_frequencies_prefix_sum[1+i];

				double defaultweight = 1.0 - sqrt(arc.nOps / (arc.overlap * maxErrorRate));
				const int frequency = freqSumNext - freqSum;
				assert(frequency >= 1);
				for(int j = block.thread_rank(); j < querylength; j += block.size()){
					const char base = buffers->d_sequences[(1+i) * buffers->max_seq_length + j];		
					const int globalIndex = subjectColumnsBegin_incl + arc.shift + j;
					double weightincrement = 0;

					for(int f = 0; f < frequency; f++){
						const int qualityindex = freqSum + f;
						double qw = defaultweight;
						if(useQScores)
							qw *= d_qscore_to_weight[(unsigned char)(buffers->d_qualityscores[(qualityindex+1) * buffers->max_seq_length + j])];
						weightincrement += qw;
					}

					switch(base){
						case 'A': buffers->d_Aweights[globalIndex] += weightincrement; buffers->d_As[globalIndex] += frequency; break;
						case 'C': buffers->d_Cweights[globalIndex] += weightincrement; buffers->d_Cs[globalIndex] += frequency; break;
						case 'G': buffers->d_Gweights[globalIndex] += weightincrement; buffers->d_Gs[globalIndex] += frequency; break;
						case 'T': buffers->d_Tweights[globalIndex] += weightincrement; buffers->d_Ts[globalIndex] += frequency; break;
						default:  printf("this should not happen in pileup\n"); break;
					}

					buffers->d_coverage[globalIndex] += frequency;
				}

				block.sync();
			}


			//find consensus and support in each column
			for(int i = block.thread_rank(); i < columnsToCheck; i += block.size()){
				const double aw = buffers->d_Aweights[i];
				const double cw = buffers->d_Cweights[i];
				const double gw = buffers->d_Gweights[i];
				const double tw = buffers->d_Tweights[i];

				char cons = 'A';
				double consWeight = aw;
				if(cw > consWeight){
					cons = 'C';
					consWeight = cw;
				}
				if(gw > consWeight){
					cons = 'G';
					consWeight = gw;
				}
				if(tw > consWeight){
					cons = 'T';
					consWeight = tw;
				}
				buffers->d_consensus[i] = cons;

				const double columnWeight = aw + cw + gw + tw;
				buffers->d_support[i] = consWeight / columnWeight;

				//printf("i %i, %f %f %f %f\n",i, aw, cw,gw,tw);

				switch(cons){
					case 'A': buffers->d_origCoverage[i] = buffers->d_As[i]; buffers->d_origWeights[i] = aw; break;
					case 'C': buffers->d_origCoverage[i] = buffers->d_Cs[i]; buffers->d_origWeights[i] = cw; break;
					case 'G': buffers->d_origCoverage[i] = buffers->d_Gs[i]; buffers->d_origWeights[i] = gw; break;
					case 'T': buffers->d_origCoverage[i] = buffers->d_Ts[i]; buffers->d_origWeights[i] = tw; break;
					default:  printf("this should not happen in pileup\n"); break;
				}	
			}

			block.sync();

			if(block.thread_rank() == 0){
				double avg_support = 0;
				double min_support = 1.0;
				int max_coverage = 0;
				int min_coverage = INT_MAX;
				//get stats for subject columns
				//for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += blockDim.x){
				for(int i = subjectColumnsBegin_incl; i < subjectColumnsEnd_excl; i += 1){
					const double sup = buffers->d_support[i];
					const int cov = buffers->d_coverage[i];
					assert(sup >= 0);
					assert(cov >= 0);
					//printf("i %d sup %f cov %d\n", i, sup, cov);

					avg_support += sup;
					min_support = sup < min_support? sup : min_support;
					max_coverage = cov > max_coverage ? cov : max_coverage;
					min_coverage = cov < min_coverage ? cov : min_coverage;
				}
				avg_support /= subjectlength;

				buffers->avg_support = avg_support;
				buffers->min_support = min_support;
				buffers->max_coverage = max_coverage;
				buffers->min_coverage = min_coverage;
			}
#endif

// multiple blocks, with atomics
#if 0
			thread_block block = this_thread_block();
			const int seqid = block.group_index.x;
			/*double Aweight = 0;
			double Cweight = 0;
			double Gweight = 0;
			double Tweight = 0;
			int As = 0;
			int Cs = 0;
			int Gs = 0;
			int Ts = 0;*/
			//add subject weights
			const int subjectlength = buffers->d_lengths[0];
		
			if(seqid == 0){
				for(int i = block.thread_rank(); i < subjectlength; i += block.size()){
					const char base = buffers->d_sequences[0 * buffers->max_seq_length + i];
					const int globalIndex = subjectColumnsBegin_incl + i;
					double qw = 1.0;
					if(useQScores)
						qw *= d_qscore_to_weight[(unsigned char)(buffers->d_qualityscores[0 * buffers->max_seq_length + i])];
					switch(base){
						case 'A': atomicAdd(buffers->d_Aweights + globalIndex, qw); atomicAdd(buffers->d_As + globalIndex, 1); break;
						case 'C': atomicAdd(buffers->d_Cweights + globalIndex, qw); atomicAdd(buffers->d_Cs + globalIndex, 1); break;
						case 'G': atomicAdd(buffers->d_Gweights + globalIndex, qw); atomicAdd(buffers->d_Gs + globalIndex, 1); break;
						case 'T': atomicAdd(buffers->d_Tweights + globalIndex, qw); atomicAdd(buffers->d_Ts + globalIndex, 1); break;
						default:  printf("this should not happen in pileup\n"); break;
					}
					atomicAdd(buffers->d_coverage + globalIndex, 1);
				}
			}


			//add candidate weights
			for(int i = seqid; i < nQueries; i += gridDim.x){
				const AlignResultCompact arc = buffers->d_alignments[i];
				const int querylength = buffers->d_lengths[1+i];
				const int freqSum = buffers->d_frequencies_prefix_sum[i];
				const int freqSumNext = buffers->d_frequencies_prefix_sum[1+i];

				double defaultweight = 1.0 - sqrt(arc.nOps / (arc.overlap * maxErrorRate));
				const int frequency = freqSumNext - freqSum;
				assert(frequency >= 1);
				for(int j = block.thread_rank(); j < querylength; j += block.size()){
					const char base = buffers->d_sequences[(1+i) * buffers->max_seq_length + j];		
					const int globalIndex = subjectColumnsBegin_incl + arc.shift + j;
					double weightincrement = 0;

					for(int f = 0; f < frequency; f++){
						const int qualityindex = freqSum + f;
						double qw = defaultweight;
						if(useQScores)
							qw *= d_qscore_to_weight[(unsigned char)(buffers->d_qualityscores[(qualityindex+1) * buffers->max_seq_length + j])];
						weightincrement += qw;
					}

					switch(base){
						case 'A': atomicAdd(buffers->d_Aweights + globalIndex, weightincrement); atomicAdd(buffers->d_As + globalIndex, frequency); break;
						case 'C': atomicAdd(buffers->d_Cweights + globalIndex, weightincrement); atomicAdd(buffers->d_Cs + globalIndex, frequency); break;
						case 'G': atomicAdd(buffers->d_Gweights + globalIndex, weightincrement); atomicAdd(buffers->d_Gs + globalIndex, frequency); break;
						case 'T': atomicAdd(buffers->d_Tweights + globalIndex, weightincrement); atomicAdd(buffers->d_Ts + globalIndex, frequency); break;
						default:  printf("this should not happen in pileup\n"); break;
					}

					atomicAdd(buffers->d_coverage + globalIndex, frequency);
				}
			}


			//find consensus and support in each column
			for(int i = block.thread_rank(); i < columnsToCheck; i += block.size()){
				const double aw = buffers->d_Aweights[i];
				const double cw = buffers->d_Cweights[i];
				const double gw = buffers->d_Gweights[i];
				const double tw = buffers->d_Tweights[i];

				char cons = 'A';
				double consWeight = aw;
				if(cw > consWeight){
					cons = 'C';
					consWeight = cw;
				}
				if(gw > consWeight){
					cons = 'G';
					consWeight = gw;
				}
				if(tw > consWeight){
					cons = 'T';
					consWeight = tw;
				}
				buffers->d_consensus[i] = cons;

				const double columnWeight = aw + cw + gw + tw;
				buffers->d_support[i] = consWeight / columnWeight;

				//printf("i %i, %f %f %f %f\n",i, aw, cw,gw,tw);

				switch(cons){
					case 'A': buffers->d_origCoverage[i] = buffers->d_As[i]; buffers->d_origWeights[i] = aw; break;
					case 'C': buffers->d_origCoverage[i] = buffers->d_Cs[i]; buffers->d_origWeights[i] = cw; break;
					case 'G': buffers->d_origCoverage[i] = buffers->d_Gs[i]; buffers->d_origWeights[i] = gw; break;
					case 'T': buffers->d_origCoverage[i] = buffers->d_Ts[i]; buffers->d_origWeights[i] = tw; break;
					default:  printf("this should not happen in pileup\n"); break;
				}	
			}

			block.sync();

			if(block.thread_rank() == 0){
				double avg_support = 0;
				double min_support = 1.0;
				int max_coverage = 0;
				int min_coverage = INT_MAX;
				//get stats for subject columns
				//for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += blockDim.x){
				for(int i = subjectColumnsBegin_incl; i < subjectColumnsEnd_excl; i += 1){
					const double sup = buffers->d_support[i];
					const int cov = buffers->d_coverage[i];
					assert(sup >= 0);
					assert(cov >= 0);
					//printf("i %d sup %f cov %d\n", i, sup, cov);

					avg_support += sup;
					min_support = sup < min_support? sup : min_support;
					max_coverage = cov > max_coverage ? cov : max_coverage;
					min_coverage = cov < min_coverage ? cov : min_coverage;
				}
				avg_support /= subjectlength;

				buffers->avg_support = avg_support;
				buffers->min_support = min_support;
				buffers->max_coverage = max_coverage;
				buffers->min_coverage = min_coverage;
			}
#endif

		}

		void call_cuda_pileup_vote_kernel_async(CorrectionBuffers* buffers, const int nQueries, const int columnsToCheck, 
						const int subjectColumnsBegin_incl, const int subjectColumnsEnd_excl,
						const double maxErrorRate, 
						const bool useQScores){

			cudaMemcpyAsync(buffers->d_this, 
					buffers, 
					sizeof(CorrectionBuffers), 
					H2D, buffers->stream); CUERR;

			constexpr int blockdim = 128;
			constexpr int griddim = 1;
			
			cuda_pileup_vote_kernel<<<griddim, blockdim, 0, buffers->stream>>>(buffers->d_this, nQueries, columnsToCheck, 
											subjectColumnsBegin_incl, subjectColumnsEnd_excl, 
											maxErrorRate, useQScores); CUERR;
		}

		void call_cuda_pileup_vote_kernel(CorrectionBuffers* buffers, const int nQueries, const int columnsToCheck, 
						const int subjectColumnsBegin_incl, const int subjectColumnsEnd_excl,
						const double maxErrorRate, 
						const bool useQScores){

			call_cuda_pileup_vote_kernel_async(buffers, nQueries, columnsToCheck, 
							subjectColumnsBegin_incl, subjectColumnsEnd_excl, 
							maxErrorRate, useQScores);
			cudaStreamSynchronize(buffers->stream);
		}

#endif



	}
}
