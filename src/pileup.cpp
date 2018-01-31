#include "../inc/pileup.hpp"
#include "../inc/alignment.hpp"

#include <vector>
#include <string>
#include <cassert>
#include <chrono>

namespace hammingtools{

	namespace correction{

		double	qscore_to_error_prob[256];
		double	qscore_to_weight[256];

		void init_once(){

			constexpr int ASCII_BASE = 33;
			constexpr double MIN_GRAPH_WEIGHT = 0.001;

			for(int i = 0; i < 256; i++){
				if(i < ASCII_BASE)
					qscore_to_error_prob[i] = 1.0;
				else
					qscore_to_error_prob[i] = std::pow(10.0, -(i-ASCII_BASE)/10.0);
			}

			for(int i = 0; i < 256; i++){
				qscore_to_weight[i] = std::max(MIN_GRAPH_WEIGHT, 1.0 - qscore_to_error_prob[i]);
			}
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

			constexpr int candidate_correction_new_cols = 2;
	
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
				double qw = 1.0 - std::sqrt(alignments[i].nOps / (alignments[i].overlap * maxErrorRate));

				for(int j = 0; j < int(queries[i].length()); j++){					
					const int globalIndex = subjectColumnsBegin_incl + alignments[i].shift + j;

					for(int f = 0; f < frequenciesPrefixSum[i+1] - frequenciesPrefixSum[i]; f++){
						const int qualityindex = frequenciesPrefixSum[i] + f;
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
			}
			avg_support /= subject.length();


		#if 0
			std::cout << "anchorsupport\n";
			for(columnindex = subjectColumnsBegin_incl; columnindex < subjectColumnsEnd_excl; columnindex++){
				std::cout << support[columnindex] << " ";
			}
			std::cout << '\n';
			std::cout << "anchorcoverage\n";
			for(columnindex = subjectColumnsBegin_incl; columnindex < subjectColumnsEnd_excl; columnindex++){
				std::cout << coverage[columnindex] << " ";
			}
			std::cout << '\n';

			std::cout << "avgsup " << avg_support << " >= " << (1-errorrate) << '\n';
			std::cout << "minsup " << min_support << " >= " << (1-2*errorrate) << '\n';
			std::cout << "mincov " << min_coverage << " >= " << (m) << '\n';
			std::cout << "maxcov " << max_coverage << " <= " << (3*m) << '\n';
			std::cout << "------------------------------------------------" << '\n';
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
							for(columnindex = subjectColumnsBegin_incl - candidate_correction_new_cols; 
								columnindex < subjectColumnsBegin_incl;
								columnindex++){

								assert(columnindex < columnsToCheck);
								if(queryColumnsBegin_incl <= columnindex){
									newColMinSupport = support[columnindex] < newColMinSupport ? support[columnindex] : newColMinSupport;		
									newColMinCov = coverage[columnindex] < newColMinCov ? coverage[columnindex] : newColMinCov;
								}
							}
							//check new columns right of subject
							for(columnindex = subjectColumnsEnd_excl; 
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
									columnindex = queryColumnsBegin_incl + j;
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
		#if 1
						subject[i] = consensus[globalIndex];
						foundAColumn = true;
		#endif
					}else{
		#if 1
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
		#endif
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


	}
}
