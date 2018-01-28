#include "../inc/pileup.hpp"
#include "../inc/alignment.hpp"

#include <vector>
#include <string>
#include <cassert>

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

		int correct_cpu(std::string& subject,
						int nQueries, 
						std::vector<std::string>& queries,
						const std::vector<AlignResultCompact>& alignments,
						const std::string& subjectqualityScores, 
						const std::vector<std::string>& queryqualityScores,
						const std::vector<int>& frequenciesPrefixSum,
						double maxErrorRate,
						bool useQScores,
						std::vector<bool>& correctedQueries,
						bool correctQueries_,
						int estimatedCoverage,
						double errorrate,
						double m,
						int k){

			constexpr int candidate_correction_new_cols = 2;
	

			const bool correctQueries = correctQueries_;
			int status = 0;

			int startindex = 0;
			int endindex = subject.length();
			std::vector<double> defaultWeightsPerQuery(queries.size());
			for(int i = 0; i < nQueries; i++){
				startindex = alignments[i].shift < startindex ? alignments[i].shift : startindex;
				int queryEndsAt = queryqualityScores[i].length() + alignments[i].shift;
				endindex = queryEndsAt > endindex ? queryEndsAt : endindex;
				defaultWeightsPerQuery[i] = 1.0 - std::sqrt(alignments[i].nOps / (alignments[i].overlap * maxErrorRate));
				correctedQueries[i] = false;
			}

			int columnsToCheck = endindex - startindex;

			// the column index range for the subject begins at max(-leftOfSubjectBegin,0);
			std::vector<char> consensus(columnsToCheck);
			std::vector<double> support(columnsToCheck);
			std::vector<int> coverage(columnsToCheck);
			std::vector<int> origWeights(columnsToCheck);
			std::vector<int> origCoverage(columnsToCheck);
			const int subjectColumnsBegin_incl = std::max(-startindex,0);
			const int subjectColumnsEnd_excl = subjectColumnsBegin_incl + subject.length();
			int columnindex = 0;

			for(int i = startindex; i < endindex; i++){
				// weights for bases A C G T N
				double weights[5]{0,0,0,0,0};
				int cov[5]{0,0,0,0,0};
				int count = 0;
		
				//count subject base
				if(i >= 0 && i < int(subject.length())){
					double qw = 1.0;
					if(useQScores)
						qw *= qscore_to_weight[(unsigned char)subjectqualityScores[i]];
					switch(subject[i]){
						case 'A': weights[0] += qw; cov[0] += 1; break;
						case 'C': weights[1] += qw; cov[1] += 1; break;
						case 'G': weights[2] += qw; cov[2] += 1; break;
						case 'T': weights[3] += qw; cov[3] += 1; break;
						case 'N': weights[4] += qw; cov[4] += 1; break;
						default: break;
					}
					count++;
				}

				//count query bases
				for(int j = 0; j < nQueries; j++){
					const int baseindex = i - alignments[j].shift;

					if(baseindex >= 0 && baseindex < int(queries[j].length())){ //check query boundary
						double qweight = defaultWeightsPerQuery[j];
						for(int f = 0; f < frequenciesPrefixSum[j+1] - frequenciesPrefixSum[j]; f++){
							const int qualityindex = frequenciesPrefixSum[j] + f;
							if(useQScores)
								qweight *= qscore_to_weight[(unsigned char)queryqualityScores[qualityindex][baseindex]];
							switch(queries[j][baseindex]){
								case 'A': weights[0] += qweight; cov[0] += 1; break;
								case 'C': weights[1] += qweight; cov[1] += 1; break;
								case 'G': weights[2] += qweight; cov[2] += 1; break;
								case 'T': weights[3] += qweight; cov[3] += 1; break;
								case 'N': weights[4] += qweight; cov[4] += 1; break;
								default: break;
							}
							count++;
						}
					}
				}

				double consensusWeight = 0;
				int maxindex = 4;
				double columnWeight = 0;
				for(int k = 0; k < 5; k++){
					columnWeight += weights[k];
					if(weights[k] > consensusWeight){
						consensusWeight = weights[k];
						maxindex = k;
					}
				}
		
				double supportvalue = consensusWeight / columnWeight;
				double origWeight = 0;
				int origCov = 0;
				if(i >= 0 && i < int(subject.length())){
					switch(subject[i]){
						case 'A': origWeight = weights[0]; origCov = cov[0]; break;
						case 'C': origWeight = weights[1]; origCov = cov[1]; break;
						case 'G': origWeight = weights[2]; origCov = cov[2]; break;
						case 'T': origWeight = weights[3]; origCov = cov[3]; break;
						case 'N': origWeight = weights[4]; origCov = cov[4]; break;
						default: break;
					}
				}

				char consensusBase = ' ';
				switch(maxindex){
					case 0: consensusBase = 'A'; break;
					case 1: consensusBase = 'C'; break;
					case 2: consensusBase = 'G'; break;
					case 3: consensusBase = 'T'; break;
					case 4: consensusBase = 'N'; break;
					default: break;
				}

				consensus[columnindex] = consensusBase;
				support[columnindex] = supportvalue;
				coverage[columnindex] = count;
				origWeights[columnindex] = origWeight;
				origCoverage[columnindex] = origCov;
				columnindex++;
			}

			double avg_support = 0;
			double min_support = 1.0;
			int max_coverage = 0;
			int min_coverage = std::numeric_limits<int>::max();
			//get stats for subject columns
			for(columnindex = subjectColumnsBegin_incl; columnindex < subjectColumnsEnd_excl; columnindex++){
				if(columnindex >= columnsToCheck){
					assert(columnindex < columnsToCheck);
				}
				avg_support += support[columnindex];
				min_support = support[columnindex] < min_support? support[columnindex] : min_support;
				max_coverage = coverage[columnindex] > max_coverage ? coverage[columnindex] : max_coverage;
				min_coverage = coverage[columnindex] < min_coverage ? coverage[columnindex] : min_coverage;
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

			bool isHQ = avg_support >= 1-errorrate
				 && min_support >= 1-3*errorrate
				 && min_coverage >= m / 2.0 * estimatedCoverage;
                 
                 
           /*      std::cout << "ishq " << isHQ << std::endl;
                 			std::cout << "avgsup " << avg_support << " >= " << (1-errorrate) << '\n';
			std::cout << "minsup " << min_support << " >= " << (1-2*errorrate) << '\n';
			std::cout << "mincov " << min_coverage << " >= " << (m) << '\n';
			std::cout << "maxcov " << max_coverage << " <= " << (3*m) << '\n';
			std::cout << "subjectColumnsBegin_incl " << subjectColumnsBegin_incl  << '\n';
			std::cout << "subjectColumnsEnd_excl " << subjectColumnsEnd_excl  << '\n';
			std::cout << "------------------------------------------------" << '\n';*/

			if(isHQ){
		#if 1
				//correct anchor
				for(int i = 0; i < int(subject.length()); i++){
					subject[i] = consensus[subjectColumnsBegin_incl + i];
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
				if(avg_support < 1-errorrate)
					status |= (1 << 0);
				if(min_support < 1-3*errorrate)
					status |= (1 << 1);
				if(min_coverage < m / 2.0 * estimatedCoverage)
					status |= (1 << 2);
		#if 1
				//correct anchor
				bool foundAColumn = false;
				for(int i = 0; i < int(subject.length()); i++){
					columnindex = subjectColumnsBegin_incl + i;

					if(support[columnindex] >= 1-3*errorrate){
		#if 1
						subject[i] = consensus[columnindex];
						foundAColumn = true;
		#endif
					}else{
		#if 1
						if(support[columnindex] > 0.5 && origCoverage[columnindex] < m / 2.0 * estimatedCoverage){
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
							if(kregioncoverageisgood && avgsupportkregion / c >= 1-errorrate){
								subject[i] = consensus[columnindex];
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


			return status;
		}


	}

}
