#include "../inc/hammingvote.hpp"
#include "../inc/alignment.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <vector>
#include <limits>
#include <cassert>

constexpr double min_bases_for_candidate_correction_factor = 0.0;

static_assert(0.0 <= min_bases_for_candidate_correction_factor && min_bases_for_candidate_correction_factor <= 1.0, "");

double	qscore_to_error_prob2[256];
double	qscore_to_graph_weight2[256];


void hamming_vote_global_init(){

	constexpr int ASCII_BASE = 33;
	constexpr double MIN_GRAPH_WEIGHT = 0.001;

	for(int i = 0; i < 256; i++){
		if(i < ASCII_BASE)
			qscore_to_error_prob2[i] = 1.0;
		else
			qscore_to_error_prob2[i] = std::pow(10.0, -(i-ASCII_BASE)/10.0);
	}

	for(int i = 0; i < 256; i++){
		qscore_to_graph_weight2[i] = std::max(MIN_GRAPH_WEIGHT, 1.0 - qscore_to_error_prob2[i]);
	}
}

int cpu_hamming_vote_new(std::string& subject, 
				std::vector<std::string>& queries, 
				const std::vector<AlignResult>& alignments,
				const std::string& subjectqualityScores, 
				const std::vector<std::string>& queryqualityScores,
				double maxErrorRate,
				bool useQScores,
				std::vector<bool>& correctedQueries,
				bool correctQueries_){

	constexpr int estimatedCoverage = 255;
	constexpr double errorrate = 0.03;
	constexpr double m = 0.6;
	constexpr int candidate_correction_new_cols = 3;
	constexpr int k = 16;
	

	const bool correctQueries = correctQueries_;
	int status = 0;

	int startindex = 0;
	int endindex = subject.length();
	std::vector<double> defaultWeightsPerQuery(queries.size());
	for(size_t i = 0; i < alignments.size(); i++){
		startindex = alignments[i].arc.shift < startindex ? alignments[i].arc.shift : startindex;
		int queryEndsAt = queryqualityScores[i].length() + alignments[i].arc.shift;
		endindex = queryEndsAt > endindex ? queryEndsAt : endindex;
		defaultWeightsPerQuery[i] = 1.0 - std::sqrt(alignments[i].arc.nOps / (alignments[i].arc.overlap * maxErrorRate));
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
				qw *= qscore_to_graph_weight2[(unsigned char)subjectqualityScores[i]];
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
		for(size_t j = 0; j < queries.size(); j++){
			const int baseindex = i - alignments[j].arc.shift;

			if(baseindex >= 0 && baseindex < int(queries[j].length())){ //check query boundary
				double qweight = defaultWeightsPerQuery[j];
				if(useQScores)
					qweight *= qscore_to_graph_weight2[(unsigned char)queryqualityScores[j][baseindex]];
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

	if(isHQ){
#if 1
		//correct anchor
		for(int i = 0; i < int(subject.length()); i++){
			subject[i] = consensus[subjectColumnsBegin_incl + i];
		}
#endif
#if 1
		//correct candidates
		if(correctQueries){
			
			for(int i = 0; i < int(queries.size()); i++){
				int queryColumnsBegin_incl = alignments[i].arc.shift - startindex;
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
#if 0
		//correct anchor
		for(int i = 0; i < int(subject.length()); i++){
			columnindex = subjectColumnsBegin_incl + i;

			if(support[columnindex] >= 1-3*errorrate){
#if 1
				subject[i] = consensus[columnindex];
#endif
			}else{
#if 0
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
					}
				}
#endif
			}
		}
#endif
	}


	return status;
}

std::string cpu_hamming_vote(const std::string& subject, 
				std::vector<std::string>& queries, 
				const std::vector<AlignResult>& alignments,
				const std::string& subjectqualityScores, 
				const std::vector<std::string>& queryqualityScores,
				double maxErrorRate,
				double alpha, 
				double x,
				bool useQScores,
				const std::vector<bool> correctThisQuery,
				bool correctQueries_){

	constexpr int estimatedCoverage = 200;
	constexpr int coverageThreshold = estimatedCoverage / 3;
	

	const bool correctQueries = correctQueries_;

	// min and max shift of queries which will be consensusBase
	int minshift = subject.length();
	int maxshift = -subject.length();
	for(size_t i = 0; i < alignments.size(); i++){
		if(correctQueries && correctThisQuery[i]){
			minshift = minshift > alignments[i].arc.shift ? alignments[i].arc.shift : minshift;
			maxshift = maxshift > alignments[i].arc.shift ? alignments[i].arc.shift : maxshift;
		}
	}
	//std::cout << "minshift " << minshift << " maxshift " << maxshift << std::endl;

	std::vector<double> defaultWeightsPerQuery(queries.size());
	for(size_t i = 0; i < queries.size(); i++)
		defaultWeightsPerQuery[i] = 1.0 - std::sqrt(alignments[i].arc.nOps / (alignments[i].arc.overlap * maxErrorRate));

	std::string result = subject;
	
	int startindex = std::min(minshift, 0);
	int endindex = std::max(subject.length(), subject.length() + maxshift);

	for(int i = startindex; i < endindex; i++){
		// weights for bases A C G T N
		std::array<double, 5> weights{0,0,0,0,0};
		std::array<double, 5> counts{0,0,0,0,0};
		
		//count subject base
		if(i >= 0 && i < int(subject.length())){
			double qw = 1.0;
			if(useQScores)
				qw *= qscore_to_graph_weight2[(unsigned char)subjectqualityScores[i]];
			switch(subject[i]){
				case 'A': weights[0] += qw; counts[0]++; break;
				case 'C': weights[1] += qw; counts[1]++; break;
				case 'G': weights[2] += qw; counts[2]++; break;
				case 'T': weights[3] += qw; counts[3]++; break;
				case 'N': weights[4] += qw; counts[4]++; break;
				default: break;
			}
		}

		//count query bases
		for(size_t j = 0; j < queries.size(); j++){
			const int baseindex = i - alignments[j].arc.shift;
			//if(baseindex >= alignments[j].arc.query_begin_incl && baseindex < alignments[j].arc.query_begin_incl + alignments[j].arc.overlap){
			if(baseindex >= 0 && baseindex < int(queries[j].length())){
				double qweight = defaultWeightsPerQuery[j];
				if(useQScores)
					qweight *= qscore_to_graph_weight2[(unsigned char)queryqualityScores[j][baseindex]];
				switch(queries[j][baseindex]){
					case 'A': weights[0] += qweight; counts[0]++; break;
					case 'C': weights[1] += qweight; counts[1]++; break;
					case 'G': weights[2] += qweight; counts[2]++; break;
					case 'T': weights[3] += qweight; counts[3]++; break;
					case 'N': weights[4] += qweight; counts[4]++; break;
					default: break;
				}
			}
		}

		//find final count of subject base
		int origcount = 0;
		if(i >= 0 && i < int(subject.length()))
			switch(subject[i]){
				case 'A': origcount = counts[0]; break;
				case 'C': origcount = counts[1]; break;
				case 'G': origcount = counts[2]; break;
				case 'T': origcount = counts[3]; break;
				case 'N': origcount = counts[4]; break;
				default: break;
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

		char consensusBase = ' ';
		switch(maxindex){
			case 0: consensusBase = 'A'; break;
			case 1: consensusBase = 'C'; break;
			case 2: consensusBase = 'G'; break;
			case 3: consensusBase = 'T'; break;
			case 4: consensusBase = 'N'; break;
			default: break;
		}

		double restweight = (columnWeight - consensusWeight);
		if(origcount <= coverageThreshold && (consensusWeight - restweight) >= alpha * std::pow(x, restweight)){
			// we correct this position
			if(i >= 0 && i < int(subject.length()))
				result[i] = consensusBase;

			if(correctQueries){
				for(size_t j = 0; j < queries.size(); j++){
					if(correctThisQuery[j]){
						const int baseindex = i - alignments[j].arc.shift;

						if(baseindex >= 0 && baseindex < int(queries[j].length())){
							queries[j][baseindex] = consensusBase;
						}
					}
				}
			}
		}



	}

	return result;
}




#if 0
def __NVCC__

/*
void cuda_shifted_hamming_distance(CudaAlignResult* result_out, AlignOp* ops_out,
					int max_ops,
					const char* subjects, 
					int nsubjects,				
					const int* subjectBytesPrefixSum, 
					const int* subjectLengths, 
					const int* isEncodedSubjects,
					const char* queries,
					int ncandidates,
					const int* queryBytesPrefixSum, 
					const int* queryLengths, 
					const int* isEncodedQueries,
					const int* queriesPerSubjectPrefixSum,
					int maxSubjectLength){
*/

/*
constexpr int ASCII_BASE = 33;
	constexpr double MIN_GRAPH_WEIGHT = 0.001;

	for(int i = 0; i < 256; i++){
		if(i < ASCII_BASE)
			qscore_to_error_prob2[i] = 1.0;
		else
			qscore_to_error_prob2[i] = std::pow(10.0, -(i-ASCII_BASE)/10.0);
	}

	for(int i = 0; i < 256; i++){
		qscore_to_graph_weight2[i] = std::max(MIN_GRAPH_WEIGHT, 1.0 - qscore_to_error_prob2[i]);
	}
*/

__global__
void hamming_vote_kernel(char* subject, int subjectbytes, int subjectlength, bool subjectIsEncoded_,
				char* queries,
				int ncandidates,
				const int* queryBytesPrefixSum, 
				const int* queryLengths, 
				const int* isEncodedQueries,
				const CudaAlignResult* alignments,
				const int* overlapErrors, 
				const int* overlapSizes,
				const char* subjectqualityScores, 
				const char* queryqualityScores,
				double maxErrorRate,
				double alpha, 
				double x,
				bool useQScores_,
				const char* correctThisQuery,
				bool correctQueries_){

	const bool correctQueries = correctQueries_;
	const bool useQScores = useQScores_;
	const bool subjectIsEncoded = subjectIsEncoded_;

	extern __shared__ double defaultWeightPerCandidate[];
	extern __shared__ bool candidateIsEncoded[];

	for(int i = threadIdx.x; i < ncandidates; i += blockDim.x){
		defaultWeightPerCandidate[i] = 1.0 - sqrt(overlapErrors[i] / (overlapSizes[i] * maxErrorRate));
		candidateIsEncoded[i] = (isEncodedQueries[i] == 1);
	}
	
	__syncthreads();

	// loop over length of subject. each thread is responsible for vote in column i
	for(int i = threadIdx.x; i < ncandidates; i += blockDim.x){
		double weights[5]{0,0,0,0,0}; // weights for bases A C G T N
		char subjectbase = subjectIsEncoded ? cuda_encoded_accessor(subject, subjectlength, i) : cuda_ordinary_accessor(subject, i);
		//number of reads overlapping with subject at position i ( including subject )
		int readcount = 1;

		//count subject base
		double qw = 1.0;
		if(useQScores)
			qw *= qscore_to_graph_weight2[(unsigned char)subjectqualityScores[i]]; //TODO
		switch(subjectbase){
			case 'A': weights[0] += qw; break;
			case 'C': weights[1] += qw; break;
			case 'G': weights[2] += qw; break;
			case 'T': weights[3] += qw; break;
			case 'N': weights[4] += qw; break;
			default: break;
		}

		//count query bases which overlap with subject[i]
		for(int j = 0; j < ncandidates; j++){
			if(alignments[j].subject_begin_incl <= i && i < alignments[j].subject_end_excl){
				readcount++;
				const int baseindex = i - alignments[j].shift;
				double qweight = defaultWeightPerCandidate[j];
				if(useQScores)
					qweight *= qscore_to_graph_weight2[(unsigned char)queryqualityScores[j][baseindex]]; //TODO
				const char* query = queries + queryBytesPrefixSum[j];
				const char querybase = candidateIsEncoded[j] ? cuda_encoded_accessor(query, queryLengths[j], baseindex) 
										: cuda_ordinary_accessor(query, baseindex);
				switch(querybase){
					case 'A': weights[0] += qweight; break;
					case 'C': weights[1] += qweight; break;
					case 'G': weights[2] += qweight; break;
					case 'T': weights[3] += qweight; break;
					case 'N': weights[4] += qweight; break;
					default: break;
				}
			}
		}

		//find final count of subject base
		double origbaseweight = 0;
		switch(subjectbase){
			case 'A': origbaseweight = weights[0]; break;
			case 'C': origbaseweight = weights[1]; break;
			case 'G': origbaseweight = weights[2]; break;
			case 'T': origbaseweight = weights[3]; break;
			case 'N': origbaseweight = weights[4]; break;
			default: break;
		}

		double consensusWeight = 0;
		int maxindex = 4;

		#pragma unroll
		for(int k = 0; k < 5; k++){
			if(weights[k] > consensusWeight){
				consensusWeight = weights[k];
				maxindex = k;
			}
		}

		char consensusBase = ' ';
		switch(maxindex){
			case 0: consensusBase = 'A'; break;
			case 1: consensusBase = 'C'; break;
			case 2: consensusBase = 'G'; break;
			case 3: consensusBase = 'T'; break;
			case 4: consensusBase = 'N'; break;
		}

		//if this is true, correct
		if((consensusWeight - origbaseweight) >= alpha * pow(x, origbaseweight) ){
			subject[i] = consensusBase; //TODO write encoded
		}

		//correct queries too if wanted
		if(correctQueries){
			for(size_t j = 0; j < queries.size(); j++){
				if(correctThisQuery[j] && alignments[j].subject_begin_incl <= i && i < alignments[j].subject_end_excl){
					const int baseindex = i - alignments[j].shift;
					const char* query = queries + queryBytesPrefixSum[j];
					const char querybase = candidateIsEncoded[j] ? cuda_encoded_accessor(query, queryLengths[j], baseindex) 
											: cuda_ordinary_accessor(query, baseindex);
					origbaseweight = 0.0;
					switch(querybase){
						case 'A': origbaseweight = weights[0]; break;
						case 'C': origbaseweight = weights[1]; break;
						case 'G': origbaseweight = weights[2]; break;
						case 'T': origbaseweight = weights[3]; break;
						case 'N': origbaseweight = weights[4]; break;
						default: break;
					}
					if((consensusWeight - origbaseweight) >= alpha * std::pow(x, origbaseweight) ){
						query[baseindex] = consensusBase; //TODO write encoded
					}
				}
			}
		}
	}




	}

	return result;
}

#endif



