#include "../inc/hammingvote.hpp"
#include "../inc/alignment.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <vector>

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

	const bool correctQueries = correctQueries_;

	// min and max shift of queries which will be corrected
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
		
		//count subject base
		if(i >= 0 && i < int(subject.length())){
			double qw = 1.0;
			if(useQScores)
				qw *= qscore_to_graph_weight2[(unsigned char)subjectqualityScores[i]];
			switch(subject[i]){
				case 'A': weights[0] += qw; break;
				case 'C': weights[1] += qw; break;
				case 'G': weights[2] += qw; break;
				case 'T': weights[3] += qw; break;
				case 'N': weights[4] += qw; break;
				default: break;
			}
		}

		//count query bases
		for(size_t j = 0; j < queries.size(); j++){
			const int baseindex = i - alignments[j].arc.shift;
			if(baseindex >= alignments[j].arc.query_begin_incl && baseindex < alignments[j].arc.query_begin_incl + alignments[j].arc.overlap){
				double qweight = defaultWeightsPerQuery[j];
				if(useQScores)
					qweight *= qscore_to_graph_weight2[(unsigned char)queryqualityScores[j][baseindex]];
				switch(queries[j][baseindex]){
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
		switch(subject[i]){
			case 'A': origbaseweight = weights[0]; break;
			case 'C': origbaseweight = weights[1]; break;
			case 'G': origbaseweight = weights[2]; break;
			case 'T': origbaseweight = weights[3]; break;
			case 'N': origbaseweight = weights[4]; break;
			default: break;
		}

		double maxweight = 0;
		int maxindex = 4;
		for(int k = 0; k < 5; k++){
			if(weights[k] > maxweight){
				maxweight = weights[k];
				maxindex = k;
			}
		}

		char corrected = ' ';
		switch(maxindex){
			case 0: corrected = 'A'; break;
			case 1: corrected = 'C'; break;
			case 2: corrected = 'G'; break;
			case 3: corrected = 'T'; break;
			case 4: corrected = 'N'; break;
		}

		//if this is true, correct
		if((maxweight - origbaseweight) >= alpha * std::pow(x, origbaseweight) ){
			result[i] = corrected;
		}

		//correct queries too if wanted
		if(correctQueries){
			for(size_t j = 0; j < queries.size(); j++){
				if(correctThisQuery[j]){
					const int baseindex = i - alignments[j].arc.shift;

					if(baseindex >= 0 && baseindex < int(queries[j].length())){
						origbaseweight = 0.0;
						switch(queries[j][baseindex]){
							case 'A': origbaseweight = weights[0]; break;
							case 'C': origbaseweight = weights[1]; break;
							case 'G': origbaseweight = weights[2]; break;
							case 'T': origbaseweight = weights[3]; break;
							case 'N': origbaseweight = weights[4]; break;
							default: break;
						}
						if((maxweight - origbaseweight) >= alpha * std::pow(x, origbaseweight) ){
							queries[j][baseindex] = corrected;
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

		double maxweight = 0;
		int maxindex = 4;

		#pragma unroll
		for(int k = 0; k < 5; k++){
			if(weights[k] > maxweight){
				maxweight = weights[k];
				maxindex = k;
			}
		}

		char corrected = ' ';
		switch(maxindex){
			case 0: corrected = 'A'; break;
			case 1: corrected = 'C'; break;
			case 2: corrected = 'G'; break;
			case 3: corrected = 'T'; break;
			case 4: corrected = 'N'; break;
		}

		//if this is true, correct
		if((maxweight - origbaseweight) >= alpha * pow(x, origbaseweight) ){
			subject[i] = corrected; //TODO write encoded
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
					if((maxweight - origbaseweight) >= alpha * std::pow(x, origbaseweight) ){
						query[baseindex] = corrected; //TODO write encoded
					}
				}
			}
		}
	}




	}

	return result;
}

#endif



