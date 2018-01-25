#include "../inc/aligner.hpp"

#include "../inc/alignment.hpp"
#include "../inc/alignment_semi_global.hpp"
#include "../inc/ganja/hpc_helpers.cuh"
#include "../inc/hamming.hpp"

#include <vector>

SemiGlobalAligner::SemiGlobalAligner(int m, int s, int i, int d) : matchscore(m), subscore(s), insertscore(i), delscore(d){
	this->type = AlignerType::SemiGlobal;
}

AlignResult SemiGlobalAligner::cpu_alignment(const char* subject, const char* query, int ns, int nq, bool subjectIsEncoded, bool queryIsEncoded) const{
	return cpu_semi_global_align(subject, query, ns, nq, subjectIsEncoded, queryIsEncoded, matchscore, subscore, insertscore, delscore);
}

#ifdef __NVCC__	
std::vector<std::vector<AlignResult>> SemiGlobalAligner::cuda_alignment(const AlignerDataArrays& mybuffers, 
			std::vector<bool> activeBatches, int max_ops_per_alignment, 
			int nsubjects, int ncandidates, int maxSubjectLength, int maxQueryLength) const{

	call_cuda_semi_global_align_kernel(
			mybuffers.d_results.get(),
			mybuffers.d_ops.get(),
			max_ops_per_alignment, 
			nsubjects, 
			ncandidates,
			mybuffers.d_subjectsdata.get(), 
			mybuffers.d_queriesdata.get(),
			mybuffers.d_rBytesPrefixSum.get(),
			mybuffers.d_rLengths.get(),
			mybuffers.d_rIsEncoded.get(),
			mybuffers.d_cBytesPrefixSum.get(),
			mybuffers.d_cLengths.get(),
			mybuffers.d_cIsEncoded.get(),
			mybuffers.d_r2PerR1.get(),
			maxSubjectLength, maxQueryLength,
			matchscore, subscore, insertscore, delscore,
			mybuffers.stream);

	// copy result to host
	cudaMemcpyAsync(mybuffers.h_results.get(), 
			mybuffers.d_results.get(), 
			sizeof(AlignResultCompact) * ncandidates, 
			D2H, 
			mybuffers.stream); CUERR;
	cudaMemcpyAsync(mybuffers.h_ops.get(), 
			mybuffers.d_ops.get(), 
			sizeof(AlignOp) * max_ops_per_alignment * ncandidates, 
			D2H, 
			mybuffers.stream); CUERR;

	cudaStreamSynchronize(mybuffers.stream); CUERR;

	// store alignments in result vector

	int candidateIndex = 0;

	std::vector<std::vector<AlignResult>> alignments(activeBatches.size());
	int previousOutputIndex = -1;

	for(int i = 0; i < nsubjects; i++){

		int outputindex = previousOutputIndex + 1;
		while(!activeBatches[outputindex]) outputindex++;
		previousOutputIndex = outputindex;

		int nqueriesForThisSubject = mybuffers.h_r2PerR1.get()[i+1] - mybuffers.h_r2PerR1.get()[i];

		alignments[outputindex].resize(nqueriesForThisSubject);

		for(int j = 0; j < nqueriesForThisSubject; j++){
			const auto& currentResult = mybuffers.h_results.get()[candidateIndex];
			alignments[outputindex][j].setOpsAndDataFromAlignResultCompact(currentResult, 
										mybuffers.h_ops.get() + candidateIndex * max_ops_per_alignment, 
										true);
			candidateIndex++; 
		}			
	}

	return alignments;
}
#endif



ShiftedHammingDistance::ShiftedHammingDistance(){
	this->type = AlignerType::ShiftedHamming;
}

AlignResult ShiftedHammingDistance::cpu_alignment(const char* subject, const char* query, int ns, int nq, bool subjectIsEncoded, bool queryIsEncoded) const{
	return cpu_shifted_hamming_distance(subject, query, ns, nq, subjectIsEncoded, queryIsEncoded);
}

#ifdef __NVCC__	
std::vector<std::vector<AlignResult>> ShiftedHammingDistance::cuda_alignment(const AlignerDataArrays& mybuffers, 
			std::vector<bool> activeBatches, int max_ops_per_alignment, 
			int nsubjects, int ncandidates, int maxSubjectLength, int maxQueryLength) const{

	call_cuda_shifted_hamming_distance_kernel(
			mybuffers.d_results.get(),
			mybuffers.d_ops.get(),
			max_ops_per_alignment, 
			mybuffers.d_subjectsdata.get(),
			nsubjects, 
			mybuffers.d_rBytesPrefixSum.get(),
			mybuffers.d_rLengths.get(),
			mybuffers.d_rIsEncoded.get(),
			mybuffers.d_queriesdata.get(),
			ncandidates,
			mybuffers.d_cBytesPrefixSum.get(),
			mybuffers.d_cLengths.get(),
			mybuffers.d_cIsEncoded.get(),
			mybuffers.d_r2PerR1.get(),
			maxSubjectLength, maxQueryLength,
			mybuffers.stream); cudaStreamSynchronize(mybuffers.stream); CUERR;

	// copy result to host
	cudaMemcpyAsync(mybuffers.h_results.get(), 
			mybuffers.d_results.get(), 
			sizeof(AlignResultCompact) * ncandidates, 
			D2H, 
			mybuffers.stream); CUERR; cudaStreamSynchronize(mybuffers.stream); CUERR;
	/*cudaMemcpyAsync(mybuffers.h_ops.get(), 
			mybuffers.d_ops.get(), 
			sizeof(AlignOp) * max_ops_per_alignment * ncandidates, 
			D2H, 
			mybuffers.stream); CUERR; cudaStreamSynchronize(mybuffers.stream); CUERR;*/

	cudaStreamSynchronize(mybuffers.stream); CUERR;

	// store alignments in result vector

	int candidateIndex = 0;

	std::vector<std::vector<AlignResult>> alignments(activeBatches.size());
	int previousOutputIndex = -1;

	for(int i = 0; i < nsubjects; i++){

		int outputindex = previousOutputIndex + 1;
		while(!activeBatches[outputindex]) outputindex++;
		previousOutputIndex = outputindex;

		int nqueriesForThisSubject = mybuffers.h_r2PerR1.get()[i+1] - mybuffers.h_r2PerR1.get()[i];

		alignments[outputindex].resize(nqueriesForThisSubject);

		for(int j = 0; j < nqueriesForThisSubject; j++){
			const auto& currentResult = mybuffers.h_results.get()[candidateIndex];
			/*alignments[outputindex][j].setOpsAndDataFromAlignResultCompact(currentResult, 
										mybuffers.h_ops.get() + candidateIndex * max_ops_per_alignment, 
										true);*/
			alignments[outputindex][j].setDataFromAlignResultCompact(currentResult);
			candidateIndex++; 
		}			
	}

	return alignments;
}
#endif

