#ifndef ALIGNER_HPP
#define ALIGNER_HPP

#include "alignment.hpp"
//#include "alignment_semi_global.hpp"
//#include "hpc_helpers.cuh"
//#include "hamming.hpp"

#include <stdexcept>

enum class AlignerType {None, ShiftedHamming, SemiGlobal};

struct Aligner{
	AlignerType type = AlignerType::None;

	virtual AlignResult cpu_alignment(const char* subject, const char* query, int ns, int nq, bool subjectIsEncoded, bool queryIsEncoded) const {
		throw std::runtime_error("Aligner::cpu_alignment not implemented. Use derived class");
	}
	
#ifdef __NVCC__	
	virtual std::vector<std::vector<AlignResult>> cuda_alignment(const AlignerDataArrays& mybuffers, 
				std::vector<bool> activeBatches, int max_ops_per_alignment, 
				int nsubjects, int ncandidates, int maxSubjectLength, int maxQueryLength) const {
		throw std::runtime_error("Aligner::cuda_alignment not implemented. Use derived class");
	}
#endif
};

struct SemiGlobalAligner : public Aligner{

	int matchscore = 1;
	int subscore = -1;
	int insertscore = -1;
	int delscore = -1;

	SemiGlobalAligner(int m, int s, int i, int d);

	AlignResult cpu_alignment(const char* subject, const char* query, int ns, int nq, bool subjectIsEncoded, bool queryIsEncoded) const override;

#ifdef __NVCC__	
	std::vector<std::vector<AlignResult>> cuda_alignment(const AlignerDataArrays& mybuffers, 
				std::vector<bool> activeBatches, int max_ops_per_alignment, 
				int nsubjects, int ncandidates, int maxSubjectLength, int maxQueryLength) const override;
#endif
};


struct ShiftedHammingDistance : public Aligner{

	ShiftedHammingDistance();

	AlignResult cpu_alignment(const char* subject, const char* query, int ns, int nq, bool subjectIsEncoded, bool queryIsEncoded) const override;

#ifdef __NVCC__	
	std::vector<std::vector<AlignResult>> cuda_alignment(const AlignerDataArrays& mybuffers, 
				std::vector<bool> activeBatches, int max_ops_per_alignment, 
				int nsubjects, int ncandidates, int maxSubjectLength, int maxQueryLength) const override;
#endif
};



#endif
