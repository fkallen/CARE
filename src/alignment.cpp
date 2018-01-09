#include "../inc/alignment.hpp"

#include <algorithm>
#include <tuple>

// split substitutions in alignment into deletion + insertion
int split_subs(AlignResult& alignment, const char* subject){
	auto& ops = alignment.operations;
	int splitted_subs = 0;
	for(auto it = ops.begin(); it != ops.end(); it++){
		if(it->type == ALIGNTYPE_SUBSTITUTE){
			AlignOp del = *it;
			del.base = subject[it->position];
			del.type = ALIGNTYPE_DELETE;

			AlignOp ins = *it;
			ins.type = ALIGNTYPE_INSERT;

			it = ops.erase(it);
			it = ops.insert(it, del);
			it = ops.insert(it, ins);
			splitted_subs++;
		}
	}
	return splitted_subs;
};



// Given AlignmentResults for a read and its reverse complement, find the "best" of both alignments

//first tuple entry is 0 if fwdAlignment, 1 if revcmplAlignment, -1 if none
//second tuple entry is overlapsize
//third tuple entry is number of mismatches in overlap
std::tuple<BestAlignment_t,int,int> get_best_alignment(const AlignResult& fwdAlignment, const AlignResult& revcmplAlignment, 
				int querylength, int candidatelength,
				double MAX_MISMATCH_RATIO, int MIN_OVERLAP, double MIN_OVERLAP_RATIO){

	const int overlap = fwdAlignment.arc.subject_end_excl - fwdAlignment.arc.subject_begin_incl;
	const int revcomploverlap = revcmplAlignment.arc.subject_end_excl - revcmplAlignment.arc.subject_begin_incl;

	BestAlignment_t retval = BestAlignment_t::None;
	int retoverlap = 0;
	int retnmismatch = 0;

	MIN_OVERLAP = std::max(int(querylength * MIN_OVERLAP_RATIO), MIN_OVERLAP);

// if 1, then size of overlap is used to chose between sequence and reverse complement
// else the mismatch ratio is used

#if 0
	// choose longest overlap
	if(overlap > revcomploverlap){
		if(overlap >= MIN_OVERLAP){
			int nMismatch = std::count_if(fwdAlignment.operations.cbegin(), fwdAlignment.operations.cend(),
						      [](const AlignOp& op){
				return op.type == ALIGNTYPE_SUBSTITUTE 
					|| op.type == ALIGNTYPE_INSERT 
					|| op.type == ALIGNTYPE_DELETE;
			});
			if((double)nMismatch / overlap < MAX_MISMATCH_RATIO){
				retval = BestAlignment_t::Forward;
				retoverlap = overlap;
				retnmismatch = nMismatch;
			}
		}
	}else if(overlap < revcomploverlap){
		if(revcomploverlap >= MIN_OVERLAP){
			int nMismatch = std::count_if(revcmplAlignment.operations.cbegin(), revcmplAlignment.operations.cend(),
						      [](const AlignOp& op){
				return op.type == ALIGNTYPE_SUBSTITUTE || op.type == ALIGNTYPE_INSERT || op.type == ALIGNTYPE_DELETE;
			});
			if((double)nMismatch / overlap < MAX_MISMATCH_RATIO){
				retval = BestAlignment_t::ReverseComplement;
				retoverlap = revcomploverlap;
				retnmismatch = nMismatch;
			}
		}
	}else{
		if(overlap >= MIN_OVERLAP){
			// overlaps are of equal size, choose lowest mismatch ratio
			const int nMismatch = std::count_if(fwdAlignment.operations.cbegin(), fwdAlignment.operations.cend(),
						      [](const AlignOp& op){
				return op.type == ALIGNTYPE_SUBSTITUTE || op.type == ALIGNTYPE_INSERT || op.type == ALIGNTYPE_DELETE;
			});
			const int revcomplnMismatch = std::count_if(revcmplAlignment.operations.cbegin(), revcmplAlignment.operations.cend(),
						      [](const AlignOp& op){
				return op.type == ALIGNTYPE_SUBSTITUTE || op.type == ALIGNTYPE_INSERT || op.type == ALIGNTYPE_DELETE;
			});
			const double ratio = (double)nMismatch / overlap;
			const double revcomplratio = (double)revcomplnMismatch / revcomploverlap;

			if(ratio < revcomplratio){
				if(ratio < MAX_MISMATCH_RATIO){
					retval = BestAlignment_t::Forward;
					retoverlap = overlap;
					retnmismatch = nMismatch;
				}				
			}else{
				if(revcomplratio < MAX_MISMATCH_RATIO){
					retval = BestAlignment_t::ReverseComplement;
					retoverlap = revcomploverlap;
					retnmismatch = revcomplnMismatch;
				}
			}
		}
	}

#else

	if(fwdAlignment.arc.isValid && overlap >= MIN_OVERLAP){
		if(revcmplAlignment.arc.isValid && revcomploverlap >= MIN_OVERLAP){
			const int nMismatch = std::count_if(fwdAlignment.operations.cbegin(), fwdAlignment.operations.cend(),
						      [](const AlignOp& op){
				return op.type == ALIGNTYPE_SUBSTITUTE || op.type == ALIGNTYPE_INSERT || op.type == ALIGNTYPE_DELETE;
			});
			const int revcomplnMismatch = std::count_if(revcmplAlignment.operations.cbegin(), revcmplAlignment.operations.cend(),
						      [](const AlignOp& op){
				return op.type == ALIGNTYPE_SUBSTITUTE || op.type == ALIGNTYPE_INSERT || op.type == ALIGNTYPE_DELETE;
			});
			const double ratio = (double)nMismatch / overlap;
			const double revcomplratio = (double)revcomplnMismatch / revcomploverlap;

			if(ratio < revcomplratio){
				if(ratio < MAX_MISMATCH_RATIO){
					retval = BestAlignment_t::Forward;
					retoverlap = overlap;
					retnmismatch = nMismatch;
				}			
			}else if(revcomplratio < ratio){
				if(revcomplratio < MAX_MISMATCH_RATIO){
					retval = BestAlignment_t::ReverseComplement;
					retoverlap = revcomploverlap;
					retnmismatch = revcomplnMismatch;
				}
			}else{
				if(ratio < MAX_MISMATCH_RATIO){
					// both have same mismatch ratio, choose longest overlap
					if(overlap > revcomploverlap){
						retval = BestAlignment_t::Forward;
						retoverlap = overlap;
						retnmismatch = nMismatch;
					}else{
						retval = BestAlignment_t::ReverseComplement;
						retoverlap = revcomploverlap;
						retnmismatch = revcomplnMismatch;
					}
				}
			}
		}else{
			int nMismatch = std::count_if(fwdAlignment.operations.cbegin(), fwdAlignment.operations.cend(),
						      [](const AlignOp& op){
				return op.type == ALIGNTYPE_SUBSTITUTE || op.type == ALIGNTYPE_INSERT || op.type == ALIGNTYPE_DELETE;
			});
			if((double)nMismatch / overlap < MAX_MISMATCH_RATIO){
				retval = BestAlignment_t::Forward;
				retoverlap = overlap;
				retnmismatch = nMismatch;
			}
		}
	}else{
		if(revcmplAlignment.arc.isValid && revcomploverlap >= MIN_OVERLAP){
			int nMismatch = std::count_if(revcmplAlignment.operations.cbegin(), revcmplAlignment.operations.cend(),
						      [](const AlignOp& op){
				return op.type == ALIGNTYPE_SUBSTITUTE || op.type == ALIGNTYPE_INSERT || op.type == ALIGNTYPE_DELETE;
			});
			if((double)nMismatch / revcomploverlap < MAX_MISMATCH_RATIO){
				retval = BestAlignment_t::ReverseComplement;
				retoverlap = revcomploverlap;
				retnmismatch = nMismatch;
			}
		}
	}
#endif

	return std::make_tuple(retval, retoverlap, retnmismatch);	
}
