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

	const int overlap = fwdAlignment.arc.overlap;
	const int revcomploverlap = revcmplAlignment.arc.overlap;
	const int fwdMismatches = fwdAlignment.arc.nOps;
	const int revcmplMismatches = revcmplAlignment.arc.nOps;

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
			if((double)fwdMismatches / overlap < MAX_MISMATCH_RATIO){
				retval = BestAlignment_t::Forward;
				retoverlap = overlap;
				retnmismatch = fwdMismatches;
			}
		}
	}else if(overlap < revcomploverlap){
		if(revcomploverlap >= MIN_OVERLAP){
			if((double)revcmplMismatches / overlap < MAX_MISMATCH_RATIO){
				retval = BestAlignment_t::ReverseComplement;
				retoverlap = revcomploverlap;
				retnmismatch = revcmplMismatches;
			}
		}
	}else{
		if(overlap >= MIN_OVERLAP){
			// overlaps are of equal size, choose lowest mismatch ratio

			const double ratio = (double)fwdMismatches / overlap;
			const double revcomplratio = (double)revcmplMismatches / revcomploverlap;

			if(ratio < revcomplratio){
				if(ratio < MAX_MISMATCH_RATIO){
					retval = BestAlignment_t::Forward;
					retoverlap = overlap;
					retnmismatch = fwdMismatches;
				}				
			}else{
				if(revcomplratio < MAX_MISMATCH_RATIO){
					retval = BestAlignment_t::ReverseComplement;
					retoverlap = revcomploverlap;
					retnmismatch = revcmplMismatches;
				}
			}
		}
	}

#else

	if(fwdAlignment.arc.isValid && overlap >= MIN_OVERLAP){
		if(revcmplAlignment.arc.isValid && revcomploverlap >= MIN_OVERLAP){
			const double ratio = (double)fwdMismatches / overlap;
			const double revcomplratio = (double)revcmplMismatches / revcomploverlap;

			if(ratio < revcomplratio){
				if(ratio < MAX_MISMATCH_RATIO){
					retval = BestAlignment_t::Forward;
					retoverlap = overlap;
					retnmismatch = fwdMismatches;
				}			
			}else if(revcomplratio < ratio){
				if(revcomplratio < MAX_MISMATCH_RATIO){
					retval = BestAlignment_t::ReverseComplement;
					retoverlap = revcomploverlap;
					retnmismatch = revcmplMismatches;
				}
			}else{
				if(ratio < MAX_MISMATCH_RATIO){
					// both have same mismatch ratio, choose longest overlap
					if(overlap > revcomploverlap){
						retval = BestAlignment_t::Forward;
						retoverlap = overlap;
						retnmismatch = fwdMismatches;
					}else{
						retval = BestAlignment_t::ReverseComplement;
						retoverlap = revcomploverlap;
						retnmismatch = revcmplMismatches;
					}
				}
			}
		}else{
			if((double)fwdMismatches / overlap < MAX_MISMATCH_RATIO){
				retval = BestAlignment_t::Forward;
				retoverlap = overlap;
				retnmismatch = fwdMismatches;
			}
		}
	}else{
		if(revcmplAlignment.arc.isValid && revcomploverlap >= MIN_OVERLAP){
			if((double)revcmplMismatches / revcomploverlap < MAX_MISMATCH_RATIO){
				retval = BestAlignment_t::ReverseComplement;
				retoverlap = revcomploverlap;
				retnmismatch = revcmplMismatches;
			}
		}
	}
#endif

	return std::make_tuple(retval, retoverlap, retnmismatch);	
}
