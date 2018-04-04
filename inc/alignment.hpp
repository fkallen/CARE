#ifndef ALIGNMENT_HPP
#define ALIGNMENT_HPP

#include "ganja/hpc_helpers.cuh"
#ifdef __NVCC__
	#include "cuda_unique.cuh"
#endif

#include <vector>
#include <algorithm>
#include <tuple>
#include <ostream>

namespace care{

enum class BestAlignment_t {Forward, ReverseComplement, None};
enum class AlignmentDevice {CPU, GPU, None};

enum AlignType : char{
	ALIGNTYPE_MATCH,
	ALIGNTYPE_SUBSTITUTE,
	ALIGNTYPE_INSERT,
	ALIGNTYPE_DELETE,
};

struct AlignOp {
	short position;
	AlignType type;
	char base;

	HOSTDEVICEQUALIFIER
	AlignOp(){}

	HOSTDEVICEQUALIFIER
	AlignOp(short p, AlignType t, char b)
		: position(p), type(t), base(b){}

    HOSTDEVICEQUALIFIER
	AlignOp(const AlignOp& other){
        *this = other;
    }

    HOSTDEVICEQUALIFIER
	AlignOp(AlignOp&& other){
        *this = std::move(other);
    }

    HOSTDEVICEQUALIFIER
	AlignOp& operator=(const AlignOp& other){
        position = other.position;
        type = other.type;
        base = other.base;
        return *this;
    }

    HOSTDEVICEQUALIFIER
	AlignOp& operator=(AlignOp&& other){
        position = other.position;
        type = other.type;
        base = other.base;
        return *this;
    }

	HOSTDEVICEQUALIFIER
	bool operator==(const AlignOp& other) const{
		return (position == other.position && type == other.type && base == other.base);
	}

	HOSTDEVICEQUALIFIER
	bool operator!=(const AlignOp& other) const{
		return !(*this == other);
	}

	/*void writeOpToStream(std::ostream& stream){
		auto TYPE = [](AlignType i){
			switch(i){
			case ALIGNTYPE_MATCH : return "M"; break;
			case ALIGNTYPE_SUBSTITUTE : return "S"; break;
			case ALIGNTYPE_INSERT : return "I"; break;
			case ALIGNTYPE_DELETE : return "D"; break;
			}
		};
		stream << op.position << " " << TYPE(op.type) << " " << op.base;
		return stream;
	}*/

	friend std::ostream& operator<<(std::ostream& stream, const AlignOp& op){
		auto TYPE = [](AlignType i){
			switch(i){
			case ALIGNTYPE_MATCH : return "M";
			case ALIGNTYPE_SUBSTITUTE : return "S";
			case ALIGNTYPE_INSERT : return "I";
			case ALIGNTYPE_DELETE : return "D";
			default: return "INVALID";
			}
		};
		stream << op.position << " " << TYPE(op.type) << " " << op.base;
		return stream;
	}
};


struct AlignResultCompact{
	int score;
	int subject_begin_incl;
	int query_begin_incl;
	int overlap;
	int shift;
	int nOps; //edit distance / number of operations
	bool isNormalized;
	bool isValid;

    bool operator==(const AlignResultCompact& rhs){
        return score == rhs.score
            && subject_begin_incl == rhs.subject_begin_incl
            && query_begin_incl == rhs.query_begin_incl
            && overlap == rhs.overlap
            && shift == rhs.shift
            && nOps == rhs.nOps
            && isNormalized == rhs.isNormalized
            && isValid == rhs.isValid;
    }
    bool operator!=(const AlignResultCompact& rhs){
        return !(*this == rhs);
    }
};


struct AlignResult{
	AlignResultCompact arc;
	std::vector<AlignOp> operations;

	bool operator==(const AlignResult& other) const{
		if(arc.score != other.arc.score) return false;
		if(arc.subject_begin_incl != other.arc.subject_begin_incl) return false;
		if(arc.query_begin_incl != other.arc.query_begin_incl) return false;
		if(arc.overlap != other.arc.overlap) return false;
		if(arc.shift != other.arc.shift) return false;
		if(arc.nOps != other.arc.nOps) return false;
		if(arc.isNormalized != other.arc.isNormalized) return false;
		if(arc.isValid != other.arc.isValid) return false;
		if(operations != other.operations) return false;
		return true;
	}

	bool operator!=(const AlignResult& other) const{
		return !(*this == other);
	}

	void writeOpsToStream(std::ostream& stream){
		for(size_t i = 0; operations.size() != 0 && i < operations.size()-1; i++)
			stream << operations[i] << '\n';
		stream << operations.back();
	}

	void writeArcToStream(std::ostream& stream){
		stream << "score: " << arc.score << '\n';
		stream << "subject_begin_incl: " << arc.subject_begin_incl << '\n';
		stream << "query_begin_incl: " << arc.query_begin_incl << '\n';
		stream << "overlap: " << arc.overlap << '\n';
		stream << "shift: " << arc.shift << '\n';
		stream << "nOps: " << arc.nOps << '\n';
		stream << "isNormalized: " << arc.isNormalized << '\n';
		stream << "isValid: " << arc.isValid << '\n';
	}

	void setOpsAndDataFromAlignResultCompact(const AlignResultCompact& cudaresult, const AlignOp* h_ops, bool opsAreReversed){
		setDataFromAlignResultCompact(cudaresult);

		setOpsFromAlignResultCompact(cudaresult, h_ops, opsAreReversed);
	}

	void setDataFromAlignResultCompact(const AlignResultCompact& cudaresult){
		arc = cudaresult;
	}

	void setOpsFromAlignResultCompact(const AlignResultCompact& cudaresult, const AlignOp* h_ops, bool opsAreReversed){
		// reserve space for operations
		operations.resize(cudaresult.nOps);
		// set operations
		if(opsAreReversed)
			std::reverse_copy(h_ops,
				  	h_ops + cudaresult.nOps,
				  	operations.begin());
		else
			std::copy(h_ops,
				  	h_ops + cudaresult.nOps,
				  	operations.begin());
	}
};


// split substitutions in alignment into deletion + insertion
int split_subs(AlignResult& alignment, const char* subject);



// Given AlignmentResults for a read and its reverse complement, find the "best" of both alignments
BestAlignment_t get_best_alignment(const AlignResultCompact& fwdAlignment, const AlignResultCompact& revcmplAlignment,
				int querylength, int candidatelength,
				double MAX_MISMATCH_RATIO, int MIN_OVERLAP, double MIN_OVERLAP_RATIO);


}

#endif
