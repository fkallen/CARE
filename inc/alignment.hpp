#ifndef ALIGNMENT_HPP
#define ALIGNMENT_HPP

#include "hpc_helpers.cuh"
#ifdef __NVCC__
	#include "cuda_unique.cuh"
#endif

#include <vector>
#include <algorithm>
#include <tuple>
#include <ostream>

namespace care{


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

    int get_score() const{
        return score;
    }
    int get_subject_begin_incl() const{
        return subject_begin_incl;
    }
    int get_query_begin_incl() const{
        return query_begin_incl;
    }
    int get_overlap() const{
        return overlap;
    }
    int get_shift() const{
        return shift;
    }
    int get_nOps() const{
        return nOps;
    }
    bool get_isNormalized() const{
        return isNormalized;
    }
    bool get_isValid() const{
        return isValid;
    }

    int& get_score(){
        return score;
    }
    int& get_subject_begin_incl(){
        return subject_begin_incl;
    }
    int& get_query_begin_incl(){
        return query_begin_incl;
    }
    int& get_overlap(){
        return overlap;
    }
    int& get_shift(){
        return shift;
    }
    int& get_nOps(){
        return nOps;
    }
    bool& get_isNormalized(){
        return isNormalized;
    }
    bool& get_isValid(){
        return isValid;
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

    int get_score() const{
        return arc.score;
    }
    int get_subject_begin_incl() const{
        return arc.subject_begin_incl;
    }
    int get_query_begin_incl() const{
        return arc.query_begin_incl;
    }
    int get_overlap() const{
        return arc.overlap;
    }
    int get_shift() const{
        return arc.shift;
    }
    int get_nOps() const{
        return arc.nOps;
    }
    bool get_isNormalized() const{
        return arc.isNormalized;
    }
    bool get_isValid() const{
        return arc.isValid;
    }

    int& get_score(){
        return arc.score;
    }
    int& get_subject_begin_incl(){
        return arc.subject_begin_incl;
    }
    int& get_query_begin_incl(){
        return arc.query_begin_incl;
    }
    int& get_overlap(){
        return arc.overlap;
    }
    int& get_shift(){
        return arc.shift;
    }
    int& get_nOps(){
        return arc.nOps;
    }
    bool& get_isNormalized(){
        return arc.isNormalized;
    }
    bool& get_isValid(){
        return arc.isValid;
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



}

#endif
