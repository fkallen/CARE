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

enum class BestAlignment_t {Forward, ReverseComplement, None};

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



#ifdef __CUDACC__

struct AlignerDataArrays{

	unique_dev_ptr<AlignResultCompact> d_results;
	unique_dev_ptr<AlignOp> d_ops;
	unique_dev_ptr<char> d_subjectsdata;
	unique_dev_ptr<char> d_queriesdata;
	unique_dev_ptr<int> d_rBytesPrefixSum;
	unique_dev_ptr<int> d_rLengths;
	unique_dev_ptr<int> d_rIsEncoded;
	unique_dev_ptr<int> d_cBytesPrefixSum;
	unique_dev_ptr<int> d_cLengths;
	unique_dev_ptr<int> d_cIsEncoded;
	unique_dev_ptr<int> d_r2PerR1;

	unique_pinned_ptr<AlignResultCompact> h_results;
	unique_pinned_ptr<AlignOp> h_ops;
	unique_pinned_ptr<char> h_subjectsdata;
	unique_pinned_ptr<char> h_queriesdata;
	unique_pinned_ptr<int> h_rBytesPrefixSum;
	unique_pinned_ptr<int> h_rLengths;
	unique_pinned_ptr<int> h_rIsEncoded;
	unique_pinned_ptr<int> h_cBytesPrefixSum;
	unique_pinned_ptr<int> h_cLengths;
	unique_pinned_ptr<int> h_cIsEncoded;
	unique_pinned_ptr<int> h_r2PerR1;

	size_t results_size = 0;
	size_t ops_size = 0;
	size_t r_size = 0;
	size_t c_size = 0;
	size_t n_subjects = 0;
	size_t n_candidates = 0;

	cudaStream_t stream = nullptr;
	int deviceId;

	AlignerDataArrays() : deviceId(0){}

	AlignerDataArrays(int deviceId_) : deviceId(deviceId_){
		//printf("AlignerDataArrays::AlignerDataArrays(%d)\n", deviceId);
		cudaSetDevice(deviceId); CUERR;
		cudaStreamCreate(&stream); CUERR;
	};

	AlignerDataArrays(const AlignerDataArrays& other){
		//printf("AlignerDataArrays::AlignerDataArrays(other)\n");
		*this = other; CUERR;
	}

	~AlignerDataArrays(){
		//printf("AlignerDataArrays::~AlignerDataArrays\n");
		cudaSetDevice(deviceId); CUERR;

		clear();
		if(stream != nullptr)
			cudaStreamDestroy(stream); CUERR;
	}

	void clear(){
		d_results.reset();
		d_ops.reset();
		d_subjectsdata.reset();
		d_queriesdata.reset();
		d_rBytesPrefixSum.reset();
		d_rLengths.reset();
		d_rIsEncoded.reset();
		d_cBytesPrefixSum.reset();
		d_cLengths.reset();
		d_cIsEncoded.reset();
		d_r2PerR1.reset();

		h_results.reset();
		h_ops.reset();
		h_subjectsdata.reset();
		h_queriesdata.reset();
		h_rBytesPrefixSum.reset();
		h_rLengths.reset();
		h_rIsEncoded.reset();
		h_cBytesPrefixSum.reset();
		h_cLengths.reset();
		h_cIsEncoded.reset();
		h_r2PerR1.reset();

		results_size = 0;
		ops_size = 0;
		r_size = 0;
		c_size = 0;
	}

	/*
		res = number of alignments
		ops = maximum total number of alignment operations for res alignments
		r = number of bytes to store all subjects
		c = number of bytes to store all candidates
	*/
	void resize(size_t res, size_t ops, size_t r, size_t c){
		//printf("AlignerDataArrays::resize\n");
		cudaSetDevice(deviceId); 

		if(res > results_size){
			d_results = make_unique_dev<AlignResultCompact>(deviceId, res);
			d_cBytesPrefixSum = make_unique_dev<int>(deviceId, res+1);
			d_cLengths = make_unique_dev<int>(deviceId, res);
			d_cIsEncoded = make_unique_dev<int>(deviceId, res);
			d_rBytesPrefixSum = make_unique_dev<int>(deviceId, res+1);
			d_rLengths = make_unique_dev<int>(deviceId, res);
			d_rIsEncoded = make_unique_dev<int>(deviceId, res);
			d_r2PerR1 = make_unique_dev<int>(deviceId, res);

			h_results = make_unique_pinned<AlignResultCompact>(res);
			h_cBytesPrefixSum = make_unique_pinned<int>(res+1);
			h_cLengths = make_unique_pinned<int>(res);
			h_cIsEncoded = make_unique_pinned<int>(res);
			h_rBytesPrefixSum = make_unique_pinned<int>(res+1);
			h_rLengths = make_unique_pinned<int>(res);
			h_rIsEncoded = make_unique_pinned<int>(res);
			h_r2PerR1 = make_unique_pinned<int>(res);

			results_size = res;
		}

		if(ops > ops_size){
			d_ops = make_unique_dev<AlignOp>(deviceId, ops);
			h_ops = make_unique_pinned<AlignOp>(ops);

			ops_size = ops;
		}

		if(r > r_size){
			d_subjectsdata = make_unique_dev<char>(deviceId, r);
			h_subjectsdata = make_unique_pinned<char>(r);

			r_size = r;
		}

		if(c > c_size){
			d_queriesdata = make_unique_dev<char>(deviceId, c);
			h_queriesdata = make_unique_pinned<char>(c);

			c_size = c;
		}
	}

	AlignerDataArrays& operator=(const AlignerDataArrays& other){
		//printf("AlignerDataArrays::operator=()\n");
		deviceId = other.deviceId;

		cudaSetDevice(deviceId); CUERR;

		clear(); CUERR;

		resize(other.results_size, other.ops_size, other.r_size, other.c_size); CUERR;

		cudaMemcpy(d_results.get(), other.d_results.get(), other.results_size, D2D); CUERR;
		//printf("%p %p %lu\n", d_cBytesPrefixSum, other.d_cBytesPrefixSum, other.results_size + 1);
		if(other.results_size > 0){
			cudaMemcpy(d_cBytesPrefixSum.get(), other.d_cBytesPrefixSum.get(), other.results_size + 1, D2D); CUERR;
			cudaMemcpy(d_rBytesPrefixSum.get(), other.d_rBytesPrefixSum.get(), other.results_size + 1, D2D); CUERR;
		}
		cudaMemcpy(d_cLengths.get(), other.d_cLengths.get(), other.results_size, D2D); CUERR;
		cudaMemcpy(d_cIsEncoded.get(), other.d_cIsEncoded.get(), other.results_size, D2D); CUERR;
		cudaMemcpy(d_rLengths.get(), other.d_rLengths.get(), other.results_size, D2D); CUERR;
		cudaMemcpy(d_rIsEncoded.get(), other.d_rIsEncoded.get(), other.results_size, D2D); CUERR;
		cudaMemcpy(d_ops.get(), other.d_ops.get(), other.ops_size, D2D); CUERR;
		cudaMemcpy(d_subjectsdata.get(), other.d_subjectsdata.get(), other.r_size, D2D); CUERR;
		cudaMemcpy(d_queriesdata.get(), other.d_queriesdata.get(), other.c_size, D2D); CUERR;
		cudaMemcpy(d_r2PerR1.get(), other.d_r2PerR1.get(), other.results_size, D2D); CUERR;

		memcpy(h_results.get(), other.h_results.get(), other.results_size);
		if(other.results_size > 0){
			memcpy(h_cBytesPrefixSum.get(), other.h_cBytesPrefixSum.get(), other.results_size + 1);
			memcpy(h_rBytesPrefixSum.get(), other.h_rBytesPrefixSum.get(), other.results_size + 1);
		}
		memcpy(h_cLengths.get(), other.h_cLengths.get(), other.results_size);
		memcpy(h_cIsEncoded.get(), other.h_cIsEncoded.get(), other.results_size);
		memcpy(h_rLengths.get(), other.h_rLengths.get(), other.results_size);
		memcpy(h_rIsEncoded.get(), other.h_rIsEncoded.get(), other.results_size);
		memcpy(h_ops.get(), other.h_ops.get(), other.ops_size);
		memcpy(h_subjectsdata.get(), other.h_subjectsdata.get(), other.r_size);
		memcpy(h_queriesdata.get(), other.h_queriesdata.get(), other.c_size);
		memcpy(h_r2PerR1.get(), other.h_r2PerR1.get(), other.results_size); CUERR;

		if(stream == nullptr)
			cudaStreamCreate(&stream); CUERR;

		return *this;
	}

	AlignerDataArrays& operator=(AlignerDataArrays&& other){
		deviceId = other.deviceId;

		cudaSetDevice(deviceId); CUERR;

		std::swap(d_results, other.d_results);
		std::swap(d_cBytesPrefixSum, other.d_cBytesPrefixSum);
		std::swap(d_cLengths, other.d_cLengths);
		std::swap(d_cIsEncoded, other.d_cIsEncoded);
		std::swap(d_rBytesPrefixSum, other.d_rBytesPrefixSum);
		std::swap(d_rLengths, other.d_rLengths);
		std::swap(d_rIsEncoded, other.d_rIsEncoded);
		std::swap(d_ops, other.d_ops);
		std::swap(d_subjectsdata, other.d_subjectsdata);
		std::swap(d_queriesdata, other.d_queriesdata);
		std::swap(d_r2PerR1, other.d_r2PerR1);
		std::swap(h_results, other.h_results);
		std::swap(h_cBytesPrefixSum, other.h_cBytesPrefixSum);
		std::swap(h_cLengths, other.h_cLengths);
		std::swap(h_cIsEncoded, other.h_cIsEncoded);
		std::swap(h_rBytesPrefixSum, other.h_rBytesPrefixSum);
		std::swap(h_rLengths, other.h_rLengths);
		std::swap(h_rIsEncoded, other.h_rIsEncoded);
		std::swap(h_ops, other.h_ops);
		std::swap(h_subjectsdata, other.h_subjectsdata);
		std::swap(h_queriesdata, other.h_queriesdata);
		std::swap(results_size, other.results_size);
		std::swap(ops_size, other.ops_size);
		std::swap(r_size, other.r_size);
		std::swap(c_size, other.c_size);
		std::swap(h_r2PerR1, other.h_r2PerR1);

		other.clear(); CUERR;

		if(stream == nullptr)
			cudaStreamCreate(&stream); CUERR;

		return *this;
	}

};















#endif


struct OrdinaryAccessor{
	#ifdef __CUDACC__
	__host__ __device__
	#endif
	OrdinaryAccessor(const char* buf, int bases) : buffer(buf), nBases(bases){}

	#ifdef __CUDACC__
	__host__ __device__
	#endif
	char operator[](int i) const{
		return buffer[i];
	}


	const char* const buffer;
	const int nBases;
};

struct EncodedAccessor{
	#ifdef __CUDACC__
	__host__ __device__
	#endif
	EncodedAccessor(const char* buf, int bases) : buffer(buf), nBases(bases), UNUSED_BYTE_SPACE((4 - (bases % 4)) % 4){}

	#ifdef __CUDACC__
	__host__ __device__
	#endif
	char operator[](int i) const{

		const int byte = (i + UNUSED_BYTE_SPACE) / 4;
		const int basepos = (i + UNUSED_BYTE_SPACE) % 4;

		switch((buffer[byte] >> (3-basepos) * 2) & 0x03) {
                        case BASE_A: return 'A';
                        case BASE_C: return 'C';
                        case BASE_G: return 'G';
                        case BASE_T: return 'T';
			default: return '_'; // cannot happen
		}
	}

	const char* const buffer;
	const int nBases;
	const int UNUSED_BYTE_SPACE;

	static constexpr std::uint8_t BASE_A = 0x00;
	static constexpr std::uint8_t BASE_C = 0x01;
	static constexpr std::uint8_t BASE_G = 0x02;
	static constexpr std::uint8_t BASE_T = 0x03;
};


// split substitutions in alignment into deletion + insertion
int split_subs(AlignResult& alignment, const char* subject);



// Given AlignmentResults for a read and its reverse complement, find the "best" of both alignments
BestAlignment_t get_best_alignment(const AlignResultCompact& fwdAlignment, const AlignResultCompact& revcmplAlignment, 
				int querylength, int candidatelength,
				double MAX_MISMATCH_RATIO, int MIN_OVERLAP, double MIN_OVERLAP_RATIO);




#endif
