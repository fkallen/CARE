#include "../inc/semi_global_alignment.hpp"

#include <cassert>

namespace sga{

HOSTDEVICEQUALIFIER
bool AlignmentAttributes::operator==(const AlignmentAttributes& rhs) const{
    return score == rhs.score
        && subject_begin_incl == rhs.subject_begin_incl
        && query_begin_incl == rhs.query_begin_incl
        && overlap == rhs.overlap
        && shift == rhs.shift
        && nOps == rhs.nOps
        && isNormalized == rhs.isNormalized
        && isValid == rhs.isValid;
}
HOSTDEVICEQUALIFIER
bool AlignmentAttributes::operator!=(const AlignmentAttributes& rhs) const{
    return !(*this == rhs);
}
HOSTDEVICEQUALIFIER
int AlignmentAttributes::get_score() const{
    return score;
}
HOSTDEVICEQUALIFIER
int AlignmentAttributes::get_subject_begin_incl() const{
    return subject_begin_incl;
}
HOSTDEVICEQUALIFIER
int AlignmentAttributes::get_query_begin_incl() const{
    return query_begin_incl;
}
HOSTDEVICEQUALIFIER
int AlignmentAttributes::get_overlap() const{
    return overlap;
}
HOSTDEVICEQUALIFIER
int AlignmentAttributes::get_shift() const{
    return shift;
}
HOSTDEVICEQUALIFIER
int AlignmentAttributes::get_nOps() const{
    return nOps;
}
HOSTDEVICEQUALIFIER
bool AlignmentAttributes::get_isNormalized() const{
    return isNormalized;
}
HOSTDEVICEQUALIFIER
bool AlignmentAttributes::get_isValid() const{
    return isValid;
}
HOSTDEVICEQUALIFIER
int& AlignmentAttributes::get_score(){
    return score;
}
HOSTDEVICEQUALIFIER
int& AlignmentAttributes::get_subject_begin_incl(){
    return subject_begin_incl;
}
HOSTDEVICEQUALIFIER
int& AlignmentAttributes::get_query_begin_incl(){
    return query_begin_incl;
}
HOSTDEVICEQUALIFIER
int& AlignmentAttributes::get_overlap(){
    return overlap;
}
HOSTDEVICEQUALIFIER
int& AlignmentAttributes::get_shift(){
    return shift;
}
HOSTDEVICEQUALIFIER
int& AlignmentAttributes::get_nOps(){
    return nOps;
}
HOSTDEVICEQUALIFIER
bool& AlignmentAttributes::get_isNormalized(){
    return isNormalized;
}
HOSTDEVICEQUALIFIER
bool& AlignmentAttributes::get_isValid(){
    return isValid;
}

bool AlignmentResult::operator==(const AlignmentResult& rhs) const{
    return attributes == rhs.attributes && operations == rhs.operations;
}
bool AlignmentResult::operator!=(const AlignmentResult& rhs) const{
    return !(*this == rhs);
}
HOSTDEVICEQUALIFIER
int AlignmentResult::get_score() const{
    return attributes.get_score();
}
HOSTDEVICEQUALIFIER
int AlignmentResult::get_subject_begin_incl() const{
    return attributes.get_subject_begin_incl();
}
HOSTDEVICEQUALIFIER
int AlignmentResult::get_query_begin_incl() const{
    return attributes.get_query_begin_incl();
}
HOSTDEVICEQUALIFIER
int AlignmentResult::get_overlap() const{
    return attributes.get_overlap();
}
HOSTDEVICEQUALIFIER
int AlignmentResult::get_shift() const{
    return attributes.get_shift();
}
HOSTDEVICEQUALIFIER
int AlignmentResult::get_nOps() const{
    return attributes.get_nOps();
}
HOSTDEVICEQUALIFIER
bool AlignmentResult::get_isNormalized() const{
    return attributes.get_isNormalized();
}
HOSTDEVICEQUALIFIER
bool AlignmentResult::get_isValid() const{
    return attributes.get_isValid();
}
HOSTDEVICEQUALIFIER
int& AlignmentResult::get_score(){
    return attributes.get_score();
}
HOSTDEVICEQUALIFIER
int& AlignmentResult::get_subject_begin_incl(){
    return attributes.get_subject_begin_incl();
}
HOSTDEVICEQUALIFIER
int& AlignmentResult::get_query_begin_incl(){
    return attributes.get_query_begin_incl();
}
HOSTDEVICEQUALIFIER
int& AlignmentResult::get_overlap(){
    return attributes.get_overlap();
}
HOSTDEVICEQUALIFIER
int& AlignmentResult::get_shift(){
    return attributes.get_shift();
}
HOSTDEVICEQUALIFIER
int& AlignmentResult::get_nOps(){
    return attributes.get_nOps();
}
HOSTDEVICEQUALIFIER
bool& AlignmentResult::get_isNormalized(){
    return attributes.get_isNormalized();
}
HOSTDEVICEQUALIFIER
bool& AlignmentResult::get_isValid(){
    return attributes.get_isValid();
}

void SGAdata::resize(int n_sub, int n_quer){
#ifdef __NVCC__
	cudaSetDevice(deviceId); CUERR;

	bool resizeResult = false;

	if(n_sub > max_n_subjects){
		const int newmax = 1.5 * n_sub;
		std::size_t oldpitch = sequencepitch;
		cudaFree(d_subjectsdata); CUERR;
		cudaMallocPitch(&d_subjectsdata, &sequencepitch, max_sequence_bytes, newmax); CUERR;
		assert(!oldpitch || oldpitch == sequencepitch);

		cudaFreeHost(h_subjectsdata); CUERR;
		cudaMallocHost(&h_subjectsdata, sequencepitch * newmax); CUERR;

		cudaFree(d_subjectlengths); CUERR;
		cudaMalloc(&d_subjectlengths, sizeof(int) * newmax); CUERR;

		cudaFreeHost(h_subjectlengths); CUERR;
		cudaMallocHost(&h_subjectlengths, sizeof(int) * newmax); CUERR;

        cudaFree(d_NqueriesPrefixSum); CUERR;
        cudaMalloc(&d_NqueriesPrefixSum, sizeof(int) * (n_sub+1)); CUERR;

        cudaFreeHost(h_NqueriesPrefixSum); CUERR;
        cudaMallocHost(&h_NqueriesPrefixSum, sizeof(int) * (n_sub+1)); CUERR;

		max_n_subjects = newmax;
		resizeResult = true;
	}


	if(n_quer > max_n_queries){
		const int newmax = 1.5 * n_quer;
		size_t oldpitch = sequencepitch;
		cudaFree(d_queriesdata); CUERR;
		cudaMallocPitch(&d_queriesdata, &sequencepitch, max_sequence_bytes, newmax); CUERR;
		assert(!oldpitch || oldpitch == sequencepitch);

		cudaFreeHost(h_queriesdata); CUERR;
		cudaMallocHost(&h_queriesdata, sequencepitch * newmax); CUERR;

		cudaFree(d_querylengths); CUERR;
		cudaMalloc(&d_querylengths, sizeof(int) * newmax); CUERR;

		cudaFreeHost(h_querylengths); CUERR;
		cudaMallocHost(&h_querylengths, sizeof(int) * newmax); CUERR;

		max_n_queries = newmax;
		resizeResult = true;
	}

	if(resizeResult){
		cudaFree(d_results); CUERR;
		cudaMalloc(&d_results, sizeof(Attributes_t) * max_n_subjects * max_n_queries); CUERR;

		cudaFreeHost(h_results); CUERR;
		cudaMallocHost(&h_results, sizeof(Attributes_t) * max_n_subjects * max_n_queries); CUERR;

		cudaFree(d_ops); CUERR;
		cudaFreeHost(h_ops); CUERR;

		cudaMalloc(&d_ops, sizeof(Op_t) * max_n_queries * max_ops_per_alignment); CUERR;
		cudaMallocHost(&h_ops, sizeof(Op_t) * max_n_queries * max_ops_per_alignment); CUERR;
	}
#endif

	n_subjects = n_sub;
	n_queries = n_quer;

    throw std::runtime_error("don't use me");
}

void SGAdata::resize(int n_sub, int n_quer, int n_res, double factor){
    #ifdef __NVCC__

    cudaSetDevice(deviceId); CUERR;

    n_subjects = n_sub;
    n_queries = n_quer;
    n_results = n_res;

    const std::size_t memSubjects = n_sub * sequencepitch;
    const std::size_t memSubjectLengths = SDIV(n_sub * sizeof(int), sequencepitch) * sequencepitch;
    const std::size_t memNqueriesPrefixSum = SDIV((n_sub+1) * sizeof(int), sequencepitch) * sequencepitch;
    const std::size_t memQueries = n_quer * sequencepitch;
    const std::size_t memQueryLengths = SDIV(n_quer * sizeof(int), sequencepitch) * sequencepitch;
    const std::size_t memResults = SDIV(sizeof(Attributes_t) * n_results, sequencepitch) * sequencepitch;
    const std::size_t memBestAlignmentFlags = SDIV(sizeof(BestAlignment_t) * n_results, sequencepitch) * sequencepitch;
    const std::size_t memOps = SDIV(sizeof(Op_t) * n_results * max_ops_per_alignment, sequencepitch) * sequencepitch;

    const std::size_t requiredMem = memSubjects + memSubjectLengths + memNqueriesPrefixSum
                                    + memQueries + memQueryLengths + memResults
                                    + memBestAlignmentFlags + memOps;

    if(requiredMem > allocatedMem){
        cudaFree(deviceptr); CUERR;
        cudaFreeHost(hostptr); CUERR;
        cudaMalloc(&deviceptr, std::size_t(requiredMem * factor)); CUERR;
        cudaMallocHost(&hostptr, std::size_t(requiredMem * factor)); CUERR;

        allocatedMem = requiredMem * factor;
    }

    transfersizeH2D = memSubjects; // d_subjectsdata
    transfersizeH2D += memSubjectLengths; // d_subjectlengths
    transfersizeH2D += memNqueriesPrefixSum; // d_NqueriesPrefixSum
    transfersizeH2D += memQueries; // d_queriesdata
    transfersizeH2D += memQueryLengths; // d_querylengths

    transfersizeD2H = memResults; //d_results
    transfersizeD2H += memBestAlignmentFlags; // d_bestAlignmentFlags
    transfersizeD2H += sizeof(Op_t) * n_results * max_ops_per_alignment; // d_ops

    d_subjectsdata = (char*)deviceptr;
    d_subjectlengths = (int*)(((char*)d_subjectsdata) + memSubjects);
    d_NqueriesPrefixSum = (int*)(((char*)d_subjectlengths) + memSubjectLengths);
    d_queriesdata = (char*)(((char*)d_NqueriesPrefixSum) + memNqueriesPrefixSum);
    d_querylengths = (int*)(((char*)d_queriesdata) + memQueries);
    d_results = (Attributes_t*)(((char*)d_querylengths) + memQueryLengths);
    d_bestAlignmentFlags = (BestAlignment_t*)(((char*)d_results) + memResults);
    d_ops = (Op_t*)(((char*)d_bestAlignmentFlags) + memBestAlignmentFlags);

    h_subjectsdata = (char*)hostptr;
    h_subjectlengths = (int*)(((char*)h_subjectsdata) + memSubjects);
    h_NqueriesPrefixSum = (int*)(((char*)h_subjectlengths) + memSubjectLengths);
    h_queriesdata = (char*)(((char*)h_NqueriesPrefixSum) + memNqueriesPrefixSum);
    h_querylengths = (int*)(((char*)h_queriesdata) + memQueries);
    h_results = (Attributes_t*)(((char*)h_querylengths) + memQueryLengths);
    h_bestAlignmentFlags = (BestAlignment_t*)(((char*)h_results) + memResults);
    h_ops = (Op_t*)(((char*)h_bestAlignmentFlags) + memBestAlignmentFlags);

    #endif
}


void cuda_init_SGAdata(SGAdata& data,
                       int deviceId,
                       int max_sequence_length,
                       int max_sequence_bytes,
                       int gpuThreshold){

    data.deviceId = deviceId;
    data.max_sequence_length = 32 * SDIV(max_sequence_length, 32); //round up to multiple of 32;
    data.max_sequence_bytes = max_sequence_bytes;
    data.max_ops_per_alignment = 2 * (data.max_sequence_length + 1);
    data.gpuThreshold = gpuThreshold;

#ifdef __NVCC__
    cudaSetDevice(deviceId); CUERR;

    void* ptr;
    std::size_t pitch;
    cudaMallocPitch(&ptr, &pitch, max_sequence_bytes, 1); CUERR;
    cudaFree(ptr); CUERR;
    data.sequencepitch = pitch;

    for(int i = 0; i < SGAdata::n_streams; i++)
        cudaStreamCreate(&(data.streams[i])); CUERR;
#endif
}

void cuda_cleanup_SGAdata(SGAdata& data){

	#ifdef __NVCC__
		cudaSetDevice(data.deviceId); CUERR;
#if 0
		cudaFree(data.d_results); CUERR;
		cudaFree(data.d_ops); CUERR;
		cudaFree(data.d_subjectsdata); CUERR;
		cudaFree(data.d_queriesdata); CUERR;
		cudaFree(data.d_subjectlengths); CUERR;
		cudaFree(data.d_querylengths); CUERR;
        cudaFree(data.d_NqueriesPrefixSum); CUERR;

		cudaFreeHost(data.h_results); CUERR;
		cudaFreeHost(data.h_ops); CUERR;
		cudaFreeHost(data.h_subjectsdata); CUERR;
		cudaFreeHost(data.h_queriesdata); CUERR;
		cudaFreeHost(data.h_subjectlengths); CUERR;
		cudaFreeHost(data.h_querylengths); CUERR;
        cudaFreeHost(data.h_NqueriesPrefixSum); CUERR;
#else
        cudaFree(data.deviceptr); CUERR;
        cudaFreeHost(data.hostptr); CUERR;
        data.d_ops = nullptr;
        data.h_ops = nullptr;

        data.d_results = nullptr;
        data.d_subjectsdata = nullptr;
        data.d_queriesdata = nullptr;
        data.d_subjectlengths = nullptr;
        data.d_querylengths = nullptr;
        data.d_NqueriesPrefixSum = nullptr;
        data.d_bestAlignmentFlags = nullptr;

        data.h_results = nullptr;
        data.h_subjectsdata = nullptr;
        data.h_queriesdata = nullptr;
        data.h_subjectlengths = nullptr;
        data.h_querylengths = nullptr;
        data.h_NqueriesPrefixSum = nullptr;
        data.h_bestAlignmentFlags = nullptr;
#endif
		for(int i = 0; i < SGAdata::n_streams; i++)
			cudaStreamDestroy(data.streams[i]); CUERR;
	#endif
}


}
