#include "../inc/semi_global_alignment.hpp"

namespace shd{

bool AlignmentResult::operator==(const AlignmentResult& rhs) const{
    return attributes == rhs.attributes && operations == rhs.operations;
}
bool AlignmentResult::operator!=(const AlignmentResult& rhs) const{
    return !(*this == rhs);
}
int AlignmentResult::get_score() const{
    return attributes.score;
}
int AlignmentResult::get_subject_begin_incl() const{
    return attributes.subject_begin_incl;
}
int AlignmentResult::get_query_begin_incl() const{
    return attributes.query_begin_incl;
}
int AlignmentResult::get_overlap() const{
    return attributes.overlap;
}
int AlignmentResult::get_shift() const{
    return attributes.shift;
}
int AlignmentResult::get_nOps() const{
    return attributes.nOps;
}
bool AlignmentResult::get_isNormalized() const{
    return attributes.isNormalized;
}
bool AlignmentResult::get_isValid() const{
    return attributes.isValid;
}
int& AlignmentResult::get_score(){
    return attributes.score;
}
int& AlignmentResult::get_subject_begin_incl(){
    return attributes.subject_begin_incl;
}
int& AlignmentResult::get_query_begin_incl(){
    return attributes.query_begin_incl;
}
int& AlignmentResult::get_overlap(){
    return attributes.overlap;
}
int& AlignmentResult::get_shift(){
    return attributes.shift;
}
int& AlignmentResult::get_nOps(){
    return attributes.nOps;
}
bool& AlignmentResult::get_isNormalized(){
    return attributes.isNormalized;
}
bool& AlignmentResult::get_isValid(){
    return attributes.isValid;
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

    for(int i = 0; i < SGAdata::n_streams; i++)
        cudaStreamCreate(&(data.streams[i])); CUERR;
#endif
}

void cuda_cleanup_SGAdata(SGAdata& data){
	#ifdef __NVCC__
		cudaSetDevice(data.deviceId); CUERR;

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

		for(int i = 0; i < SGAdata::n_streams; i++)
			cudaStreamDestroy(data.streams[i]); CUERR;
	#endif
}


}
