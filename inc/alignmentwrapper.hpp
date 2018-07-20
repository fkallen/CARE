#ifndef CARE_ALIGNMENT_HPP
#define CARE_ALIGNMENT_HPP

#include "hpc_helpers.cuh"

#include "shifted_hamming_distance.hpp"
#include "semi_global_alignment.hpp"
#include "tasktiming.hpp"

#include <vector>
#include <chrono>
#include <memory>
#include <cstring>
#include <cassert>

namespace care{


enum class AlignmentDevice {CPU, GPU, None};

template<class T>
struct AlignmentHandle{
    T buffers;
    TaskTimings timings;
};


/*

    ########## SHIFTED HAMMING DISTANCE ##########

*/

using SHDhandle = AlignmentHandle<shd::SHDdata>;
using SHDResult = shd::Result_t;

void init_SHDhandle(SHDhandle& handle, int deviceId,
                        int max_sequence_length,
                        int max_sequence_bytes,
                        int gpuThreshold);

//free buffers
void destroy_SHDhandle(SHDhandle& handle);

/*
    Wrapper functions for shifted hamming distance calls
    which derive the required accessor from Sequence_t
*/
template<class Sequence_t>
SHDResult cpu_shifted_hamming_distance(const char* subject,
                                        int subjectlength,
										const char* query,
										int querylength,
                                        int min_overlap,
                                        double maxErrorRate,
                                        double min_overlap_ratio){

    auto accessor = [] (const char* data, int length, int index){
        return Sequence_t::get(data, length, index);
    };

    return shd::cpu_shifted_hamming_distance(subject, subjectlength, query, querylength,
                        min_overlap, maxErrorRate, min_overlap_ratio, accessor);
}

#ifdef __NVCC__

template<class Sequence_t>
void call_shd_kernel_async(const shd::SHDdata& shddata,
                      int min_overlap,
                      double maxErrorRate,
                      double min_overlap_ratio,
                      int maxSubjectLength,
                      int maxQueryLength){

    auto accessor = [] __device__ (const char* data, int length, int index){
        return Sequence_t::get(data, length, index);
    };

    shd::call_shd_kernel_async(shddata, min_overlap, maxErrorRate, min_overlap_ratio, maxSubjectLength, maxQueryLength, accessor);
}

template<class Sequence_t>
void call_shd_kernel(const shd::SHDdata& shddata,
                      int min_overlap,
                      double maxErrorRate,
                      double min_overlap_ratio,
                      int maxSubjectLength,
                      int maxQueryLength){

    auto accessor = [] __device__ (const char* data, int length, int index){
        return Sequence_t::get(data, length, index);
    };

    shd::call_shd_kernel(shddata, min_overlap, maxErrorRate, min_overlap_ratio, maxSubjectLength, maxQueryLength, accessor);
}

#endif



/*
    Begin calculation of alignments. Results are only present after a subsequent
    call to shifted_hamming_distance_get_results

    SubjectIter,QueryIter: Iterator to const Sequence_t*
    AlignmentIter: Iterator to shd::Result_t
*/
template<class Sequence_t, class SubjectIter, class QueryIter, class AlignmentIter>
AlignmentDevice shifted_hamming_distance_async(SHDhandle& handle,
                                SubjectIter subjectsbegin,
                                SubjectIter subjectsend,
                                QueryIter queriesbegin,
                                QueryIter queriesend,
                                AlignmentIter alignmentsbegin,
                                AlignmentIter alignmentsend,
                                const std::vector<int>& queriesPerSubject,
                                int min_overlap,
                                double maxErrorRate,
                                double min_overlap_ratio,
                                bool canUseGpu){

    static_assert(std::is_same<typename AlignmentIter::value_type, shd::Result_t>::value, "shifted hamming distance unexpected Alignment type");

    auto& mybuffers = handle.buffers;
    auto& timings = handle.timings;

    AlignmentDevice device = AlignmentDevice::None;

    const int numberOfAlignments = std::distance(alignmentsbegin, alignmentsend);
    assert(numberOfAlignments == std::distance(queriesbegin, queriesend));

    //nothing to do here
    if(numberOfAlignments == 0)
        return device;
#ifdef __NVCC__

    const int numberOfSubjects = queriesPerSubject.size();

    if(canUseGpu && numberOfAlignments >= mybuffers.gpuThreshold){ // use gpu for alignment
        device = AlignmentDevice::GPU;

        timings.preprocessingBegin();

        cudaSetDevice(mybuffers.deviceId); CUERR;

        mybuffers.resize(numberOfSubjects, numberOfAlignments);

        mybuffers.n_subjects = numberOfSubjects;
        mybuffers.n_queries = numberOfAlignments;

        int maxSubjectLength = 0;

        for(auto t = std::make_pair(0, subjectsbegin); t.second != subjectsend; t.first++, t.second++){
            auto& count = t.first;
            auto& it = t.second;

            assert((*it)->length() <= mybuffers.max_sequence_length);
            assert((*it)->getNumBytes() <= mybuffers.max_sequence_bytes);

            std::memcpy(mybuffers.h_subjectsdata + count * mybuffers.sequencepitch,
                        (*it)->begin(),
                        (*it)->getNumBytes());

            mybuffers.h_subjectlengths[count] = (*it)->length();
            maxSubjectLength = std::max(int((*it)->length()), maxSubjectLength);
        }

        int maxQueryLength = 0;

        for(auto t = std::make_pair(0, queriesbegin); t.second != queriesend; t.first++, t.second++){
            auto& count = t.first;
            auto& it = t.second;

            assert((*it)->length() <= mybuffers.max_sequence_length);
            assert((*it)->getNumBytes() <= mybuffers.max_sequence_bytes);

            std::memcpy(mybuffers.h_queriesdata + count * mybuffers.sequencepitch,
                        (*it)->begin(),
                        (*it)->getNumBytes());

            mybuffers.h_querylengths[count] = (*it)->length();
            maxQueryLength = std::max(int((*it)->length()), maxQueryLength);
        }

        mybuffers.h_NqueriesPrefixSum[0] = 0;
        for(std::size_t i = 0; i < queriesPerSubject.size(); i++)
            mybuffers.h_NqueriesPrefixSum[i+1] = mybuffers.h_NqueriesPrefixSum[i] + queriesPerSubject[i];

        assert(numberOfAlignments == mybuffers.h_NqueriesPrefixSum[queriesPerSubject.size()]);

        timings.preprocessingEnd();

        timings.executionBegin();


        // copy data to gpu
#if 0
        cudaMemcpyAsync(mybuffers.d_subjectsdata,
                mybuffers.h_subjectsdata,
                numberOfSubjects * mybuffers.sequencepitch,
                H2D,
                mybuffers.streams[0]); CUERR;
        cudaMemcpyAsync(mybuffers.d_queriesdata,
                mybuffers.h_queriesdata,
                numberOfAlignments * mybuffers.sequencepitch,
                H2D,
                mybuffers.streams[0]); CUERR;
        cudaMemcpyAsync(mybuffers.d_subjectlengths,
                mybuffers.h_subjectlengths,
                numberOfSubjects * sizeof(int),
                H2D,
                mybuffers.streams[0]); CUERR;
        cudaMemcpyAsync(mybuffers.d_querylengths,
                mybuffers.h_querylengths,
                numberOfAlignments * sizeof(int),
                H2D,
                mybuffers.streams[0]); CUERR;
        cudaMemcpyAsync(mybuffers.d_NqueriesPrefixSum,
                mybuffers.h_NqueriesPrefixSum,
                (numberOfSubjects+1) * sizeof(int),
                H2D,
                mybuffers.streams[0]); CUERR;
#else
        cudaMemcpyAsync(mybuffers.deviceptr,
                        mybuffers.hostptr,
                        mybuffers.transfersizeH2D,
                        H2D,
                        mybuffers.streams[0]); CUERR;

#endif
        call_shd_kernel_async<Sequence_t>(mybuffers,
                                    min_overlap,
                                    maxErrorRate,
                                    min_overlap_ratio,
                                    maxSubjectLength,
                                    maxQueryLength); CUERR;

        shd::Result_t* results = mybuffers.h_results;
        shd::Result_t* d_results = mybuffers.d_results;
#if 0
        cudaMemcpyAsync(results,
            d_results,
            sizeof(shd::Result_t) * numberOfAlignments,
            D2H,
            mybuffers.streams[0]); CUERR;
#else
        cudaMemcpyAsync(results,
            d_results,
            mybuffers.transfersizeD2H,
            D2H,
            mybuffers.streams[0]); CUERR;
#endif

    }else{ // use cpu for alignment

#endif
        device = AlignmentDevice::CPU;

        timings.executionBegin();

        auto queryIt = queriesbegin;
        auto alignmentsIt = alignmentsbegin;

        for(auto t = std::make_pair(0, subjectsbegin); t.second != subjectsend; t.first++, t.second++){
            auto& subjectcount = t.first;
            auto& subjectIt = t.second;

            assert((*subjectIt)->length() <= mybuffers.max_sequence_length);
            assert((*subjectIt)->getNumBytes() <= mybuffers.max_sequence_bytes);

            const char* const subject = (const char*)(*subjectIt)->begin();
            const int subjectLength = (*subjectIt)->length();

            const int nQueries = queriesPerSubject[subjectcount];

            for(int i = 0; i < nQueries; i++){
                const char* query =  (const char*)(*queryIt)->begin();
                const int queryLength = (*queryIt)->length();

                *alignmentsIt = cpu_shifted_hamming_distance<Sequence_t>(subject, subjectLength, query, queryLength,
                                                                        min_overlap,
                                                                        maxErrorRate,
                                                                        min_overlap_ratio);

                queryIt++;
                alignmentsIt++;
            }
        }

        timings.executionEnd();

#ifdef __NVCC__
    }
#endif

    return device;
}

/*
Ensures that all alignment results from the preceding call to shifted_hamming_distance_async
have been stored in the range [alignmentsbegin, alignmentsend[

must be called with the same mybuffers, alignmentsbegin, alignmentsend, canUseGpu
as the call to shifted_hamming_distance_async
*/

template<class AlignmentIter>
void shifted_hamming_distance_get_results(SHDhandle& handle,
                                AlignmentIter alignmentsbegin,
                                AlignmentIter alignmentsend,
                                bool canUseGpu){

    static_assert(std::is_same<typename AlignmentIter::value_type, shd::Result_t>::value, "shifted hamming distance unexpected Alignement type");

    const int numberOfAlignments = std::distance(alignmentsbegin, alignmentsend);

    //nothing to do here
    if(numberOfAlignments == 0)
        return;
#ifdef __NVCC__

    auto& mybuffers = handle.buffers;
    auto& timings = handle.timings;

    if(canUseGpu && numberOfAlignments >= mybuffers.gpuThreshold){ // use gpu for alignment
        cudaSetDevice(mybuffers.deviceId); CUERR;

        shd::Result_t* results = mybuffers.h_results;

        cudaStreamSynchronize(mybuffers.streams[0]); CUERR;

        timings.executionEnd();

        timings.postprocessingBegin();

        for(auto t = std::make_pair(0, alignmentsbegin); t.second != alignmentsend; t.first++, t.second++){
            auto& count = t.first;
            auto& it = t.second;

            *it = results[count];
        }

        timings.postprocessingEnd();


    }else{ // cpu already done

#endif

#ifdef __NVCC__
    }
#endif

    return;
}

template<class Sequence_t, class SubjectIter, class QueryIter, class AlignmentIter>
AlignmentDevice shifted_hamming_distance(SHDhandle& handle,
                                SubjectIter subjectsbegin,
                                SubjectIter subjectsend,
                                QueryIter queriesbegin,
                                QueryIter queriesend,
                                AlignmentIter alignmentsbegin,
                                AlignmentIter alignmentsend,
                                const std::vector<int>& queriesPerSubject,
                                int min_overlap,
                                double maxErrorRate,
                                double min_overlap_ratio,
                                bool canUseGpu){

    AlignmentDevice device = shifted_hamming_distance_async(handle, subjectsbegin, subjectsend,
                                    queriesbegin, queriesend,
                                    alignmentsbegin, alignmentsend,
                                    queriesPerSubject, min_overlap,
                                    maxErrorRate, min_overlap_ratio, canUseGpu);

    shifted_hamming_distance_get_results(handle,
                                    alignmentsbegin,
                                    alignmentsend,
                                    canUseGpu);

    return device;
}







/*

    ########### SHIFTED HAMMING DISTANCE WITH REVERSE COMPLEMENT

*/






#ifdef __NVCC__

template<class Sequence_t>
void call_shd_with_revcompl_kernel_async(const shd::SHDdata& shddata,
                      int min_overlap,
                      double maxErrorRate,
                      double min_overlap_ratio,
                      int maxSubjectLength,
                      int maxQueryLength){

    auto accessor = [] __device__ (const char* data, int length, int index){
        return Sequence_t::get(data, length, index);
    };

    shd::call_shd_with_revcompl_kernel_async(shddata, min_overlap, maxErrorRate, min_overlap_ratio, maxSubjectLength, maxQueryLength, accessor);
}

template<class Sequence_t>
void call_shd_with_revcompl_kernel(const shd::SHDdata& shddata,
                      int min_overlap,
                      double maxErrorRate,
                      double min_overlap_ratio,
                      int maxSubjectLength,
                      int maxQueryLength){

    auto accessor = [] __device__ (const char* data, int length, int index){
        return Sequence_t::get(data, length, index);
    };

    shd::call_shd_with_revcompl_kernel(shddata, min_overlap, maxErrorRate, min_overlap_ratio, maxSubjectLength, maxQueryLength, accessor);
}

#endif



/*
    Begin calculation of alignments. Results are only present after a subsequent
    call to shifted_hamming_distance_get_results

    SubjectIter,QueryIter: Iterator to const Sequence_t*
    AlignmentIter: Iterator to shd::Result_t
*/
template<class Sequence_t, class SubjectIter, class QueryIter, class AlignmentIter>
AlignmentDevice shifted_hamming_distance_with_revcompl_async(SHDhandle& handle,
                                SubjectIter subjectsbegin,
                                SubjectIter subjectsend,
                                QueryIter queriesbegin,
                                QueryIter queriesend,
                                AlignmentIter alignmentsbegin,
                                AlignmentIter alignmentsend,
                                const std::vector<int>& queriesPerSubject,
                                int min_overlap,
                                double maxErrorRate,
                                double min_overlap_ratio,
                                bool canUseGpu){

    static_assert(std::is_same<typename AlignmentIter::value_type, shd::Result_t>::value, "shifted hamming distance unexpected Alignment type");

    auto& mybuffers = handle.buffers;
    auto& timings = handle.timings;

    AlignmentDevice device = AlignmentDevice::None;

    const int numberOfAlignments = std::distance(alignmentsbegin, alignmentsend);
    const int numberOfQueries = std::distance(queriesbegin, queriesend);
    assert(numberOfAlignments == 2*numberOfQueries);

    //nothing to do here
    if(numberOfAlignments == 0)
        return device;
#ifdef __NVCC__

    const int numberOfSubjects = queriesPerSubject.size();

    if(canUseGpu && numberOfAlignments >= mybuffers.gpuThreshold){ // use gpu for alignment
        device = AlignmentDevice::GPU;

        timings.preprocessingBegin();

        cudaSetDevice(mybuffers.deviceId); CUERR;

        mybuffers.resize(numberOfSubjects, numberOfQueries, numberOfAlignments);

        mybuffers.n_subjects = numberOfSubjects;
        mybuffers.n_queries = numberOfQueries;
        mybuffers.n_results = numberOfAlignments;

        int maxSubjectLength = 0;

        for(auto t = std::make_pair(0, subjectsbegin); t.second != subjectsend; t.first++, t.second++){
            auto& count = t.first;
            auto& it = t.second;

            assert((*it)->length() <= mybuffers.max_sequence_length);
            assert((*it)->getNumBytes() <= mybuffers.max_sequence_bytes);

            std::memcpy(mybuffers.h_subjectsdata + count * mybuffers.sequencepitch,
                        (*it)->begin(),
                        (*it)->getNumBytes());

            mybuffers.h_subjectlengths[count] = (*it)->length();
            maxSubjectLength = std::max(int((*it)->length()), maxSubjectLength);
        }

        int maxQueryLength = 0;

        for(auto t = std::make_pair(0, queriesbegin); t.second != queriesend; t.first++, t.second++){
            auto& count = t.first;
            auto& it = t.second;

            assert((*it)->length() <= mybuffers.max_sequence_length);
            assert((*it)->getNumBytes() <= mybuffers.max_sequence_bytes);

            std::memcpy(mybuffers.h_queriesdata + count * mybuffers.sequencepitch,
                        (*it)->begin(),
                        (*it)->getNumBytes());

            mybuffers.h_querylengths[count] = (*it)->length();
            maxQueryLength = std::max(int((*it)->length()), maxQueryLength);
        }

        mybuffers.h_NqueriesPrefixSum[0] = 0;
        for(std::size_t i = 0; i < queriesPerSubject.size(); i++)
            mybuffers.h_NqueriesPrefixSum[i+1] = mybuffers.h_NqueriesPrefixSum[i] + queriesPerSubject[i];

        assert(numberOfAlignments == 2*mybuffers.h_NqueriesPrefixSum[queriesPerSubject.size()]);

        timings.preprocessingEnd();

        timings.executionBegin();


        // copy data to gpu
#if 0
        cudaMemcpyAsync(mybuffers.d_subjectsdata,
                mybuffers.h_subjectsdata,
                numberOfSubjects * mybuffers.sequencepitch,
                H2D,
                mybuffers.streams[0]); CUERR;
        cudaMemcpyAsync(mybuffers.d_queriesdata,
                mybuffers.h_queriesdata,
                numberOfAlignments * mybuffers.sequencepitch,
                H2D,
                mybuffers.streams[0]); CUERR;
        cudaMemcpyAsync(mybuffers.d_subjectlengths,
                mybuffers.h_subjectlengths,
                numberOfSubjects * sizeof(int),
                H2D,
                mybuffers.streams[0]); CUERR;
        cudaMemcpyAsync(mybuffers.d_querylengths,
                mybuffers.h_querylengths,
                numberOfAlignments * sizeof(int),
                H2D,
                mybuffers.streams[0]); CUERR;
        cudaMemcpyAsync(mybuffers.d_NqueriesPrefixSum,
                mybuffers.h_NqueriesPrefixSum,
                (numberOfSubjects+1) * sizeof(int),
                H2D,
                mybuffers.streams[0]); CUERR;
#else
        cudaMemcpyAsync(mybuffers.deviceptr,
                        mybuffers.hostptr,
                        mybuffers.transfersizeH2D,
                        H2D,
                        mybuffers.streams[0]); CUERR;

#endif
        call_shd_kernel_async<Sequence_t>(mybuffers,
                                    min_overlap,
                                    maxErrorRate,
                                    min_overlap_ratio,
                                    maxSubjectLength,
                                    maxQueryLength); CUERR;

        shd::Result_t* results = mybuffers.h_results;
        shd::Result_t* d_results = mybuffers.d_results;
#if 0
        cudaMemcpyAsync(results,
            d_results,
            sizeof(shd::Result_t) * numberOfAlignments,
            D2H,
            mybuffers.streams[0]); CUERR;
#else
        cudaMemcpyAsync(results,
            d_results,
            mybuffers.transfersizeD2H,
            D2H,
            mybuffers.streams[0]); CUERR;
#endif

    }else{ // use cpu for alignment

#endif
        assert(false && "shd revcompl not available on cpu");
        
        device = AlignmentDevice::CPU;

        timings.executionBegin();

        auto queryIt = queriesbegin;
        auto alignmentsIt = alignmentsbegin;

        for(auto t = std::make_pair(0, subjectsbegin); t.second != subjectsend; t.first++, t.second++){
            auto& subjectcount = t.first;
            auto& subjectIt = t.second;

            assert((*subjectIt)->length() <= mybuffers.max_sequence_length);
            assert((*subjectIt)->getNumBytes() <= mybuffers.max_sequence_bytes);

            const char* const subject = (const char*)(*subjectIt)->begin();
            const int subjectLength = (*subjectIt)->length();

            const int nQueries = queriesPerSubject[subjectcount];

            for(int i = 0; i < nQueries; i++){
                const char* query =  (const char*)(*queryIt)->begin();
                const int queryLength = (*queryIt)->length();

                *alignmentsIt = cpu_shifted_hamming_distance<Sequence_t>(subject, subjectLength, query, queryLength,
                                                                        min_overlap,
                                                                        maxErrorRate,
                                                                        min_overlap_ratio);

                queryIt++;
                alignmentsIt++;
            }
        }

        timings.executionEnd();

#ifdef __NVCC__
    }
#endif

    return device;
}

/*
Ensures that all alignment results from the preceding call to shifted_hamming_distance_async
have been stored in the range [alignmentsbegin, alignmentsend[

must be called with the same mybuffers, alignmentsbegin, alignmentsend, canUseGpu
as the call to shifted_hamming_distance_async
*/

template<class AlignmentIter>
void shifted_hamming_distance_with_revcompl_get_results(SHDhandle& handle,
                                AlignmentIter alignmentsbegin,
                                AlignmentIter alignmentsend,
                                bool canUseGpu){

    static_assert(std::is_same<typename AlignmentIter::value_type, shd::Result_t>::value, "shifted hamming distance unexpected Alignement type");

    const int numberOfAlignments = std::distance(alignmentsbegin, alignmentsend);

    //nothing to do here
    if(numberOfAlignments == 0)
        return;
#ifdef __NVCC__

    auto& mybuffers = handle.buffers;
    auto& timings = handle.timings;

    if(canUseGpu && numberOfAlignments >= mybuffers.gpuThreshold){ // use gpu for alignment
        cudaSetDevice(mybuffers.deviceId); CUERR;

        shd::Result_t* results = mybuffers.h_results;

        cudaStreamSynchronize(mybuffers.streams[0]); CUERR;

        timings.executionEnd();

        timings.postprocessingBegin();

        for(auto t = std::make_pair(0, alignmentsbegin); t.second != alignmentsend; t.first++, t.second++){
            auto& count = t.first;
            auto& it = t.second;

            *it = results[count];
        }

        timings.postprocessingEnd();


    }else{ // cpu already done

#endif

#ifdef __NVCC__
    }
#endif

    return;
}

template<class Sequence_t, class SubjectIter, class QueryIter, class AlignmentIter>
AlignmentDevice shifted_hamming_distance_with_revcompl(SHDhandle& handle,
                                SubjectIter subjectsbegin,
                                SubjectIter subjectsend,
                                QueryIter queriesbegin,
                                QueryIter queriesend,
                                AlignmentIter alignmentsbegin,
                                AlignmentIter alignmentsend,
                                const std::vector<int>& queriesPerSubject,
                                int min_overlap,
                                double maxErrorRate,
                                double min_overlap_ratio,
                                bool canUseGpu){

    AlignmentDevice device = shifted_hamming_distance_with_revcompl_async(handle, subjectsbegin, subjectsend,
                                    queriesbegin, queriesend,
                                    alignmentsbegin, alignmentsend,
                                    queriesPerSubject, min_overlap,
                                    maxErrorRate, min_overlap_ratio, canUseGpu);

    shifted_hamming_distance_with_revcompl_get_results(handle,
                                    alignmentsbegin,
                                    alignmentsend,
                                    canUseGpu);

    return device;
}












/*

    ########## SEMI GLOBAL ALIGNMENT ##########

*/

using SGAhandle = AlignmentHandle<sga::SGAdata>;
using SGAResult = sga::Result_t;

void init_SGAhandle(SGAhandle& handle, int deviceId,
                        int max_sequence_length,
                        int max_sequence_bytes,
                        int gpuThreshold);

//free buffers
void destroy_SGAhandle(SGAhandle& handle);

/*
    Wrapper functions for semi global alignment calls
    which derive the required accessor from Sequence_t
*/
template<class Sequence_t>
SGAResult cpu_semi_global_alignment(const char* subject,
                                    int subjectlength,
                                    const char* query,
                                    int querylength,
                                    int score_match,
                                    int score_sub,
                                    int score_ins,
                                    int score_del){

    auto accessor = [] (const char* data, int length, int index){
        return Sequence_t::get(data, length, index);
    };

    return sga::cpu_semi_global_alignment(subject, subjectlength, query, querylength,
                        score_match, score_sub, score_ins, score_del, accessor);
}

#ifdef __NVCC__

template<class Sequence_t>
void call_semi_global_alignment_kernel_async(const sga::SGAdata& sgadata,
                                            const int score_match,
                                            const int score_sub,
                                            const int score_ins,
                                            const int score_del,
                                            const int maxSubjectLength,
                                            const int maxQueryLength){

    auto accessor = [] __device__ (const char* data, int length, int index){
        return Sequence_t::get(data, length, index);
    };

    sga::call_semi_global_alignment_kernel_async(sgadata, score_match, score_sub,
            score_ins, score_del, maxSubjectLength, maxQueryLength, accessor);
}

template<class Sequence_t>
void call_semi_global_alignment_kernel(const sga::SGAdata& sgadata,
                                            const int score_match,
                                            const int score_sub,
                                            const int score_ins,
                                            const int score_del,
                                            const int maxSubjectLength,
                                            const int maxQueryLength){

    auto accessor = [] __device__ (const char* data, int length, int index){
        return Sequence_t::get(data, length, index);
    };

    sga::call_semi_global_alignment_kernel(sgadata, score_match, score_sub,
            score_ins, score_del, maxSubjectLength, maxQueryLength, accessor);
}

#endif

/*
    Begin calculation of alignments. Results are only present after a subsequent
    call to semi_global_alignment_get_results

    SubjectIter,QueryIter: Iterator to const Sequence_t*
    AlignmentIter: Iterator to sga::Result_t
*/
template<class Sequence_t, class SubjectIter, class QueryIter, class AlignmentIter>
AlignmentDevice semi_global_alignment_async(SGAhandle& handle,
                                SubjectIter subjectsbegin,
                                SubjectIter subjectsend,
                                QueryIter queriesbegin,
                                QueryIter queriesend,
                                AlignmentIter alignmentsbegin,
                                AlignmentIter alignmentsend,
                                const std::vector<int>& queriesPerSubject,
                                int score_match,
                                int score_sub,
                                int score_ins,
                                int score_del,
                                bool canUseGpu){

    static_assert(std::is_same<typename AlignmentIter::value_type, sga::Result_t>::value, "semi_global_alignment unexpected Alignement type");

    auto& mybuffers = handle.buffers;
    auto& timings = handle.timings;

    AlignmentDevice device = AlignmentDevice::None;

    const int numberOfAlignments = std::distance(alignmentsbegin, alignmentsend);
    assert(numberOfAlignments == std::distance(queriesbegin, queriesend));

    //nothing to do here
    if(numberOfAlignments == 0)
        return device;
#ifdef __NVCC__

    const int numberOfSubjects = queriesPerSubject.size();

    if(canUseGpu && numberOfAlignments >= mybuffers.gpuThreshold){ // use gpu for alignment
        device = AlignmentDevice::GPU;

        timings.preprocessingBegin();

        cudaSetDevice(mybuffers.deviceId); CUERR;

        mybuffers.resize(numberOfSubjects, numberOfAlignments);

        mybuffers.n_subjects = numberOfSubjects;
        mybuffers.n_queries = numberOfAlignments;

        int maxSubjectLength = 0;

        for(auto t = std::make_pair(0, subjectsbegin); t.second != subjectsend; t.first++, t.second++){
            auto& count = t.first;
            auto& it = t.second;

            assert((*it)->length() <= mybuffers.max_sequence_length);
            assert((*it)->getNumBytes() <= mybuffers.max_sequence_bytes);

            std::memcpy(mybuffers.h_subjectsdata + count * mybuffers.sequencepitch,
                        (*it)->begin(),
                        (*it)->getNumBytes());

            mybuffers.h_subjectlengths[count] = (*it)->length();
            maxSubjectLength = std::max(int((*it)->length()), maxSubjectLength);
        }

        int maxQueryLength = 0;

        for(auto t = std::make_pair(0, queriesbegin); t.second != queriesend; t.first++, t.second++){
            auto& count = t.first;
            auto& it = t.second;

            assert((*it)->length() <= mybuffers.max_sequence_length);
            assert((*it)->getNumBytes() <= mybuffers.max_sequence_bytes);

            std::memcpy(mybuffers.h_queriesdata + count * mybuffers.sequencepitch,
                        (*it)->begin(),
                        (*it)->getNumBytes());

            mybuffers.h_querylengths[count] = (*it)->length();
            maxQueryLength = std::max(int((*it)->length()), maxQueryLength);
        }

        mybuffers.h_NqueriesPrefixSum[0] = 0;
        for(std::size_t i = 0; i < queriesPerSubject.size(); i++)
            mybuffers.h_NqueriesPrefixSum[i+1] = mybuffers.h_NqueriesPrefixSum[i] + queriesPerSubject[i];

        assert(numberOfAlignments == mybuffers.h_NqueriesPrefixSum[queriesPerSubject.size()]);

        timings.preprocessingEnd();

        timings.executionBegin();


        // copy data to gpu
        cudaMemcpyAsync(mybuffers.d_subjectsdata,
                mybuffers.h_subjectsdata,
                numberOfSubjects * mybuffers.sequencepitch,
                H2D,
                mybuffers.streams[0]); CUERR;
        cudaMemcpyAsync(mybuffers.d_queriesdata,
                mybuffers.h_queriesdata,
                numberOfAlignments * mybuffers.sequencepitch,
                H2D,
                mybuffers.streams[0]); CUERR;
        cudaMemcpyAsync(mybuffers.d_subjectlengths,
                mybuffers.h_subjectlengths,
                numberOfSubjects * sizeof(int),
                H2D,
                mybuffers.streams[0]); CUERR;
        cudaMemcpyAsync(mybuffers.d_querylengths,
                mybuffers.h_querylengths,
                numberOfAlignments * sizeof(int),
                H2D,
                mybuffers.streams[0]); CUERR;
        cudaMemcpyAsync(mybuffers.d_NqueriesPrefixSum,
                mybuffers.h_NqueriesPrefixSum,
                (numberOfSubjects+1) * sizeof(int),
                H2D,
                mybuffers.streams[0]); CUERR;

        call_semi_global_alignment_kernel_async<Sequence_t>(mybuffers,
                                                    score_match,
                                                    score_sub,
                                                    score_ins,
                                                    score_del,
                                                    maxSubjectLength,
                                                    maxQueryLength); CUERR;

        sga::Attributes_t* results = mybuffers.h_results;
        sga::Attributes_t* d_results = mybuffers.d_results;
        sga::Op_t* ops = mybuffers.h_ops;
        sga::Op_t* d_ops = mybuffers.d_ops;

        cudaMemcpyAsync(results,
            d_results,
            sizeof(sga::Attributes_t) * numberOfAlignments,
            D2H,
            mybuffers.streams[0]); CUERR;

        cudaMemcpyAsync(ops,
            d_ops,
            sizeof(sga::Op_t) * numberOfAlignments * mybuffers.max_ops_per_alignment,
            D2H,
            mybuffers.streams[0]); CUERR;

    }else{ // use cpu for alignment

#endif
        device = AlignmentDevice::CPU;

        timings.executionBegin();

        auto queryIt = queriesbegin;
        auto alignmentsIt = alignmentsbegin;

        for(auto t = std::make_pair(0, subjectsbegin); t.second != subjectsend; t.first++, t.second++){
            auto& subjectcount = t.first;
            auto& subjectIt = t.second;

            assert((*subjectIt)->length() <= mybuffers.max_sequence_length);
            assert((*subjectIt)->getNumBytes() <= mybuffers.max_sequence_bytes);

            const char* const subject = (const char*)(*subjectIt)->begin();
            const int subjectLength = (*subjectIt)->length();

            const int nQueries = queriesPerSubject[subjectcount];

            for(int i = 0; i < nQueries; i++){
                const char* query =  (const char*)(*queryIt)->begin();
                const int queryLength = (*queryIt)->length();
                *alignmentsIt = cpu_semi_global_alignment<Sequence_t>(subject,
                                                    subjectLength,
                                                    query,
                                                    queryLength,
                                                    score_match,
                                                    score_sub,
                                                    score_ins,
                                                    score_del);

                queryIt++;
                alignmentsIt++;
            }
        }

        timings.executionEnd();

#ifdef __NVCC__
    }
#endif

    return device;
}


/*
Ensures that all alignment results from the preceding call to semi_global_alignment_async
have been stored in the range [alignmentsbegin, alignmentsend[

must be called with the same mybuffers, alignmentsbegin, alignmentsend, canUseGpu
as the call to semi_global_alignment_async
*/

template<class AlignmentIter>
void semi_global_alignment_get_results(SGAhandle& handle,
                                AlignmentIter alignmentsbegin,
                                AlignmentIter alignmentsend,
                                bool canUseGpu){

    static_assert(std::is_same<typename AlignmentIter::value_type, sga::Result_t>::value, "semi_global_alignment unexpected Alignement type");

    const int numberOfAlignments = std::distance(alignmentsbegin, alignmentsend);

    //nothing to do here
    if(numberOfAlignments == 0)
        return;
#ifdef __NVCC__

    auto& mybuffers = handle.buffers;
    auto& timings = handle.timings;

    if(canUseGpu && numberOfAlignments >= mybuffers.gpuThreshold){ // use gpu for alignment
        cudaSetDevice(mybuffers.deviceId); CUERR;

        sga::Attributes_t* results = mybuffers.h_results;
        sga::Op_t* ops = mybuffers.h_ops;

        cudaStreamSynchronize(mybuffers.streams[0]); CUERR;

        timings.executionEnd();

        timings.postprocessingBegin();

        for(auto t = std::make_pair(0, alignmentsbegin); t.second != alignmentsend; t.first++, t.second++){
            auto& count = t.first;
            auto& it = t.second;

            it->attributes = results[count];

            auto& opvector = it->operations;
            const int nOps = it->get_nOps();

            opvector.resize(nOps);
            std::reverse_copy(ops + count * mybuffers.max_ops_per_alignment,
                      ops + count * mybuffers.max_ops_per_alignment + nOps,
                      opvector.begin());
        }

        timings.postprocessingEnd();

    }else{ // cpu already done

#endif

#ifdef __NVCC__
    }
#endif

    return;
}

template<class Sequence_t, class SubjectIter, class QueryIter, class AlignmentIter>
AlignmentDevice semi_global_alignment(SGAhandle& handle,
                                SubjectIter subjectsbegin,
                                SubjectIter subjectsend,
                                QueryIter queriesbegin,
                                QueryIter queriesend,
                                AlignmentIter alignmentsbegin,
                                AlignmentIter alignmentsend,
                                const std::vector<int>& queriesPerSubject,
                                int score_match,
                                int score_sub,
                                int score_ins,
                                int score_del,
                                bool canUseGpu){

    AlignmentDevice device = semi_global_alignment_async(handle,
                                                        subjectsbegin,
                                                        subjectsend,
                                                        queriesbegin,
                                                        queriesend,
                                                        alignmentsbegin,
                                                        alignmentsend,
                                                        queriesPerSubject,
                                                        score_match,
                                                        score_sub,
                                                        score_ins,
                                                        score_del,
                                                        canUseGpu);

    semi_global_alignment_get_results(handle, alignmentsbegin,
                                      alignmentsend, canUseGpu);
    return device;
}


}

#endif
