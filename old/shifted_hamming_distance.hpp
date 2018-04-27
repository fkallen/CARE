#ifndef CARE_SHIFTED_HAMMING_DISTANCE_HPP
#define CARE_SHIFTED_HAMMING_DISTANCE_HPP

#include "alignment.hpp"
#include "options.hpp"

#include "shifted_hamming_distance_impl.hpp"

#include <vector>
#include <chrono>
#include <memory>
#include <cstring>
#include <cassert>

namespace care{


template<class Sequence_t>
Result_t cpu_shifted_hamming_distance(const char* subject,
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
void call_shd_kernel_async(const SHDdata& shddata,
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
void call_shd_kernel(const SHDdata& shddata,
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
AlignmentDevice shifted_hamming_distance_async(SHDdata& mybuffers,
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

    AlignmentDevice device = AlignmentDevice::None;

    const int numberOfSubjects = queriesPerSubject.size();
    const int numberOfAlignments = std::distance(alignmentsbegin, alignmentsend);
    assert(numberOfAlignments == std::distance(queriesbegin, queriesend));

    //nothing to do here
    if(numberOfAlignments == 0)
        return device;
#ifdef __NVCC__

    std::chrono::time_point<std::chrono::system_clock> tpa = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> tpb = std::chrono::system_clock::now();

    if(canUseGpu && numberOfAlignments >= mybuffers.gpuThreshold){ // use gpu for alignment
        device = AlignmentDevice::GPU;
        tpa = std::chrono::system_clock::now();

        cudaSetDevice(mybuffers.deviceId); CUERR;

        mybuffers.resize(numberOfSubjects, numberOfAlignments);

        mybuffers.n_subjects = numberOfSubjects;
        mybuffers.n_queries = numberOfAlignments;

        tpb = std::chrono::system_clock::now();

        mybuffers.resizetime += tpb - tpa;

        tpa = std::chrono::system_clock::now();

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

        tpb = std::chrono::system_clock::now();

        mybuffers.preprocessingtime += tpb - tpa;


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

        call_shd_kernel_async<Sequence_t>(shddata,
                                    min_overlap,
                                    maxErrorRate,
                                    min_overlap_ratio,
                                    maxSubjectLength,
                                    maxQueryLength);

        AlignResultCompact* results = mybuffers.h_results;
        AlignResultCompact* d_results = mybuffers.d_results;

        cudaMemcpyAsync(results,
            d_results,
            sizeof(AlignResultCompact) * numberOfAlignments,
            D2H,
            mybuffers.streams[0]); CUERR;

    }else{ // use cpu for alignment

#endif
        device = AlignmentDevice::CPU;

        tpa = std::chrono::system_clock::now();

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

                *alignmentsIt = cpu_shifted_hamming_distance<Sequence_t>(subject, query, subjectLength, queryLength,
                                                                        min_overlap,
                                                                        maxErrorRate,
                                                                        min_overlap_ratio);

                queryIt++;
                alignmentsIt++;
            }
        }

        tpb = std::chrono::system_clock::now();

        mybuffers.alignmenttime += tpb - tpa;

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
void shifted_hamming_distance_get_results(SHDdata& mybuffers,
                                AlignmentIter alignmentsbegin,
                                AlignmentIter alignmentsend,
                                bool canUseGpu){

    static_assert(std::is_same<typename AlignmentIter::value_type, AlignResultCompact>::value, "shifted hamming distance unexpected Alignement type");

    const int numberOfAlignments = std::distance(alignmentsbegin, alignmentsend);

    //nothing to do here
    if(numberOfAlignments == 0)
        return;
#ifdef __NVCC__

    std::chrono::time_point<std::chrono::system_clock> tpa = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> tpb = std::chrono::system_clock::now();

    if(canUseGpu && numberOfAlignments >= mybuffers.gpuThreshold){ // use gpu for alignment
        cudaSetDevice(mybuffers.deviceId); CUERR;

        AlignResultCompact* results = mybuffers.h_results;

        cudaStreamSynchronize(mybuffers.streams[0]); CUERR;

        tpa = std::chrono::system_clock::now();

        for(auto t = std::make_pair(0, alignmentsbegin); t.second != alignmentsend; t.first++, t.second++){
            auto& count = t.first;
            auto& it = t.second;

            *it = results[count];
        }

        tpb = std::chrono::system_clock::now();
        mybuffers.postprocessingtime += tpb - tpa;


    }else{ // cpu already done

#endif

#ifdef __NVCC__
    }
#endif

    return;
}

template<class Sequence_t, class SubjectIter, class QueryIter, class AlignmentIter>
AlignmentDevice shifted_hamming_distance(SHDdata& mybuffers,
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

    shifted_hamming_distance_async(mybuffers, subjectsbegin, subjectsend,
                                    queriesbegin, queriesend,
                                    alignmentsbegin, alignmentsend,
                                    queriesPerSubject, min_overlap,
                                    maxErrorRate, min_overlap_ratio, canUseGpu);

    shifted_hamming_distance_get_results(mybuffers,
                                    alignmentsbegin,
                                    alignmentsend,
                                    canUseGpu);

}








#if 0
template<class Sequence_t>
int find_shifted_hamming_distance_gpu_threshold(int deviceId, int minsequencelength, int minsequencebytes){
    int threshold = std::numeric_limits<int>::max();
#ifdef __NVCC__
    SHDdata shddata;
    cuda_init_SHDdata(shddata, deviceId, minsequencelength, minsequencebytes, 0);

    const int increment = 20;
    int nalignments = 0;

    std::chrono::time_point<std::chrono::system_clock> gpustart;
    std::chrono::time_point<std::chrono::system_clock> gpuend;
    std::chrono::time_point<std::chrono::system_clock> cpustart;
    std::chrono::time_point<std::chrono::system_clock> cpuend;

    std::string seqstring = "";
    for(int i = 0; i < minsequencelength; i++)
        seqstring += "C";

    Sequence_t sequence(seqstring);

    GoodAlignmentProperties alignProps;

    do{
        nalignments += increment;
        std::vector<Sequence_t> sequences(nalignments, sequence);
        std::vector<AlignResultCompact> gpuresults(nalignments);
        std::vector<AlignResultCompact> cpuresults(nalignments);

        gpustart = std::chrono::system_clock::now();

        shddata.resize(1, nalignments);
        shdparams params;

        params.props = alignProps;
        params.max_sequence_bytes = shddata.max_sequence_bytes;
        params.sequencepitch = shddata.sequencepitch;
        params.subjectlength = minsequencelength;
        params.n_queries = nalignments;
        params.querylengths = shddata.d_querylengths;
        params.subjectdata = shddata.d_subjectsdata;
        params.queriesdata = shddata.d_queriesdata;
        params.results = shddata.d_results;

        int* querylengths = shddata.h_querylengths;
        char* subjectdata = shddata.h_subjectsdata;
        char* queriesdata = shddata.h_queriesdata;

        std::memcpy(subjectdata, sequences[0].begin(), sequences[0].getNumBytes());
        for(int count = 0; count < nalignments; count++){
            const auto& seq = sequences[count];

            std::memcpy(queriesdata + count * shddata.sequencepitch,
                    seq.begin(),
                    seq.getNumBytes());

            querylengths[count] = seq.length();
        }
        cudaMemcpyAsync(const_cast<char*>(params.subjectdata),
                subjectdata,
                shddata.sequencepitch,
                H2D,
                shddata.streams[0]); CUERR;
        cudaMemcpyAsync(const_cast<char*>(params.queriesdata),
                queriesdata,
                shddata.sequencepitch * params.n_queries,
                H2D,
                shddata.streams[0]); CUERR;
        cudaMemcpyAsync(const_cast<int*>(params.querylengths),
                querylengths,
                sizeof(int) * params.n_queries,
                H2D,
                shddata.streams[0]); CUERR;

        // start kernel
        call_shd_kernel_async<Sequence_t>(params, minsequencelength, shddata.streams[0]);

        AlignResultCompact* results = shddata.h_results;

        cudaMemcpyAsync(results,
            params.results,
            sizeof(AlignResultCompact) * params.n_queries,
            D2H,
            shddata.streams[0]); CUERR;

        cudaStreamSynchronize(shddata.streams[0]); CUERR;

        for(int count = 0; count < nalignments; count++){
            gpuresults[count] = results[count];
        }

        gpuend = std::chrono::system_clock::now();


        cpustart = std::chrono::system_clock::now();

        const char* const subject = (const char*)sequences[0].begin();
        const int subjectLength = sequences[0].length();

        for(int i = 0; i < nalignments; i++){
            const char* query =  (const char*)sequences[i].begin();
            const int queryLength = sequences[i].length();
            cpuresults[i] = cpu_shifted_hamming_distance<Sequence_t>(alignProps, subject, query, subjectLength, queryLength);
        }

        cpuend = std::chrono::system_clock::now();


    }while(gpuend - gpustart > cpuend - cpustart || nalignments == 1000);

    if(gpuend - gpustart <= cpuend - cpustart){
        threshold = nalignments;
    }

    cuda_cleanup_SHDdata(shddata);
#endif
    return threshold;
}
#endif



}


#endif
