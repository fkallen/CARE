#ifndef CARE_ALIGNMENT_HPP
#define CARE_ALIGNMENT_HPP

#include "hpc_helpers.cuh"

#include "shifted_hamming_distance.hpp"
#include "semi_global_alignment.hpp"
#include "tasktiming.hpp"
#include "bestalignment.hpp"
#include "sequence.hpp"

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
template<class Sequence_t, typename std::enable_if<!std::is_same<Sequence_t, Sequence2BitHiLo>::value, int>::type = 0>
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

template<class Sequence_t, typename std::enable_if<std::is_same<Sequence_t, Sequence2BitHiLo>::value, int*>::type = nullptr>
SHDResult cpu_shifted_hamming_distance(const char* subject,
                                        int subjectlength,
										const char* query,
										int querylength,
                                        int min_overlap,
                                        double maxErrorRate,
                                        double min_overlap_ratio){

    auto getNumBytes = [] (int nbases){
        return Sequence_t::getNumBytes(nbases);
    };

    return shd::cpu_shifted_hamming_distance_popcount(subject, subjectlength, query, querylength,
                        min_overlap, maxErrorRate, min_overlap_ratio, getNumBytes);
}




/*

	########## SHIFTED HAMMING DISTANCE CANONICAL batched
*/


#ifdef __NVCC__

template<class Sequence_t>
typename std::enable_if<!std::is_same<Sequence_t, Sequence2BitHiLo>::value, void>::type
call_shd_canonical_kernel_async(const shd::SHDdata& shddata,
                      int min_overlap,
                      double maxErrorRate,
                      double min_overlap_ratio,
                      int maxSubjectLength,
                      int maxQueryLength){

    auto accessor = [] __device__ (const char* data, int length, int index){
        return Sequence_t::get(data, length, index);
    };

    auto nucleotide_accessor = [] __device__ (const char* data, int length, int index){
        return Sequence_t::get_as_nucleotide(data, length, index);
    };

    auto make_reverse_complement_inplace = [] __device__ (std::uint8_t* sequence, int sequencelength){
        return Sequence_t::make_reverse_complement_inplace(sequence, sequencelength);
    };

    auto make_unpacked_reverse_complement_inplace = [] __device__ (std::uint8_t* sequence, int sequencelength){
        return SequenceString::make_reverse_complement_inplace(sequence, sequencelength);
    };

    auto comp = [=] __device__ (const SHDResult& fwdAlignment,
                               const SHDResult& revcmplAlignment,
                               int subjectlength,
                               int querylength) -> BestAlignment_t{
        return choose_best_alignment(fwdAlignment,
                              revcmplAlignment,
                              subjectlength,
                              querylength,
                              min_overlap_ratio,
                              min_overlap,
                              maxErrorRate);
    };

    shd::call_shd_with_revcompl_kernel_async(shddata,
                                        min_overlap,
                                        maxErrorRate,
                                        min_overlap_ratio,
                                        maxSubjectLength,
                                        maxQueryLength,
                                        accessor,
                                        make_reverse_complement_inplace);

    call_cuda_find_best_alignment_and_unpack_good_sequences_kernel_async(
                                        shddata.d_results,
                                        shddata.d_bestAlignmentFlags,
                                        shddata.d_unpacked_subjects,
                                        shddata.d_unpacked_queries,
                                        shddata.d_subjectlengths,
                                        shddata.d_querylengths,
                                        shddata.d_NqueriesPrefixSum,
                                        shddata.n_subjects,
                                        shddata.n_queries,
                                        shddata.d_subjectsdata,
                                        shddata.d_queriesdata,
                                        shddata.max_sequence_length,
                                        shddata.sequencepitch,
                                        comp,
                                        nucleotide_accessor,
                                        make_unpacked_reverse_complement_inplace,
                                        shddata.streams[0]);
}

template<class Sequence_t>
typename std::enable_if<std::is_same<Sequence_t, Sequence2BitHiLo>::value, void>::type
call_shd_canonical_kernel_async(const shd::SHDdata& shddata,
                      int min_overlap,
                      double maxErrorRate,
                      double min_overlap_ratio,
                      int maxSubjectLength,
                      int maxQueryLength){

    auto getNumBytes = [] __device__ (int length){
        return Sequence_t::getNumBytes(length);
    };

    auto nucleotide_accessor = [] __device__ (const char* data, int length, int index){
        return Sequence_t::get_as_nucleotide(data, length, index);
    };

    auto make_unpacked_reverse_complement_inplace = [] __device__ (std::uint8_t* sequence, int sequencelength){
        return SequenceString::make_reverse_complement_inplace(sequence, sequencelength);
    };

    auto comp = [=] __device__ (const SHDResult& fwdAlignment,
                             const SHDResult& revcmplAlignment,
                             int subjectlength,
                             int querylength) -> BestAlignment_t{
        return choose_best_alignment(fwdAlignment,
                            revcmplAlignment,
                            subjectlength,
                            querylength,
                            min_overlap_ratio,
                            min_overlap,
                            maxErrorRate);
    };

    shd::call_popcount_shd_with_revcompl_kernel_async(shddata,
                              min_overlap,
                              maxErrorRate,
                              min_overlap_ratio,
                              maxSubjectLength,
                              maxQueryLength,
                              getNumBytes);


    call_cuda_find_best_alignment_and_unpack_good_sequences_kernel_async(
        shddata.d_results,
        shddata.d_bestAlignmentFlags,
        shddata.d_unpacked_queries,
        shddata.d_subjectlengths,
        shddata.d_querylengths,
        shddata.d_NqueriesPrefixSum,
        shddata.n_subjects,
        shddata.n_queries,
        shddata.d_queriesdata,
        shddata.max_sequence_length,
        shddata.sequencepitch,
        comp,
        nucleotide_accessor,
        make_unpacked_reverse_complement_inplace,
        shddata.streams[0]);
}

#endif


/*
    Calculates both subject/query alignment and subject/reversecomplementquery alignment
    and keeps only the better one according to choose_best_alignment(...).
    Flags indicate whether the better one is forward alignment, reverse complement alignment, or if both alignments are bad.
*/

template<class Sequence_t, class SubjectIter, class QueryIter, class AlignmentIter, class FlagsIter, class StringIter,
    typename std::enable_if<!std::is_same<typename AlignmentIter::value_type, shd::Result_t>::value, int>::type = 0>
AlignmentDevice shifted_hamming_distance_canonical_batched_async(SHDhandle& handle,
                                const std::vector<SubjectIter>& subjectsbegin,
                                const std::vector<SubjectIter>& subjectsend,
                                const std::vector<QueryIter>& queriesbegin,
                                const std::vector<QueryIter>& queriesend,
                                std::vector<AlignmentIter>& alignmentsbegin,
                                std::vector<AlignmentIter>& alignmentsend,
                                std::vector<FlagsIter>& flagsbegin,
                                std::vector<FlagsIter>& flagsend,
                                std::vector<StringIter>& bestSequenceStringsbegin,
                                std::vector<StringIter>& bestSequenceStringsend,
                                const std::vector<int>& queriesPerSubject,
                                int min_overlap,
                                double maxErrorRate,
                                double min_overlap_ratio,
                                bool canUseGpu){
    return AlignmentDevice::None;
}

template<class Sequence_t, class SubjectIter, class QueryIter, class AlignmentIter, class FlagsIter, class StringIter,
    typename std::enable_if<std::is_same<typename AlignmentIter::value_type, shd::Result_t>::value, int*>::type = nullptr>
AlignmentDevice shifted_hamming_distance_canonical_batched_async(SHDhandle& handle,
                                const std::vector<SubjectIter>& subjectsbegin,
                                const std::vector<SubjectIter>& subjectsend,
                                const std::vector<QueryIter>& queriesbegin,
                                const std::vector<QueryIter>& queriesend,
                                std::vector<AlignmentIter>& alignmentsbegin,
                                std::vector<AlignmentIter>& alignmentsend,
                                std::vector<FlagsIter>& flagsbegin,
                                std::vector<FlagsIter>& flagsend,
                                std::vector<StringIter>& bestSequenceStringsbegin,
                                std::vector<StringIter>& bestSequenceStringsend,
                                const std::vector<int>& queriesPerSubject,
                                int min_overlap,
                                double maxErrorRate,
                                double min_overlap_ratio,
                                bool canUseGpu){

    //static_assert(std::is_same<typename AlignmentIter::value_type, shd::Result_t>::value, "shifted hamming distance unexpected Alignment type");

    auto& mybuffers = handle.buffers;
    auto& timings = handle.timings;

    AlignmentDevice device = AlignmentDevice::None;

    int numberOfFlags = 0;
    int numberOfAlignments = 0;
    int numberOfQueries = 0;
    int numberOfSubjects = 0;

    assert(subjectsbegin.size() == subjectsend.size());
    assert(subjectsbegin.size() == queriesbegin.size());
    assert(subjectsbegin.size() == queriesend.size());
    assert(subjectsbegin.size() == alignmentsbegin.size());
    assert(subjectsbegin.size() == alignmentsend.size());
    assert(subjectsbegin.size() == flagsbegin.size());
    assert(subjectsbegin.size() == flagsend.size());
    assert(subjectsbegin.size() == bestSequenceStringsbegin.size());
    assert(subjectsbegin.size() == bestSequenceStringsend.size());

    for(std::size_t i = 0; i < flagsbegin.size(); i++){
        numberOfFlags += std::distance(flagsbegin[i], flagsend[i]);
    }
    for(std::size_t i = 0; i < alignmentsbegin.size(); i++){
        numberOfAlignments += std::distance(alignmentsbegin[i], alignmentsend[i]);
    }
    for(std::size_t i = 0; i < queriesbegin.size(); i++){
        numberOfQueries += std::distance(queriesbegin[i], queriesend[i]);
    }
    assert(numberOfAlignments == numberOfQueries);
    assert(numberOfAlignments == numberOfFlags);

    for(std::size_t i = 0; i < subjectsbegin.size(); i++){
        numberOfSubjects += std::distance(subjectsbegin[i], subjectsend[i]);
    }
    assert(numberOfSubjects == (int)queriesPerSubject.size());

    //nothing to do here
    if(numberOfAlignments == 0)
        return device;
#ifdef __NVCC__

    if(canUseGpu && numberOfAlignments >= mybuffers.gpuThreshold){ // use gpu for alignment
        device = AlignmentDevice::GPU;

        timings.preprocessingBegin();

        cudaSetDevice(mybuffers.deviceId); CUERR;

        mybuffers.resize(numberOfSubjects, numberOfQueries, 2*numberOfAlignments);

        int maxSubjectLength = 0;

        for(std::size_t i = 0, count = 0; i < subjectsbegin.size(); i++){

            for(auto it = subjectsbegin[i]; it != subjectsend[i]; ++it, ++count){

                assert((*it)->length() <= mybuffers.max_sequence_length);
                assert((*it)->getNumBytes() <= mybuffers.max_sequence_bytes);

                std::memcpy(mybuffers.h_subjectsdata + count * mybuffers.sequencepitch,
                            (*it)->begin(),
                            (*it)->getNumBytes());

                mybuffers.h_subjectlengths[count] = (*it)->length();
                maxSubjectLength = std::max(int((*it)->length()), maxSubjectLength);
            }

        }

        int maxQueryLength = 0;

        for(std::size_t i = 0, count = 0; i < queriesbegin.size(); i++){

            for(auto it = queriesbegin[i]; it != queriesend[i]; ++it, ++count){

                assert((*it)->length() <= mybuffers.max_sequence_length);
                assert((*it)->getNumBytes() <= mybuffers.max_sequence_bytes);

                std::memcpy(mybuffers.h_queriesdata + count * mybuffers.sequencepitch,
                            (*it)->begin(),
                            (*it)->getNumBytes());

                mybuffers.h_querylengths[count] = (*it)->length();
                maxQueryLength = std::max(int((*it)->length()), maxQueryLength);
            }

        }

        mybuffers.h_NqueriesPrefixSum[0] = 0;
        for(std::size_t i = 0; i < queriesPerSubject.size(); i++)
            mybuffers.h_NqueriesPrefixSum[i+1] = mybuffers.h_NqueriesPrefixSum[i] + queriesPerSubject[i];

        assert(numberOfAlignments == mybuffers.h_NqueriesPrefixSum[queriesPerSubject.size()]);

        timings.preprocessingEnd();

        timings.executionBegin();


        // copy data to gpu
        std::size_t transfersizeH2D = mybuffers.memSubjects; // d_subjectsdata
        transfersizeH2D += mybuffers.memSubjectLengths; // d_subjectlengths
        transfersizeH2D += mybuffers.memNqueriesPrefixSum; // d_NqueriesPrefixSum
        transfersizeH2D += mybuffers.memQueries; // d_queriesdata
        transfersizeH2D += mybuffers.memQueryLengths; // d_querylengths

        cudaMemcpyAsync(mybuffers.deviceptr,
                        mybuffers.hostptr,
                        transfersizeH2D,
                        H2D,
                        mybuffers.streams[0]); CUERR;


        call_shd_canonical_kernel_async<Sequence_t>(mybuffers,
                                    min_overlap,
                                    maxErrorRate,
                                    min_overlap_ratio,
                                    maxSubjectLength,
                                    maxQueryLength); CUERR;

        std::size_t transfersizeD2H = mybuffers.memResults; //d_results
        transfersizeD2H += mybuffers.memBestAlignmentFlags; // d_bestAlignmentFlags
        transfersizeD2H += mybuffers.memUnpackedSubjects; // d_unpacked_subjects
        transfersizeD2H += mybuffers.memUnpackedQueries; // d_unpacked_queries

        //copy results to host

        cudaMemcpyAsync(mybuffers.h_results,
            mybuffers.d_results,
            transfersizeD2H,
            D2H,
            mybuffers.streams[0]); CUERR;
#if 0
        cudaStreamSynchronize(mybuffers.streams[0]);

        for(std::size_t i = 0, count = 0; i < queriesbegin.size(); i++){

            for(auto it = queriesbegin[i]; it != queriesend[i]; ++it, ++count){
                auto querystring = (*it)->toString();
                auto rcquerystring = (*it)->reverseComplement().toString();

                auto newstring = std::string{mybuffers.h_unpacked_queries + count * mybuffers.max_sequence_length,
                                                    mybuffers.h_unpacked_queries + count * mybuffers.max_sequence_length + (*it)->length()};
                auto flag = mybuffers.h_bestAlignmentFlags[count];
                if((flag == BestAlignment_t::Forward && querystring != newstring) || (flag == BestAlignment_t::ReverseComplement && rcquerystring != newstring)){
                    std::cout << (flag == BestAlignment_t::Forward ? "fwd" : "rc") << std::endl;
                    std::cout << querystring << std::endl;
                    std::cout << rcquerystring << std::endl;
                    std::cout << newstring << std::endl;
                    assert(false);
                }
            }
        }
#endif

    }else{ // use cpu for alignment

#endif
        device = AlignmentDevice::CPU;

        timings.executionBegin();

        for(std::size_t i = 0, subjectcount = 0; i < subjectsbegin.size(); i++){

            auto queryIt = queriesbegin[i];
            auto alignmentsIt = alignmentsbegin[i];
            auto flagsIt = flagsbegin[i];
            auto bestSequenceStringsIt = bestSequenceStringsbegin[i];

            for(auto subjectIt = subjectsbegin[i]; subjectIt != subjectsend[i]; ++subjectIt, ++subjectcount){

                assert((*subjectIt)->length() <= mybuffers.max_sequence_length);
                assert((*subjectIt)->getNumBytes() <= mybuffers.max_sequence_bytes);

                const char* const subject = (const char*)(*subjectIt)->begin();
                const int subjectLength = (*subjectIt)->length();

                const int nQueries = queriesPerSubject[subjectcount];

                for(int i = 0; i < nQueries; i++){
                    const char* query =  (const char*)(*queryIt)->begin();
                    const int queryLength = (*queryIt)->length();

                    Sequence_t revcomplsequence = (*queryIt)->reverseComplement();
                    const char* const revcomplQuery = (const char*)revcomplsequence.begin();

                    auto fwdAlignment = cpu_shifted_hamming_distance<Sequence_t>(subject, subjectLength, query, queryLength,
                                                                            min_overlap,
                                                                            maxErrorRate,
                                                                            min_overlap_ratio);

                    auto revComplAlignment = cpu_shifted_hamming_distance<Sequence_t>(subject, subjectLength, revcomplQuery, queryLength,
                                                                            min_overlap,
                                                                            maxErrorRate,
                                                                            min_overlap_ratio);

                    BestAlignment_t bestAlignmentFlag = choose_best_alignment(fwdAlignment,
                                                                              revComplAlignment,
                                                                              subjectLength,
                                                                              queryLength,
                                                                              min_overlap_ratio,
                                                                              min_overlap,
                                                                              maxErrorRate);


                    *flagsIt = bestAlignmentFlag;

                    if(bestAlignmentFlag == BestAlignment_t::Forward){
                        *alignmentsIt = fwdAlignment;
                        *bestSequenceStringsIt = std::move((*queryIt)->toString());
                    }else if(bestAlignmentFlag == BestAlignment_t::ReverseComplement){
                        *alignmentsIt = revComplAlignment;
                        *bestSequenceStringsIt = std::move(revcomplsequence.toString());
                    }else
                        ; //BestAlignment_t::None

                    ++queryIt;
                    ++alignmentsIt;
                    ++flagsIt;
                    ++bestSequenceStringsIt;
                }
            }
        }

        timings.executionEnd();

#ifdef __NVCC__
    }
#endif

    return device;
}


/*
Ensures that all alignment results from the preceding call to shifted_hamming_distance_canonical_batched_async
have been stored in the range [alignmentsbegin[i], alignmentsend[i][

must be called with the same handle, alignmentsbegin, alignmentsend, flagsbegin, flagsend, canUseGpu
as the call to shifted_hamming_distance_canonical_batched_async
*/

template<class AlignmentIter, class FlagsIter, class StringIter,
    typename std::enable_if<!std::is_same<typename AlignmentIter::value_type, shd::Result_t>::value, int>::type = 0>
void shifted_hamming_distance_canonical_get_results_batched(SHDhandle& handle,
                                std::vector<AlignmentIter>& alignmentsbegin,
                                std::vector<AlignmentIter>& alignmentsend,
                                std::vector<FlagsIter>& flagsbegin,
                                std::vector<FlagsIter>& flagsend,
                                std::vector<StringIter>& bestSequenceStringsbegin,
                                std::vector<StringIter>& bestSequenceStringsend,
                                bool canUseGpu){}

template<class AlignmentIter, class FlagsIter, class StringIter,
    typename std::enable_if<std::is_same<typename AlignmentIter::value_type, shd::Result_t>::value, int*>::type = nullptr>
void shifted_hamming_distance_canonical_get_results_batched(SHDhandle& handle,
                                std::vector<AlignmentIter>& alignmentsbegin,
                                std::vector<AlignmentIter>& alignmentsend,
                                std::vector<FlagsIter>& flagsbegin,
                                std::vector<FlagsIter>& flagsend,
                                std::vector<StringIter>& bestSequenceStringsbegin,
                                std::vector<StringIter>& bestSequenceStringsend,
                                bool canUseGpu){

    //static_assert(std::is_same<typename AlignmentIter::value_type, shd::Result_t>::value, "shifted hamming distance unexpected Alignment type");
    static_assert(std::is_same<typename FlagsIter::value_type, BestAlignment_t>::value, "shifted hamming distance unexpected flag type");

    assert(alignmentsbegin.size() == alignmentsend.size());
    assert(flagsbegin.size() == flagsend.size());
    assert(bestSequenceStringsbegin.size() == bestSequenceStringsend.size());

    int numberOfAlignments = 0;
    int numberOfFlags = 0;
    int numberOfBestSequenceStrings = 0;

    for(std::size_t i = 0; i < alignmentsbegin.size(); i++){
        numberOfAlignments += std::distance(alignmentsbegin[i], alignmentsend[i]);
    }

    for(std::size_t i = 0; i < flagsbegin.size(); i++){
        numberOfFlags += std::distance(flagsbegin[i], flagsend[i]);
    }

    for(std::size_t i = 0; i < bestSequenceStringsbegin.size(); i++){
        numberOfBestSequenceStrings += std::distance(bestSequenceStringsbegin[i], bestSequenceStringsend[i]);
    }

    assert(numberOfAlignments == numberOfFlags);
    assert(numberOfAlignments == numberOfBestSequenceStrings);

    //nothing to do here
    if(numberOfAlignments == 0)
        return;
#ifdef __NVCC__

    auto& mybuffers = handle.buffers;
    auto& timings = handle.timings;

    if(canUseGpu && numberOfAlignments >= mybuffers.gpuThreshold){ // use gpu for alignment
        cudaSetDevice(mybuffers.deviceId); CUERR;

        cudaStreamSynchronize(mybuffers.streams[0]); CUERR;

        const shd::Result_t* const results = mybuffers.h_results;
        const BestAlignment_t* const bestAlignmentFlags = mybuffers.h_bestAlignmentFlags;
        const char* const bestSequenceStrings = mybuffers.h_unpacked_queries;

        timings.executionEnd();

        timings.postprocessingBegin();

        for(std::size_t i = 0, count = 0; i < alignmentsbegin.size(); ++i){

			for(auto t = std::make_tuple(alignmentsbegin[i], flagsbegin[i], bestSequenceStringsbegin[i]);
                std::get<0>(t) != alignmentsend[i]
                && std::get<1>(t) != flagsend[i]
                && std::get<2>(t) != bestSequenceStringsend[i];
                ++std::get<0>(t), ++std::get<1>(t), ++std::get<2>(t), ++count){

                auto rit = std::get<0>(t);
                auto fit = std::get<1>(t);
                auto bit = std::get<2>(t);

                *rit = results[count];
                *fit = bestAlignmentFlags[count];
                *bit = std::move(std::string{bestSequenceStrings + count * mybuffers.max_sequence_length,
                                            bestSequenceStrings + count * mybuffers.max_sequence_length + mybuffers.h_querylengths[count]});
            }

		}

        timings.postprocessingEnd();


    }else{ // cpu already done

#endif

#ifdef __NVCC__
    }
#endif

    return;
}

template<class Sequence_t, class SubjectIter, class QueryIter, class AlignmentIter, class FlagsIter,
    typename std::enable_if<!std::is_same<typename AlignmentIter::value_type, shd::Result_t>::value, int>::type = 0>
AlignmentDevice shifted_hamming_distance_canonical_batched(SHDhandle& handle,
                                const std::vector<SubjectIter>& subjectsbegin,
                                const std::vector<SubjectIter>& subjectsend,
                                const std::vector<QueryIter>& queriesbegin,
                                const std::vector<QueryIter>& queriesend,
                                std::vector<AlignmentIter>& alignmentsbegin,
                                std::vector<AlignmentIter>& alignmentsend,
                                std::vector<FlagsIter>& flagsbegin,
                                std::vector<FlagsIter>& flagsend,
                                const std::vector<int>& queriesPerSubject,
                                int min_overlap,
                                double maxErrorRate,
                                double min_overlap_ratio,
                                bool canUseGpu){
    return AlignmentDevice::None;
}

template<class Sequence_t, class SubjectIter, class QueryIter, class AlignmentIter, class FlagsIter,
    typename std::enable_if<std::is_same<typename AlignmentIter::value_type, shd::Result_t>::value, int*>::type = nullptr>
AlignmentDevice shifted_hamming_distance_canonical_batched(SHDhandle& handle,
                                const std::vector<SubjectIter>& subjectsbegin,
                                const std::vector<SubjectIter>& subjectsend,
                                const std::vector<QueryIter>& queriesbegin,
                                const std::vector<QueryIter>& queriesend,
                                std::vector<AlignmentIter>& alignmentsbegin,
                                std::vector<AlignmentIter>& alignmentsend,
                                std::vector<FlagsIter>& flagsbegin,
                                std::vector<FlagsIter>& flagsend,
                                const std::vector<int>& queriesPerSubject,
                                int min_overlap,
                                double maxErrorRate,
                                double min_overlap_ratio,
                                bool canUseGpu){

    AlignmentDevice device = shifted_hamming_distance_canonical_batched_async(handle, subjectsbegin, subjectsend,
                                    queriesbegin, queriesend,
                                    alignmentsbegin, alignmentsend,
                                    flagsbegin, flagsend,
                                    queriesPerSubject, min_overlap,
                                    maxErrorRate, min_overlap_ratio, canUseGpu);

    shifted_hamming_distance_canonical_get_results_batched(handle,
                                    alignmentsbegin,
                                    alignmentsend,
                                    flagsbegin,
                                    flagsend,
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
                                    int min_overlap,
                                    double maxErrorRate,
                                    double min_overlap_ratio,
                                    int score_match,
                                    int score_sub,
                                    int score_ins,
                                    int score_del){

    auto accessor = [] (const char* data, int length, int index){
        return Sequence_t::get(data, length, index);
    };
#if 1
    return sga::cpu_semi_global_alignment(subject, subjectlength, query, querylength,
                        score_match, score_sub, score_ins, score_del, accessor);
#else
return sga::cpu_semi_global_alignment_banded(subject, subjectlength, query, querylength,
                    min_overlap, maxErrorRate, min_overlap_ratio,
                    score_match, score_sub, score_ins, score_del, accessor);
#endif
}





/*
    SEMI GLOBAL ALIGNMENT CANONICAL batched
*/

#ifdef __NVCC__


template<class Sequence_t>
void call_sga_canonical_kernel_async(const sga::SGAdata& sgadata,
                      int min_overlap,
                      double maxErrorRate,
                      double min_overlap_ratio,
                      int maxSubjectLength,
                      int maxQueryLength,
                      int score_match,
                      int score_sub,
                      int score_ins,
                      int score_del){

    auto accessor = [] __device__ (const char* data, int length, int index){
        return Sequence_t::get(data, length, index);
    };

    auto make_reverse_complement = [] __device__ (std::uint8_t* reverseComplement, const std::uint8_t* sequence, int sequencelength){
        return Sequence_t::make_reverse_complement(reverseComplement, sequence, sequencelength);
    };

    auto nucleotide_accessor = [] __device__ (const char* data, int length, int index){
        return Sequence_t::get_as_nucleotide(data, length, index);
    };

    auto make_unpacked_reverse_complement_inplace = [] __device__ (std::uint8_t* sequence, int sequencelength){
        return SequenceString::make_reverse_complement_inplace(sequence, sequencelength);
    };

    auto comp = [=] __device__ (const sga::Attributes_t& fwdAlignment,
                               const sga::Attributes_t& revcmplAlignment,
                               int subjectlength,
                               int querylength) -> BestAlignment_t{
        return choose_best_alignment(fwdAlignment,
                              revcmplAlignment,
                              subjectlength,
                              querylength,
                              min_overlap_ratio,
                              min_overlap,
                              maxErrorRate);
    };

    sga::call_semi_global_alignment_with_revcompl_kernel_async(sgadata, score_match, score_sub, score_ins, score_del, maxSubjectLength, maxQueryLength, accessor, make_reverse_complement);

    /*call_cuda_find_best_alignment_kernel_async(sgadata.d_results,
                              sgadata.d_bestAlignmentFlags,
                              sgadata.d_subjectlengths,
                              sgadata.d_querylengths,
                              sgadata.d_NqueriesPrefixSum,
                              sgadata.n_subjects,
                              comp,
                              sgadata.n_queries,
                              sgadata.streams[0]);*/

    call_cuda_find_best_alignment_and_unpack_good_sequences_kernel_async(
                                      sgadata.d_results,
                                      sgadata.d_bestAlignmentFlags,
                                      sgadata.d_unpacked_queries,
                                      sgadata.d_subjectlengths,
                                      sgadata.d_querylengths,
                                      sgadata.d_NqueriesPrefixSum,
                                      sgadata.n_subjects,
                                      sgadata.n_queries,
                                      sgadata.d_queriesdata,
                                      sgadata.max_sequence_length,
                                      sgadata.sequencepitch,
                                      comp,
                                      nucleotide_accessor,
                                      make_unpacked_reverse_complement_inplace,
                                      sgadata.streams[0]);
}

#endif

/*
    Calculates both subject/query alignment and subject/reversecomplementquery alignment
    and keeps only the better one according to choose_best_alignment(...).
    Flags indicate whether the better one is forward alignment, reverse complement alignment, or if both alignments are bad.
*/

template<class Sequence_t, class SubjectIter, class QueryIter, class AlignmentIter, class FlagsIter, class StringIter,
        typename std::enable_if<!std::is_same<typename AlignmentIter::value_type, sga::Result_t>::value, int>::type = 0>
AlignmentDevice semi_global_alignment_canonical_batched_async(SGAhandle& handle,
                                const std::vector<SubjectIter>& subjectsbegin,
                                const std::vector<SubjectIter>& subjectsend,
                                const std::vector<QueryIter>& queriesbegin,
                                const std::vector<QueryIter>& queriesend,
                                std::vector<AlignmentIter>& alignmentsbegin,
                                std::vector<AlignmentIter>& alignmentsend,
                                std::vector<FlagsIter>& flagsbegin,
                                std::vector<FlagsIter>& flagsend,
                                std::vector<StringIter>& bestSequenceStringsbegin,
                                std::vector<StringIter>& bestSequenceStringsend,
                                const std::vector<int>& queriesPerSubject,
                                int min_overlap,
                                double maxErrorRate,
                                double min_overlap_ratio,
                                int score_match,
                                int score_sub,
                                int score_ins,
                                int score_del,
                                bool canUseGpu){

    return AlignmentDevice::None;
}

template<class Sequence_t, class SubjectIter, class QueryIter, class AlignmentIter, class FlagsIter, class StringIter,
        typename std::enable_if<std::is_same<typename AlignmentIter::value_type, sga::Result_t>::value, int*>::type = nullptr>
AlignmentDevice semi_global_alignment_canonical_batched_async(SGAhandle& handle,
                                const std::vector<SubjectIter>& subjectsbegin,
                                const std::vector<SubjectIter>& subjectsend,
                                const std::vector<QueryIter>& queriesbegin,
                                const std::vector<QueryIter>& queriesend,
                                std::vector<AlignmentIter>& alignmentsbegin,
                                std::vector<AlignmentIter>& alignmentsend,
                                std::vector<FlagsIter>& flagsbegin,
                                std::vector<FlagsIter>& flagsend,
                                std::vector<StringIter>& bestSequenceStringsbegin,
                                std::vector<StringIter>& bestSequenceStringsend,
                                const std::vector<int>& queriesPerSubject,
                                int min_overlap,
                                double maxErrorRate,
                                double min_overlap_ratio,
                                int score_match,
                                int score_sub,
                                int score_ins,
                                int score_del,
                                bool canUseGpu){

    //static_assert(std::is_same<typename AlignmentIter::value_type, sga::Result_t>::value, "semi global alignment unexpected Alignment type");

    auto& mybuffers = handle.buffers;
    auto& timings = handle.timings;

    AlignmentDevice device = AlignmentDevice::None;

    int numberOfFlags = 0;
    int numberOfAlignments = 0;
    int numberOfQueries = 0;
    int numberOfSubjects = 0;

    assert(subjectsbegin.size() == subjectsend.size());
    assert(subjectsbegin.size() == queriesbegin.size());
    assert(subjectsbegin.size() == queriesend.size());
    assert(subjectsbegin.size() == alignmentsbegin.size());
    assert(subjectsbegin.size() == alignmentsend.size());
    assert(subjectsbegin.size() == flagsbegin.size());
    assert(subjectsbegin.size() == flagsend.size());
    assert(subjectsbegin.size() == bestSequenceStringsbegin.size());
    assert(subjectsbegin.size() == bestSequenceStringsend.size());

    for(std::size_t i = 0; i < flagsbegin.size(); i++){
        numberOfFlags += std::distance(flagsbegin[i], flagsend[i]);
    }
    for(std::size_t i = 0; i < alignmentsbegin.size(); i++){
        numberOfAlignments += std::distance(alignmentsbegin[i], alignmentsend[i]);
    }
    for(std::size_t i = 0; i < queriesbegin.size(); i++){
        numberOfQueries += std::distance(queriesbegin[i], queriesend[i]);
    }
    assert(numberOfAlignments == numberOfQueries);
    assert(numberOfAlignments == numberOfFlags);

    for(std::size_t i = 0; i < subjectsbegin.size(); i++){
        numberOfSubjects += std::distance(subjectsbegin[i], subjectsend[i]);
    }
    assert(numberOfSubjects == (int)queriesPerSubject.size());

    //nothing to do here
    if(numberOfAlignments == 0)
        return device;

#ifdef __NVCC__

    if(canUseGpu && numberOfAlignments >= mybuffers.gpuThreshold){ // use gpu for alignment
        device = AlignmentDevice::GPU;

        timings.preprocessingBegin();

        cudaSetDevice(mybuffers.deviceId); CUERR;

        mybuffers.resize(numberOfSubjects, numberOfQueries, 2*numberOfAlignments);

        int maxSubjectLength = 0;

        for(std::size_t i = 0, count = 0; i < subjectsbegin.size(); i++){

            for(auto it = subjectsbegin[i]; it != subjectsend[i]; ++it, ++count){

                assert((*it)->length() <= mybuffers.max_sequence_length);
                assert((*it)->getNumBytes() <= mybuffers.max_sequence_bytes);

                std::memcpy(mybuffers.h_subjectsdata + count * mybuffers.sequencepitch,
                            (*it)->begin(),
                            (*it)->getNumBytes());

                mybuffers.h_subjectlengths[count] = (*it)->length();
                maxSubjectLength = std::max(int((*it)->length()), maxSubjectLength);
            }

        }

        int maxQueryLength = 0;

        for(std::size_t i = 0, count = 0; i < queriesbegin.size(); i++){

            for(auto it = queriesbegin[i]; it != queriesend[i]; ++it, ++count){

                assert((*it)->length() <= mybuffers.max_sequence_length);
                assert((*it)->getNumBytes() <= mybuffers.max_sequence_bytes);

                std::memcpy(mybuffers.h_queriesdata + count * mybuffers.sequencepitch,
                            (*it)->begin(),
                            (*it)->getNumBytes());

                mybuffers.h_querylengths[count] = (*it)->length();
                maxQueryLength = std::max(int((*it)->length()), maxQueryLength);
            }

        }

        mybuffers.h_NqueriesPrefixSum[0] = 0;
        for(std::size_t i = 0; i < queriesPerSubject.size(); i++)
            mybuffers.h_NqueriesPrefixSum[i+1] = mybuffers.h_NqueriesPrefixSum[i] + queriesPerSubject[i];

        assert(numberOfAlignments == mybuffers.h_NqueriesPrefixSum[queriesPerSubject.size()]);

        timings.preprocessingEnd();

        timings.executionBegin();


        // copy data to gpu

        cudaMemcpyAsync(mybuffers.deviceptr,
                        mybuffers.hostptr,
                        mybuffers.transfersizeH2D,
                        H2D,
                        mybuffers.streams[0]); CUERR;

        call_sga_canonical_kernel_async<Sequence_t>(mybuffers,
                                            min_overlap,
                                            maxErrorRate,
                                            min_overlap_ratio,
                                            maxSubjectLength,
                                            maxQueryLength,
                                            score_match,
                                            score_sub,
                                            score_ins,
                                            score_del); CUERR;

        //copy results to host

        cudaMemcpyAsync(mybuffers.h_results,
            mybuffers.d_results,
            mybuffers.transfersizeD2H,
            D2H,
            mybuffers.streams[0]); CUERR;

    }else{ // use cpu for alignment

#endif
        device = AlignmentDevice::CPU;

        timings.executionBegin();

        for(std::size_t i = 0, subjectcount = 0; i < subjectsbegin.size(); i++){

            auto queryIt = queriesbegin[i];
            auto alignmentsIt = alignmentsbegin[i];
            auto flagsIt = flagsbegin[i];
            auto bestSequenceStringsIt = bestSequenceStringsbegin[i];

            for(auto subjectIt = subjectsbegin[i]; subjectIt != subjectsend[i]; ++subjectIt, ++subjectcount){

                assert((*subjectIt)->length() <= mybuffers.max_sequence_length);
                assert((*subjectIt)->getNumBytes() <= mybuffers.max_sequence_bytes);

                const char* const subject = (const char*)(*subjectIt)->begin();
                const int subjectLength = (*subjectIt)->length();

                const int nQueries = queriesPerSubject[subjectcount];

                for(int i = 0; i < nQueries; i++){
                    const char* query =  (const char*)(*queryIt)->begin();
                    const int queryLength = (*queryIt)->length();

                    Sequence_t revcomplsequence = (*queryIt)->reverseComplement();
                    const char* const revcomplQuery = (const char*)revcomplsequence.begin();

                    auto fwdAlignment = cpu_semi_global_alignment<Sequence_t>(subject,
                                                        subjectLength,
                                                        query,
                                                        queryLength,
                                                        min_overlap,
                                                        maxErrorRate,
                                                        min_overlap_ratio,
                                                        score_match,
                                                        score_sub,
                                                        score_ins,
                                                        score_del);

                    auto revComplAlignment = cpu_semi_global_alignment<Sequence_t>(subject,
                                                        subjectLength,
                                                        revcomplQuery,
                                                        queryLength,
                                                        min_overlap,
                                                        maxErrorRate,
                                                        min_overlap_ratio,
                                                        score_match,
                                                        score_sub,
                                                        score_ins,
                                                        score_del);

                    BestAlignment_t bestAlignmentFlag = choose_best_alignment(fwdAlignment,
                                                                              revComplAlignment,
                                                                              subjectLength,
                                                                              queryLength,
                                                                              min_overlap_ratio,
                                                                              min_overlap,
                                                                              maxErrorRate);


                    *flagsIt = bestAlignmentFlag;

                    if(bestAlignmentFlag == BestAlignment_t::Forward){
                        *alignmentsIt = fwdAlignment;
                        *bestSequenceStringsIt = std::move((*queryIt)->toString());
                    }else if(bestAlignmentFlag == BestAlignment_t::ReverseComplement){
                        *alignmentsIt = revComplAlignment;
                        *bestSequenceStringsIt = std::move(revcomplsequence.toString());
                    }else
                        ; //BestAlignment_t::None

                    ++queryIt;
                    ++alignmentsIt;
                    ++flagsIt;
                    ++bestSequenceStringsIt;
                }
            }
        }

        timings.executionEnd();

#ifdef __NVCC__
    }
#endif

    return device;
}


/*
Ensures that all alignment results from the preceding call to semi_global_alignment_canonical_batched_async
have been stored in the range [alignmentsbegin[i], alignmentsend[i][

must be called with the same handle, alignmentsbegin, alignmentsend, flagsbegin, flagsend, canUseGpu
as the call to semi_global_alignment_canonical_batched_async
*/
template<class AlignmentIter, class FlagsIter, class StringIter,
    typename std::enable_if<!std::is_same<typename AlignmentIter::value_type, sga::Result_t>::value, int>::type = 0>
void semi_global_alignment_canonical_get_results_batched(SGAhandle& handle,
                                    std::vector<AlignmentIter>& alignmentsbegin,
                                    std::vector<AlignmentIter>& alignmentsend,
                                    std::vector<FlagsIter>& flagsbegin,
                                    std::vector<FlagsIter>& flagsend,
                                    std::vector<StringIter>& bestSequenceStringsbegin,
                                    std::vector<StringIter>& bestSequenceStringsend,
                                    bool canUseGpu){

}

template<class AlignmentIter, class FlagsIter, class StringIter,
    typename std::enable_if<std::is_same<typename AlignmentIter::value_type, sga::Result_t>::value, int*>::type = nullptr>
void semi_global_alignment_canonical_get_results_batched(SGAhandle& handle,
                                std::vector<AlignmentIter>& alignmentsbegin,
                                std::vector<AlignmentIter>& alignmentsend,
                                std::vector<FlagsIter>& flagsbegin,
                                std::vector<FlagsIter>& flagsend,
                                std::vector<StringIter>& bestSequenceStringsbegin,
                                std::vector<StringIter>& bestSequenceStringsend,
                                bool canUseGpu){

    //static_assert(std::is_same<typename AlignmentIter::value_type, sga::Result_t>::value, "semi global alignment unexpected Alignment type");
    static_assert(std::is_same<typename FlagsIter::value_type, BestAlignment_t>::value, "semi global alignment unexpected flag type");

    assert(alignmentsbegin.size() == alignmentsend.size());
    assert(flagsbegin.size() == flagsend.size());
    assert(bestSequenceStringsbegin.size() == bestSequenceStringsend.size());

    int numberOfAlignments = 0;
    int numberOfFlags = 0;
    int numberOfBestSequenceStrings = 0;

    for(std::size_t i = 0; i < alignmentsbegin.size(); i++){
        numberOfAlignments += std::distance(alignmentsbegin[i], alignmentsend[i]);
    }

    for(std::size_t i = 0; i < flagsbegin.size(); i++){
        numberOfFlags += std::distance(flagsbegin[i], flagsend[i]);
    }

    for(std::size_t i = 0; i < bestSequenceStringsbegin.size(); i++){
        numberOfBestSequenceStrings += std::distance(bestSequenceStringsbegin[i], bestSequenceStringsend[i]);
    }

    assert(numberOfAlignments == numberOfFlags);
    assert(numberOfAlignments == numberOfBestSequenceStrings);

    //nothing to do here
    if(numberOfAlignments == 0)
        return;
#ifdef __NVCC__

    auto& mybuffers = handle.buffers;
    auto& timings = handle.timings;

    if(canUseGpu && numberOfAlignments >= mybuffers.gpuThreshold){ // use gpu for alignment
        cudaSetDevice(mybuffers.deviceId); CUERR;

        cudaStreamSynchronize(mybuffers.streams[0]); CUERR;

        const sga::Attributes_t* results = mybuffers.h_results;
        const BestAlignment_t* bestAlignmentFlags = mybuffers.h_bestAlignmentFlags;
        const sga::Op_t* ops = mybuffers.h_ops;
        const char* const bestSequenceStrings = mybuffers.h_unpacked_queries;

        timings.executionEnd();

        timings.postprocessingBegin();

        for(std::size_t i = 0, count = 0; i < alignmentsbegin.size(); ++i){

            for(auto t = std::make_tuple(alignmentsbegin[i], flagsbegin[i], bestSequenceStringsbegin[i]);
                std::get<0>(t) != alignmentsend[i]
                && std::get<1>(t) != flagsend[i]
                && std::get<2>(t) != bestSequenceStringsend[i];
                ++std::get<0>(t), ++std::get<1>(t), ++std::get<2>(t), ++count){

                auto rit = std::get<0>(t);
                auto fit = std::get<1>(t);
                auto bit = std::get<2>(t);

                rit->attributes = results[count];

                *bit = std::move(std::string{bestSequenceStrings + count * mybuffers.max_sequence_length,
                                            bestSequenceStrings + count * mybuffers.max_sequence_length + mybuffers.h_querylengths[count]});

                *fit = bestAlignmentFlags[count];

                auto& opvector = rit->operations;
                const int nOps = rit->get_nOps();
                opvector.resize(nOps);

                if(*fit == BestAlignment_t::Forward)
                    std::reverse_copy(ops + count * mybuffers.max_ops_per_alignment,
                          ops + count * mybuffers.max_ops_per_alignment + nOps,
                          opvector.begin());
                else if(*fit == BestAlignment_t::ReverseComplement)
                    //when determining the best alignment, the alignment operations are not modified.
                    //Thus we need to make sure to copy the correct operations for reverse complements
                    std::reverse_copy(ops + (count + numberOfAlignments) * mybuffers.max_ops_per_alignment,
                          ops + (count + numberOfAlignments) * mybuffers.max_ops_per_alignment + nOps,
                          opvector.begin());
                else
                    ; // BestAlignment_t::None, do nothing


            }

		}

        timings.postprocessingEnd();


    }else{ // cpu already done

#endif

#ifdef __NVCC__
    }
#endif

    return;
}


}

#endif
