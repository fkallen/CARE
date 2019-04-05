#ifndef CARE_CPU_CORRECTION_CORE
#define CARE_CPU_CORRECTION_CORE

#include "bestalignment.hpp"
#include "cpu_alignment.hpp"

#include <config.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <vector>

namespace care{
namespace cpu{


/*
    subjectptr / candidateptrs point to (possibly encoded) sequences with a
    (decoded) length of subjectLength / candidateLengths nucleotides

    Returns the alignments of each candidate to the subject.
*/
template<class Sequence_t>
std::vector<SHDResult>
calculate_shd_alignments(const char* subjectptr,
                    const int subjectLength,
                    const std::vector<char*>& candidateptrs,
                    const std::vector<int>& candidateLengths,
                    const int min_overlap,
                    const float maxErrorRate,
                    const float min_overlap_ratio){

    std::vector<SHDResult> results;
    results.reserve(candidateptrs.size());

    for(size_t i = 0; i < candidateptrs.size(); i++){
        const char* candidateptr = candidateptrs[i];
        const int candidateLength = candidateLengths[i];

        const SHDResult alignmentresult = CPUShiftedHammingDistanceChooser<Sequence_t>
                                                        ::cpu_shifted_hamming_distance(
                                                            subjectptr,
                                                            subjectLength,
                                                            candidateptr,
                                                            candidateLength,
                                                            min_overlap,
                                                            maxErrorRate,
                                                            min_overlap_ratio);

        results.emplace_back(alignmentresult);
    }

    return results;

}









/*
    alignmentresults contains alignments results of candidates aligned to the subject
    The first half of alignmentresults contains the forward candidate alignments.
    The second half of alignmentresults contains the reverse complement candidate alignments.

    returns a list of BestAlignment_t to indicate whether the forward alignment,
    reverse complement alignment or neither alignment is better.

    alignmentresults.size() == n
    candidateLengths.size() == n/2
    returnvalue.size() == n/2
*/
template<int dummy=0>
std::vector<BestAlignment_t> findBestAlignmentDirection(const std::vector<SHDResult>& alignmentresults,
                                    const int subjectLength,
                                    const std::vector<int>& candidateLengths,
                                    const int min_overlap,
                                    const float maxErrorRate,
                                    const float min_overlap_ratio){

    assert(alignmentresults.size() % 2 == 0);
    assert(alignmentresults.size() / 2 == candidateLengths.size());

    const size_t half_size = alignmentresults.size() / 2;

    std::vector<BestAlignment_t> result;
    result.reserve(half_size);

    for(size_t i = 0; i < half_size; i++){
        const SHDResult& forwardAlignment = alignmentresults[i];
        const SHDResult& revcAlignment = alignmentresults[i + half_size];
        const int candidateLength = candidateLengths[i];

        BestAlignment_t bestAlignmentFlag = care::cpu::choose_best_alignment(forwardAlignment,
                                                                          revcAlignment,
                                                                          subjectLength,
                                                                          candidateLength,
                                                                          min_overlap_ratio,
                                                                          min_overlap,
                                                                          maxErrorRate);

        result.emplace_back(bestAlignmentFlag);
    }

    return result;
}

template<int dummy=0>
std::vector<BestAlignment_t> findBestAlignmentDirection(const std::vector<SHDResult>& forwardAlignments,
                                    const std::vector<SHDResult>& revcAlignments,
                                    const int subjectLength,
                                    const std::vector<int>& candidateLengths,
                                    const int min_overlap,
                                    const float maxErrorRate,
                                    const float min_overlap_ratio){

    assert(forwardAlignments.size() == revcAlignments.size());
    assert(forwardAlignments.size() == candidateLengths.size());

    std::vector<BestAlignment_t> result;
    result.reserve(forwardAlignments.size());

    for(size_t i = 0; i < forwardAlignments.size(); i++){
        const SHDResult& forwardAlignment = forwardAlignments[i];
        const SHDResult& revcAlignment = revcAlignments[i];
        const int candidateLength = candidateLengths[i];

        BestAlignment_t bestAlignmentFlag = care::cpu::choose_best_alignment(forwardAlignment,
                                                                          revcAlignment,
                                                                          subjectLength,
                                                                          candidateLength,
                                                                          min_overlap_ratio,
                                                                          min_overlap,
                                                                          maxErrorRate);

        result.emplace_back(bestAlignmentFlag);
    }

    return result;
}

/*
    Filters alignments by good mismatch ratio.

    Returns an sorted index list to alignments which pass the filter.
*/

template<class Func>
std::vector<int>
filterAlignmentsByMismatchRatio(const std::vector<SHDResult>& alignments,
                                    const float estimatedErrorrate,
                                    const int estimatedCoverage,
                                    const float m_coverage,
                                    Func lastResortFunc){

    const float mismatchratioBaseFactor = estimatedErrorrate * 1.0f;
    const float goodAlignmentsCountThreshold = estimatedCoverage * m_coverage;

    std::array<int, 3> counts({0,0,0});

    for(const auto& alignment : alignments){
        const float mismatchratio = float(alignment.nOps) / float(alignment.overlap);

        if (mismatchratio < 2 * mismatchratioBaseFactor) {
            counts[0] += 1;
        }
        if (mismatchratio < 3 * mismatchratioBaseFactor) {
            counts[1] += 1;
        }
        if (mismatchratio < 4 * mismatchratioBaseFactor) {
            counts[2] += 1;
        }
    }

    //no correction possible without enough candidates
    if(!std::any_of(counts.begin(), counts.end(), [](auto c){return c > 0;})){
        return {};
    }

    //std::cerr << "Read " << task.readId << ", good alignments after bining: " << std::accumulate(counts.begin(), counts.end(), int(0)) << '\n';
    //std::cerr << "Read " << task.readId << ", bins: " << counts[0] << " " << counts[1] << " " << counts[2] << '\n';


    float mismatchratioThreshold = 0;
    if (counts[0] >= goodAlignmentsCountThreshold) {
        mismatchratioThreshold = 2 * mismatchratioBaseFactor;
    } else if (counts[1] >= goodAlignmentsCountThreshold) {
        mismatchratioThreshold = 3 * mismatchratioBaseFactor;
    } else if (counts[2] >= goodAlignmentsCountThreshold) {
        mismatchratioThreshold = 4 * mismatchratioBaseFactor;
    } else {
        if(lastResortFunc()){
            mismatchratioThreshold = 4 * mismatchratioBaseFactor;
        }else{
            return {}; //no correction possible without good candidates
        }
    }

    std::vector<int> result;

    result.reserve(alignments.size());

    for(int i = 0; i < int(alignments.size()); i++){
        auto& alignment = alignments[i];
        const float mismatchratio = float(alignment.nOps) / float(alignment.overlap);
        const bool notremoved = mismatchratio < mismatchratioThreshold;

        if(notremoved){
            result.emplace_back(i);
        }
    }

    return result;
}




}
}

#endif
