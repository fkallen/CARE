#ifndef CARE_BATCH_ELEM_HPP
#define CARE_BATCH_ELEM_HPP

#include "alignment.hpp"
#include "tasktiming.hpp"
#include "options.hpp"

#include <cassert>
#include <vector>
#include <string>

namespace care{

    enum class BestAlignment_t {Forward, ReverseComplement, None};

struct DetermineGoodAlignmentStats{
    int correctionCases[4]{0,0,0,0};
    int uniqueCandidatesWithoutGoodAlignment=0;
};


template<class readStorage_t,
		 class alignment_result_t>
struct BatchElem{

	using ReadStorage_t = readStorage_t;
	using Sequence_t = typename ReadStorage_t::Sequence_t;
	using ReadId_t = typename ReadStorage_t::ReadId_t;
    using AlignmentResult_t = alignment_result_t;

	static constexpr bool canUseQualityScores = ReadStorage_t::hasQualityScores;

    bool active;

    //query properties
    bool corrected;
    ReadId_t readId;
    const Sequence_t* fwdSequence;
	std::string fwdSequenceString;
    const std::string* fwdQuality;
    std::string correctedSequence;

	std::uint64_t n_unique_candidates;
	std::uint64_t n_candidates;

    //candidates properties
    std::vector<ReadId_t> candidateIds;
    std::vector<int> candidateCountsPrefixSum;
    std::vector<int> candidateCounts;

    std::vector<bool> activeCandidates;
    std::vector<const Sequence_t*> fwdSequences;
    std::vector<const Sequence_t*> revcomplSequences;
    std::vector<AlignmentResult_t> fwdAlignments;
    std::vector<AlignmentResult_t> revcomplAlignments;

    std::vector<AlignmentResult_t*> bestAlignments;
    std::vector<const Sequence_t*> bestSequences;
	std::vector<std::string> bestSequenceStrings;
	std::vector<const std::string*> bestQualities;
    std::vector<bool> bestIsForward;

    double mismatchratioThreshold;

    const ReadStorage_t* readStorage;

    double goodAlignmentsCountThreshold;
    double mismatchratioBaseFactor;
    CorrectionOptions correctionOptions;

    TaskTimings findCandidatesTiming;
    std::vector<std::pair<ReadId_t, const Sequence_t*>> make_unique_sequences_numseqpairs;

    int counts[3] { 0, 0, 0 }; //count number of cases of mismatchratio < 2*errorrate, 3*errorrate, 4*errorrate

    BatchElem(){}

    BatchElem(const ReadStorage_t& rs,
                const CorrectionOptions& CO)
                :   readStorage(&rs),
                goodAlignmentsCountThreshold(CO.estimatedCoverage * CO.m_coverage),
                mismatchratioBaseFactor(CO.estimatedErrorrate*1.0),
                correctionOptions(CO){


    }
};


template<class BE>
std::uint64_t get_number_of_duplicate_sequences(const BE& b){
    return b.candidateIds.size() - b.fwdSequences.size();
}

template<class BE>
void clear(BE& b){
    b.active = false;
    b.corrected = false;
    b.fwdQuality = nullptr;
    b.fwdSequence = nullptr;
    b.candidateIds.clear();
    b.candidateCounts.clear();
    b.candidateCountsPrefixSum.clear();
    b.fwdSequences.clear();
    b.activeCandidates.clear();
    b.revcomplSequences.clear();
    b.fwdAlignments.clear();
    b.revcomplAlignments.clear();
    b.bestAlignments.clear();
    b.bestSequences.clear();
    b.bestSequenceStrings.clear();
    b.bestIsForward.clear();
    b.bestQualities.clear();

    b.counts[0] = 0;
    b.counts[1] = 0;
    b.counts[2] = 0;
}

template<class BE>
void set_number_of_sequences(BE& b, std::uint64_t num){
    b.bestQualities.resize(num, nullptr);
}

template<class BE>
void set_number_of_unique_sequences(BE& b, std::uint64_t num){
    b.candidateCounts.resize(num);
    b.candidateCountsPrefixSum.resize(num+1);
    b.activeCandidates.resize(num, false);
    b.fwdSequences.resize(num);
    b.revcomplSequences.resize(num);
    b.fwdAlignments.resize(num);
    b.revcomplAlignments.resize(num);
    b.bestAlignments.resize(num);
    b.bestSequences.resize(num);
    b.bestSequenceStrings.resize(num);
    b.bestIsForward.resize(num);
}

template<class BE>
void set_read_id(BE& b, typename BE::ReadId_t id){
    clear(b);
    b.readId = id;
    b.corrected = false;
    b.active = true;
}

template<class BE, class Func>
void findCandidates(BE& b, Func get_candidates){
    //get data of sequence which should be corrected
    fetch_query_data_from_readstorage(b);

    b.findCandidatesTiming.preprocessingBegin();

    b.candidateIds = get_candidates(b.fwdSequenceString);

    //remove self from candidates
    b.candidateIds.erase(std::find(b.candidateIds.begin(), b.candidateIds.end(), b.readId));

    set_number_of_sequences(b, b.candidateIds.size());
    set_number_of_unique_sequences(b, b.candidateIds.size());

    for(std::size_t k = 0; k < b.activeCandidates.size(); k++)
        b.activeCandidates[k] = false;

    b.findCandidatesTiming.preprocessingEnd();
    if(b.candidateIds.size() == 0){
        //no need for further processing without candidates
        b.active = false;
    }else{
        b.findCandidatesTiming.executionBegin();
        //find unique candidate sequences
        make_unique_sequences(b);
        b.findCandidatesTiming.executionEnd();

        b.findCandidatesTiming.postprocessingBegin();
        //get reverse complements of unique candidate sequences
        fetch_revcompl_sequences_from_readstorage(b);
        b.findCandidatesTiming.postprocessingEnd();
    }
}

template<class BE>
void fetch_query_data_from_readstorage(BE& b){
    b.fwdSequence = b.readStorage->fetchSequence_ptr(b.readId);
    if(BE::canUseQualityScores){
        b.fwdQuality = b.readStorage->fetchQuality_ptr(b.readId);
    }
    b.fwdSequenceString = b.fwdSequence->toString();
}

template<class BE>
void make_unique_sequences(BE& b){
    //vector
    auto& numseqpairs = b.make_unique_sequences_numseqpairs;
    numseqpairs.clear();
    numseqpairs.reserve(b.candidateIds.size()*1.3);

    //std::chrono::time_point < std::chrono::system_clock > t1 =
    //        std::chrono::system_clock::now();
    for(const auto& id : b.candidateIds){
        numseqpairs.emplace_back(id, b.readStorage->fetchSequence_ptr(id));
    }
    //std::chrono::time_point < std::chrono::system_clock > t2 =
    //        std::chrono::system_clock::now();
    //mapminhashresultsfetch += (t2 - t1);

    //t1 = std::chrono::system_clock::now();

    //sort pairs by sequence
    b.findCandidatesTiming.h2dBegin();
    std::sort(numseqpairs.begin(), numseqpairs.end(), [](const auto& l, const auto& r){return l.second < r.second;});
    b.findCandidatesTiming.h2dEnd();

    b.findCandidatesTiming.d2hBegin();
    std::uint64_t n_unique_elements = 1;
    b.candidateCounts[0] = 1;
    b.candidateCountsPrefixSum[0] = 0;
    b.candidateCountsPrefixSum[1] = 1;
    b.candidateIds[0] = numseqpairs[0].first;
    b.fwdSequences[0] = numseqpairs[0].second;

    using Sequence_t = typename BE::Sequence_t;
    const Sequence_t* prevSeq = numseqpairs[0].second;

    for (std::size_t k = 1; k < numseqpairs.size(); k++) {
        const auto& pair = numseqpairs[k];
        b.candidateIds[k] = pair.first;
        const Sequence_t* curSeq = pair.second;
        if (prevSeq == curSeq) {
            b.candidateCounts[n_unique_elements-1]++;
            b.candidateCountsPrefixSum[n_unique_elements]++;
        }else {
            b.candidateCounts[n_unique_elements]++;
            b.candidateCountsPrefixSum[n_unique_elements+1] = 1 + b.candidateCountsPrefixSum[n_unique_elements];
            b.fwdSequences[n_unique_elements] = curSeq;
            n_unique_elements++;
        }
        prevSeq = curSeq;
    }
    b.findCandidatesTiming.d2hEnd();

    set_number_of_unique_sequences(b, n_unique_elements);
    b.n_unique_candidates = b.fwdSequences.size();
    b.n_candidates = b.candidateIds.size();

    assert(b.candidateCountsPrefixSum.back() == int(b.candidateIds.size()));
}

template<class BE>
void fetch_revcompl_sequences_from_readstorage(BE& b){
    for(std::size_t k = 0; k < b.fwdSequences.size(); k++){
        int first = b.candidateCountsPrefixSum[k];
        b.revcomplSequences[k] = b.readStorage->fetchReverseComplementSequence_ptr(b.candidateIds[first]);
    }
}

template<class BE, class Func>
void determine_good_alignments(BE& b, Func get_best_alignment){
    determine_good_alignments(b, 0, b.fwdSequences.size(), get_best_alignment);
}

/*
    Processes at most N alignments from
    alignments[firstIndex]
    to
    alignments[min(firstIndex + N-1, number of alignments - 1)]
*/
template<class BE, class Func>
void determine_good_alignments(BE& b, int firstIndex, int N, Func get_best_alignment){

    using ReadId_t = typename BE::ReadId_t;

    const int querylength = b.fwdSequence->length();
    const int lastIndex_excl = std::min(std::size_t(firstIndex + N), b.fwdSequences.size());

    for(int i = firstIndex; i < lastIndex_excl; i++){
        const auto& res = b.fwdAlignments[i];
        const auto& revcomplres = b.revcomplAlignments[i];
        const int candidatelength = b.fwdSequences[i]->length();

        BestAlignment_t bestAlignment = get_best_alignment(res,
                                                    revcomplres, querylength, candidatelength);

        if(bestAlignment == BestAlignment_t::None){
            //both alignments are bad, cannot use this candidate for correction
            b.activeCandidates[i] = false;
        }else{
            const double mismatchratio = [&](){
                if(bestAlignment == BestAlignment_t::Forward)
                    return double(res.get_nOps()) / double(res.get_overlap());
                else
                    return double(revcomplres.get_nOps()) / double(revcomplres.get_overlap());
            }();
            const int candidateCount = b.candidateCounts[i];//b.candidateCountsPrefixSum[i+1] - b.candidateCountsPrefixSum[i];
            if(mismatchratio >= 4 * b.mismatchratioBaseFactor){
                //best alignments is still not good enough, cannot use this candidate for correction
                b.activeCandidates[i] = false;
            }else{
                b.activeCandidates[i] = true;

                if (mismatchratio < 2 * b.mismatchratioBaseFactor) {
                    b.counts[0] += candidateCount;
                }
                if (mismatchratio < 3 * b.mismatchratioBaseFactor) {
                    b.counts[1] += candidateCount;
                }
                if (mismatchratio < 4 * b.mismatchratioBaseFactor) {
                    b.counts[2] += candidateCount;
                }

                const int begin = b.candidateCountsPrefixSum[i];
                if(bestAlignment == BestAlignment_t::Forward){
                    b.bestIsForward[i] = true;
                    b.bestSequences[i] = b.fwdSequences[i];
                    b.bestAlignments[i] = &b.fwdAlignments[i];

                    if(BE::canUseQualityScores){
                        for(int j = 0; j < candidateCount; j++){
                            const ReadId_t id = b.candidateIds[begin + j];
                            b.bestQualities[begin + j] = b.readStorage->fetchQuality_ptr(id);
                        }
                    }
                }else{
                    b.bestIsForward[i] = false;
                    b.bestSequences[i] = b.revcomplSequences[i];
                    b.bestAlignments[i] = &b.revcomplAlignments[i];

                    if(BE::canUseQualityScores){
                        for(int j = 0; j < candidateCount; j++){
                            const ReadId_t id = b.candidateIds[begin + j];
                            b.bestQualities[begin + j] = b.readStorage->fetchReverseComplementQuality_ptr(id);
                        }
                    }
                }
            }
        }
    }
}

template<class BE>
bool hasEnoughGoodCandidates(const BE& b){
    if (b.counts[0] >= b.goodAlignmentsCountThreshold
        || b.counts[1] >= b.goodAlignmentsCountThreshold
        || b.counts[2] >= b.goodAlignmentsCountThreshold)

        return true;

    return false;
}

template<class BE>
void prepare_good_candidates(BE& b){
    DetermineGoodAlignmentStats stats;

    b.mismatchratioThreshold = 0;
    if (b.counts[0] >= b.goodAlignmentsCountThreshold) {
        b.mismatchratioThreshold = 2 * b.mismatchratioBaseFactor;
        stats.correctionCases[0]++;
    } else if (b.counts[1] >= b.goodAlignmentsCountThreshold) {
        b.mismatchratioThreshold = 3 * b.mismatchratioBaseFactor;
        stats.correctionCases[1]++;
    } else if (b.counts[2] >= b.goodAlignmentsCountThreshold) {
        b.mismatchratioThreshold = 4 * b.mismatchratioBaseFactor;
        stats.correctionCases[2]++;
    } else { //no correction possible
        stats.correctionCases[3]++;
        b.active = false;
        return;
    }

    std::size_t activeposition_unique = 0;
    std::size_t activeposition = 0;

    //stable_partition on struct of arrays with condition (activeCandidates[i] && notremoved) ?
    for(std::size_t i = 0; i < b.activeCandidates.size(); i++){
        if(b.activeCandidates[i]){
            const double mismatchratio = double(b.bestAlignments[i]->get_nOps()) / double(b.bestAlignments[i]->get_overlap());
            const bool notremoved = mismatchratio < b.mismatchratioThreshold;
            if(notremoved){
                b.fwdSequences[activeposition_unique] = b.fwdSequences[i];
                b.revcomplSequences[activeposition_unique] = b.revcomplSequences[i];
                b.bestAlignments[activeposition_unique] = b.bestAlignments[i];
                b.bestSequences[activeposition_unique] = b.bestSequences[i];
                b.bestSequenceStrings[activeposition_unique] = b.bestSequences[i]->toString();
                b.bestIsForward[activeposition_unique] = b.bestIsForward[i];

                const int begin = b.candidateCountsPrefixSum[i];
                const int count = b.candidateCountsPrefixSum[i+1] - begin;
                for(int j = 0; j < count; j++){
                    b.candidateIds[activeposition] = b.candidateIds[begin + j];
                    if(BE::canUseQualityScores){
                        b.bestQualities[activeposition] = b.bestQualities[begin + j];
                    }
                    activeposition++;
                }
                b.candidateCounts[activeposition_unique] = count;
                b.candidateCountsPrefixSum[activeposition_unique+1] = b.candidateCountsPrefixSum[activeposition_unique] + count;
                activeposition_unique++;
            }
        }
    }

    b.fwdSequences.resize(activeposition_unique);
    b.revcomplSequences.resize(activeposition_unique);
    b.bestAlignments.resize(activeposition_unique);
    b.bestSequences.resize(activeposition_unique);
    b.bestSequenceStrings.resize(activeposition_unique);
    b.bestIsForward.resize(activeposition_unique);
    b.candidateCounts.resize(activeposition_unique);
    b.candidateCountsPrefixSum.resize(activeposition_unique+1);

    b.candidateIds.resize(activeposition);
    b.bestQualities.resize(activeposition);

    b.activeCandidates.clear(); //no longer need this

    b.n_unique_candidates = activeposition_unique;
    b.n_candidates = activeposition;
}

}
#endif
