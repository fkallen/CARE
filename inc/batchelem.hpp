#ifndef CARE_BATCH_ELEM_HPP
#define CARE_BATCH_ELEM_HPP

#include "tasktiming.hpp"
#include "options.hpp"
#include "bestalignment.hpp"

#include <cassert>
#include <vector>
#include <string>
#include <numeric>
#include <chrono>

namespace care{

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

	bool canUseQualityScores;

    bool active;

    //query properties
    bool corrected;
    ReadId_t readId;
    const Sequence_t* fwdSequence;
	std::string fwdSequenceString;
    const std::string* fwdQuality;
    std::string correctedSequence;

	std::uint64_t n_candidates;

    //candidates properties
    std::vector<ReadId_t> candidateIds;

    std::vector<bool> activeCandidates;
    std::vector<const Sequence_t*> fwdSequences;

    std::vector<AlignmentResult_t*> bestAlignments;
	std::vector<std::string> bestSequenceStrings;
	std::vector<const std::string*> bestQualities;

    std::vector<std::string> reverseComplementQualities;

    std::vector<BestAlignment_t> bestAlignmentFlags;
    std::vector<AlignmentResult_t> alignments;

    double mismatchratioThreshold;

    const ReadStorage_t* readStorage;

    double goodAlignmentsCountThreshold;
    double mismatchratioBaseFactor;
    CorrectionOptions correctionOptions;

    TaskTimings findCandidatesTiming;
    std::vector<std::pair<ReadId_t, const Sequence_t*>> make_unique_sequences_numseqpairs;

    int counts[3] { 0, 0, 0 }; //count number of cases of mismatchratio < 2*errorrate, 3*errorrate, 4*errorrate

    BatchElem(){
        //std::cout << "BatchElem()" << std::endl;
    }

    BatchElem(const ReadStorage_t& rs,
                const CorrectionOptions& CO)
                :   BatchElem(rs, CO, 10){

                        //std::cout << "BatchElem(const ReadStorage_t& rs, const CorrectionOptions& CO)" << std::endl;
    }

    BatchElem(const ReadStorage_t& rs,
                const CorrectionOptions& CO, std::uint64_t approxMaxCandidates)
                :   canUseQualityScores(CO.useQualityScores),
                    readStorage(&rs),
                    goodAlignmentsCountThreshold(CO.estimatedCoverage * CO.m_coverage),
                    mismatchratioBaseFactor(CO.estimatedErrorrate*1.0),
                    correctionOptions(CO){


            activeCandidates.reserve(approxMaxCandidates);
            fwdSequences.reserve(approxMaxCandidates);
            bestAlignments.reserve(approxMaxCandidates);
            bestSequenceStrings.reserve(approxMaxCandidates);
            bestQualities.reserve(approxMaxCandidates);

            reverseComplementQualities.reserve(approxMaxCandidates);

            bestAlignmentFlags.reserve(approxMaxCandidates);
            alignments.reserve(approxMaxCandidates);
    }

    BatchElem(const BatchElem& other)
        :canUseQualityScores(other.canUseQualityScores),
        active(other.active),
        corrected(other.corrected),
        readId(other.readId),
        fwdSequence(other.fwdSequence),
        fwdSequenceString(other.fwdSequenceString),
        fwdQuality(other.fwdQuality),
        correctedSequence(other.correctedSequence),
        n_candidates(other.n_candidates),
        candidateIds(other.candidateIds),
        activeCandidates(other.activeCandidates),
        fwdSequences(other.fwdSequences),
        bestAlignments(other.bestAlignments),
        bestSequenceStrings(other.bestSequenceStrings),
        bestQualities(other.bestQualities),
        reverseComplementQualities(other.reverseComplementQualities),
        bestAlignmentFlags(other.bestAlignmentFlags),
        alignments(other.alignments),
        mismatchratioThreshold(other.mismatchratioThreshold),
        readStorage(other.readStorage),
        goodAlignmentsCountThreshold(other.goodAlignmentsCountThreshold),
        mismatchratioBaseFactor(other.mismatchratioBaseFactor),
        correctionOptions(other.correctionOptions),
        findCandidatesTiming(other.findCandidatesTiming),
        make_unique_sequences_numseqpairs(other.make_unique_sequences_numseqpairs){

            //std::cout << "BatchElem(const BatchElem& other)" << std::endl;

        counts[0] = other.counts[0];
        counts[1] = other.counts[1];
        counts[2] = other.counts[2];
    }

    BatchElem(BatchElem&& other){
        //std::cout << "BatchElem(BatchElem&& other)" << std::endl;
        operator=(other);
    }

    BatchElem& operator=(const BatchElem& other){
        //std::cout << "operator=(const BatchElem& other)" << std::endl;
    //BatchElem& operator=(BatchElem other){
        BatchElem tmp(other);
        swap(*this, tmp);
        //swap(*this, other);
        return *this;
    }

    BatchElem& operator=(BatchElem&& other){
        //std::cout << "operator=(BatchElem&& other)" << std::endl;
        swap(*this, other);
        return *this;
    }

public:
    friend void swap(BatchElem& l, BatchElem& r) noexcept{
        using std::swap;
        //std::cout << "swap(BatchElem& l, BatchElem& r)" << std::endl;

        swap(l.canUseQualityScores, r.canUseQualityScores);
        swap(l.active, r.active);
        swap(l.corrected, r.corrected);
        swap(l.readId, r.readId);
        swap(l.fwdSequence, r.fwdSequence);
        swap(l.fwdSequenceString, r.fwdSequenceString);
        swap(l.fwdQuality, r.fwdQuality);
        swap(l.correctedSequence, r.correctedSequence);
        swap(l.n_candidates, r.n_candidates);
        swap(l.candidateIds, r.candidateIds);
        swap(l.activeCandidates, r.activeCandidates);
        swap(l.fwdSequences, r.fwdSequences);
        swap(l.bestAlignments, r.bestAlignments);
        swap(l.bestSequenceStrings, r.bestSequenceStrings);
        swap(l.bestQualities, r.bestQualities);
        swap(l.reverseComplementQualities, r.reverseComplementQualities);
        swap(l.bestAlignmentFlags, r.bestAlignmentFlags);
        swap(l.alignments, r.alignments);
        swap(l.mismatchratioThreshold, r.mismatchratioThreshold);
        swap(l.readStorage, r.readStorage);
        swap(l.goodAlignmentsCountThreshold, r.goodAlignmentsCountThreshold);
        swap(l.mismatchratioBaseFactor, r.mismatchratioBaseFactor);
        swap(l.correctionOptions, r.correctionOptions);
        swap(l.findCandidatesTiming, r.findCandidatesTiming);
        swap(l.make_unique_sequences_numseqpairs, r.make_unique_sequences_numseqpairs);
        swap(l.counts, r.counts);
    }

};


template<class BE>
void clear(BE& b){
    b.active = false;
    b.corrected = false;
    b.fwdQuality = nullptr;
    b.fwdSequence = nullptr;
    b.candidateIds.clear();
    b.fwdSequences.clear();
    b.activeCandidates.clear();
    b.bestAlignments.clear();
    b.bestSequenceStrings.clear();
    b.bestQualities.clear();

    b.reverseComplementQualities.clear();

    b.bestAlignmentFlags.clear();
    b.alignments.clear();

    b.counts[0] = 0;
    b.counts[1] = 0;
    b.counts[2] = 0;
}

template<class BE>
void set_number_of_candidates(BE& b, std::uint64_t num){
    b.bestQualities.resize(num, nullptr);
    b.activeCandidates.resize(num, false);
    b.fwdSequences.resize(num);
    b.bestAlignments.resize(num);
    b.bestSequenceStrings.resize(num);

    b.bestAlignmentFlags.resize(num);
    b.alignments.resize(num);

    b.n_candidates = num;
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
	auto readIdPos = std::find(b.candidateIds.begin(), b.candidateIds.end(), b.readId);
	if(readIdPos != b.candidateIds.end())
		b.candidateIds.erase(readIdPos);

    set_number_of_candidates(b, b.candidateIds.size());

    //for(std::size_t k = 0; k < b.activeCandidates.size(); k++)
    //    b.activeCandidates[k] = false;

    b.findCandidatesTiming.preprocessingEnd();
    if(b.candidateIds.size() == 0){
        //no need for further processing without candidates
        b.active = false;
    }else{
        b.findCandidatesTiming.executionBegin();

        fetch_candidates_from_readstorage(b);

        b.findCandidatesTiming.executionEnd();

    }

}

template<class BE>
void fetch_query_data_from_readstorage(BE& b){
    b.fwdSequence = b.readStorage->fetchSequence_ptr(b.readId);
    if(b.canUseQualityScores){
        b.fwdQuality = b.readStorage->fetchQuality_ptr(b.readId);
    }
    b.fwdSequenceString = b.fwdSequence->toString();
}

template<class BE>
void fetch_candidates_from_readstorage(BE& b){
    assert(b.candidateIds.size() == b.fwdSequences.size());

    for(std::size_t i = 0; i <  b.candidateIds.size(); i++)
        b.fwdSequences[i] = b.readStorage->fetchSequence_ptr(b.candidateIds[i]);
}



template<class BE, class Func>
void determine_good_alignments(BE& b, Func get_best_alignment){
    determine_good_alignments(b, 0, b.fwdSequences.size(), get_best_alignment);
}



template<class BE, class Func>
void determine_good_alignments(BE& b, int firstIndex, int N, Func get_best_alignment){
    using AlignmentResult_t = typename BE::AlignmentResult_t;

    const int lastIndex_excl = std::min(std::size_t(firstIndex + N), b.fwdSequences.size());

    if(!BE::ReadStorage_t::has_reverse_complement){
        b.reverseComplementQualities.reserve(b.candidateIds.size());
    }
    for(int i = firstIndex; i < lastIndex_excl; i++){

        BestAlignment_t alignmentFlag = b.bestAlignmentFlags[i];
        const AlignmentResult_t& alignment = b.alignments[i];

        if(alignmentFlag == BestAlignment_t::None){
            //cannot use this candidate for correction
            b.activeCandidates[i] = false;
        }else{
            const double mismatchratio = double(alignment.get_nOps()) / double(alignment.get_overlap());

            if(mismatchratio >= 4 * b.mismatchratioBaseFactor){
                //best alignments is still not good enough, cannot use this candidate for correction
                b.activeCandidates[i] = false;
            }else{
                b.activeCandidates[i] = true;

                if (mismatchratio < 2 * b.mismatchratioBaseFactor) {
                    b.counts[0] += 1;
                }
                if (mismatchratio < 3 * b.mismatchratioBaseFactor) {
                    b.counts[1] += 1;
                }
                if (mismatchratio < 4 * b.mismatchratioBaseFactor) {
                    b.counts[2] += 1;
                }

                if(alignmentFlag == BestAlignment_t::Forward){
                    b.bestAlignments[i] = &b.alignments[i];
                    //new
                    //b.bestSequenceStrings[i] = b.fwdSequences[i]->toString();

                    if(b.canUseQualityScores){
                        b.bestQualities[i] = b.readStorage->fetchQuality_ptr(b.candidateIds[i]);
                    }
                }else{

                    //new
                    //b.bestSequenceStrings[i] = b.reverseComplements[i].toString();

                    b.bestAlignments[i] = &b.alignments[i];

                    if(b.canUseQualityScores){
                        if(BE::ReadStorage_t::has_reverse_complement){
                            b.bestQualities[i] = b.readStorage->fetchReverseComplementQuality_ptr(b.candidateIds[i]);
                        }else{
                            std::string qualitystring = *(b.readStorage->fetchQuality_ptr(b.candidateIds[i]));
                            std::reverse(qualitystring.begin(), qualitystring.end());
                            b.reverseComplementQualities.emplace_back(std::move(qualitystring));
                            b.bestQualities[i] = &b.reverseComplementQualities.back();
                        }
                    }
                }

                //assert(b.bestSequences[i]->toString() == b.bestSequenceStrings[i]);
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
std::tuple<std::chrono::duration<double>, std::chrono::duration<double>, std::chrono::duration<double>>
prepare_good_candidates(BE& b){

    std::chrono::duration<double> da, db, dc;


    std::chrono::time_point<std::chrono::system_clock> tpa, tpb;

    tpa = std::chrono::system_clock::now();
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
        return {da,db,dc};
    }

    tpb = std::chrono::system_clock::now();
    da = tpb-tpa;
    tpa = std::chrono::system_clock::now();

    std::size_t activeposition = 0;

    for(std::size_t i = 0; i < b.activeCandidates.size(); i++){
        if(b.activeCandidates[i]){
            const double mismatchratio = double(b.bestAlignments[i]->get_nOps()) / double(b.bestAlignments[i]->get_overlap());
            const bool notremoved = mismatchratio < b.mismatchratioThreshold;
            if(notremoved){

                //avoid self assignment / move
                if(activeposition != i){
                    b.fwdSequences[activeposition] = b.fwdSequences[i];
                    b.bestAlignments[activeposition] = b.bestAlignments[i];
                    //new
                    b.bestSequenceStrings[activeposition] = std::move(b.bestSequenceStrings[i]);
                    b.bestAlignmentFlags[activeposition] = b.bestAlignmentFlags[i];
                    b.candidateIds[activeposition] = b.candidateIds[i];

                    if(b.canUseQualityScores){
                        b.bestQualities[activeposition] = b.bestQualities[i];
                    }
                }

                activeposition++;
            }
        }
    }

    //std::cout << b.activeCandidates.size() << " " <<

    tpb = std::chrono::system_clock::now();
    db = tpb-tpa;
    tpa = std::chrono::system_clock::now();

    b.activeCandidates.clear(); //no longer need this, all remaining candidates are active

    b.candidateIds.resize(activeposition);
    b.bestQualities.resize(activeposition);
    b.fwdSequences.resize(activeposition);
    b.bestAlignments.resize(activeposition);
    b.bestSequenceStrings.resize(activeposition);
    b.bestAlignmentFlags.resize(activeposition);

    b.n_candidates = activeposition;

    tpb = std::chrono::system_clock::now();
    dc = tpb-tpa;

    return {da,db,dc};
}


}


#endif
