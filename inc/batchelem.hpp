#ifndef CARE_BATCH_ELEM
#define CARE_BATCH_ELEM

#include "read.hpp"
#include "alignment.hpp"

#include <vector>
#include <string>

struct CorrectedCandidate{
    std::uint64_t index;
    std::string sequence;
}

struct BatchElem{
    bool active;
    bool corrected;
    std::uint64_t readId;
    const Sequence* fwdSequence;
	std::string fwdSequenceString;
    const std::string* fwdQuality;
    std::string correctedSequence;

    std::vector<std::uint64_t> candidateIds;
    std::vector<int> candidateCountsPrefixSum;

    std::vector<bool> activeCandidates;
    std::vector<const Sequence*> fwdSequences;
    std::vector<const Sequence*> revcomplSequences;
    std::vector<const std::string*> fwdQualities;
	std::vector<const std::string*> revcomplQualities;
    std::vector<AlignResultCompact> fwdAlignments;
    std::vector<AlignResultCompact> revcomplAlignments;
    std::vector<AlignResultCompact*> bestAlignments;
    std::vector<const Sequence*> bestSequences;
    std::vector<bool> bestIsForward;
    std::vector<CorrectedCandidate> correctedCandidates;

    void clear(){
        active = false;
        corrected = false;
        fwdQuality = nullptr;
        fwdSequence = nullptr;
        candidateIds.clear();
        candidateCountsPrefixSum.clear();
        fwdSequences.clear();
        activeCandidates.clear();
        fwdQualities.clear();
        revcomplSequences.clear();
        revcomplQualities.clear();
        fwdAlignments.clear();
        revcomplAlignments.clear();
        bestAlignments.clear();
        bestSequences.clear();
        bestIsForward.clear();
        correctedCandidates.clear();
    }

    void set_number_of_sequences(std::uint64_t num){
        fwdQualities.resize(num);
        revcomplQualities.resize(num);
    }

    void set_number_of_unique_sequences(std::uint64_t num){
        candidateCountsPrefixSum.resize(num+1);
        activeCandidates.resize(num);
        fwdSequences.resize(num);
        revcomplSequences.resize(num);
        fwdQualities.resize(num);
        revcomplQualities.resize(num);
        fwdAlignments.resize(num);
        revcomplAlignments.resize(num);
        bestAlignments.resize(num);
        bestSequences.resize(num);
        bestIsForward.resize(num);
    }
};

#endif
