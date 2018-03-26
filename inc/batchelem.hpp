#ifndef CARE_BATCH_ELEM_HPP
#define CARE_BATCH_ELEM_HPP

#include "read.hpp"
#include "alignment.hpp"
#include "readstorage.hpp"

#include <vector>
#include <string>

struct CorrectedCandidate{
    std::uint64_t index;
    std::string sequence;
    CorrectedCandidate(){}
    CorrectedCandidate(std::uint64_t index, const std::string& sequence)
        : index(index), sequence(sequence){}

    CorrectedCandidate(const CorrectedCandidate& other) = default;
    CorrectedCandidate(CorrectedCandidate&& other) = default;
    CorrectedCandidate& operator=(const CorrectedCandidate& other) = default;
    CorrectedCandidate& operator=(CorrectedCandidate&& other) = default;
};

struct DetermineGoodAlignmentStats{
    int correctionCases[4]{0,0,0,0};
    int uniqueCandidatesWithoutGoodAlignment=0;
};

struct BatchElem{
    bool active;

    //query properties
    bool corrected;
    std::uint64_t readId;
    const Sequence* fwdSequence;
	std::string fwdSequenceString;
    const std::string* fwdQuality;
    std::string correctedSequence;

	size_t n_unique_candidates;
	size_t n_candidates;

    //candidates properties
    std::vector<std::uint64_t> candidateIds;
    std::vector<int> candidateCountsPrefixSum;

    std::vector<bool> activeCandidates;
    std::vector<const Sequence*> fwdSequences;
    std::vector<const Sequence*> revcomplSequences;
    //std::vector<const std::string*> fwdQualities;
	//std::vector<const std::string*> revcomplQualities;
    std::vector<AlignResultCompact> fwdAlignments;
    std::vector<AlignResultCompact> revcomplAlignments;
    std::vector<AlignResultCompact> bestAlignments;
    std::vector<const Sequence*> bestSequences;
	std::vector<std::string> bestSequenceStrings;
	std::vector<const std::string*> bestQualities;
    std::vector<bool> bestIsForward;
    std::vector<CorrectedCandidate> correctedCandidates;

    std::vector<std::vector<AlignOp>> fwdAlignOps;
    std::vector<std::vector<AlignOp>> revcomplAlignOps;
    std::vector<std::vector<AlignOp>*> bestAlignOps;

    double mismatchratioThreshold;

    //constant batch independent data
    const ReadStorage* readStorage;
    double errorrate;
    int estimatedCoverage;
    double m_coverage;
    double goodAlignmentsCountThreshold;
    double MAX_MISMATCH_RATIO;
    int MIN_OVERLAP;
    double MIN_OVERLAP_RATIO;

    //BatchElem() : BatchElem(nullptr, 0.0, 0.0){}

    BatchElem(const ReadStorage* rs, double errorrate_,
                int estimatedCoverage_, double m_coverage_,
                double MAX_MISMATCH_RATIO_, int MIN_OVERLAP_, double MIN_OVERLAP_RATIO_);

    std::uint64_t get_number_of_duplicate_sequences() const;
    void clear();
    void set_number_of_sequences(std::uint64_t num);
    void set_number_of_unique_sequences(std::uint64_t num);
    void set_read_id(std::uint64_t id);
    void fetch_query_data_from_readstorage();
    void set_candidate_ids(const std::vector<std::uint64_t>& ids);
    void make_unique_sequences();
    void fetch_revcompl_sequences_from_readstorage();
    DetermineGoodAlignmentStats determine_good_alignments();
    void prepare_good_candidates();
};

#endif
