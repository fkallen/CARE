#ifndef CARE_BATCH_ELEM
#define CARE_BATCH_ELEM

#include "read.hpp"
#include "alignment.hpp"
#include "readstorage.hpp"

#include <vector>
#include <string>

struct CorrectedCandidate{
    std::uint64_t index;
    std::string sequence;
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
                double MAX_MISMATCH_RATIO_, int MIN_OVERLAP_, double MIN_OVERLAP_RATIO_)
            :   readStorage(rs),
                errorrate(errorrate_),
                estimatedCoverage(estimatedCoverage_),
                m_coverage(m_coverage_),
                goodAlignmentsCountThreshold(estimatedCoverage_ * m_coverage_),
                MAX_MISMATCH_RATIO(MAX_MISMATCH_RATIO_),
                MIN_OVERLAP(MIN_OVERLAP_),
                MIN_OVERLAP_RATIO(MIN_OVERLAP_RATIO_){


    }

    std::uint64_t get_number_of_duplicate_sequences() const{
        return candidateIds.size() - fwdSequences.size();
    }

    void clear(){
        active = false;
        corrected = false;
        fwdQuality = nullptr;
        fwdSequence = nullptr;
        candidateIds.clear();
        candidateCountsPrefixSum.clear();
        fwdSequences.clear();
        activeCandidates.clear();
        //fwdQualities.clear();
        revcomplSequences.clear();
        //revcomplQualities.clear();
        fwdAlignments.clear();
        revcomplAlignments.clear();
        bestAlignments.clear();
        bestSequences.clear();
		bestSequenceStrings.clear();
        bestIsForward.clear();
        correctedCandidates.clear();
		bestQualities.clear();
    }

    void set_number_of_sequences(std::uint64_t num){
        //fwdQualities.resize(num);
        //revcomplQualities.resize(num);
		bestQualities.resize(num);
    }

    void set_number_of_unique_sequences(std::uint64_t num){
        candidateCountsPrefixSum.resize(num+1);
        activeCandidates.resize(num);
        fwdSequences.resize(num);
        revcomplSequences.resize(num);
        fwdAlignments.resize(num);
        revcomplAlignments.resize(num);
        bestAlignments.resize(num);
        bestSequences.resize(num);
		bestSequenceStrings.resize(num);
        bestIsForward.resize(num);
    }

    void set_read_id(std::uint64_t id){
        clear();
        readId = id;
        corrected = false;
        active = true;
    }

    void fetch_query_data_from_readstorage(){
        fwdSequence = readStorage->fetchSequence_ptr(readId);
        fwdQuality = readStorage->fetchQuality_ptr(readId);
        fwdSequenceString = fwdSequence->toString();
    }

    void set_candidate_ids(const std::vector<std::uint64_t>& ids){
        candidateIds = ids;
        //remove self from candidates
        candidateIds.erase(std::find(candidateIds.begin(), candidateIds.end(), readId));

        set_number_of_sequences(candidateIds.size());
        set_number_of_unique_sequences(candidateIds.size());

        for(size_t k = 0; k < activeCandidates.size(); k++)
            activeCandidates[k] = true;
    }

    void make_unique_sequences(){
        std::vector<std::pair<std::uint64_t, const Sequence*>> numseqpairs;
        numseqpairs.reserve(candidateIds.size());

        //std::chrono::time_point < std::chrono::system_clock > t1 =
        //        std::chrono::system_clock::now();
        for(const auto id : candidateIds){
            numseqpairs.emplace_back(id, readStorage->fetchSequence_ptr(id));
        }
        //std::chrono::time_point < std::chrono::system_clock > t2 =
        //        std::chrono::system_clock::now();
        //mapminhashresultsfetch += (t2 - t1);

        //t1 = std::chrono::system_clock::now();

        //sort pairs by sequence
        std::sort(numseqpairs.begin(), numseqpairs.end(), [](auto l, auto r){return l.second < r.second;});

        std::uint64_t n_unique_elements = 1;
        candidateCountsPrefixSum[0] = 0;
        candidateCountsPrefixSum[1] = 1;
        candidateIds[0] = numseqpairs[0].first;
        fwdSequences[0] = numseqpairs[0].second;

        const Sequence* prevSeq = numseqpairs[0].second;

        for (size_t k = 1; k < numseqpairs.size(); k++) {
            auto pair = numseqpairs[k];
            candidateIds[k] = pair.first;
            const Sequence* curSeq = pair.second;
            if (prevSeq == curSeq) {
                candidateCountsPrefixSum[n_unique_elements]++;
            }else {
                candidateCountsPrefixSum[n_unique_elements+1] = 1 + candidateCountsPrefixSum[n_unique_elements];
                fwdSequences[n_unique_elements] = curSeq;
                n_unique_elements++;
            }
            prevSeq = curSeq;
        }

        set_number_of_unique_sequences(n_unique_elements);
        n_unique_candidates = fwdSequences.size();
        n_candidates = candidateIds.size();

        assert(candidateCountsPrefixSum.back() == int(candidateIds.size()));
    }

    void fetch_revcompl_sequences_from_readstorage(){
        for(size_t k = 0; k < fwdSequences.size(); k++){
            int first = candidateCountsPrefixSum[k];
            revcomplSequences[k] = readStorage->fetchReverseComplementSequence_ptr(candidateIds[first]);
        }
    }


    // sets active to false if not enough good alignments
    DetermineGoodAlignmentStats determine_good_alignments(){
        DetermineGoodAlignmentStats stats;

        int counts[3] { 0, 0, 0 };

        const int querylength = fwdSequence->length();

        for(size_t i = 0; i < fwdSequences.size(); i++){
            const auto& res = fwdAlignments[i];
            const auto& revcomplres = revcomplAlignments[i];
            const int candidatelength = fwdSequences[i]->length();

            BestAlignment_t bestAlignment = get_best_alignment(res,
                    revcomplres, querylength, candidatelength,
                    MAX_MISMATCH_RATIO, MIN_OVERLAP,
                    MIN_OVERLAP_RATIO);

            if(bestAlignment == BestAlignment_t::None){
                activeCandidates[i] = false;
                stats.uniqueCandidatesWithoutGoodAlignment++;
            }else{
                const double mismatchratio = [&](){
                    if(bestAlignment == BestAlignment_t::Forward)
                        return double(res.nOps) / double(res.overlap);
                    else
                        return double(revcomplres.nOps) / double(revcomplres.overlap);
                }();
                const int candidateCount = candidateCountsPrefixSum[i+1] - candidateCountsPrefixSum[i];
                if(mismatchratio >= 4 * errorrate){
                    activeCandidates[i] = false;
                }else{
                    if (mismatchratio < 2 * errorrate) {
                        counts[0] += candidateCount;
                    }
                    if (mismatchratio < 3 * errorrate) {
                        counts[1] += candidateCount;
                    }
                    if (mismatchratio < 4 * errorrate) {
                        counts[2] += candidateCount;
                    }

                    const int begin = candidateCountsPrefixSum[i];
                    if(bestAlignment == BestAlignment_t::Forward){
                        bestIsForward[i] = true;
                        bestSequences[i] = fwdSequences[i];
                        bestAlignments[i] = res;
                        for(int j = 0; j < candidateCount; j++){
                            const std::uint64_t id = candidateIds[begin + j];
                            bestQualities[begin + j] = readStorage->fetchQuality_ptr(id);
                        }
                    }else{
                        bestIsForward[i] = false;
                        bestSequences[i] = revcomplSequences[i];
                        bestAlignments[i] = revcomplres;
                        for(int j = 0; j < candidateCount; j++){
                            const std::uint64_t id = candidateIds[begin + j];
                            bestQualities[begin + j] = readStorage->fetchReverseComplementQuality_ptr(id);
                        }
                    }
                }
            }
        }

        // check errorrate of good alignments. we want at least m_coverage * estimatedCoverage alignments.
        // if possible, we want to use only alignments with a max mismatch ratio of 2*errorrate
        // if there are not enough alignments, use max mismatch ratio of 3*errorrate
        // if there are not enough alignments, use max mismatch ratio of 4*errorrate
        // if there are not enough alignments, do not correct

        mismatchratioThreshold = 0;
        if (counts[0] >= goodAlignmentsCountThreshold) {
            mismatchratioThreshold = 2 * errorrate;
            stats.correctionCases[0]++;
        } else if (counts[1] >= goodAlignmentsCountThreshold) {
            mismatchratioThreshold = 3 * errorrate;
            stats.correctionCases[1]++;
        } else if (counts[2] >= goodAlignmentsCountThreshold) {
            mismatchratioThreshold = 4 * errorrate;
            stats.correctionCases[2]++;
        } else { //no correction. write original sequence to output
            stats.correctionCases[3]++;
            active = false;
        }

        return stats;
    }

    void prepare_good_candidates(){
        size_t activeposition_unique = 0;
        size_t activeposition = 0;
        //stable_partition with condition (activeCandidates[i] && notremoved) ?
        for(size_t i = 0; i < activeCandidates.size(); i++){
            if(activeCandidates[i]){
                const double mismatchratio = double(bestAlignments[i].nOps) / double(bestAlignments[i].overlap);
                const bool notremoved = mismatchratio < mismatchratioThreshold;
                if(notremoved){
                    fwdSequences[activeposition_unique] = fwdSequences[i];
                    revcomplSequences[activeposition_unique] = revcomplSequences[i];
                    bestAlignments[activeposition_unique] = bestAlignments[i];
                    bestSequences[activeposition_unique] = bestSequences[i];
                    bestSequenceStrings[activeposition_unique] = bestSequences[i]->toString();
                    bestIsForward[activeposition_unique] = bestIsForward[i];

                    const int begin = candidateCountsPrefixSum[i];
                    const int count = candidateCountsPrefixSum[i+1] - begin;
                    for(int j = 0; j < count; j++){
                        candidateIds[activeposition] = candidateIds[begin + j];
                        bestQualities[activeposition] = bestQualities[begin + j];
                        activeposition++;
                    }
                    candidateCountsPrefixSum[activeposition_unique+1] = candidateCountsPrefixSum[activeposition_unique] + count;
                    activeposition_unique++;
                }
            }
        }
        n_unique_candidates = activeposition_unique;
        n_candidates = activeposition;
    }

};
#endif
