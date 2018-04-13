#include "../inc/batchelem.hpp"

#include "../inc/alignment.hpp"
#include "../inc/readstorage.hpp"
#include "../inc/sequence.hpp"
#include "../inc/types.hpp"

#include <vector>
#include <string>

namespace care{
    BatchElem::BatchElem(const ReadStorage* rs,
                        const Minhasher* minhasher,
                        double errorrate_,
                int estimatedCoverage_, double m_coverage_,
                double MAX_MISMATCH_RATIO_, int MIN_OVERLAP_, double MIN_OVERLAP_RATIO_)
            :   readStorage(rs),
                minhasher(minhasher),
                errorrate(errorrate_),
                estimatedCoverage(estimatedCoverage_),
                m_coverage(m_coverage_),
                goodAlignmentsCountThreshold(estimatedCoverage_ * m_coverage_),
                MAX_MISMATCH_RATIO(MAX_MISMATCH_RATIO_),
                MIN_OVERLAP(MIN_OVERLAP_),
                MIN_OVERLAP_RATIO(MIN_OVERLAP_RATIO_){


    }

    std::uint64_t BatchElem::get_number_of_duplicate_sequences() const{
        return candidateIds.size() - fwdSequences.size();
    }

    void BatchElem::clear(){
        active = false;
        corrected = false;
        fwdQuality = nullptr;
        fwdSequence = nullptr;
        candidateIds.clear();
        candidateCountsPrefixSum.clear();
        fwdSequences.clear();
        activeCandidates.clear();
        revcomplSequences.clear();
        fwdAlignments.clear();
        revcomplAlignments.clear();
        bestAlignments.clear();
        bestSequences.clear();
		bestSequenceStrings.clear();
        bestIsForward.clear();
        correctedCandidates.clear();
		bestQualities.clear();

        fwdAlignOps.clear();
        revcomplAlignOps.clear();
        bestAlignOps.clear();

        counts[0] = 0;
        counts[1] = 0;
        counts[2] = 0;
    }

    void BatchElem::set_number_of_sequences(std::uint64_t num){
		bestQualities.resize(num);
    }

    void BatchElem::set_number_of_unique_sequences(std::uint64_t num){
        candidateCountsPrefixSum.resize(num+1);
        activeCandidates.resize(num, false);
        fwdSequences.resize(num);
        revcomplSequences.resize(num);
        fwdAlignments.resize(num);
        revcomplAlignments.resize(num);
        bestAlignments.resize(num);
        bestSequences.resize(num);
		bestSequenceStrings.resize(num);
        bestIsForward.resize(num);
        bestAlignOps.resize(num);

        fwdAlignOps.resize(num);
        revcomplAlignOps.resize(num);
    }

    void BatchElem::set_read_id(ReadId_t id){
        clear();
        readId = id;
        corrected = false;
        active = true;
    }

    void BatchElem::findCandidates(std::uint64_t max_number_candidates){
        //get data of sequence which should be corrected
        fetch_query_data_from_readstorage();

        //get candidate ids from minhasher
        set_candidate_ids(minhasher->getCandidates(fwdSequenceString, max_number_candidates));
        if(candidateIds.size() == 0){
            //no need for further processing without candidates
            active = false;
        }else{
            //find unique candidate sequences
            make_unique_sequences();
            //get reverse complements of unique candidate sequences
            fetch_revcompl_sequences_from_readstorage();
        }
    }

    void BatchElem::fetch_query_data_from_readstorage(){
        fwdSequence = readStorage->fetchSequence_ptr(readId);
        fwdQuality = readStorage->fetchQuality_ptr(readId);
        fwdSequenceString = fwdSequence->toString();
    }

    void BatchElem::set_candidate_ids(std::vector<Minhasher::Result_t>&& minhashResults){
        candidateIds = std::move(minhashResults);

        //remove self from candidates
        candidateIds.erase(std::find(candidateIds.begin(), candidateIds.end(), readId));

        set_number_of_sequences(candidateIds.size());
        set_number_of_unique_sequences(candidateIds.size());

        for(size_t k = 0; k < activeCandidates.size(); k++)
            activeCandidates[k] = false;
    }

    void BatchElem::make_unique_sequences(){
        std::vector<std::pair<ReadId_t, const Sequence_t*>> numseqpairs;
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

        const Sequence_t* prevSeq = numseqpairs[0].second;

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

    void BatchElem::fetch_revcompl_sequences_from_readstorage(){
        for(size_t k = 0; k < fwdSequences.size(); k++){
            int first = candidateCountsPrefixSum[k];
            revcomplSequences[k] = readStorage->fetchReverseComplementSequence_ptr(candidateIds[first]);
        }
    }

    void BatchElem::determine_good_alignments(){
        determine_good_alignments(0, fwdSequences.size());
    }

    /*
        Processes at most N alignments from
        alignments[firstIndex]
        to
        alignments[min(firstIndex + N-1, number of alignments - 1)]
    */
    void BatchElem::determine_good_alignments(int firstIndex, int N){
        const int querylength = fwdSequence->length();
        const int lastIndex_excl = std::min(size_t(firstIndex + N), fwdSequences.size());

        for(int i = firstIndex; i < lastIndex_excl; i++){
            const auto& res = fwdAlignments[i];
            const auto& revcomplres = revcomplAlignments[i];
            const int candidatelength = fwdSequences[i]->length();

            BestAlignment_t bestAlignment = get_best_alignment(res,
                    revcomplres, querylength, candidatelength,
                    MAX_MISMATCH_RATIO, MIN_OVERLAP,
                    MIN_OVERLAP_RATIO);

            if(bestAlignment == BestAlignment_t::None){
                //both alignments are bad, cannot use this candidate for correction
                activeCandidates[i] = false;
            }else{
                const double mismatchratio = [&](){
                    if(bestAlignment == BestAlignment_t::Forward)
                        return double(res.nOps) / double(res.overlap);
                    else
                        return double(revcomplres.nOps) / double(revcomplres.overlap);
                }();
                const int candidateCount = candidateCountsPrefixSum[i+1] - candidateCountsPrefixSum[i];
                if(mismatchratio >= 4 * errorrate){
                    //best alignments is still not good enough, cannot use this candidate for correction
                    activeCandidates[i] = false;
                }else{
                    activeCandidates[i] = true;

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
                        bestAlignOps[i] = &fwdAlignOps[i];
                        for(int j = 0; j < candidateCount; j++){
                            const ReadId_t id = candidateIds[begin + j];
                            bestQualities[begin + j] = readStorage->fetchQuality_ptr(id);
                        }
                    }else{
                        bestIsForward[i] = false;
                        bestSequences[i] = revcomplSequences[i];
                        bestAlignments[i] = revcomplres;
                        bestAlignOps[i] = &revcomplAlignOps[i];
                        for(int j = 0; j < candidateCount; j++){
                            const ReadId_t id = candidateIds[begin + j];
                            bestQualities[begin + j] = readStorage->fetchReverseComplementQuality_ptr(id);
                        }
                    }
                }
            }
        }
    }

    bool BatchElem::hasEnoughGoodCandidates() const{
        if (counts[0] >= goodAlignmentsCountThreshold
            || counts[1] >= goodAlignmentsCountThreshold
            || counts[2] >= goodAlignmentsCountThreshold)

            return true;

        return false;
    }

    void BatchElem::prepare_good_candidates(){
        DetermineGoodAlignmentStats stats;

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
        } else { //no correction possible
            stats.correctionCases[3]++;
            active = false;
            return;
        }

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
                    bestAlignOps[activeposition_unique] = bestAlignOps[i];
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

}
