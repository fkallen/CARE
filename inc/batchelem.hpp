#ifndef CARE_BATCH_ELEM_HPP
#define CARE_BATCH_ELEM_HPP

#include "alignment.hpp"
#include "tasktiming.hpp"
#include "options.hpp"

#include <cassert>
#include <vector>
#include <string>

namespace care{

struct CorrectedCandidate{
    std::uint64_t index;
    std::string sequence;
    CorrectedCandidate(){}
    CorrectedCandidate(std::uint64_t index, const std::string& sequence)
        : index(index), sequence(sequence){}
};

struct DetermineGoodAlignmentStats{
    int correctionCases[4]{0,0,0,0};
    int uniqueCandidatesWithoutGoodAlignment=0;
};

template<class minhasher_t,
		 class readStorage_t,
		 class alignment_result_t>
struct BatchElem{

	using Minhasher_t = minhasher_t;
	using ReadStorage_t = readStorage_t;
	using Sequence_t = typename ReadStorage_t::Sequence_t;
	using ReadId_t = typename ReadStorage_t::ReadId_t;
    using AlignmentResult_t = alignment_result_t;

	static constexpr bool canUseQualityScores = ReadStorage_t::hasQualityScores;

	enum class BestAlignment_t {Forward, ReverseComplement, None};

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

    std::vector<CorrectedCandidate> correctedCandidates;

    double mismatchratioThreshold;

    const ReadStorage_t* readStorage;
    const Minhasher_t* minhasher;

    double goodAlignmentsCountThreshold;
    double mismatchratioBaseFactor;
    CorrectionOptions correctionOptions;
    GoodAlignmentProperties goodAlignmentProperties;

    TaskTimings findCandidatesTiming;
    std::vector<std::pair<ReadId_t, const Sequence_t*>> make_unique_sequences_numseqpairs;

    int counts[3] { 0, 0, 0 }; //count number of cases of mismatchratio < 2*errorrate, 3*errorrate, 4*errorrate

    BatchElem(const ReadStorage_t& rs, const Minhasher_t& minhasher,
                const CorrectionOptions& CO,
                const GoodAlignmentProperties& GAP)
	            :   readStorage(&rs),
                minhasher(&minhasher),
                goodAlignmentsCountThreshold(CO.estimatedCoverage * CO.m_coverage),
                mismatchratioBaseFactor(CO.estimatedErrorrate*1.0),
                correctionOptions(CO),
                goodAlignmentProperties(GAP){


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
        revcomplSequences.clear();
        fwdAlignments.clear();
        revcomplAlignments.clear();
        bestAlignments.clear();
        bestSequences.clear();
		bestSequenceStrings.clear();
        bestIsForward.clear();
        correctedCandidates.clear();
		bestQualities.clear();

        counts[0] = 0;
        counts[1] = 0;
        counts[2] = 0;
    }

   void set_number_of_sequences(std::uint64_t num){
		if(canUseQualityScores){
			bestQualities.resize(num);
		}
    }

    void set_number_of_unique_sequences(std::uint64_t num){
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
    }

    void set_read_id(ReadId_t id){
        clear();
        readId = id;
        corrected = false;
        active = true;
    }

    void findCandidates(std::uint64_t max_number_candidates){
        //get data of sequence which should be corrected
        fetch_query_data_from_readstorage();

        findCandidatesTiming.preprocessingBegin();
        //get candidate ids from minhasher
        set_candidate_ids(minhasher->getCandidates(fwdSequenceString, max_number_candidates));
        findCandidatesTiming.preprocessingEnd();
        if(candidateIds.size() == 0){
            //no need for further processing without candidates
            active = false;
        }else{
            findCandidatesTiming.executionBegin();
            //find unique candidate sequences
            make_unique_sequences();
            findCandidatesTiming.executionEnd();

            findCandidatesTiming.postprocessingBegin();
            //get reverse complements of unique candidate sequences
            fetch_revcompl_sequences_from_readstorage();
            findCandidatesTiming.postprocessingEnd();
        }
    }

    void fetch_query_data_from_readstorage(){
        fwdSequence = readStorage->fetchSequence_ptr(readId);
		if(canUseQualityScores){
			fwdQuality = readStorage->fetchQuality_ptr(readId);
		}
        fwdSequenceString = fwdSequence->toString();
    }

    void set_candidate_ids(std::vector<typename Minhasher_t::Result_t>&& minhashResults){
        candidateIds = std::move(minhashResults);

        //remove self from candidates
        candidateIds.erase(std::find(candidateIds.begin(), candidateIds.end(), readId));

        set_number_of_sequences(candidateIds.size());
        set_number_of_unique_sequences(candidateIds.size());

        for(std::size_t k = 0; k < activeCandidates.size(); k++)
            activeCandidates[k] = false;
    }

    void make_unique_sequences(){
        //vector
        auto& numseqpairs = make_unique_sequences_numseqpairs;
        numseqpairs.clear();
        numseqpairs.reserve(candidateIds.size()*1.3);

        //std::chrono::time_point < std::chrono::system_clock > t1 =
        //        std::chrono::system_clock::now();
        for(const auto& id : candidateIds){
            numseqpairs.emplace_back(id, readStorage->fetchSequence_ptr(id));
        }
        //std::chrono::time_point < std::chrono::system_clock > t2 =
        //        std::chrono::system_clock::now();
        //mapminhashresultsfetch += (t2 - t1);

        //t1 = std::chrono::system_clock::now();

        //sort pairs by sequence
        findCandidatesTiming.h2dBegin();
        std::sort(numseqpairs.begin(), numseqpairs.end(), [](const auto& l, const auto& r){return l.second < r.second;});
        findCandidatesTiming.h2dEnd();

        findCandidatesTiming.d2hBegin();
        std::uint64_t n_unique_elements = 1;
        candidateCountsPrefixSum[0] = 0;
        candidateCountsPrefixSum[1] = 1;
        candidateIds[0] = numseqpairs[0].first;
        fwdSequences[0] = numseqpairs[0].second;

        const Sequence_t* prevSeq = numseqpairs[0].second;

        for (std::size_t k = 1; k < numseqpairs.size(); k++) {
            const auto& pair = numseqpairs[k];
            candidateIds[k] = pair.first;
            const Sequence_t* curSeq = pair.second;
            if (prevSeq == curSeq) {
                candidateCountsPrefixSum[n_unique_elements]++;
            }else {
                candidateCountsPrefixSum[n_unique_elements+1] = 1 + candidateCountsPrefixSum[n_unique_elements];
                fwdSequences[n_unique_elements] = curSeq;
                n_unique_elements++;
            }
            prevSeq = curSeq;
        }
        findCandidatesTiming.d2hEnd();

        set_number_of_unique_sequences(n_unique_elements);
        n_unique_candidates = fwdSequences.size();
        n_candidates = candidateIds.size();

        assert(candidateCountsPrefixSum.back() == int(candidateIds.size()));
    }

    void fetch_revcompl_sequences_from_readstorage(){
        for(std::size_t k = 0; k < fwdSequences.size(); k++){
            int first = candidateCountsPrefixSum[k];
            revcomplSequences[k] = readStorage->fetchReverseComplementSequence_ptr(candidateIds[first]);
        }
    }

    void determine_good_alignments(){
        determine_good_alignments(0, fwdSequences.size());
    }

    /*
        Processes at most N alignments from
        alignments[firstIndex]
        to
        alignments[min(firstIndex + N-1, number of alignments - 1)]
    */
    void determine_good_alignments(int firstIndex, int N){

		// Given AlignmentResults for a read and its reverse complement, find the "best" of both alignments
		auto get_best_alignment = [this](const AlignmentResult_t& fwdAlignment,
									 const AlignmentResult_t& revcmplAlignment,
									 int querylength,
									 int candidatelength) -> BestAlignment_t{
			const int overlap = fwdAlignment.get_overlap();
			const int revcomploverlap = revcmplAlignment.get_overlap();
			const int fwdMismatches = fwdAlignment.get_nOps();
			const int revcmplMismatches = revcmplAlignment.get_nOps();

			BestAlignment_t retval = BestAlignment_t::None;

			const int minimumOverlap = int(querylength * goodAlignmentProperties.min_overlap_ratio) > goodAlignmentProperties.min_overlap
                            ? int(querylength * goodAlignmentProperties.min_overlap_ratio) : goodAlignmentProperties.min_overlap;

			//find alignment with lowest mismatch ratio. if both have same ratio choose alignment with longer overlap

			if(fwdAlignment.get_isValid() && overlap >= minimumOverlap){
				if(revcmplAlignment.get_isValid() && revcomploverlap >= minimumOverlap){
					const double ratio = (double)fwdMismatches / overlap;
					const double revcomplratio = (double)revcmplMismatches / revcomploverlap;

					if(ratio < revcomplratio){
						if(ratio < goodAlignmentProperties.maxErrorRate){
							retval = BestAlignment_t::Forward;
						}
					}else if(revcomplratio < ratio){
						if(revcomplratio < goodAlignmentProperties.maxErrorRate){
							retval = BestAlignment_t::ReverseComplement;
						}
					}else{
						if(ratio < goodAlignmentProperties.maxErrorRate){
							// both have same mismatch ratio, choose longest overlap
							if(overlap > revcomploverlap){
								retval = BestAlignment_t::Forward;
							}else{
								retval = BestAlignment_t::ReverseComplement;
							}
						}
					}
				}else{
					if((double)fwdMismatches / overlap < goodAlignmentProperties.maxErrorRate){
						retval = BestAlignment_t::Forward;
					}
				}
			}else{
				if(revcmplAlignment.get_isValid() && revcomploverlap >= minimumOverlap){
					if((double)revcmplMismatches / revcomploverlap < goodAlignmentProperties.maxErrorRate){
						retval = BestAlignment_t::ReverseComplement;
					}
				}
			}

			return retval;
		};


        const int querylength = fwdSequence->length();
        const int lastIndex_excl = std::min(std::size_t(firstIndex + N), fwdSequences.size());

        for(int i = firstIndex; i < lastIndex_excl; i++){
            const auto& res = fwdAlignments[i];
            const auto& revcomplres = revcomplAlignments[i];
            const int candidatelength = fwdSequences[i]->length();

            BestAlignment_t bestAlignment = get_best_alignment(res,
														revcomplres, querylength, candidatelength);

            if(bestAlignment == BestAlignment_t::None){
                //both alignments are bad, cannot use this candidate for correction
                activeCandidates[i] = false;
            }else{
                const double mismatchratio = [&](){
                    if(bestAlignment == BestAlignment_t::Forward)
                        return double(res.get_nOps()) / double(res.get_overlap());
                    else
                        return double(revcomplres.get_nOps()) / double(revcomplres.get_overlap());
                }();
                const int candidateCount = candidateCountsPrefixSum[i+1] - candidateCountsPrefixSum[i];
                if(mismatchratio >= 4 * mismatchratioBaseFactor){
                    //best alignments is still not good enough, cannot use this candidate for correction
                    activeCandidates[i] = false;
                }else{
                    activeCandidates[i] = true;

                    if (mismatchratio < 2 * mismatchratioBaseFactor) {
                        counts[0] += candidateCount;
                    }
                    if (mismatchratio < 3 * mismatchratioBaseFactor) {
                        counts[1] += candidateCount;
                    }
                    if (mismatchratio < 4 * mismatchratioBaseFactor) {
                        counts[2] += candidateCount;
                    }

                    const int begin = candidateCountsPrefixSum[i];
                    if(bestAlignment == BestAlignment_t::Forward){
                        bestIsForward[i] = true;
                        bestSequences[i] = fwdSequences[i];
                        bestAlignments[i] = &fwdAlignments[i];

						if(canUseQualityScores){
							for(int j = 0; j < candidateCount; j++){
								const ReadId_t id = candidateIds[begin + j];
								bestQualities[begin + j] = readStorage->fetchQuality_ptr(id);
							}
						}
                    }else{
                        bestIsForward[i] = false;
                        bestSequences[i] = revcomplSequences[i];
                        bestAlignments[i] = &revcomplAlignments[i];

						if(canUseQualityScores){
							for(int j = 0; j < candidateCount; j++){
								const ReadId_t id = candidateIds[begin + j];
								bestQualities[begin + j] = readStorage->fetchReverseComplementQuality_ptr(id);
							}
						}
                    }
                }
            }
        }
    }

    bool hasEnoughGoodCandidates() const{
        if (counts[0] >= goodAlignmentsCountThreshold
            || counts[1] >= goodAlignmentsCountThreshold
            || counts[2] >= goodAlignmentsCountThreshold)

            return true;

        return false;
    }

    void prepare_good_candidates(){
        DetermineGoodAlignmentStats stats;

        mismatchratioThreshold = 0;
        if (counts[0] >= goodAlignmentsCountThreshold) {
            mismatchratioThreshold = 2 * mismatchratioBaseFactor;
            stats.correctionCases[0]++;
        } else if (counts[1] >= goodAlignmentsCountThreshold) {
            mismatchratioThreshold = 3 * mismatchratioBaseFactor;
            stats.correctionCases[1]++;
        } else if (counts[2] >= goodAlignmentsCountThreshold) {
            mismatchratioThreshold = 4 * mismatchratioBaseFactor;
            stats.correctionCases[2]++;
        } else { //no correction possible
            stats.correctionCases[3]++;
            active = false;
            return;
        }

        std::size_t activeposition_unique = 0;
        std::size_t activeposition = 0;

        //stable_partition with condition (activeCandidates[i] && notremoved) ?
        for(std::size_t i = 0; i < activeCandidates.size(); i++){
            if(activeCandidates[i]){
                const double mismatchratio = double(bestAlignments[i]->get_nOps()) / double(bestAlignments[i]->get_overlap());
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
						if(canUseQualityScores){
							bestQualities[activeposition] = bestQualities[begin + j];
						}
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

}
#endif
