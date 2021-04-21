#ifndef READ_EXTENDER_HPP
#define READ_EXTENDER_HPP



#include <config.hpp>
#include <sequencehelpers.hpp>
#include <cpuminhasher.hpp>
#include <cpureadstorage.hpp>
#include <options.hpp>
#include <cpu_alignment.hpp>
#include <bestalignment.hpp>
#include <msa.hpp>

#include <hpc_helpers.cuh>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>
#include <iostream>
#include <string>
#include <memory>
#include <mutex>
#include <numeric>
#include <limits>

#include <readextension_cpu.hpp>
#include <extensionresultprocessing.hpp>
#include <rangegenerator.hpp>
#include <threadpool.hpp>
#include <memoryfile.hpp>
#include <util.hpp>
#include <filehelpers.hpp>


namespace care{

enum class ExtensionDirection {LR, RL};

enum class AbortReason{
    MsaNotExtended, 
    NoPairedCandidates, 
    NoPairedCandidatesAfterAlignment, 
    PairedAnchorFinished,
    None
};

struct ExtendInput{
    int readLength1{};
    int readLength2{};
    read_number readId1{};
    read_number readId2{};
    std::vector<unsigned int> encodedRead1{};
    std::vector<unsigned int> encodedRead2{};
};

struct ExtendResult{
    bool mateHasBeenFound = false;
    bool success = false;
    bool aborted = false;
    int numIterations = 0;
    int originalLength = 0;
    int originalMateLength = 0;
    int read1begin = 0;
    int read2begin = 0;
    ExtensionDirection direction = ExtensionDirection::LR;
    AbortReason abortReason = AbortReason::None;

    read_number readId1{}; //same as input ids
    read_number readId2{}; //same as input ids

    std::string extendedRead{};
    

    read_number getReadPairId() const noexcept{
        return readId1 / 2;
    }
};

struct ReadExtenderBase{
public:

    

    struct ReadPairIds{
        read_number first;
        read_number second;
    };   

    ReadExtenderBase(
        int insertSize,
        int insertSizeStddev,
        int maxextensionPerStep,
        int maximumSequenceLength,
        const CorrectionOptions& coropts,
        const GoodAlignmentProperties& gap        
    ) : insertSize(insertSize), 
        insertSizeStddev(insertSizeStddev),
        maxextensionPerStep(maxextensionPerStep),
        maximumSequenceLength(maximumSequenceLength),
        correctionOptions(coropts),
        goodAlignmentProperties(gap),
        hashTimer{"Hash timer"},
        collectTimer{"Collect timer"},
        alignmentTimer{"Alignment timer"},
        alignmentFilterTimer{"Alignment filter timer"},
        msaTimer{"MSA timer"}
    {

        encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength);
        decodedSequencePitchInBytes = maximumSequenceLength;
        qualityPitchInBytes = maximumSequenceLength;


    }

    virtual ~ReadExtenderBase() {}


    

    std::vector<ExtendResult> extendPairedReadBatch(
        const std::vector<ExtendInput>& inputs
    );

    std::vector<ExtendResult> extendSingleEndReadBatch(
        const std::vector<ExtendInput>& inputs
    );


    void printTimers(){
        hashTimer.print();
        collectTimer.print();
        alignmentTimer.print();
        alignmentFilterTimer.print();
        msaTimer.print();
    }


    struct Task{
        bool dataIsAvailable = false;
        bool pairedEnd = false;
        bool abort = false;
        bool mateHasBeenFound = false;
        bool mateRemovedFromCandidates = false;
        AbortReason abortReason = AbortReason::None;
        int id = 0;
        int myLength = 0;
        int currentAnchorLength = 0;
        int accumExtensionLengths = 0;
        int iteration = 0;
        int mateLength = 0;
        int numRemainingCandidates = 0;
        int splitDepth = 0;
        ExtensionDirection direction{};
        read_number myReadId = 0;
        read_number mateReadId = 0;
        read_number currentAnchorReadId = 0;
        std::string decodedMate;
        std::string decodedMateRevC;
        std::string resultsequence;
        std::vector<read_number> candidateReadIds;
        std::vector<read_number>::iterator mateIdLocationIter{};
        std::vector<unsigned int> currentAnchor;
        std::vector<unsigned int> encodedMate;
        std::vector<int> candidateSequenceLengths;
        std::vector<unsigned int> candidateSequencesFwdData;
        std::vector<unsigned int> candidateSequencesRevcData;
        std::vector<unsigned int> candidateSequenceData;
        std::vector<care::cpu::SHDResult> alignments;
        std::vector<BestAlignment_t> alignmentFlags;
        std::vector<std::string> totalDecodedAnchors;
        std::vector<int> totalAnchorBeginInExtendedRead;
        std::vector<std::vector<read_number>> usedCandidateReadIdsPerIteration;
        std::vector<std::vector<care::cpu::SHDResult>> usedAlignmentsPerIteration;
        std::vector<std::vector<BestAlignment_t>> usedAlignmentFlagsPerIteration;
        std::vector<read_number> allUsedCandidateReadIdPairs; //sorted
        std::vector<char> candidateStrings;
        std::vector<int> candidateShifts;
        std::vector<float> candidateOverlapWeights;

        bool operator==(const Task& rhs) const noexcept{
            #if 1
                if(pairedEnd != rhs.pairedEnd) std::cerr << "pairedEnd differs\n";
                if(abort != rhs.abort) std::cerr << "abort differs\n";
                if(mateHasBeenFound != rhs.mateHasBeenFound) std::cerr << "mateHasBeenFound differs\n";
                if(mateRemovedFromCandidates != rhs.mateRemovedFromCandidates) std::cerr << "mateRemovedFromCandidates differs\n";
                if(abortReason != rhs.abortReason) std::cerr << "abortReason differs\n";
                if(id != rhs.id) std::cerr << "id differs\n";
                if(myLength != rhs.myLength) std::cerr << "myLength differs\n";
                if(currentAnchorLength != rhs.currentAnchorLength) std::cerr << "currentAnchorLength differs\n";
                if(accumExtensionLengths != rhs.accumExtensionLengths) std::cerr << "accumExtensionLengths differs\n";
                if(iteration != rhs.iteration) std::cerr << "iteration differs\n";
                if(mateLength != rhs.mateLength) std::cerr << "mateLength differs\n";
                if(numRemainingCandidates != rhs.numRemainingCandidates) std::cerr << "numRemainingCandidates differs\n";
                if(splitDepth != rhs.splitDepth) std::cerr << "splitDepth differs\n";
                if(direction != rhs.direction) std::cerr << "direction differs\n";
                if(myReadId != rhs.myReadId) std::cerr << "myReadId differs\n";
                if(mateReadId != rhs.mateReadId) std::cerr << "mateReadId differs\n";
                if(currentAnchorReadId != rhs.currentAnchorReadId) std::cerr << "currentAnchorReadId differs\n";
                if(decodedMate != rhs.decodedMate) std::cerr << "decodedMate differs\n";
                if(decodedMateRevC != rhs.decodedMateRevC) std::cerr << "decodedMateRevC differs\n";
                if(resultsequence != rhs.resultsequence) std::cerr << "resultsequence differs\n";
                if(candidateReadIds != rhs.candidateReadIds) std::cerr << "candidateReadIds differs\n";
                if(mateIdLocationIter != rhs.mateIdLocationIter) std::cerr << "mateIdLocationIter differs\n";
                if(currentAnchor != rhs.currentAnchor) std::cerr << "currentAnchor differs\n";
                if(encodedMate != rhs.encodedMate) std::cerr << "encodedMate differs\n";
                if(candidateSequenceLengths != rhs.candidateSequenceLengths) std::cerr << "candidateSequenceLengths differs\n";
                if(candidateSequencesFwdData != rhs.candidateSequencesFwdData) std::cerr << "candidateSequencesFwdData differs\n";
                if(candidateSequencesRevcData != rhs.candidateSequencesRevcData) std::cerr << "candidateSequencesRevcData differs\n";
                if(candidateSequenceData != rhs.candidateSequenceData) std::cerr << "candidateSequenceData differs\n";
                if(alignments != rhs.alignments) std::cerr << "alignments differs\n";
                if(alignmentFlags != rhs.alignmentFlags) std::cerr << "alignmentFlags differs\n";
                if(totalDecodedAnchors != rhs.totalDecodedAnchors) std::cerr << "totalDecodedAnchors differs\n";
                if(totalAnchorBeginInExtendedRead != rhs.totalAnchorBeginInExtendedRead) std::cerr << "totalAnchorBeginInExtendedRead differs\n";
                if(usedCandidateReadIdsPerIteration != rhs.usedCandidateReadIdsPerIteration) std::cerr << "usedCandidateReadIdsPerIteration differs\n";
                if(usedAlignmentsPerIteration != rhs.usedAlignmentsPerIteration) std::cerr << "usedAlignmentsPerIteration differs\n";
                if(usedAlignmentFlagsPerIteration != rhs.usedAlignmentFlagsPerIteration) std::cerr << "usedAlignmentFlagsPerIteration differs\n";
                if(allUsedCandidateReadIdPairs != rhs.allUsedCandidateReadIdPairs) std::cerr << "allUsedCandidateReadIdPairs differs\n";
                if(candidateStrings != rhs.candidateStrings) std::cerr << "candidateStrings differs\n";
                if(candidateShifts != rhs.candidateShifts) std::cerr << "candidateShifts differs\n";
                if(candidateOverlapWeights != rhs.candidateOverlapWeights) std::cerr << "candidateOverlapWeights differs\n";
            #endif
            if(pairedEnd != rhs.pairedEnd) return false;
            if(abort != rhs.abort) return false;
            if(mateHasBeenFound != rhs.mateHasBeenFound) return false;
            if(mateRemovedFromCandidates != rhs.mateRemovedFromCandidates) return false;
            if(abortReason != rhs.abortReason) return false;
            if(id != rhs.id) return false;
            if(myLength != rhs.myLength) return false;
            if(currentAnchorLength != rhs.currentAnchorLength) return false;
            if(accumExtensionLengths != rhs.accumExtensionLengths) return false;
            if(iteration != rhs.iteration) return false;
            if(mateLength != rhs.mateLength) return false;
            if(numRemainingCandidates != rhs.numRemainingCandidates) return false;
            if(splitDepth != rhs.splitDepth) return false;
            if(direction != rhs.direction) return false;
            if(myReadId != rhs.myReadId) return false;
            if(mateReadId != rhs.mateReadId) return false;
            if(currentAnchorReadId != rhs.currentAnchorReadId) return false;
            if(decodedMate != rhs.decodedMate) return false;
            if(decodedMateRevC != rhs.decodedMateRevC) return false;
            if(resultsequence != rhs.resultsequence) return false;
            if(candidateReadIds != rhs.candidateReadIds) return false;
            if(mateIdLocationIter != rhs.mateIdLocationIter) return false;
            if(currentAnchor != rhs.currentAnchor) return false;
            if(encodedMate != rhs.encodedMate) return false;
            if(candidateSequenceLengths != rhs.candidateSequenceLengths) return false;
            if(candidateSequencesFwdData != rhs.candidateSequencesFwdData) return false;
            if(candidateSequencesRevcData != rhs.candidateSequencesRevcData) return false;
            if(candidateSequenceData != rhs.candidateSequenceData) return false;
            if(alignments != rhs.alignments) return false;
            if(alignmentFlags != rhs.alignmentFlags) return false;
            if(totalDecodedAnchors != rhs.totalDecodedAnchors) return false;
            if(totalAnchorBeginInExtendedRead != rhs.totalAnchorBeginInExtendedRead) return false;
            if(usedCandidateReadIdsPerIteration != rhs.usedCandidateReadIdsPerIteration) return false;
            if(usedAlignmentsPerIteration != rhs.usedAlignmentsPerIteration) return false;
            if(usedAlignmentFlagsPerIteration != rhs.usedAlignmentFlagsPerIteration) return false;
            if(allUsedCandidateReadIdPairs != rhs.allUsedCandidateReadIdPairs) return false;
            if(candidateStrings != rhs.candidateStrings) return false;
            if(candidateShifts != rhs.candidateShifts) return false;
            if(candidateOverlapWeights != rhs.candidateOverlapWeights) return false;

            return true;
        }

        bool operator!=(const Task& rhs) const noexcept{
            return !operator==(rhs);
        }
        

        bool isActive(int insertSize, int insertSizeStddev) const noexcept{
            return (iteration < insertSize 
                && accumExtensionLengths < insertSize - (mateLength) + insertSizeStddev
                && !abort 
                && !mateHasBeenFound);
        }

        void reset(){
            auto clear = [](auto& vec){vec.clear();};

            dataIsAvailable = false;
            pairedEnd = false;
            abort = false;
            mateHasBeenFound = false;
            mateRemovedFromCandidates = false;
            abortReason = AbortReason::None;
            id = 0;
            myLength = 0;
            currentAnchorLength = 0;
            accumExtensionLengths = 0;
            iteration = 0;
            mateLength = 0;
            direction = ExtensionDirection::LR;
            myReadId = 0;
            mateReadId = 0;
            currentAnchorReadId = 0;
            numRemainingCandidates = 0;
            splitDepth = 0;

            clear(decodedMate);
            clear(decodedMateRevC);
            clear(resultsequence);
            clear(candidateReadIds);
            mateIdLocationIter = candidateReadIds.end();
            clear(currentAnchor);
            clear(encodedMate);
            clear(candidateSequenceLengths);
            clear(candidateSequencesFwdData);
            clear(candidateSequencesRevcData);
            clear(candidateSequenceData);
            clear(alignments);
            clear(alignmentFlags);
            clear(totalDecodedAnchors);
            clear(totalAnchorBeginInExtendedRead);
            clear(usedCandidateReadIdsPerIteration);
            clear(usedAlignmentsPerIteration);
            clear(usedAlignmentFlagsPerIteration);
            clear(allUsedCandidateReadIdPairs);
            clear(candidateStrings);
            clear(candidateShifts);
            clear(candidateOverlapWeights);
        }
    };

    static Task makePairedEndTask(ExtendInput input, ExtensionDirection direction){
        if(direction == ExtensionDirection::LR){
            Task task;
            task.pairedEnd = true;
            task.direction = direction;

            task.currentAnchor = std::move(input.encodedRead1);
            task.encodedMate = std::move(input.encodedRead2);

            task.currentAnchorLength = input.readLength1;
            task.currentAnchorReadId = input.readId1;
            task.accumExtensionLengths = 0;
            task.iteration = 0;

            task.myLength = input.readLength1;
            task.myReadId = input.readId1;

            task.mateLength = input.readLength2;
            task.mateReadId = input.readId2;

            task.decodedMate.resize(task.mateLength);
            SequenceHelpers::decode2BitSequence(
                task.decodedMate.data(),
                task.encodedMate.data(),
                task.mateLength
            );

            task.decodedMateRevC = SequenceHelpers::reverseComplementSequenceDecoded(task.decodedMate.data(), task.mateLength);

            task.resultsequence.resize(input.readLength1);
            SequenceHelpers::decode2BitSequence(
                task.resultsequence.data(),
                task.currentAnchor.data(),
                task.myLength
            );

            return task;
        }else if(direction == ExtensionDirection::RL){
            Task task;
            task.pairedEnd = true;
            task.direction = direction;

            task.currentAnchor = std::move(input.encodedRead2);
            task.encodedMate = std::move(input.encodedRead1);

            task.currentAnchorLength = input.readLength2;
            task.currentAnchorReadId = input.readId2;
            task.accumExtensionLengths = 0;
            task.iteration = 0;

            task.myLength = input.readLength2;
            task.myReadId = input.readId2;

            task.mateLength = input.readLength1;
            task.mateReadId = input.readId1;

            task.decodedMate.resize(input.readLength1);
            SequenceHelpers::decode2BitSequence(
                task.decodedMate.data(),
                task.encodedMate.data(),
                task.mateLength
            );
            task.decodedMateRevC = SequenceHelpers::reverseComplementSequenceDecoded(task.decodedMate.data(), task.mateLength);

            task.resultsequence.resize(input.readLength2);
            SequenceHelpers::decode2BitSequence(
                task.resultsequence.data(),
                task.currentAnchor.data(),
                task.myLength
            );

            return task;
        }else{
            assert(false);
            return Task{};
        }
    }


    static Task makeSingleEndTask(ExtendInput input, ExtensionDirection direction){
        if(direction == ExtensionDirection::LR){
            Task task;
            task.pairedEnd = false;
            task.direction = direction;

            task.currentAnchor = std::move(input.encodedRead1);

            task.currentAnchorLength = input.readLength1;
            task.currentAnchorReadId = input.readId1;
            task.accumExtensionLengths = 0;
            task.iteration = 0;

            task.myLength = input.readLength1;
            task.myReadId = input.readId1;

            task.mateLength = 0;
            task.mateReadId = std::numeric_limits<read_number>::max();

            task.resultsequence.resize(input.readLength1);
            SequenceHelpers::decode2BitSequence(
                task.resultsequence.data(),
                task.currentAnchor.data(),
                task.myLength
            );

            return task;
        }else if(direction == ExtensionDirection::RL){
            
            Task task;
            task.pairedEnd = false;
            task.direction = direction;

            task.currentAnchor = std::move(input.encodedRead1);

            //to extend a single-end read to the left, its reverse complement will be extended to the right
            SequenceHelpers::reverseComplementSequenceInplace2Bit(task.currentAnchor.data(), input.readLength1);

            task.currentAnchorLength = input.readLength1;
            task.currentAnchorReadId = input.readId1;
            task.accumExtensionLengths = 0;
            task.iteration = 0;

            task.myLength = input.readLength1;
            task.myReadId = input.readId1;

            task.mateLength = 0;
            task.mateReadId = std::numeric_limits<read_number>::max();

            task.resultsequence.resize(input.readLength1);
            SequenceHelpers::decode2BitSequence(
                task.resultsequence.data(),
                task.currentAnchor.data(),
                task.myLength
            );

            return task;
        }else{
            assert(false);
            return Task{};
        }
    }

#if 0
    struct SingleEndTask{
        bool abort = false;
        AbortReason abortReason = AbortReason::None;
        int currentAnchorLength = 0;
        int myReadLength = 0;
        int accumExtensionLengths = 0;
        int iteration = 0;
        ExtensionDirection direction{};
        read_number myReadId = 0;
        read_number currentAnchorReadId = 0;
        std::vector<read_number> candidateReadIds;
        std::vector<read_number>::iterator mateIdLocationIter{};
        std::vector<unsigned int> currentAnchor;
        std::vector<int> candidateSequenceLengths;
        std::vector<unsigned int> candidateSequencesFwdData;
        std::vector<unsigned int> candidateSequencesRevcData;
        std::vector<care::cpu::SHDResult> alignments;
        std::vector<BestAlignment_t> alignmentFlags;
        std::vector<std::string> totalDecodedAnchors;
        std::vector<int> totalAnchorBeginInExtendedRead;
        std::vector<std::vector<read_number>> usedCandidateReadIdsPerIteration;
        std::vector<std::vector<care::cpu::SHDResult>> usedAlignmentsPerIteration;
        std::vector<std::vector<BestAlignment_t>> usedAlignmentFlagsPerIteration;
        std::vector<read_number> allUsedCandidateReadIdPairs; //sorted

        bool isActive(int insertSize, int insertSizeStddev) const noexcept{
            return (iteration < insertSize 
                //TODO && accumExtensionLengths < insertSize - (mateLength) + insertSizeStddev
                && !abort);
        }

        void reset(){
            auto clear = [](auto& vec){vec.clear();};

            abort = false;
            abortReason = AbortReason::None;
            myReadLength = 0;
            currentAnchorLength = 0;
            accumExtensionLengths = 0;
            iteration = 0;
            direction = ExtensionDirection::LR;
            myReadId = 0;
            currentAnchorReadId = 0;
            
            clear(candidateReadIds);
            mateIdLocationIter = candidateReadIds.end();
            clear(currentAnchor);
            clear(candidateSequenceLengths);
            clear(candidateSequencesFwdData);
            clear(candidateSequencesRevcData);
            clear(alignments);
            clear(alignmentFlags);
            clear(totalDecodedAnchors);
            clear(totalAnchorBeginInExtendedRead);
            clear(usedCandidateReadIdsPerIteration);
            clear(usedAlignmentsPerIteration);
            clear(usedAlignmentFlagsPerIteration);
            clear(allUsedCandidateReadIdPairs);
        }
    };

    SingleEndTask makeSingleEndTask(const ExtendInput& input, ExtensionDirection direction){
        SingleEndTask task;
        task.direction = direction;

        task.currentAnchor.resize(input.numInts1);
        std::copy_n(input.encodedRead1, input.numInts1, task.currentAnchor.begin());

        task.currentAnchorLength = input.readLength1;
        task.currentAnchorReadId = input.readId1;
        task.accumExtensionLengths = 0;
        task.iteration = 0;

        task.myReadLength = input.readLength1;
        task.myReadId = input.readId1;

        return task;
    }
#endif

    static std::vector<ExtendResult> combinePairedEndDirectionResults(
        std::vector<ExtendResult>& lr_and_rl,
        int insertSize,
        int insertSizeStddev
    );

    static std::vector<ExtendResult> combinePairedEndDirectionResults2(
        std::vector<ExtendResult>& lr_and_rl,
        int insertSize,
        int insertSizeStddev
    );

    static std::vector<ExtendResult> combinePairedEndDirectionResults4(
        std::vector<ExtendResult>& lr_and_rl,
        int insertSize,
        int insertSizeStddev
    );

    std::vector<ExtendResult> combinePairedEndDirectionResults(
        std::vector<ExtendResult>& lr_and_rl
    );

    std::vector<ExtendResult> combinePairedEndDirectionResults(
        std::vector<ExtendResult>& lr,
        std::vector<ExtendResult>& rl
    );

    std::vector<ExtendResult> combineSingleEndDirectionResults(
        std::vector<ExtendResult>& lr,
        std::vector<ExtendResult>& rl,
        const std::vector<Task>& tasks
    );

    bool isSameReadPair(read_number id1, read_number id2) const noexcept{
        //read pairs have consecutive read ids. (0,1) (2,3) ...
        //map to the smaller id within a pair, and compare those

        const auto firstId1 = id1 - id1 % 2;
        const auto firstId2 = id2 - id2 % 2;

        return firstId1 == firstId2;
    }

    bool isSameReadPair(ReadPairIds ids1, ReadPairIds ids2) const noexcept{
        assert(isSameReadPair(ids1.first, ids1.second));
        assert(isSameReadPair(ids2.first, ids2.second));
        
        return isSameReadPair(ids1.first, ids2.first);
    }

    template<class InputIt1, class InputIt2,
         class OutputIt1, class OutputIt2,
         class Compare>
    std::pair<OutputIt1, OutputIt2> filterIdsByMate1(
        InputIt1 first1, 
        InputIt1 last1,
        InputIt2 first2, 
        InputIt2 last2,
        OutputIt1 d_first1, 
        OutputIt2 d_first2, 
        Compare isSameReadPair
    ) const noexcept {
        while (first1 != last1 && first2 != last2) {
            const auto nextfirst1 = std::next(first1);
            const auto nextfirst2 = std::next(first2);

            int elems1 = 1;
            if(nextfirst1 != last1){
                //if equal(*first1, *nextfirst1)
                if(isSameReadPair(*first1, *nextfirst1)){
                    elems1 = 2;
                }
            }

            int elems2 = 1;
            if(nextfirst2 != last2){
                //if equal(*first2, *nextfirst2)
                if(isSameReadPair(*first2, *nextfirst2)){
                    elems2 = 2;
                }
            }

            if(elems1 == 1 && elems2 == 1){
                if(isSameReadPair(*first1, *first2)){
                    if(*first1 != *first2){
                        *d_first1++ = *first1++;
                        *d_first2++ = *first2++;
                    }else{
                        ++first1;
                        ++first2;
                    }
                }else{
                    if(*first1 < *first2){
                        ++first1;
                    }else{
                        ++first2;
                    }
                }
            }else if (elems1 == 2 && elems2 == 2){
                if(isSameReadPair(*first1, *first2)){
                    *d_first1++ = *first1++;
                    *d_first2++ = *first2++;
                    *d_first1++ = *first1++;
                    *d_first2++ = *first2++;
                }else{
                    if(*first1 < *first2){
                        ++first1;
                        ++first1;
                    }else{
                        ++first2;
                        ++first2;
                    }
                }

            }else if (elems1 == 2 && elems2 == 1){
                if(isSameReadPair(*first1, *first2)){
                    if(*first1 == *first2){
                        //discard first entry of first range, keep rest
                        ++first1;
                        *d_first1++ = *first1++;
                        *d_first2++ = *first2++;
                    }else{
                        //keep first entry of first range, discard second entry
                        *d_first1++ = *first1++;
                        *d_first2++ = *first2++;
                        ++first1;
                    }
                }else{
                    if(*first1 < *first2){
                        ++first1;
                        ++first1;
                    }else{
                        ++first2;
                    }
                }
                
            }else {
                //(elems1 == 1 && elems2 == 2)

                if(isSameReadPair(*first1, *first2)){
                    if(*first1 == *first2){
                        //discard first entry of first range, keep rest
                        ++first2;
                        *d_first1++ = *first1++;
                        *d_first2++ = *first2++;
                    }else{
                        //keep first entry of first range, discard second entry
                        *d_first1++ = *first1++;
                        *d_first2++ = *first2++;
                        ++first2;
                    }
                }else{
                    if(*first1 < *first2){
                        ++first1;
                    }else{
                        ++first2;
                        ++first2;
                    }
                }            
            }
        }
        return std::make_pair(d_first1, d_first2);
    }

    /*
        Given candidate read ids for read A and read B which are the read pair (A,B),
        remove candidate read ids of list A which have no mate in list B, and vice-versa
    */
    std::vector<ReadPairIds> removeCandidateIdsWithoutMate(
        const std::vector<read_number>& candidateIdsA,
        const std::vector<read_number>& candidateIdsB        
    ) const {
        std::vector<read_number> outputCandidateIdsA;
        std::vector<read_number> outputCandidateIdsB;

        outputCandidateIdsA.resize(candidateIdsA.size());
        outputCandidateIdsB.resize(candidateIdsB.size());

        auto isSame = [this](read_number id1, read_number id2){
            return isSameReadPair(id1, id2);
        };

        auto endIterators = filterIdsByMate1(
            candidateIdsA.begin(), 
            candidateIdsA.end(),
            candidateIdsB.begin(), 
            candidateIdsB.end(),
            outputCandidateIdsA.begin(), 
            outputCandidateIdsB.begin(), 
            isSame
        );

        outputCandidateIdsA.erase(endIterators.first, outputCandidateIdsA.end());
        outputCandidateIdsB.erase(endIterators.second, outputCandidateIdsB.end());

        assert(outputCandidateIdsA.size() == outputCandidateIdsB.size());

        std::vector<ReadPairIds> returnValue;

        const int numIds = outputCandidateIdsA.size();
        returnValue.resize(numIds);

        for(int i = 0; i < numIds; i++){
            if(outputCandidateIdsA[i] != outputCandidateIdsB[i]){
                returnValue[i].first = outputCandidateIdsA[i];
                returnValue[i].second = outputCandidateIdsB[i];

                assert(isSameReadPair(returnValue[i].first, returnValue[i].second));
            }else{
                //if both ids are equal, it must be a pair in both A and B. reverse pair of B to avoid a readIdPair with two equal ids
                returnValue[i].first = outputCandidateIdsA[i];
                returnValue[i].second = outputCandidateIdsB[i+1];
                returnValue[i+1].first = outputCandidateIdsA[i+1];
                returnValue[i+1].second = outputCandidateIdsB[i];

                assert(isSameReadPair(returnValue[i].first, returnValue[i].second));
                assert(isSameReadPair(returnValue[i+1].first, returnValue[i+1].second));
                assert(isSameReadPair(returnValue[i], returnValue[i+1]));

                i++;
            }
        }

        return returnValue;
    }


    virtual std::vector<ExtendResult> processPairedEndTasks(
        std::vector<Task> tasks
    ) = 0;

    virtual std::vector<ExtendResult> processSingleEndTasks(
        std::vector<Task> tasks
    ) = 0;


    int insertSize{};
    int insertSizeStddev{};
    int maxextensionPerStep{};
    int maximumSequenceLength{};
    std::size_t encodedSequencePitchInInts{};
    std::size_t decodedSequencePitchInBytes{};
    std::size_t qualityPitchInBytes{};

    CorrectionOptions correctionOptions{};
    GoodAlignmentProperties goodAlignmentProperties{};

    helpers::CpuTimer hashTimer{};
    helpers::CpuTimer collectTimer{};
    helpers::CpuTimer alignmentTimer{};
    helpers::CpuTimer alignmentFilterTimer{};
    helpers::CpuTimer msaTimer{};
};


using ReadExtender = ReadExtenderBase;








__inline__
std::string to_string(AbortReason r){
    using ar = AbortReason;

    switch(r){
        case ar::MsaNotExtended: return "MsaNotExtended";
        case ar::NoPairedCandidates: return "NoPairedCandidates";
        case ar::NoPairedCandidatesAfterAlignment: return "NoPairedCandidatesAfterAlignment";
        default: return "None";
    }
}

__inline__
std::string to_string(ExtensionDirection r){
    using ar = ExtensionDirection;

    switch(r){
        case ar::LR: return "LR";
        case ar::RL: return "RL";
    }

    return "INVALID ENUM VALUE to_string(ExtensionDirection)";
}



}


#endif