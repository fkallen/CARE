#ifndef READ_EXTENDER_HPP
#define READ_EXTENDER_HPP

#if 0

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

#include <readextender_common.hpp>

namespace care{



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


    

    std::vector<extension::ExtendResult> extendPairedReadBatch(
        const std::vector<extension::ExtendInput>& inputs
    );

    std::vector<extension::ExtendResult> extendSingleEndReadBatch(
        const std::vector<extension::ExtendInput>& inputs
    );


    void printTimers(){
        hashTimer.print();
        collectTimer.print();
        alignmentTimer.print();
        alignmentFilterTimer.print();
        msaTimer.print();
    }


    static extension::Task makePairedEndTask(extension::ExtendInput input, extension::ExtensionDirection direction){
        if(direction == extension::ExtensionDirection::LR){
            extension::Task task;
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
        }else if(direction == extension::ExtensionDirection::RL){
            extension::Task task;
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
            return extension::Task{};
        }
    }


    static extension::Task makeSingleEndTask(extension::ExtendInput input, extension::ExtensionDirection direction){
        if(direction == extension::ExtensionDirection::LR){
            extension::Task task;
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
        }else if(direction == extension::ExtensionDirection::RL){
            
            extension::Task task;
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
            return extension::Task{};
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

    static std::vector<extension::ExtendResult> combinePairedEndDirectionResults(
        std::vector<extension::ExtendResult>& lr_and_rl,
        int insertSize,
        int insertSizeStddev
    );

    static std::vector<extension::ExtendResult> combinePairedEndDirectionResults2(
        std::vector<extension::ExtendResult>& lr_and_rl,
        int insertSize,
        int insertSizeStddev
    );

    static std::vector<extension::ExtendResult> combinePairedEndDirectionResults4(
        std::vector<extension::ExtendResult>& lr_and_rl,
        int insertSize,
        int insertSizeStddev
    );

    std::vector<extension::ExtendResult> combinePairedEndDirectionResults(
        std::vector<extension::ExtendResult>& lr_and_rl
    );

    std::vector<extension::ExtendResult> combinePairedEndDirectionResults(
        std::vector<extension::ExtendResult>& lr,
        std::vector<extension::ExtendResult>& rl
    );

    std::vector<extension::ExtendResult> combineSingleEndDirectionResults(
        std::vector<extension::ExtendResult>& lr,
        std::vector<extension::ExtendResult>& rl,
        const std::vector<extension::Task>& tasks
    );



    virtual std::vector<extension::ExtendResult> processPairedEndTasks(
        std::vector<extension::Task> tasks
    ) = 0;

    virtual std::vector<extension::ExtendResult> processSingleEndTasks(
        std::vector<extension::Task> tasks
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
std::string to_string(extension::AbortReason r){
    using ar = extension::AbortReason;

    switch(r){
        case ar::MsaNotExtended: return "MsaNotExtended";
        case ar::NoPairedCandidates: return "NoPairedCandidates";
        case ar::NoPairedCandidatesAfterAlignment: return "NoPairedCandidatesAfterAlignment";
        default: return "None";
    }
}

__inline__
std::string to_string(extension::ExtensionDirection r){
    using ar = extension::ExtensionDirection;

    switch(r){
        case ar::LR: return "LR";
        case ar::RL: return "RL";
    }

    return "INVALID ENUM VALUE to_string(ExtensionDirection)";
}



}


#endif


#endif