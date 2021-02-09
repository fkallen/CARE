#include <readextenderbase.hpp>
#include <config.hpp>

#include <cpu_alignment.hpp>
#include <cpuminhasher.hpp>
#include <cpureadstorage.hpp>
#include <options.hpp>
#include <bestalignment.hpp>

namespace care{

struct ReadExtenderCpu final : public ReadExtenderBase{
public:


    ReadExtenderCpu(
        int insertSize,
        int insertSizeStddev,
        int maxextensionPerStep,
        int maximumSequenceLength,
        const CpuReadStorage& rs, 
        const CpuMinhasher& mh,
        const CorrectionOptions& coropts,
        const GoodAlignmentProperties& gap        
    ) : ReadExtenderBase(insertSize, insertSizeStddev, maxextensionPerStep, maximumSequenceLength, coropts, gap),
        readStorage(&rs), minhasher(&mh), readStorageHandle{rs.makeHandle()}, minhashHandle{mh.makeQueryHandle()}{

    }

    ~ReadExtenderCpu(){
        readStorage->destroyHandle(readStorageHandle);
        //minhasher->destroyHandle(minhashHandle);
    }
     
private:

    void getCandidateReadIdsSingle(
        std::vector<read_number>& result, 
        const unsigned int* encodedRead, 
        int readLength, 
        read_number readId,
        int beginPos = 0 // only positions [beginPos, readLength] are hashed
    ){

        result.clear();

        bool containsN = false;
        readStorage->areSequencesAmbiguous(readStorageHandle, &containsN, &readId, 1);

        //exclude anchors with ambiguous bases
        if(!(correctionOptions.excludeAmbiguousReads && containsN)){

            int numValuesPerSequence = 0;
            int totalNumValues = 0;

            minhasher->determineNumValues(
                minhashHandle,encodedRead,
                encodedSequencePitchInInts,
                &readLength,
                1,
                &numValuesPerSequence,
                totalNumValues
            );

            result.resize(totalNumValues);
            std::array<int, 2> offsets{};

            minhasher->retrieveValues(
                minhashHandle,
                nullptr, //do not remove selfid
                1,
                totalNumValues,
                result.data(),
                &numValuesPerSequence,
                offsets.data()
            );

            result.erase(result.begin() + numValuesPerSequence, result.end());

            //exclude candidates with ambiguous bases

            if(correctionOptions.excludeAmbiguousReads){
                auto minhashResultsEnd = std::remove_if(
                    result.begin(),
                    result.end(),
                    [&](read_number readId){
                        bool containsN = false;
                        readStorage->areSequencesAmbiguous(readStorageHandle, &containsN, &readId, 1);
                        return containsN;
                    }
                );

                result.erase(minhashResultsEnd, result.end());
            }            

        }else{
            ; // no candidates
        }
    }

    void getCandidateReadIds(std::vector<Task>& tasks, const std::vector<int>& indicesOfActiveTasks) override{
        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            getCandidateReadIdsSingle(
                task.candidateReadIds, 
                task.currentAnchor.data(), 
                task.currentAnchorLength,
                task.currentAnchorReadId
            );

        }
    }


    void loadCandidateSequenceData(std::vector<Task>& tasks, const std::vector<int>& indicesOfActiveTasks) override{
        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            const int numCandidates = task.candidateReadIds.size();

            task.candidateSequenceLengths.resize(numCandidates);
            task.candidateSequencesFwdData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);
            task.candidateSequencesRevcData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);

            readStorage->gatherSequenceLengths(
                readStorageHandle,
                task.candidateSequenceLengths.data(),
                task.candidateReadIds.data(),
                task.candidateReadIds.size()
            );

            readStorage->gatherSequences(
                readStorageHandle,
                task.candidateSequencesFwdData.data(),
                encodedSequencePitchInInts,
                task.candidateReadIds.data(),
                task.candidateReadIds.size()
            );

            for(int c = 0; c < numCandidates; c++){
                const unsigned int* const seqPtr = task.candidateSequencesFwdData.data() 
                                                    + std::size_t(encodedSequencePitchInInts) * c;
                unsigned int* const seqrevcPtr = task.candidateSequencesRevcData.data() 
                                                    + std::size_t(encodedSequencePitchInInts) * c;

                SequenceHelpers::reverseComplementSequence2Bit(
                    seqrevcPtr,  
                    seqPtr,
                    task.candidateSequenceLengths[c]
                );
            }
        }
    }

    void eraseDataOfRemovedMates(std::vector<Task>& tasks, const std::vector<int>& indicesOfActiveTasks) override{
        auto vecAccess = [](auto& vec, auto index) -> decltype(vec.at(index)){
            return vec.at(index);
        };
        
        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = vecAccess(tasks, indexOfActiveTask);

            if(task.mateRemovedFromCandidates){
                const int numCandidates = task.candidateReadIds.size();

                std::vector<int> positionsOfCandidatesToKeep;
                positionsOfCandidatesToKeep.reserve(numCandidates);

                for(int c = 0; c < numCandidates; c++){
                    const unsigned int* const seqPtr = task.candidateSequencesFwdData.data() 
                                                    + std::size_t(encodedSequencePitchInInts) * c;

                    auto mismatchIters = std::mismatch(
                        task.encodedMate.begin(), task.encodedMate.end(),
                        seqPtr, seqPtr + encodedSequencePitchInInts
                    );

                    //candidate differs from mate
                    if(mismatchIters.first != task.encodedMate.end()){                            
                        positionsOfCandidatesToKeep.emplace_back(c);
                    }else{
                        ;//std::cerr << "";
                    }
                }

                //compact
                const int toKeep = positionsOfCandidatesToKeep.size();
                for(int c = 0; c < toKeep; c++){
                    const int index = vecAccess(positionsOfCandidatesToKeep, c);

                    vecAccess(task.candidateReadIds, c) = vecAccess(task.candidateReadIds, index);
                    vecAccess(task.candidateSequenceLengths, c) = vecAccess(task.candidateSequenceLengths, index);                        

                    std::copy_n(
                        task.candidateSequencesFwdData.data() + index * encodedSequencePitchInInts,
                        encodedSequencePitchInInts,
                        task.candidateSequencesFwdData.data() + c * encodedSequencePitchInInts
                    );

                    std::copy_n(
                        task.candidateSequencesRevcData.data() + index * encodedSequencePitchInInts,
                        encodedSequencePitchInInts,
                        task.candidateSequencesRevcData.data() + c * encodedSequencePitchInInts
                    );

                    
                }

                //erase
                task.candidateReadIds.erase(
                    task.candidateReadIds.begin() + toKeep, 
                    task.candidateReadIds.end()
                );
                task.candidateSequenceLengths.erase(
                    task.candidateSequenceLengths.begin() + toKeep, 
                    task.candidateSequenceLengths.end()
                );
                task.candidateSequencesFwdData.erase(
                    task.candidateSequencesFwdData.begin() + toKeep * encodedSequencePitchInInts, 
                    task.candidateSequencesFwdData.end()
                );
                task.candidateSequencesRevcData.erase(
                    task.candidateSequencesRevcData.begin() + toKeep * encodedSequencePitchInInts, 
                    task.candidateSequencesRevcData.end()
                );

                task.mateRemovedFromCandidates = false;
            }

        }
    }

    void calculateAlignments(std::vector<ReadExtenderBase::Task>& tasks, const std::vector<int>& indicesOfActiveTasks) override{
        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            const int numCandidates = task.candidateReadIds.size();

            std::vector<care::cpu::SHDResult> forwardAlignments;
            std::vector<care::cpu::SHDResult> revcAlignments;

            forwardAlignments.resize(numCandidates);
            revcAlignments.resize(numCandidates);
            task.alignmentFlags.resize(numCandidates);
            task.alignments.resize(numCandidates);

            care::cpu::shd::cpuShiftedHammingDistancePopcount2BitWithDirection<care::cpu::shd::ShiftDirection::Right>(
                alignmentHandle,
                forwardAlignments.data(),
                task.currentAnchor.data(),
                task.currentAnchorLength,
                task.candidateSequencesFwdData.data(),
                encodedSequencePitchInInts,
                task.candidateSequenceLengths.data(),
                numCandidates,
                goodAlignmentProperties.min_overlap,
                goodAlignmentProperties.maxErrorRate,
                goodAlignmentProperties.min_overlap_ratio
            );

            care::cpu::shd::cpuShiftedHammingDistancePopcount2BitWithDirection<care::cpu::shd::ShiftDirection::Right>(
                alignmentHandle,
                revcAlignments.data(),
                task.currentAnchor.data(),
                task.currentAnchorLength,
                task.candidateSequencesRevcData.data(),
                encodedSequencePitchInInts,
                task.candidateSequenceLengths.data(),
                numCandidates,
                goodAlignmentProperties.min_overlap,
                goodAlignmentProperties.maxErrorRate,
                goodAlignmentProperties.min_overlap_ratio
            );

            //decide whether to keep forward or reverse complement, and keep it

            for(int c = 0; c < numCandidates; c++){
                const auto& forwardAlignment = forwardAlignments[c];
                const auto& revcAlignment = revcAlignments[c];
                const int candidateLength = task.candidateSequenceLengths[c];

                task.alignmentFlags[c] = care::choose_best_alignment(
                    forwardAlignment,
                    revcAlignment,
                    task.currentAnchorLength,
                    candidateLength,
                    goodAlignmentProperties.min_overlap_ratio,
                    goodAlignmentProperties.min_overlap,
                    correctionOptions.estimatedErrorrate
                );

                if(task.alignmentFlags[c] == BestAlignment_t::Forward){
                    task.alignments[c] = forwardAlignment;
                }else{
                    task.alignments[c] = revcAlignment;
                }
            }
        }
    }

    const CpuReadStorage* readStorage;
    const CpuMinhasher* minhasher;

    ReadStorageHandle readStorageHandle;
    CpuMinhasher::QueryHandle minhashHandle;
    cpu::shd::CpuAlignmentHandle alignmentHandle;

};

}

