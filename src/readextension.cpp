
#include <config.hpp>
#include <sequence.hpp>
#include <minhasher.hpp>
#include <readstorage.hpp>
#include <options.hpp>
#include <cpu_alignment.hpp>
#include <bestalignment.hpp>
#include <msa.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>
#include <iostream>
#include <string>
#include <memory>
#include <mutex>



#include <readextension_cpu.hpp>
#include <extensionresultprocessing.hpp>
#include <correctionresultprocessing.hpp>
#include <rangegenerator.hpp>
#include <threadpool.hpp>
#include <memoryfile.hpp>
#include <util.hpp>
#include <filehelpers.hpp>

#include <omp.h>


namespace care{


constexpr int maxextension = 20;


struct ReadExtender{
public:
    enum class AbortReason{
        MsaNotExtended, 
        NoPairedCandidates, 
        NoPairedCandidatesAfterAlignment, 
        None
    };

    struct ReadPairIds{
        read_number first;
        read_number second;
    };

    struct ExtendResult{
        bool mateHasBeenFound = false;
        bool success = false;
        bool aborted = false;
        int numIterations = 0;

        AbortReason abortReason = AbortReason::None;
        std::vector<int> extensionLengths;
        // (read number of forward strand read, extended read for forward strand)
        std::vector<std::pair<read_number, std::string>> extendedReads;

    };

    struct WorkingSet{
        std::array<std::string, 2> decodedPairedRead;
        std::array<std::vector<unsigned int>, 2> candidates;
        std::array<std::vector<int>, 2> candidateLengths;
        std::array<std::vector<unsigned int>, 2> reverseComplementCandidates;

        std::array<std::vector<read_number>, 2> candidateReadIds;
        std::array<int, 2> numCandidates;
    };

    struct ExtendInput{
        read_number readId1{};
        const unsigned int* encodedRead1{};
        int readLength1{};
        int numInts1{};
        read_number readId2{};
        const unsigned int* encodedRead2{};
        int readLength2{};
        int numInts2{};
        bool verbose{};
        std::mutex* verboseMutex;
    };

    ReadExtender(
        int insertSize,
        int insertSizeStddev,
        int maximumSequenceLength,
        const cpu::ContiguousReadStorage& rs, 
        const Minhasher& mh,
        const CorrectionOptions& coropts,
        const GoodAlignmentProperties& gap        
    ) : insertSize(insertSize), 
        insertSizeStddev(insertSizeStddev),
            maximumSequenceLength(maximumSequenceLength),
            minhasher(&mh), readStorage(&rs), 
            correctionOptions(coropts),
            goodAlignmentProperties(gap){

        encodedSequencePitchInInts = getEncodedNumInts2Bit(maximumSequenceLength);
        decodedSequencePitchInBytes = maximumSequenceLength;
        qualityPitchInBytes = maximumSequenceLength;
    }

    /*
        Assumes read1 is on the forward strand, read2 is on the reverse strand
    */
    ExtendResult extendPairedRead(
        const ExtendInput& input
    ){

        std::array<std::vector<unsigned int>, 2> currentAnchor;
        std::array<int, 2> currentAnchorLength;
        std::array<read_number, 2> currentAnchorReadId;
        std::array<int, 2> accumExtensionLengths;        
        
        //for each iteration of the while-loop, saves the currentAnchor (decoded), 
        //the current accumExtensionLength,
        //the new candidate read ids
        std::array<std::vector<std::string>, 2> totalDecodedAnchors;
        std::array<std::vector<int>, 2> totalAnchorBeginInExtendedRead;
        std::vector<std::vector<ReadPairIds>> usedCandidateReadIdsPerIteration;

        std::vector<ReadPairIds> allUsedCandidateReadIdPairs; //sorted

        auto readPairIdComparator = [](const auto& pair1, const auto& pair2){
            if(pair1.first < pair2.first) return true;
            if(pair1.first > pair2.first) return false;
            if(pair1.second < pair2.second) return true;
            return false;
        };

        bool abort = false;
        AbortReason abortReason = AbortReason::None;
        bool mateHasBeenFound = false;
        std::vector<read_number>::iterator mateIdLocationIter;

        //setup input of first loop iteration
        currentAnchor[0].resize(input.numInts1);
        std::copy_n(input.encodedRead1, input.numInts1, currentAnchor[0].begin());
        currentAnchor[1].resize(input.numInts2);
        std::copy_n(input.encodedRead2, input.numInts2, currentAnchor[1].begin());

        currentAnchorLength[0] = input.readLength1;
        currentAnchorLength[1] = input.readLength2;

        currentAnchorReadId[0] = input.readId1;
        currentAnchorReadId[1] = input.readId2;

        accumExtensionLengths[0] = 0;
        accumExtensionLengths[1] = 0;

        std::stringstream verboseStream;

        if(input.verbose){
            verboseStream << "readId1 " << input.readId1 << ", readId2 " << input.readId2 << "\n";
        }

        int iter = 0;
        while(iter < insertSize && accumExtensionLengths[0] < insertSize - input.readLength2 + insertSizeStddev && !abort && !mateHasBeenFound){

            //update "total" arrays
            for(int i = 0; i < 2; i++){
                std::string decodedAnchor(currentAnchorLength[i], '\0');

                decode2BitSequence(
                    &decodedAnchor[0],
                    currentAnchor[i].data(),
                    currentAnchorLength[i]
                );

                totalDecodedAnchors[i].emplace_back(std::move(decodedAnchor));
                totalAnchorBeginInExtendedRead[i].emplace_back(accumExtensionLengths[i]);
            }
            
            if(input.verbose){
                verboseStream << "Iteration " << iter << "\n";
            }

            if(input.verbose){    
                verboseStream << "anchor0: " << totalDecodedAnchors[0].back() << ", anchor1: " << totalDecodedAnchors[1].back() << "\n";
            }

            if(input.verbose){    
                for(int i = 0; i < 2; i++){
                    verboseStream << "totalAnchorBeginInExtendedRead["<< i << "]: ";
                    std::copy(
                        totalAnchorBeginInExtendedRead[i].begin(),
                        totalAnchorBeginInExtendedRead[i].end(),
                        std::ostream_iterator<int>(verboseStream, ", ")
                    );
                    verboseStream << "\n";
                }
            }
            

            

            std::array<std::vector<read_number>, 2> newCandidateReadIds;

            getCandidates(
                newCandidateReadIds[0], 
                currentAnchor[0].data(), 
                currentAnchorLength[0],
                currentAnchorReadId[0]
            );

            getCandidates(
                newCandidateReadIds[1], 
                currentAnchor[1].data(), 
                currentAnchorLength[1],
                currentAnchorReadId[1]
            );

            if(iter == 0){
                // remove self from candidate list
                for(int i = 0; i < 2; i++){
                    auto readIdPos = std::lower_bound(
                        newCandidateReadIds[i].begin(),                                            
                        newCandidateReadIds[i].end(),
                        currentAnchorReadId[i]
                    );

                    if(readIdPos != newCandidateReadIds[i].end() && *readIdPos == currentAnchorReadId[i]){
                        newCandidateReadIds[i].erase(readIdPos);
                    }
                }
                
            }

            //remove mate of input from candidate list if it is not possible that mate could be reached at the current iteration
            if(input.readLength2 + accumExtensionLengths[0] < insertSize){
                auto readIdPos = std::lower_bound(
                    newCandidateReadIds[0].begin(),                                            
                    newCandidateReadIds[0].end(),
                    input.readId2
                );

                if(readIdPos != newCandidateReadIds[0].end() && *readIdPos == input.readId2){
                    newCandidateReadIds[0].erase(readIdPos);
                }
            }

            //remove mate of input from candidate list if it is not possible that mate could be reached at the current iteration
            if(input.readLength1 + accumExtensionLengths[1] < insertSize){
                auto readIdPos = std::lower_bound(
                    newCandidateReadIds[1].begin(),                                            
                    newCandidateReadIds[1].end(),
                    input.readId1
                );

                if(readIdPos != newCandidateReadIds[1].end() && *readIdPos == input.readId1){
                    newCandidateReadIds[1].erase(readIdPos);
                }
            }

            if(input.verbose){    
                verboseStream << "initial candidate read ids for anchor 0:\n";
                std::copy(
                    newCandidateReadIds[0].begin(),
                    newCandidateReadIds[0].end(),
                    std::ostream_iterator<read_number>(verboseStream, ",")
                );
                verboseStream << "\n";

                verboseStream << "initial candidate read ids for anchor 1:\n";
                std::copy(
                    newCandidateReadIds[1].begin(),
                    newCandidateReadIds[1].end(),
                    std::ostream_iterator<read_number>(verboseStream, ",")
                );
                verboseStream << "\n";
            }

            /*
                Remove candidates whose mates are not candidates of the other read.
            */

            std::vector<ReadPairIds> candidateIdsWithMate = removeCandidateIdsWithoutMate(
                newCandidateReadIds[0],
                newCandidateReadIds[1]      
            );

            std::sort(
                candidateIdsWithMate.begin(),
                candidateIdsWithMate.end(),
                readPairIdComparator
            );

            /*
                Remove candidate pairs which have already been used for extension
            */

            {
                std::vector<ReadPairIds> tmp(candidateIdsWithMate.size());

                auto end = std::set_difference(
                    candidateIdsWithMate.begin(),
                    candidateIdsWithMate.end(),
                    allUsedCandidateReadIdPairs.begin(),
                    allUsedCandidateReadIdPairs.end(),
                    tmp.begin(),
                    readPairIdComparator
                );

                tmp.erase(end, tmp.end());

                std::swap(candidateIdsWithMate, tmp);
            }

            newCandidateReadIds[0].clear();
            newCandidateReadIds[1].clear();

            std::for_each(
                candidateIdsWithMate.begin(),
                candidateIdsWithMate.end(),
                [&](const auto& pair){
                    newCandidateReadIds[0].emplace_back(pair.first);
                    newCandidateReadIds[1].emplace_back(pair.second);
                }
            );

            

            if(input.verbose){    
                verboseStream << "new candidate read ids for anchor 0:\n";
                std::copy(
                    newCandidateReadIds[0].begin(),
                    newCandidateReadIds[0].end(),
                    std::ostream_iterator<read_number>(verboseStream, ",")
                );
                verboseStream << "\n";

                verboseStream << "new candidate read ids for anchor 1:\n";
                std::copy(
                    newCandidateReadIds[1].begin(),
                    newCandidateReadIds[1].end(),
                    std::ostream_iterator<read_number>(verboseStream, ",")
                );
                verboseStream << "\n";
            }

            /*
                Load candidate sequences and compute reverse complements
            */

            cpu::ContiguousReadStorage::GatherHandle readStorageGatherHandle;

            std::array<std::vector<int>, 2> newCandidateSequenceLengths;
            std::array<std::vector<unsigned int>, 2> newCandidateSequenceFwdData;
            std::array<std::vector<unsigned int>, 2> newCandidateSequenceRevcData;

            for(int i = 0; i < 2; i++){
                const int numCandidates = newCandidateReadIds[i].size();

                newCandidateSequenceLengths[i].resize(numCandidates);
                newCandidateSequenceFwdData[i].resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);
                newCandidateSequenceRevcData[i].resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);

                readStorage->gatherSequenceLengths(
                    readStorageGatherHandle,
                    newCandidateReadIds[i].data(),
                    newCandidateReadIds[i].size(),
                    newCandidateSequenceLengths[i].data()
                );

                readStorage->gatherSequenceData(
                    readStorageGatherHandle,
                    newCandidateReadIds[i].data(),
                    newCandidateReadIds[i].size(),
                    newCandidateSequenceFwdData[i].data(),
                    encodedSequencePitchInInts
                );

                for(int c = 0; c < numCandidates; c++){
                    const unsigned int* const seqPtr = newCandidateSequenceFwdData[i].data() 
                                                        + std::size_t(encodedSequencePitchInInts) * c;
                    unsigned int* const seqrevcPtr = newCandidateSequenceRevcData[i].data() 
                                                        + std::size_t(encodedSequencePitchInInts) * c;

                    reverseComplement2Bit(
                        seqrevcPtr,  
                        seqPtr,
                        newCandidateSequenceLengths[i][c]
                    );
                }
            }


            /*
                Compute alignments
            */

            cpu::shd::CpuAlignmentHandle alignmentHandle;

            std::array<std::vector<care::cpu::SHDResult>, 2> newAlignments;
            std::array<std::vector<BestAlignment_t>, 2> newAlignmentFlags;

            assert(newCandidateReadIds[0].size() == newCandidateReadIds[1].size());

            for(int i = 0; i < 2; i++){

                const int numCandidates = newCandidateReadIds[i].size();

                std::vector<care::cpu::SHDResult> newForwardAlignments;
                std::vector<care::cpu::SHDResult> newRevcAlignments;

                newForwardAlignments.resize(numCandidates);
                newRevcAlignments.resize(numCandidates);
                newAlignmentFlags[i].resize(numCandidates);
                newAlignments[i].resize(numCandidates);

                care::cpu::shd::cpuShiftedHammingDistancePopcount2Bit<care::cpu::shd::ShiftDirection::LeftRight>(
                    alignmentHandle,
                    newForwardAlignments.data(),
                    currentAnchor[i].data(),
                    currentAnchorLength[i],
                    newCandidateSequenceFwdData[i].data(),
                    encodedSequencePitchInInts,
                    newCandidateSequenceLengths[i].data(),
                    numCandidates,
                    goodAlignmentProperties.min_overlap,
                    //currentAnchorLength[i] - maxextension,
                    goodAlignmentProperties.maxErrorRate,
                    goodAlignmentProperties.min_overlap_ratio
                );

                care::cpu::shd::cpuShiftedHammingDistancePopcount2Bit<care::cpu::shd::ShiftDirection::LeftRight>(
                    alignmentHandle,
                    newRevcAlignments.data(),
                    currentAnchor[i].data(),
                    currentAnchorLength[i],
                    newCandidateSequenceRevcData[i].data(),
                    encodedSequencePitchInInts,
                    newCandidateSequenceLengths[i].data(),
                    numCandidates,
                    goodAlignmentProperties.min_overlap,
                    //currentAnchorLength[i] - maxextension,
                    goodAlignmentProperties.maxErrorRate,
                    goodAlignmentProperties.min_overlap_ratio
                );

                //decide whether to keep forward or reverse complement, and keep it

                for(int c = 0; c < numCandidates; c++){
                    const auto& forwardAlignment = newForwardAlignments[c];
                    const auto& revcAlignment = newRevcAlignments[c];
                    const int candidateLength = newCandidateSequenceLengths[i][c];

                    newAlignmentFlags[i][c] = care::choose_best_alignment(
                        forwardAlignment,
                        revcAlignment,
                        currentAnchorLength[i],
                        candidateLength,
                        goodAlignmentProperties.min_overlap_ratio,
                        goodAlignmentProperties.min_overlap,
                        correctionOptions.estimatedErrorrate
                    );

                    if(newAlignmentFlags[i][c] == BestAlignment_t::Forward){
                        newAlignments[i][c] = forwardAlignment;
                    }else{
                        newAlignments[i][c] = revcAlignment;
                    }
                }

            }

            /*
                Remove bad alignments and the corresponding alignments of their mate
            */        

            assert(newAlignments[0].size() == newAlignments[1].size());
            const int size = newAlignments[0].size();

            // std::array<std::vector<care::cpu::SHDResult>, 2> newAlignmentsTmp;
            // std::array<std::vector<BestAlignment_t>, 2> newAlignmentFlagsTmp;

            // newAlignmentsTmp[0].resize(size);
            // newAlignmentFlagsTmp[0].resize(size);

            // newAlignmentsTmp[1].resize(size);
            // newAlignmentFlagsTmp[1].resize(size);

            std::vector<int> positionsOfCandidatesToKeep(size);
            std::vector<int> tmpPositionsOfCandidatesToKeep(size);

            int numRemainingCandidates = 0;
            int numRemainingCandidatesTmp = 0;

            //select remaining candidates by alignment flag
            //if any of the mates aligns badly, remove both of them
            for(int c = 0; c < size; c++){
                const BestAlignment_t alignmentFlag0 = newAlignmentFlags[0][c];
                const BestAlignment_t alignmentFlag1 = newAlignmentFlags[1][c];
                
                if(!(alignmentFlag0 == BestAlignment_t::None || alignmentFlag1 == BestAlignment_t::None)){
                    positionsOfCandidatesToKeep[numRemainingCandidates] = c;
                    numRemainingCandidates++;
                }else{
                    ; //if any of the mates aligns badly, remove both of them
                }
            }

            //remove candidates whose mate aligns with a conflicting shift direction
            for(int c = 0; c < numRemainingCandidates; c++){
                const int index = positionsOfCandidatesToKeep[c];
                const auto& alignmentresult0 = newAlignments[0][index];
                const auto& alignmentresult1 = newAlignments[1][index];
                
                //only keep candidates for both read0 and read1 
                //which align "to the right" relative to the forward strand
                // => positive shift on forward strand, negative shift on reverse strand
                if(alignmentresult0.shift >= 0 && alignmentresult1.shift <= 0){
                    //keep
                    tmpPositionsOfCandidatesToKeep[numRemainingCandidatesTmp] = index;
                    numRemainingCandidatesTmp++;
                }else{
                    ;
                }
            }

            std::swap(tmpPositionsOfCandidatesToKeep, positionsOfCandidatesToKeep);
            std::swap(numRemainingCandidates, numRemainingCandidatesTmp);

            if(numRemainingCandidates == 0){
                abort = true;
                abortReason = AbortReason::NoPairedCandidatesAfterAlignment;
                break; //terminate while loop
            }

            //compact selected candidates inplace
            for(int c = 0; c < numRemainingCandidates; c++){
                const int index = positionsOfCandidatesToKeep[c];
                candidateIdsWithMate[c] = candidateIdsWithMate[index];
            }

            candidateIdsWithMate.erase(
                candidateIdsWithMate.begin() + numRemainingCandidates, 
                candidateIdsWithMate.end()
            );

            std::array<std::vector<unsigned int>, 2> newCandidateSequenceData;

            for(int i = 0; i < 2; i++){
                newCandidateSequenceData[i].resize(numRemainingCandidates * encodedSequencePitchInInts);

                for(int c = 0; c < numRemainingCandidates; c++){
                    const int index = positionsOfCandidatesToKeep[c];

                    newAlignments[i][c] = newAlignments[i][index];
                    newAlignmentFlags[i][c] = newAlignmentFlags[i][index];
                    newCandidateReadIds[i][c] = newCandidateReadIds[i][index];
                    newCandidateSequenceLengths[i][c] = newCandidateSequenceLengths[i][index];
                    
                    assert(newAlignmentFlags[i][index] != BestAlignment_t::None);

                    if(newAlignmentFlags[i][index] == BestAlignment_t::Forward){
                        std::copy_n(
                            newCandidateSequenceFwdData[i].data() + index * encodedSequencePitchInInts,
                            encodedSequencePitchInInts,
                            newCandidateSequenceData[i].data() + c * encodedSequencePitchInInts
                        );
                    }else{
                        //BestAlignment_t::ReverseComplement

                        std::copy_n(
                            newCandidateSequenceRevcData[i].data() + index * encodedSequencePitchInInts,
                            encodedSequencePitchInInts,
                            newCandidateSequenceData[i].data() + c * encodedSequencePitchInInts
                        );
                    }

                    //not sure if these 2 arrays will be required further on
                    std::copy_n(
                        newCandidateSequenceFwdData[i].data() + index * encodedSequencePitchInInts,
                        encodedSequencePitchInInts,
                        newCandidateSequenceFwdData[i].data() + c * encodedSequencePitchInInts
                    );

                    std::copy_n(
                        newCandidateSequenceRevcData[i].data() + index * encodedSequencePitchInInts,
                        encodedSequencePitchInInts,
                        newCandidateSequenceRevcData[i].data() + c * encodedSequencePitchInInts
                    );
                    
                }

                //erase past-end elements
                newAlignments[i].erase(
                    newAlignments[i].begin() + numRemainingCandidates, 
                    newAlignments[i].end()
                );
                newAlignmentFlags[i].erase(
                    newAlignmentFlags[i].begin() + numRemainingCandidates, 
                    newAlignmentFlags[i].end()
                );
                newCandidateReadIds[i].erase(
                    newCandidateReadIds[i].begin() + numRemainingCandidates, 
                    newCandidateReadIds[i].end()
                );
                newCandidateSequenceLengths[i].erase(
                    newCandidateSequenceLengths[i].begin() + numRemainingCandidates, 
                    newCandidateSequenceLengths[i].end()
                );
                //not sure if these 2 arrays will be required further on
                newCandidateSequenceFwdData[i].erase(
                    newCandidateSequenceFwdData[i].begin() + numRemainingCandidates * encodedSequencePitchInInts, 
                    newCandidateSequenceFwdData[i].end()
                );
                newCandidateSequenceRevcData[i].erase(
                    newCandidateSequenceRevcData[i].begin() + numRemainingCandidates * encodedSequencePitchInInts, 
                    newCandidateSequenceRevcData[i].end()
                );
                
            }

            if(input.verbose){    
                verboseStream << "new candidate read ids for anchor 0 after alignments:\n";
                std::copy(
                    newCandidateReadIds[0].begin(),
                    newCandidateReadIds[0].end(),
                    std::ostream_iterator<read_number>(verboseStream, ",")
                );
                verboseStream << "\n";

                verboseStream << "new candidate read ids for anchor 1 after alignments:\n";
                std::copy(
                    newCandidateReadIds[1].begin(),
                    newCandidateReadIds[1].end(),
                    std::ostream_iterator<read_number>(verboseStream, ",")
                );
                verboseStream << "\n";
            }

            //check if mate has been reached
            mateIdLocationIter = std::lower_bound(
                newCandidateReadIds[0].begin(),
                newCandidateReadIds[0].end(),
                input.readId2
            );

            mateHasBeenFound = (mateIdLocationIter != newCandidateReadIds[0].end() && *mateIdLocationIter == input.readId2);

            if(input.verbose){    
                verboseStream << "mate has been found ? " << (mateHasBeenFound ? "yes":"no") << "\n";
            }

            /*
                Construct MSAs
            */

            for(int i = 0; i < 2; i++){
                const std::string& decodedAnchor = totalDecodedAnchors[i].back();

                auto calculateOverlapWeight = [](int anchorlength, int nOps, int overlapsize){
                    constexpr float maxErrorPercentInOverlap = 0.2f;

                    return 1.0f - sqrtf(nOps / (overlapsize * maxErrorPercentInOverlap));
                };

                std::vector<int> candidateShifts(numRemainingCandidates);
                std::vector<float> candidateOverlapWeights(numRemainingCandidates);

                for(int c = 0; c < numRemainingCandidates; c++){
                    candidateShifts[c] = newAlignments[i][c].shift;

                    candidateOverlapWeights[c] = calculateOverlapWeight(
                        currentAnchorLength[i], 
                        newAlignments[i][c].nOps,
                        newAlignments[i][c].overlap
                    );
                }

                std::vector<char> candidateStrings(decodedSequencePitchInBytes * numRemainingCandidates, '\0');

                for(int c = 0; c < numRemainingCandidates; c++){
                    decode2BitSequence(
                        candidateStrings.data() + c * decodedSequencePitchInBytes,
                        newCandidateSequenceData[i].data() + c * encodedSequencePitchInInts,
                        newCandidateSequenceLengths[i][c]
                    );
                }

                MultipleSequenceAlignment::InputData msaInput;
                msaInput.useQualityScores = false;
                msaInput.subjectLength = currentAnchorLength[i];
                msaInput.nCandidates = numRemainingCandidates;
                msaInput.candidatesPitch = decodedSequencePitchInBytes;
                msaInput.candidateQualitiesPitch = 0;
                msaInput.subject = decodedAnchor.c_str();
                msaInput.candidates = candidateStrings.data();
                msaInput.subjectQualities = nullptr;
                msaInput.candidateQualities = nullptr;
                msaInput.candidateLengths = newCandidateSequenceLengths[i].data();
                msaInput.candidateShifts = candidateShifts.data();
                msaInput.candidateDefaultWeightFactors = candidateOverlapWeights.data();

                MultipleSequenceAlignment msa;

                msa.build(msaInput);

                if(input.verbose){    
                    verboseStream << "msa for anchor " << i << "\n";
                    msa.print(verboseStream);
                    verboseStream << "\n";
                }


                if(!mateHasBeenFound){
                    //mate not found. prepare next while-loop iteration
                    int consensusLength = msa.consensus.size();

                    //the first currentAnchorLength[i] columns are occupied by anchor. try to extend read 
                    //by at most maxextension bp. In case consensuslength == anchorlength, abort

                    if(consensusLength == currentAnchorLength[i]){
                        abort = true;
                        abortReason = AbortReason::MsaNotExtended;                        
                    }else{
                        assert(consensusLength > currentAnchorLength[i]);
                        

                        const int extendBy = std::min(consensusLength - currentAnchorLength[i], maxextension);
                        accumExtensionLengths[i] += extendBy;

                        if(input.verbose){
                            verboseStream << "extended by " << extendBy << ", total extension length " << accumExtensionLengths[i] << "\n";
                        }

                        //update data for next iteration of outer while loop
                        if(i == 0){
                            const std::string nextDecodedAnchor(msa.consensus.data() + extendBy, currentAnchorLength[i]);
                            const int numInts = getEncodedNumInts2Bit(nextDecodedAnchor.size());

                            currentAnchor[i].resize(numInts);
                            encodeSequence2Bit(
                                currentAnchor[i].data(), 
                                nextDecodedAnchor.c_str(), 
                                nextDecodedAnchor.size()
                            );
                            currentAnchorLength[i] = nextDecodedAnchor.size();

                            if(input.verbose){
                                verboseStream << "next anchor: " << nextDecodedAnchor << "\n";
                            }
                        }else{
                            //i == 1
                            const char* const data = msa.consensus.data() 
                                + consensusLength - currentAnchorLength[i] - extendBy;
                            const std::string nextDecodedAnchor(data, currentAnchorLength[i]);
                            const int numInts = getEncodedNumInts2Bit(nextDecodedAnchor.size());

                            currentAnchor[i].resize(numInts);
                            encodeSequence2Bit(
                                currentAnchor[i].data(), 
                                nextDecodedAnchor.c_str(), 
                                nextDecodedAnchor.size()
                            );
                            currentAnchorLength[i] = nextDecodedAnchor.size();

                            if(input.verbose){
                                verboseStream << "next anchor: " << nextDecodedAnchor << "\n";
                            }
                        }
                    }
                }else{
                    if(i == 0){
                        //find end of mate in msa
                        const int index = std::distance(newCandidateReadIds[i].begin(), mateIdLocationIter);
                        const int shift = newAlignments[i][index].shift;
                        const int clength = newCandidateSequenceLengths[i][index];
                        assert(shift >= 0);
                        const int endcolumn = shift + clength;

                        const int extendby = shift;
                        assert(extendby >= 0);
                        accumExtensionLengths[i] += extendby;

                        std::string decodedAnchor(msa.consensus.data() + extendby, endcolumn - extendby);

                        if(input.verbose){
                            verboseStream << "consensus until end of mate: " << decodedAnchor << "\n";
                        }

                        totalDecodedAnchors[i].emplace_back(std::move(decodedAnchor));
                        totalAnchorBeginInExtendedRead[i].emplace_back(accumExtensionLengths[i]);
                    }
                }

            }

            /*
                update book-keeping of used candidates
            */                        
            {
                std::vector<ReadPairIds> tmp(allUsedCandidateReadIdPairs.size() + candidateIdsWithMate.size());
                auto tmp_end = std::merge(
                    allUsedCandidateReadIdPairs.begin(),
                    allUsedCandidateReadIdPairs.end(),
                    candidateIdsWithMate.begin(),
                    candidateIdsWithMate.end(),
                    tmp.begin(),
                    readPairIdComparator
                );

                tmp.erase(tmp_end, tmp.end());

                std::swap(allUsedCandidateReadIdPairs, tmp);
            }

            usedCandidateReadIdsPerIteration.emplace_back(std::move(candidateIdsWithMate));

            iter++; //control outer while-loop
        }

        ExtendResult extendResult;
        extendResult.numIterations = iter;
        extendResult.aborted = abort;
        extendResult.abortReason = abortReason;
        extendResult.extensionLengths.emplace_back(totalAnchorBeginInExtendedRead[0].back());

        if(abort){
            ; //no read extension possible
        }else{
            if(mateHasBeenFound){
                //construct extended read
                //build msa of all saved totalDecodedAnchors[0]

                const int numsteps = totalDecodedAnchors[0].size();
                int maxlen = 0;
                for(const auto& s: totalDecodedAnchors[0]){
                    const int len = s.length();
                    if(len > maxlen){
                        maxlen = len;
                    }
                }
                const std::string& decodedAnchor = totalDecodedAnchors[0][0];

                const std::vector<int> shifts(totalAnchorBeginInExtendedRead[0].begin() + 1, totalAnchorBeginInExtendedRead[0].end());
                std::vector<float> initialWeights(numsteps-1, 1.0f);


                std::vector<char> stepstrings(maxlen * (numsteps-1), '\0');
                std::vector<int> stepstringlengths(numsteps-1);
                for(int c = 1; c < numsteps; c++){
                    std::copy(
                        totalDecodedAnchors[0][c].begin(),
                        totalDecodedAnchors[0][c].end(),
                        stepstrings.begin() + (c-1) * maxlen
                    );
                    stepstringlengths[c-1] = totalDecodedAnchors[0][c].size();
                }

                MultipleSequenceAlignment::InputData msaInput;
                msaInput.useQualityScores = false;
                msaInput.subjectLength = decodedAnchor.length();
                msaInput.nCandidates = numsteps-1;
                msaInput.candidatesPitch = maxlen;
                msaInput.candidateQualitiesPitch = 0;
                msaInput.subject = decodedAnchor.c_str();
                msaInput.candidates = stepstrings.data();
                msaInput.subjectQualities = nullptr;
                msaInput.candidateQualities = nullptr;
                msaInput.candidateLengths = stepstringlengths.data();
                msaInput.candidateShifts = shifts.data();
                msaInput.candidateDefaultWeightFactors = initialWeights.data();

                MultipleSequenceAlignment msa;

                msa.build(msaInput);

                if(input.verbose){    
                    verboseStream << "msa of partial results \n";
                    msa.print(verboseStream);
                    verboseStream << "\n";
                }

                extendResult.success = true;

                std::string extendedRead(msa.consensus.begin(), msa.consensus.end());

                if(input.verbose){    
                    verboseStream << "extended read: " << extendedRead << "\n";
                }

                extendResult.extendedReads.emplace_back(input.readId1, std::move(extendedRead));

                

                if(input.verbose){
                    if(input.verboseMutex != nullptr){
                        std::lock_guard<std::mutex> lg(*input.verboseMutex);

                        std::cerr << verboseStream.rdbuf();
                    }else{
                        std::cerr << verboseStream.rdbuf();
                    }
                }
            }else{
                ; //no read extension possible
            }
        }

        return extendResult;
    }




    ExtendResult extendPairedRead2(
        const ExtendInput& input
    ){

        std::vector<unsigned int> currentAnchor;
        int currentAnchorLength;
        read_number currentAnchorReadId;
        int accumExtensionLengths;        
        
        //for each iteration of the while-loop, saves the currentAnchor (decoded), 
        //the current accumExtensionLength,
        //the new candidate read ids
        std::vector<std::string> totalDecodedAnchors;
        std::vector<int> totalAnchorBeginInExtendedRead;
        std::vector<std::vector<read_number>> usedCandidateReadIdsPerIteration;

        std::vector<read_number> allUsedCandidateReadIdPairs; //sorted

        bool abort = false;
        AbortReason abortReason = AbortReason::None;
        bool mateHasBeenFound = false;
        std::vector<read_number>::iterator mateIdLocationIter;

        //setup input of first loop iteration
        currentAnchor.resize(input.numInts1);
        std::copy_n(input.encodedRead1, input.numInts1, currentAnchor.begin());

        currentAnchorLength = input.readLength1;

        currentAnchorReadId = input.readId1;

        accumExtensionLengths = 0;

        std::stringstream verboseStream;

        if(input.verbose){
            verboseStream << "readId1 " << input.readId1 << ", readId2 " << input.readId2 << "\n";
        }

        int iter = 0;
        while(iter < insertSize && accumExtensionLengths < insertSize - (input.readLength2) + insertSizeStddev
                && !abort && !mateHasBeenFound){

            //update "total" arrays
            {
                std::string decodedAnchor(currentAnchorLength, '\0');

                decode2BitSequence(
                    &decodedAnchor[0],
                    currentAnchor.data(),
                    currentAnchorLength
                );

                totalDecodedAnchors.emplace_back(std::move(decodedAnchor));
                totalAnchorBeginInExtendedRead.emplace_back(accumExtensionLengths);
            }
            
            if(input.verbose){
                verboseStream << "Iteration " << iter << "\n";
            }

            if(input.verbose){    
                verboseStream << "anchor0: " << totalDecodedAnchors.back() << "\n";
            }

            if(input.verbose){    
                {
                    verboseStream << "totalAnchorBeginInExtendedRead: ";
                    std::copy(
                        totalAnchorBeginInExtendedRead.begin(),
                        totalAnchorBeginInExtendedRead.end(),
                        std::ostream_iterator<int>(verboseStream, ", ")
                    );
                    verboseStream << "\n";
                }
            }
            

            

            std::vector<read_number> newCandidateReadIds;

            getCandidates(
                newCandidateReadIds, 
                currentAnchor.data(), 
                currentAnchorLength,
                currentAnchorReadId
            );

            if(iter == 0){
                // remove self from candidate list
                {
                    auto readIdPos = std::lower_bound(
                        newCandidateReadIds.begin(),                                            
                        newCandidateReadIds.end(),
                        currentAnchorReadId
                    );

                    if(readIdPos != newCandidateReadIds.end() && *readIdPos == currentAnchorReadId){
                        newCandidateReadIds.erase(readIdPos);
                    }
                }
                
            }

            //remove mate of input from candidate list if it is not possible that mate could be reached at the current iteration
            if(input.readLength2 + accumExtensionLengths < insertSize - insertSizeStddev){
                auto readIdPos = std::lower_bound(
                    newCandidateReadIds.begin(),                                            
                    newCandidateReadIds.end(),
                    input.readId2
                );

                if(readIdPos != newCandidateReadIds.end() && *readIdPos == input.readId2){
                    newCandidateReadIds.erase(readIdPos);
                }
            }

            if(input.verbose){    
                verboseStream << "initial candidate read ids for anchor 0:\n";
                std::copy(
                    newCandidateReadIds.begin(),
                    newCandidateReadIds.end(),
                    std::ostream_iterator<read_number>(verboseStream, ",")
                );
                verboseStream << "\n";
            }

            /*
                Remove candidate pairs which have already been used for extension
            */

            {
                std::vector<read_number> tmp(newCandidateReadIds.size());

                auto end = std::set_difference(
                    newCandidateReadIds.begin(),
                    newCandidateReadIds.end(),
                    allUsedCandidateReadIdPairs.begin(),
                    allUsedCandidateReadIdPairs.end(),
                    tmp.begin()
                );

                tmp.erase(end, tmp.end());

                std::swap(newCandidateReadIds, tmp);
            }

            if(input.verbose){    
                verboseStream << "new candidate read ids for anchor 0:\n";
                std::copy(
                    newCandidateReadIds.begin(),
                    newCandidateReadIds.end(),
                    std::ostream_iterator<read_number>(verboseStream, ",")
                );
                verboseStream << "\n";
            }

            /*
                Load candidate sequences and compute reverse complements
            */

            cpu::ContiguousReadStorage::GatherHandle readStorageGatherHandle;

            std::vector<int> newCandidateSequenceLengths;
            std::vector<unsigned int> newCandidateSequenceFwdData;
            std::vector<unsigned int> newCandidateSequenceRevcData;

            {
                const int numCandidates = newCandidateReadIds.size();

                newCandidateSequenceLengths.resize(numCandidates);
                newCandidateSequenceFwdData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);
                newCandidateSequenceRevcData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);

                readStorage->gatherSequenceLengths(
                    readStorageGatherHandle,
                    newCandidateReadIds.data(),
                    newCandidateReadIds.size(),
                    newCandidateSequenceLengths.data()
                );

                readStorage->gatherSequenceData(
                    readStorageGatherHandle,
                    newCandidateReadIds.data(),
                    newCandidateReadIds.size(),
                    newCandidateSequenceFwdData.data(),
                    encodedSequencePitchInInts
                );

                for(int c = 0; c < numCandidates; c++){
                    const unsigned int* const seqPtr = newCandidateSequenceFwdData.data() 
                                                        + std::size_t(encodedSequencePitchInInts) * c;
                    unsigned int* const seqrevcPtr = newCandidateSequenceRevcData.data() 
                                                        + std::size_t(encodedSequencePitchInInts) * c;

                    reverseComplement2Bit(
                        seqrevcPtr,  
                        seqPtr,
                        newCandidateSequenceLengths[c]
                    );
                }
            }


            /*
                Compute alignments
            */

            cpu::shd::CpuAlignmentHandle alignmentHandle;

            std::vector<care::cpu::SHDResult> newAlignments;
            std::vector<BestAlignment_t> newAlignmentFlags;

            {

                const int numCandidates = newCandidateReadIds.size();

                std::vector<care::cpu::SHDResult> newForwardAlignments;
                std::vector<care::cpu::SHDResult> newRevcAlignments;

                newForwardAlignments.resize(numCandidates);
                newRevcAlignments.resize(numCandidates);
                newAlignmentFlags.resize(numCandidates);
                newAlignments.resize(numCandidates);

                care::cpu::shd::cpuShiftedHammingDistancePopcount2Bit<care::cpu::shd::ShiftDirection::Right>(
                    alignmentHandle,
                    newForwardAlignments.data(),
                    currentAnchor.data(),
                    currentAnchorLength,
                    newCandidateSequenceFwdData.data(),
                    encodedSequencePitchInInts,
                    newCandidateSequenceLengths.data(),
                    numCandidates,
                    goodAlignmentProperties.min_overlap,
                    //currentAnchorLength - maxextension,
                    goodAlignmentProperties.maxErrorRate,
                    goodAlignmentProperties.min_overlap_ratio
                );

                care::cpu::shd::cpuShiftedHammingDistancePopcount2Bit<care::cpu::shd::ShiftDirection::Right>(
                    alignmentHandle,
                    newRevcAlignments.data(),
                    currentAnchor.data(),
                    currentAnchorLength,
                    newCandidateSequenceRevcData.data(),
                    encodedSequencePitchInInts,
                    newCandidateSequenceLengths.data(),
                    numCandidates,
                    goodAlignmentProperties.min_overlap,
                    //currentAnchorLength - maxextension,
                    goodAlignmentProperties.maxErrorRate,
                    goodAlignmentProperties.min_overlap_ratio
                );

                //decide whether to keep forward or reverse complement, and keep it

                for(int c = 0; c < numCandidates; c++){
                    const auto& forwardAlignment = newForwardAlignments[c];
                    const auto& revcAlignment = newRevcAlignments[c];
                    const int candidateLength = newCandidateSequenceLengths[c];

                    newAlignmentFlags[c] = care::choose_best_alignment(
                        forwardAlignment,
                        revcAlignment,
                        currentAnchorLength,
                        candidateLength,
                        goodAlignmentProperties.min_overlap_ratio,
                        goodAlignmentProperties.min_overlap,
                        correctionOptions.estimatedErrorrate
                    );

                    if(newAlignmentFlags[c] == BestAlignment_t::Forward){
                        newAlignments[c] = forwardAlignment;
                    }else{
                        newAlignments[c] = revcAlignment;
                    }
                }

            }

            /*
                Remove bad alignments and the corresponding alignments of their mate
            */        

            const int size = newAlignments.size();

            std::vector<int> positionsOfCandidatesToKeep(size);
            std::vector<int> tmpPositionsOfCandidatesToKeep(size);

            int numRemainingCandidates = 0;

            //select candidates with good alignment and positive shift
            for(int c = 0; c < size; c++){
                const BestAlignment_t alignmentFlag0 = newAlignmentFlags[c];
                
                if(alignmentFlag0 != BestAlignment_t::None && newAlignments[c].shift >= 0){
                    positionsOfCandidatesToKeep[numRemainingCandidates] = c;
                    numRemainingCandidates++;
                }else{
                    ; //if any of the mates aligns badly, remove both of them
                }
            }

            if(numRemainingCandidates == 0){
                abort = true;
                abortReason = AbortReason::NoPairedCandidatesAfterAlignment;

                if(input.verbose){    
                    verboseStream << "no candidates left after alignment\n";
                }
                break; //terminate while loop
            }

            //compact selected candidates inplace

            std::vector<unsigned int> newCandidateSequenceData;

            {
                newCandidateSequenceData.resize(numRemainingCandidates * encodedSequencePitchInInts);

                for(int c = 0; c < numRemainingCandidates; c++){
                    const int index = positionsOfCandidatesToKeep[c];

                    newAlignments[c] = newAlignments[index];
                    newAlignmentFlags[c] = newAlignmentFlags[index];
                    newCandidateReadIds[c] = newCandidateReadIds[index];
                    newCandidateSequenceLengths[c] = newCandidateSequenceLengths[index];
                    
                    assert(newAlignmentFlags[index] != BestAlignment_t::None);

                    if(newAlignmentFlags[index] == BestAlignment_t::Forward){
                        std::copy_n(
                            newCandidateSequenceFwdData.data() + index * encodedSequencePitchInInts,
                            encodedSequencePitchInInts,
                            newCandidateSequenceData.data() + c * encodedSequencePitchInInts
                        );
                    }else{
                        //BestAlignment_t::ReverseComplement

                        std::copy_n(
                            newCandidateSequenceRevcData.data() + index * encodedSequencePitchInInts,
                            encodedSequencePitchInInts,
                            newCandidateSequenceData.data() + c * encodedSequencePitchInInts
                        );
                    }

                    // //not sure if these 2 arrays will be required further on
                    // std::copy_n(
                    //     newCandidateSequenceFwdData.data() + index * encodedSequencePitchInInts,
                    //     encodedSequencePitchInInts,
                    //     newCandidateSequenceFwdData.data() + c * encodedSequencePitchInInts
                    // );

                    // std::copy_n(
                    //     newCandidateSequenceRevcData.data() + index * encodedSequencePitchInInts,
                    //     encodedSequencePitchInInts,
                    //     newCandidateSequenceRevcData.data() + c * encodedSequencePitchInInts
                    // );
                    
                }

                //erase past-end elements
                newAlignments.erase(
                    newAlignments.begin() + numRemainingCandidates, 
                    newAlignments.end()
                );
                newAlignmentFlags.erase(
                    newAlignmentFlags.begin() + numRemainingCandidates, 
                    newAlignmentFlags.end()
                );
                newCandidateReadIds.erase(
                    newCandidateReadIds.begin() + numRemainingCandidates, 
                    newCandidateReadIds.end()
                );
                newCandidateSequenceLengths.erase(
                    newCandidateSequenceLengths.begin() + numRemainingCandidates, 
                    newCandidateSequenceLengths.end()
                );
                // //not sure if these 2 arrays will be required further on
                // newCandidateSequenceFwdData.erase(
                //     newCandidateSequenceFwdData.begin() + numRemainingCandidates * encodedSequencePitchInInts, 
                //     newCandidateSequenceFwdData.end()
                // );
                // newCandidateSequenceRevcData.erase(
                //     newCandidateSequenceRevcData.begin() + numRemainingCandidates * encodedSequencePitchInInts, 
                //     newCandidateSequenceRevcData.end()
                // );
                
            }

            if(input.verbose){    
                verboseStream << "new candidate read ids for anchor 0 after alignments:\n";
                std::copy(
                    newCandidateReadIds.begin(),
                    newCandidateReadIds.end(),
                    std::ostream_iterator<read_number>(verboseStream, ",")
                );
                verboseStream << "\n";
            }

            //check if mate has been reached
            mateIdLocationIter = std::lower_bound(
                newCandidateReadIds.begin(),
                newCandidateReadIds.end(),
                input.readId2
            );

            mateHasBeenFound = (mateIdLocationIter != newCandidateReadIds.end() && *mateIdLocationIter == input.readId2);

            if(input.verbose){    
                verboseStream << "mate has been found ? " << (mateHasBeenFound ? "yes":"no") << "\n";
            }

            //check that extending to mate does not leave fragment
            if(mateHasBeenFound){
                const int mateIndex = std::distance(newCandidateReadIds.begin(), mateIdLocationIter);
                const auto& mateAlignment = newAlignments[mateIndex];

                if(accumExtensionLengths + input.readLength2 + mateAlignment.shift > insertSize + insertSizeStddev){
                    mateHasBeenFound = false;

                    newAlignments.erase(newAlignments.begin() + mateIndex);
                    newAlignmentFlags.erase(newAlignmentFlags.begin() + mateIndex);
                    newCandidateReadIds.erase(newCandidateReadIds.begin() + mateIndex);
                    newCandidateSequenceLengths.erase(newCandidateSequenceLengths.begin() + mateIndex);

                    newCandidateSequenceData.erase(
                        newCandidateSequenceData.begin() + mateIndex * encodedSequencePitchInInts,
                        newCandidateSequenceData.begin() + (mateIndex + 1) * encodedSequencePitchInInts
                    );

                    if(input.verbose){    
                        verboseStream << "mate has been removed again because it would reach beyond fragment\n";
                    }
                }
            }

            

            /*
                Construct MSAs
            */

            {
                const std::string& decodedAnchor = totalDecodedAnchors.back();

                auto calculateOverlapWeight = [](int anchorlength, int nOps, int overlapsize){
                    constexpr float maxErrorPercentInOverlap = 0.2f;

                    return 1.0f - sqrtf(nOps / (overlapsize * maxErrorPercentInOverlap));
                };

                std::vector<int> candidateShifts(numRemainingCandidates);
                std::vector<float> candidateOverlapWeights(numRemainingCandidates);

                for(int c = 0; c < numRemainingCandidates; c++){
                    candidateShifts[c] = newAlignments[c].shift;

                    candidateOverlapWeights[c] = calculateOverlapWeight(
                        currentAnchorLength, 
                        newAlignments[c].nOps,
                        newAlignments[c].overlap
                    );
                }

                std::vector<char> candidateStrings(decodedSequencePitchInBytes * numRemainingCandidates, '\0');

                for(int c = 0; c < numRemainingCandidates; c++){
                    decode2BitSequence(
                        candidateStrings.data() + c * decodedSequencePitchInBytes,
                        newCandidateSequenceData.data() + c * encodedSequencePitchInInts,
                        newCandidateSequenceLengths[c]
                    );
                }

                MultipleSequenceAlignment::InputData msaInput;
                msaInput.useQualityScores = false;
                msaInput.subjectLength = currentAnchorLength;
                msaInput.nCandidates = numRemainingCandidates;
                msaInput.candidatesPitch = decodedSequencePitchInBytes;
                msaInput.candidateQualitiesPitch = 0;
                msaInput.subject = decodedAnchor.c_str();
                msaInput.candidates = candidateStrings.data();
                msaInput.subjectQualities = nullptr;
                msaInput.candidateQualities = nullptr;
                msaInput.candidateLengths = newCandidateSequenceLengths.data();
                msaInput.candidateShifts = candidateShifts.data();
                msaInput.candidateDefaultWeightFactors = candidateOverlapWeights.data();

                MultipleSequenceAlignment msa;

                msa.build(msaInput);

                if(input.verbose){    
                    verboseStream << "msa for anchor 0\n";
                    msa.print(verboseStream);
                    verboseStream << "\n";
                }


                if(!mateHasBeenFound){
                    //mate not found. prepare next while-loop iteration

                    {
                        int consensusLength = msa.consensus.size();

                        //the first currentAnchorLength columns are occupied by anchor. try to extend read 
                        //by at most maxextension bp.

                        //can extend by at most maxextension bps
                        int extendBy = std::min(consensusLength - currentAnchorLength, maxextension);
                        //cannot extend over fragment 
                        extendBy = std::min(extendBy, (insertSize + insertSizeStddev - input.readLength2) - accumExtensionLengths);

                        if(extendBy == 0){
                            abort = true;
                            abortReason = AbortReason::MsaNotExtended;
                        }else{
                            accumExtensionLengths += extendBy;

                            if(input.verbose){
                                verboseStream << "extended by " << extendBy << ", total extension length " << accumExtensionLengths << "\n";
                            }

                            //update data for next iteration of outer while loop
                            const std::string nextDecodedAnchor(msa.consensus.data() + extendBy, currentAnchorLength);
                            const int numInts = getEncodedNumInts2Bit(nextDecodedAnchor.size());

                            currentAnchor.resize(numInts);
                            encodeSequence2Bit(
                                currentAnchor.data(), 
                                nextDecodedAnchor.c_str(), 
                                nextDecodedAnchor.size()
                            );
                            currentAnchorLength = nextDecodedAnchor.size();

                            if(input.verbose){
                                verboseStream << "next anchor: " << nextDecodedAnchor << "\n";
                            }
                        }
                       
                    }
                }else{
                    {
                        //find end of mate in msa
                        const int index = std::distance(newCandidateReadIds.begin(), mateIdLocationIter);
                        const int shift = newAlignments[index].shift;
                        const int clength = newCandidateSequenceLengths[index];
                        assert(shift >= 0);
                        const int endcolumn = shift + clength;

                        const int extendby = shift;
                        assert(extendby >= 0);
                        accumExtensionLengths += extendby;

                        std::string decodedAnchor(msa.consensus.data() + extendby, endcolumn - extendby);

                        if(input.verbose){
                            verboseStream << "consensus until end of mate: " << decodedAnchor << "\n";
                        }

                        totalDecodedAnchors.emplace_back(std::move(decodedAnchor));
                        totalAnchorBeginInExtendedRead.emplace_back(accumExtensionLengths);
                    }
                }

            }

            /*
                update book-keeping of used candidates
            */                        
            {
                std::vector<read_number> tmp(allUsedCandidateReadIdPairs.size() + newCandidateReadIds.size());
                auto tmp_end = std::merge(
                    allUsedCandidateReadIdPairs.begin(),
                    allUsedCandidateReadIdPairs.end(),
                    newCandidateReadIds.begin(),
                    newCandidateReadIds.end(),
                    tmp.begin()
                );

                tmp.erase(tmp_end, tmp.end());

                std::swap(allUsedCandidateReadIdPairs, tmp);
            }

            usedCandidateReadIdsPerIteration.emplace_back(std::move(newCandidateReadIds));

            iter++; //control outer while-loop
        }

        ExtendResult extendResult;
        extendResult.numIterations = iter;
        extendResult.aborted = abort;
        extendResult.abortReason = abortReason;
        extendResult.extensionLengths.emplace_back(totalAnchorBeginInExtendedRead.back());

        // if(abort){
        //     ; //no read extension possible
        // }else
        {
            //if(mateHasBeenFound){
            {
                //construct extended read
                //build msa of all saved totalDecodedAnchors[0]

                const int numsteps = totalDecodedAnchors.size();
                int maxlen = 0;
                for(const auto& s: totalDecodedAnchors){
                    const int len = s.length();
                    if(len > maxlen){
                        maxlen = len;
                    }
                }
                const std::string& decodedAnchor = totalDecodedAnchors[0];

                const std::vector<int> shifts(totalAnchorBeginInExtendedRead.begin() + 1, totalAnchorBeginInExtendedRead.end());
                std::vector<float> initialWeights(numsteps-1, 1.0f);


                std::vector<char> stepstrings(maxlen * (numsteps-1), '\0');
                std::vector<int> stepstringlengths(numsteps-1);
                for(int c = 1; c < numsteps; c++){
                    std::copy(
                        totalDecodedAnchors[c].begin(),
                        totalDecodedAnchors[c].end(),
                        stepstrings.begin() + (c-1) * maxlen
                    );
                    stepstringlengths[c-1] = totalDecodedAnchors[c].size();
                }

                MultipleSequenceAlignment::InputData msaInput;
                msaInput.useQualityScores = false;
                msaInput.subjectLength = decodedAnchor.length();
                msaInput.nCandidates = numsteps-1;
                msaInput.candidatesPitch = maxlen;
                msaInput.candidateQualitiesPitch = 0;
                msaInput.subject = decodedAnchor.c_str();
                msaInput.candidates = stepstrings.data();
                msaInput.subjectQualities = nullptr;
                msaInput.candidateQualities = nullptr;
                msaInput.candidateLengths = stepstringlengths.data();
                msaInput.candidateShifts = shifts.data();
                msaInput.candidateDefaultWeightFactors = initialWeights.data();

                MultipleSequenceAlignment msa;

                msa.build(msaInput);

                if(input.verbose){    
                    verboseStream << "msa of partial results \n";
                    msa.print(verboseStream);
                    verboseStream << "\n";
                }

                extendResult.success = true;

                std::string extendedRead(msa.consensus.begin(), msa.consensus.end());

                if(input.verbose){    
                    verboseStream << "extended read: " << extendedRead << "\n";
                }

                extendResult.extendedReads.emplace_back(input.readId1, std::move(extendedRead));

                if(mateHasBeenFound){
                    extendResult.mateHasBeenFound = mateHasBeenFound;
                }

                

                if(input.verbose){
                    if(input.verboseMutex != nullptr){
                        std::lock_guard<std::mutex> lg(*input.verboseMutex);

                        std::cerr << verboseStream.rdbuf();
                    }else{
                        std::cerr << verboseStream.rdbuf();
                    }
                }
            }
            // else{
            //     ; //no read extension possible
            // }
        }

        return extendResult;
    }

private:

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

    void getCandidates(
        std::vector<read_number>& result, 
        const unsigned int* encodedRead, 
        int readLength, 
        read_number readId
    ){
        Minhasher::Handle minhashHandle;

        result.clear();

        const bool containsN = readStorage->readContainsN(readId);

        //exclude anchors with ambiguous bases
        if(!(correctionOptions.excludeAmbiguousReads && containsN)){

            const int length = readLength;
            std::string sequence(length, '0');

            decode2BitSequence(
                &sequence[0],
                encodedRead,
                length
            );

            minhasher->getCandidates_any_map(
                minhashHandle,
                sequence,
                0
            );

            auto minhashResultsEnd = minhashHandle.result().end();
            //exclude candidates with ambiguous bases

            if(correctionOptions.excludeAmbiguousReads){
                minhashResultsEnd = std::remove_if(
                    minhashHandle.result().begin(),
                    minhashHandle.result().end(),
                    [&](read_number readId){
                        return readStorage->readContainsN(readId);
                    }
                );
            }            

            result.insert(
                result.begin(),
                minhashHandle.result().begin(),
                minhashResultsEnd
            );
        }else{
            ; // no candidates
        }
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


    int insertSize;
    int insertSizeStddev;
    int maximumSequenceLength;
    std::size_t encodedSequencePitchInInts;
    std::size_t decodedSequencePitchInBytes;
    std::size_t qualityPitchInBytes;


    const Minhasher* minhasher;
    const cpu::ContiguousReadStorage* readStorage;

    CorrectionOptions correctionOptions;
    GoodAlignmentProperties goodAlignmentProperties;
    WorkingSet ws;
};







MemoryFileFixedSize<ExtendedRead> 
//std::vector<ExtendedRead>
extend_cpu(
    const GoodAlignmentProperties& goodAlignmentProperties,
    const CorrectionOptions& correctionOptions,
    const RuntimeOptions& runtimeOptions,
    const FileOptions& fileOptions,
    const MemoryOptions& memoryOptions,
    const SequenceFileProperties& sequenceFileProperties,
    Minhasher& minhasher,
    cpu::ContiguousReadStorage& readStorage
){
    const auto rsMemInfo = readStorage.getMemoryInfo();
    const auto mhMemInfo = minhasher.getMemoryInfo();

    std::size_t memoryAvailableBytesHost = memoryOptions.memoryTotalLimit;
    if(memoryAvailableBytesHost > rsMemInfo.host){
        memoryAvailableBytesHost -= rsMemInfo.host;
    }else{
        memoryAvailableBytesHost = 0;
    }
    if(memoryAvailableBytesHost > mhMemInfo.host){
        memoryAvailableBytesHost -= mhMemInfo.host;
    }else{
        memoryAvailableBytesHost = 0;
    }

    std::unique_ptr<std::uint8_t[]> correctionStatusFlagsPerRead = std::make_unique<std::uint8_t[]>(sequenceFileProperties.nReads);

    #pragma omp parallel for
    for(read_number i = 0; i < sequenceFileProperties.nReads; i++){
        correctionStatusFlagsPerRead[i] = 0;
    }

    std::cerr << "correctionStatusFlagsPerRead bytes: " << sizeof(std::uint8_t) * sequenceFileProperties.nReads / 1024. / 1024. << " MB\n";

    if(memoryAvailableBytesHost > sizeof(std::uint8_t) * sequenceFileProperties.nReads){
        memoryAvailableBytesHost -= sizeof(std::uint8_t) * sequenceFileProperties.nReads;
    }else{
        memoryAvailableBytesHost = 0;
    }

    const std::size_t availableMemoryInBytes = memoryAvailableBytesHost; //getAvailableMemoryInKB() * 1024;
    std::size_t memoryForPartialResultsInBytes = 0;

    if(availableMemoryInBytes > 3*(std::size_t(1) << 30)){
        memoryForPartialResultsInBytes = availableMemoryInBytes - 3*(std::size_t(1) << 30);
    }

    const std::string tmpfilename{fileOptions.tempdirectory + "/" + "MemoryFileFixedSizetmp"};
    MemoryFileFixedSize<ExtendedRead> partialResults(memoryForPartialResultsInBytes, tmpfilename);

    std::vector<ExtendedRead> resultExtendedReads;

    //cpu::RangeGenerator<read_number> readIdGenerator(sequenceFileProperties.nReads);
    cpu::RangeGenerator<read_number> readIdGenerator(1000);

    BackgroundThread outputThread(true);

    

    auto showProgress = [&](auto totalCount, auto seconds){
        if(runtimeOptions.showProgress){

            printf("Processed %10u of %10lu reads (Runtime: %03d:%02d:%02d)\r",
                    totalCount, sequenceFileProperties.nReads,
                    int(seconds / 3600),
                    int(seconds / 60) % 60,
                    int(seconds) % 60);
            std::cout.flush();
        }

        if(totalCount == sequenceFileProperties.nReads){
            std::cerr << '\n';
        }
    };

    auto updateShowProgressInterval = [](auto duration){
        return duration;
    };

    ProgressThread<read_number> progressThread(sequenceFileProperties.nReads, showProgress, updateShowProgressInterval);

    
    const int insertSize = 300;
    const int insertSizeStddev = 5;
    const int maximumSequenceLength = sequenceFileProperties.maxSequenceLength;
    const std::size_t encodedSequencePitchInInts = getEncodedNumInts2Bit(maximumSequenceLength);

    std::mutex verboseMutex;

    std::int64_t totalNumSuccess0 = 0;
    std::int64_t totalNumSuccess1 = 0;
    std::int64_t totalNumSuccess01 = 0;
    std::int64_t totalNumSuccessRead = 0;

    std::map<int, int> totalExtensionLengthsMap;

    std::map<int, int> totalMismatchesBetweenMateExtensions;

    //omp_set_num_threads(1);

    #pragma omp parallel
    {
        GoodAlignmentProperties goodAlignmentProperties2 = goodAlignmentProperties;
        //goodAlignmentProperties2.maxErrorRate = 0.05;

        ReadExtender readExtender{
            insertSize,
            insertSizeStddev,
            maximumSequenceLength,
            readStorage, 
            minhasher,
            correctionOptions,
            goodAlignmentProperties2
        };

        std::int64_t numSuccess0 = 0;
        std::int64_t numSuccess1 = 0;
        std::int64_t numSuccess01 = 0;
        std::int64_t numSuccessRead = 0;

        std::map<int, int> extensionLengthsMap;
        std::map<int, int> mismatchesBetweenMateExtensions;

        cpu::ContiguousReadStorage::GatherHandle readStorageGatherHandle;

        std::vector<read_number> currentIds(2);
        std::vector<unsigned int> currentEncodedReads(2 * encodedSequencePitchInInts);
        std::array<int, 2> currentReadLengths;

        while(!(readIdGenerator.empty())){
            std::array<read_number, 2> currentIds;

            auto readIdsEnd = readIdGenerator.next_n_into_buffer(
                2, 
                currentIds.begin()
            );
            
            if(std::distance(currentIds.begin(), readIdsEnd) != 2){
                continue; //this should only happen if all reads have been processed or input file is not properly paired
            }

            readStorage.gatherSequenceLengths(
                readStorageGatherHandle,
                currentIds.data(),
                currentIds.size(),
                currentReadLengths.data()
            );

            readStorage.gatherSequenceData(
                readStorageGatherHandle,
                currentIds.data(),
                currentIds.size(),
                currentEncodedReads.data(),
                encodedSequencePitchInInts
            );

            auto processReadOrder = [&](std::array<int, 2> order){
                ReadExtender::ExtendInput input{};

                input.readId1 = currentIds[order[0]];
                input.readId2 = currentIds[order[1]];
                input.encodedRead1 = currentEncodedReads.data() + order[0] * encodedSequencePitchInInts;
                input.encodedRead2 = currentEncodedReads.data() + order[1] * encodedSequencePitchInInts;
                input.readLength1 = currentReadLengths[order[0]];
                input.readLength2 = currentReadLengths[order[1]];
                input.numInts1 = getEncodedNumInts2Bit(currentReadLengths[order[0]]);
                input.numInts2 = getEncodedNumInts2Bit(currentReadLengths[order[1]]);
                input.verbose = false;
                input.verboseMutex = &verboseMutex;

                auto extendResult = readExtender.extendPairedRead2(input);

                return extendResult;  
            };

            // it is not known which of both reads is on the forward strand / reverse complement strand. try both combinations
            auto extendResult0 = processReadOrder({0,1});

            auto extendResult1 = processReadOrder({1,0});

            if(extendResult0.success || extendResult1.success){
                numSuccessRead++;
            }

            // if(extendResult.extensionLengths.size() > 0){
            //     const int l = extendResult.extensionLengths.front();
            //     extensionLengthsMap[l]++;
            // }

            auto handleMultiResult = [&](const ReadExtender::ExtendResult* result1, const ReadExtender::ExtendResult* result2){
                ExtendedReadDebug er{};

                if(result1 != nullptr){
                    er.extendedRead1 = result1->extendedReads.front().second;
                    er.reachedMate1 = result1->mateHasBeenFound;
                }
                if(result2 != nullptr){
                    er.extendedRead2 = result2->extendedReads.front().second;
                    er.reachedMate2 = result2->mateHasBeenFound;
                }

                er.readId1 = currentIds[0];
                er.readId2 = currentIds[1];

                er.originalRead1.resize(currentReadLengths[0], '\0');

                decode2BitSequence(
                    &er.originalRead1[0],
                    currentEncodedReads.data() + 0 * encodedSequencePitchInInts,
                    currentReadLengths[0]
                );

                er.originalRead2.resize(currentReadLengths[1], '\0');

                decode2BitSequence(
                    &er.originalRead2[0],
                    currentEncodedReads.data() + 1 * encodedSequencePitchInInts,
                    currentReadLengths[1]
                );

                auto func = [&, er = std::move(er)]() mutable{
                    //resultExtendedReads.emplace_back(std::move(er));
                    //std::cerr << er.readId1 << " " << er.readId2 << "\n";
                    partialResults.storeElement(std::move(er));
                };

                outputThread.enqueue(
                    std::move(func)
                );

                
            };

            // auto handleSingleResult = [&](const auto& extendResult){
            //     const int numResults = extendResult.extendedReads.size();
            //     auto encodeddata = std::make_unique<EncodedTempCorrectedSequence[]>(numResults);

            //     for(int i = 0; i < numResults; i++){
            //         auto& pair = extendResult.extendedReads[i];

            //         TempCorrectedSequence tcs;
            //         tcs.hq = false;
            //         tcs.useEdits = false;
            //         tcs.type = TempCorrectedSequence::Type::Anchor;
            //         tcs.shift = 0;
            //         tcs.readId = pair.first;
            //         tcs.sequence = std::move(pair.second);

            //         encodeddata[i] = tcs.encode();
            //     }

            //     auto func = [&, size = numResults, encodeddata = encodeddata.release()](){
            //         for(int i = 0; i < size; i++){
            //             partialResults.storeElement(std::move(encodeddata[i]));
            //         }
            //     };

            //     outputThread.enqueue(
            //         std::move(func)
            //     );
            // };

            if(extendResult0.success && !extendResult1.success){
                //handleSingleResult(extendResult0);
                handleMultiResult(&extendResult0, nullptr);
                numSuccess0++;
            }

            if(!extendResult0.success && extendResult1.success){
                //handleSingleResult(extendResult1);
                handleMultiResult(nullptr, &extendResult1);
                numSuccess1++;
            }

            if(extendResult0.success && extendResult1.success){
                

                const auto& extendedString0 = extendResult0.extendedReads.front().second;
                const auto& extendedString1 = extendResult1.extendedReads.front().second;

                std::string mateExtendedReverseComplement = reverseComplementString(
                    extendedString1.c_str(), 
                    extendedString1.length()
                );
                const int mismatches = cpu::hammingDistance(
                    extendedString0.begin(),
                    extendedString0.end(),
                    mateExtendedReverseComplement.begin(),
                    mateExtendedReverseComplement.end()
                );

                mismatchesBetweenMateExtensions[mismatches]++;

                // if(extendedString0.length() != extendedString1.length()){
                //     std::cerr << "0:\n";
                //     std::cerr << extendedString0 << "\n";
                //     std::cerr << "1 rev compl:\n";
                //     std::cerr << mateExtendedReverseComplement << "\n";
                //     std::cerr << "1:\n";
                //     std::cerr << extendedString1 << "\n";
                // }

                //handleSingleResult(extendResult0);
                handleMultiResult(&extendResult0, &extendResult1);

                numSuccess01++;
            }

            progressThread.addProgress(2);
            
        }

        #pragma omp critical
        {
            totalNumSuccess0 += numSuccess0;
            totalNumSuccess1 += numSuccess1;
            totalNumSuccess01 += numSuccess01;
            totalNumSuccessRead += numSuccessRead;

            for(const auto& pair : extensionLengthsMap){
                totalExtensionLengthsMap[pair.first] += pair.second;
            }

            for(const auto& pair : mismatchesBetweenMateExtensions){
                totalMismatchesBetweenMateExtensions[pair.first] += pair.second;
            }
            
        }
        
    } //end omp parallel

    progressThread.finished();

    outputThread.stopThread(BackgroundThread::StopType::FinishAndStop);

    //outputstream.flush();
    partialResults.flush();

    std::cout << "totalNumSuccess0: " << totalNumSuccess0 << std::endl;
    std::cout << "totalNumSuccess1: " << totalNumSuccess1 << std::endl;
    std::cout << "totalNumSuccess01: " << totalNumSuccess01 << std::endl;
    std::cout << "totalNumSuccessRead: " << totalNumSuccessRead << std::endl;

    // std::cout << "Extension lengths:\n";

    // for(const auto& pair : totalExtensionLengthsMap){
    //     std::cout << pair.first << ": " << pair.second << "\n";
    // }

    // std::cout << "mismatches between mate extensions:\n";

    // for(const auto& pair : totalMismatchesBetweenMateExtensions){
    //     std::cout << pair.first << ": " << pair.second << "\n";
    // }



    return partialResults;
    //return resultExtendedReads;
}



















} // namespace care