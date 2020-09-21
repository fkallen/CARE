#ifndef READ_EXTENDER_HPP
#define READ_EXTENDER_HPP



#include <config.hpp>
#include <sequence.hpp>
#include <minhasher.hpp>
#include <readstorage.hpp>
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


#include <readextension_cpu.hpp>
#include <extensionresultprocessing.hpp>
#include <rangegenerator.hpp>
#include <threadpool.hpp>
#include <memoryfile.hpp>
#include <util.hpp>
#include <filehelpers.hpp>


namespace care{


struct ReadExtenderBase{
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

    ReadExtenderBase(
        int insertSize,
        int insertSizeStddev,
        int maximumSequenceLength,
        const CorrectionOptions& coropts,
        const GoodAlignmentProperties& gap        
    ) : insertSize(insertSize), 
        insertSizeStddev(insertSizeStddev),
            maximumSequenceLength(maximumSequenceLength),
            correctionOptions(coropts),
            goodAlignmentProperties(gap),
            hashTimer{"Hash timer"},
            collectTimer{"Collect timer"},
            alignmentTimer{"Alignment timer"},
            alignmentFilterTimer{"Alignment filter timer"},
            msaTimer{"MSA timer"}{

        encodedSequencePitchInInts = getEncodedNumInts2Bit(maximumSequenceLength);
        decodedSequencePitchInBytes = maximumSequenceLength;
        qualityPitchInBytes = maximumSequenceLength;


    }

    virtual ~ReadExtenderBase() = default;

#if 0
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
            

            

            std::array<std::vector<read_number>, 2> candidateReadIds;

            getCandidateReadIdsSingle(
                candidateReadIds[0], 
                currentAnchor[0].data(), 
                currentAnchorLength[0],
                currentAnchorReadId[0]
            );

            getCandidateReadIdsSingle(
                candidateReadIds[1], 
                currentAnchor[1].data(), 
                currentAnchorLength[1],
                currentAnchorReadId[1]
            );

            if(iter == 0){
                // remove self from candidate list
                for(int i = 0; i < 2; i++){
                    auto readIdPos = std::lower_bound(
                        candidateReadIds[i].begin(),                                            
                        candidateReadIds[i].end(),
                        currentAnchorReadId[i]
                    );

                    if(readIdPos != candidateReadIds[i].end() && *readIdPos == currentAnchorReadId[i]){
                        candidateReadIds[i].erase(readIdPos);
                    }
                }
                
            }

            //remove mate of input from candidate list if it is not possible that mate could be reached at the current iteration
            if(input.readLength2 + accumExtensionLengths[0] < insertSize){
                auto readIdPos = std::lower_bound(
                    candidateReadIds[0].begin(),                                            
                    candidateReadIds[0].end(),
                    input.readId2
                );

                if(readIdPos != candidateReadIds[0].end() && *readIdPos == input.readId2){
                    candidateReadIds[0].erase(readIdPos);
                }
            }

            //remove mate of input from candidate list if it is not possible that mate could be reached at the current iteration
            if(input.readLength1 + accumExtensionLengths[1] < insertSize){
                auto readIdPos = std::lower_bound(
                    candidateReadIds[1].begin(),                                            
                    candidateReadIds[1].end(),
                    input.readId1
                );

                if(readIdPos != candidateReadIds[1].end() && *readIdPos == input.readId1){
                    candidateReadIds[1].erase(readIdPos);
                }
            }

            if(input.verbose){    
                verboseStream << "initial candidate read ids for anchor 0:\n";
                std::copy(
                    candidateReadIds[0].begin(),
                    candidateReadIds[0].end(),
                    std::ostream_iterator<read_number>(verboseStream, ",")
                );
                verboseStream << "\n";

                verboseStream << "initial candidate read ids for anchor 1:\n";
                std::copy(
                    candidateReadIds[1].begin(),
                    candidateReadIds[1].end(),
                    std::ostream_iterator<read_number>(verboseStream, ",")
                );
                verboseStream << "\n";
            }

            /*
                Remove candidates whose mates are not candidates of the other read.
            */

            std::vector<ReadPairIds> candidateIdsWithMate = removeCandidateIdsWithoutMate(
                candidateReadIds[0],
                candidateReadIds[1]      
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

            candidateReadIds[0].clear();
            candidateReadIds[1].clear();

            std::for_each(
                candidateIdsWithMate.begin(),
                candidateIdsWithMate.end(),
                [&](const auto& pair){
                    candidateReadIds[0].emplace_back(pair.first);
                    candidateReadIds[1].emplace_back(pair.second);
                }
            );

            

            if(input.verbose){    
                verboseStream << "new candidate read ids for anchor 0:\n";
                std::copy(
                    candidateReadIds[0].begin(),
                    candidateReadIds[0].end(),
                    std::ostream_iterator<read_number>(verboseStream, ",")
                );
                verboseStream << "\n";

                verboseStream << "new candidate read ids for anchor 1:\n";
                std::copy(
                    candidateReadIds[1].begin(),
                    candidateReadIds[1].end(),
                    std::ostream_iterator<read_number>(verboseStream, ",")
                );
                verboseStream << "\n";
            }

            /*
                Load candidate sequences and compute reverse complements
            */

            cpu::ContiguousReadStorage::GatherHandle readStorageGatherHandle;

            std::array<std::vector<int>, 2> candidateSequenceLengths;
            std::array<std::vector<unsigned int>, 2> candidateSequencesFwdData;
            std::array<std::vector<unsigned int>, 2> candidateSequencesRevcData;

            for(int i = 0; i < 2; i++){
                const int numCandidates = candidateReadIds[i].size();

                candidateSequenceLengths[i].resize(numCandidates);
                candidateSequencesFwdData[i].resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);
                candidateSequencesRevcData[i].resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);

                readStorage->gatherSequenceLengths(
                    readStorageGatherHandle,
                    candidateReadIds[i].data(),
                    candidateReadIds[i].size(),
                    candidateSequenceLengths[i].data()
                );

                readStorage->gatherSequenceData(
                    readStorageGatherHandle,
                    candidateReadIds[i].data(),
                    candidateReadIds[i].size(),
                    candidateSequencesFwdData[i].data(),
                    encodedSequencePitchInInts
                );

                for(int c = 0; c < numCandidates; c++){
                    const unsigned int* const seqPtr = candidateSequencesFwdData[i].data() 
                                                        + std::size_t(encodedSequencePitchInInts) * c;
                    unsigned int* const seqrevcPtr = candidateSequencesRevcData[i].data() 
                                                        + std::size_t(encodedSequencePitchInInts) * c;

                    reverseComplement2Bit(
                        seqrevcPtr,  
                        seqPtr,
                        candidateSequenceLengths[i][c]
                    );
                }
            }


            /*
                Compute alignments
            */

            cpu::shd::CpuAlignmentHandle alignmentHandle;

            std::array<std::vector<care::cpu::SHDResult>, 2> alignments;
            std::array<std::vector<BestAlignment_t>, 2> alignmentFlags;

            assert(candidateReadIds[0].size() == candidateReadIds[1].size());

            for(int i = 0; i < 2; i++){

                const int numCandidates = candidateReadIds[i].size();

                std::vector<care::cpu::SHDResult> forwardAlignments;
                std::vector<care::cpu::SHDResult> revcAlignments;

                forwardAlignments.resize(numCandidates);
                revcAlignments.resize(numCandidates);
                alignmentFlags[i].resize(numCandidates);
                alignments[i].resize(numCandidates);

                care::cpu::shd::cpuShiftedHammingDistancePopcount2Bit<care::cpu::shd::ShiftDirection::LeftRight>(
                    alignmentHandle,
                    forwardAlignments.data(),
                    currentAnchor[i].data(),
                    currentAnchorLength[i],
                    candidateSequencesFwdData[i].data(),
                    encodedSequencePitchInInts,
                    candidateSequenceLengths[i].data(),
                    numCandidates,
                    goodAlignmentProperties.min_overlap,
                    //currentAnchorLength[i] - maxextension,
                    goodAlignmentProperties.maxErrorRate,
                    goodAlignmentProperties.min_overlap_ratio
                );

                care::cpu::shd::cpuShiftedHammingDistancePopcount2Bit<care::cpu::shd::ShiftDirection::LeftRight>(
                    alignmentHandle,
                    revcAlignments.data(),
                    currentAnchor[i].data(),
                    currentAnchorLength[i],
                    candidateSequencesRevcData[i].data(),
                    encodedSequencePitchInInts,
                    candidateSequenceLengths[i].data(),
                    numCandidates,
                    goodAlignmentProperties.min_overlap,
                    //currentAnchorLength[i] - maxextension,
                    goodAlignmentProperties.maxErrorRate,
                    goodAlignmentProperties.min_overlap_ratio
                );

                //decide whether to keep forward or reverse complement, and keep it

                for(int c = 0; c < numCandidates; c++){
                    const auto& forwardAlignment = forwardAlignments[c];
                    const auto& revcAlignment = revcAlignments[c];
                    const int candidateLength = candidateSequenceLengths[i][c];

                    alignmentFlags[i][c] = care::choose_best_alignment(
                        forwardAlignment,
                        revcAlignment,
                        currentAnchorLength[i],
                        candidateLength,
                        goodAlignmentProperties.min_overlap_ratio,
                        goodAlignmentProperties.min_overlap,
                        correctionOptions.estimatedErrorrate
                    );

                    if(alignmentFlags[i][c] == BestAlignment_t::Forward){
                        alignments[i][c] = forwardAlignment;
                    }else{
                        alignments[i][c] = revcAlignment;
                    }
                }

            }

            /*
                Remove bad alignments and the corresponding alignments of their mate
            */        

            assert(alignments[0].size() == alignments[1].size());
            const int size = alignments[0].size();

            // std::array<std::vector<care::cpu::SHDResult>, 2> alignmentsTmp;
            // std::array<std::vector<BestAlignment_t>, 2> alignmentFlagsTmp;

            // alignmentsTmp[0].resize(size);
            // alignmentFlagsTmp[0].resize(size);

            // alignmentsTmp[1].resize(size);
            // alignmentFlagsTmp[1].resize(size);

            std::vector<int> positionsOfCandidatesToKeep(size);
            std::vector<int> tmpPositionsOfCandidatesToKeep(size);

            int numRemainingCandidates = 0;
            int numRemainingCandidatesTmp = 0;

            //select remaining candidates by alignment flag
            //if any of the mates aligns badly, remove both of them
            for(int c = 0; c < size; c++){
                const BestAlignment_t alignmentFlag0 = alignmentFlags[0][c];
                const BestAlignment_t alignmentFlag1 = alignmentFlags[1][c];
                
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
                const auto& alignmentresult0 = alignments[0][index];
                const auto& alignmentresult1 = alignments[1][index];
                
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

            std::array<std::vector<unsigned int>, 2> candidateSequenceData;

            for(int i = 0; i < 2; i++){
                candidateSequenceData[i].resize(numRemainingCandidates * encodedSequencePitchInInts);

                for(int c = 0; c < numRemainingCandidates; c++){
                    const int index = positionsOfCandidatesToKeep[c];

                    alignments[i][c] = alignments[i][index];
                    alignmentFlags[i][c] = alignmentFlags[i][index];
                    candidateReadIds[i][c] = candidateReadIds[i][index];
                    candidateSequenceLengths[i][c] = candidateSequenceLengths[i][index];
                    
                    assert(alignmentFlags[i][index] != BestAlignment_t::None);

                    if(alignmentFlags[i][index] == BestAlignment_t::Forward){
                        std::copy_n(
                            candidateSequencesFwdData[i].data() + index * encodedSequencePitchInInts,
                            encodedSequencePitchInInts,
                            candidateSequenceData[i].data() + c * encodedSequencePitchInInts
                        );
                    }else{
                        //BestAlignment_t::ReverseComplement

                        std::copy_n(
                            candidateSequencesRevcData[i].data() + index * encodedSequencePitchInInts,
                            encodedSequencePitchInInts,
                            candidateSequenceData[i].data() + c * encodedSequencePitchInInts
                        );
                    }

                    //not sure if these 2 arrays will be required further on
                    std::copy_n(
                        candidateSequencesFwdData[i].data() + index * encodedSequencePitchInInts,
                        encodedSequencePitchInInts,
                        candidateSequencesFwdData[i].data() + c * encodedSequencePitchInInts
                    );

                    std::copy_n(
                        candidateSequencesRevcData[i].data() + index * encodedSequencePitchInInts,
                        encodedSequencePitchInInts,
                        candidateSequencesRevcData[i].data() + c * encodedSequencePitchInInts
                    );
                    
                }

                //erase past-end elements
                alignments[i].erase(
                    alignments[i].begin() + numRemainingCandidates, 
                    alignments[i].end()
                );
                alignmentFlags[i].erase(
                    alignmentFlags[i].begin() + numRemainingCandidates, 
                    alignmentFlags[i].end()
                );
                candidateReadIds[i].erase(
                    candidateReadIds[i].begin() + numRemainingCandidates, 
                    candidateReadIds[i].end()
                );
                candidateSequenceLengths[i].erase(
                    candidateSequenceLengths[i].begin() + numRemainingCandidates, 
                    candidateSequenceLengths[i].end()
                );
                //not sure if these 2 arrays will be required further on
                candidateSequencesFwdData[i].erase(
                    candidateSequencesFwdData[i].begin() + numRemainingCandidates * encodedSequencePitchInInts, 
                    candidateSequencesFwdData[i].end()
                );
                candidateSequencesRevcData[i].erase(
                    candidateSequencesRevcData[i].begin() + numRemainingCandidates * encodedSequencePitchInInts, 
                    candidateSequencesRevcData[i].end()
                );
                
            }

            if(input.verbose){    
                verboseStream << "new candidate read ids for anchor 0 after alignments:\n";
                std::copy(
                    candidateReadIds[0].begin(),
                    candidateReadIds[0].end(),
                    std::ostream_iterator<read_number>(verboseStream, ",")
                );
                verboseStream << "\n";

                verboseStream << "new candidate read ids for anchor 1 after alignments:\n";
                std::copy(
                    candidateReadIds[1].begin(),
                    candidateReadIds[1].end(),
                    std::ostream_iterator<read_number>(verboseStream, ",")
                );
                verboseStream << "\n";
            }

            //check if mate has been reached
            mateIdLocationIter = std::lower_bound(
                candidateReadIds[0].begin(),
                candidateReadIds[0].end(),
                input.readId2
            );

            mateHasBeenFound = (mateIdLocationIter != candidateReadIds[0].end() && *mateIdLocationIter == input.readId2);

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
                    candidateShifts[c] = alignments[i][c].shift;

                    candidateOverlapWeights[c] = calculateOverlapWeight(
                        currentAnchorLength[i], 
                        alignments[i][c].nOps,
                        alignments[i][c].overlap
                    );
                }

                std::vector<char> candidateStrings(decodedSequencePitchInBytes * numRemainingCandidates, '\0');

                for(int c = 0; c < numRemainingCandidates; c++){
                    decode2BitSequence(
                        candidateStrings.data() + c * decodedSequencePitchInBytes,
                        candidateSequenceData[i].data() + c * encodedSequencePitchInInts,
                        candidateSequenceLengths[i][c]
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
                msaInput.candidateLengths = candidateSequenceLengths[i].data();
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
                        const int index = std::distance(candidateReadIds[i].begin(), mateIdLocationIter);
                        const int shift = alignments[i][index].shift;
                        const int clength = candidateSequenceLengths[i][index];
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
#endif

    struct Task{
        bool abort = false;
        bool mateHasBeenFound = false;
        AbortReason abortReason = AbortReason::None;
        int currentAnchorLength = 0;
        int accumExtensionLengths = 0;
        int iteration = 0;
        int mateLength = 0;
        read_number myReadId = 0;
        read_number mateReadId = 0;
        read_number currentAnchorReadId;
        std::vector<read_number> candidateReadIds;
        std::vector<read_number>::iterator mateIdLocationIter;
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
                && accumExtensionLengths < insertSize - (mateLength) + insertSizeStddev
                && !abort 
                && !mateHasBeenFound);
        }
    };

    Task makeTask(const ExtendInput& input){
        Task task;

        task.currentAnchor.resize(input.numInts1);
        std::copy_n(input.encodedRead1, input.numInts1, task.currentAnchor.begin());

        task.currentAnchorLength = input.readLength1;
        task.currentAnchorReadId = input.readId1;
        task.accumExtensionLengths = 0;
        task.iteration = 0;

        task.myReadId = input.readId1;

        task.mateLength = input.readLength2;
        task.mateReadId = input.readId2;

        return task;
    }

    std::vector<ExtendResult> extendPairedReadBatch(
        const std::vector<ExtendInput>& inputs
    );


    void printTimers(){
        hashTimer.print();
        collectTimer.print();
        alignmentTimer.print();
        alignmentFilterTimer.print();
        msaTimer.print();
    }

protected:

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

    virtual void getCandidateReadIds(std::vector<Task>& tasks, const std::vector<int>& indicesOfActiveTasks) = 0;
    virtual void loadCandidateSequenceData(std::vector<Task>& tasks, const std::vector<int>& indicesOfActiveTasks) = 0;
    virtual void calculateAlignments(std::vector<Task>& tasks, const std::vector<int>& indicesOfActiveTasks) = 0;

    int insertSize;
    int insertSizeStddev;
    int maximumSequenceLength;
    std::size_t encodedSequencePitchInInts;
    std::size_t decodedSequencePitchInBytes;
    std::size_t qualityPitchInBytes;

    CorrectionOptions correctionOptions;
    GoodAlignmentProperties goodAlignmentProperties;

    helpers::CpuTimer hashTimer;
    helpers::CpuTimer collectTimer;
    helpers::CpuTimer alignmentTimer;
    helpers::CpuTimer alignmentFilterTimer;
    helpers::CpuTimer msaTimer;
};











struct ReadExtenderCpu final : public ReadExtenderBase{
public:


    ReadExtenderCpu(
        int insertSize,
        int insertSizeStddev,
        int maximumSequenceLength,
        const cpu::ContiguousReadStorage& rs, 
        const Minhasher& mh,
        const CorrectionOptions& coropts,
        const GoodAlignmentProperties& gap        
    ) : ReadExtenderBase(insertSize, insertSizeStddev, maximumSequenceLength, coropts, gap),
        readStorage(&rs), minhasher(&mh){

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
                sequence.c_str() + beginPos,
                std::max(0, readLength - beginPos),
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

            readStorage->gatherSequenceLengths(
                readStorageGatherHandle,
                task.candidateReadIds.data(),
                task.candidateReadIds.size(),
                task.candidateSequenceLengths.data()
            );

            readStorage->gatherSequenceData(
                readStorageGatherHandle,
                task.candidateReadIds.data(),
                task.candidateReadIds.size(),
                task.candidateSequencesFwdData.data(),
                encodedSequencePitchInInts
            );
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

            care::cpu::shd::cpuShiftedHammingDistancePopcount2Bit<care::cpu::shd::ShiftDirection::Right>(
                alignmentHandle,
                forwardAlignments.data(),
                task.currentAnchor.data(),
                task.currentAnchorLength,
                task.candidateSequencesFwdData.data(),
                encodedSequencePitchInInts,
                task.candidateSequenceLengths.data(),
                numCandidates,
                goodAlignmentProperties.min_overlap,
                //currentAnchorLength - maxextension,
                goodAlignmentProperties.maxErrorRate,
                goodAlignmentProperties.min_overlap_ratio
            );

            care::cpu::shd::cpuShiftedHammingDistancePopcount2Bit<care::cpu::shd::ShiftDirection::Right>(
                alignmentHandle,
                revcAlignments.data(),
                task.currentAnchor.data(),
                task.currentAnchorLength,
                task.candidateSequencesRevcData.data(),
                encodedSequencePitchInInts,
                task.candidateSequenceLengths.data(),
                numCandidates,
                goodAlignmentProperties.min_overlap,
                //currentAnchorLength - maxextension,
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

    cpu::ContiguousReadStorage::GatherHandle readStorageGatherHandle;
    const cpu::ContiguousReadStorage* readStorage;

    const Minhasher* minhasher;
    Minhasher::Handle minhashHandle;
    cpu::shd::CpuAlignmentHandle alignmentHandle;

};












}


#endif