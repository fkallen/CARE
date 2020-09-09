
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

            getCandidates(
                candidateReadIds[0], 
                currentAnchor[0].data(), 
                currentAnchorLength[0],
                currentAnchorReadId[0]
            );

            getCandidates(
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
            std::array<std::vector<unsigned int>, 2> candidateSequenceFwdData;
            std::array<std::vector<unsigned int>, 2> candidateSequenceRevcData;

            for(int i = 0; i < 2; i++){
                const int numCandidates = candidateReadIds[i].size();

                candidateSequenceLengths[i].resize(numCandidates);
                candidateSequenceFwdData[i].resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);
                candidateSequenceRevcData[i].resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);

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
                    candidateSequenceFwdData[i].data(),
                    encodedSequencePitchInInts
                );

                for(int c = 0; c < numCandidates; c++){
                    const unsigned int* const seqPtr = candidateSequenceFwdData[i].data() 
                                                        + std::size_t(encodedSequencePitchInInts) * c;
                    unsigned int* const seqrevcPtr = candidateSequenceRevcData[i].data() 
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
                    candidateSequenceFwdData[i].data(),
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
                    candidateSequenceRevcData[i].data(),
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
                            candidateSequenceFwdData[i].data() + index * encodedSequencePitchInInts,
                            encodedSequencePitchInInts,
                            candidateSequenceData[i].data() + c * encodedSequencePitchInInts
                        );
                    }else{
                        //BestAlignment_t::ReverseComplement

                        std::copy_n(
                            candidateSequenceRevcData[i].data() + index * encodedSequencePitchInInts,
                            encodedSequencePitchInInts,
                            candidateSequenceData[i].data() + c * encodedSequencePitchInInts
                        );
                    }

                    //not sure if these 2 arrays will be required further on
                    std::copy_n(
                        candidateSequenceFwdData[i].data() + index * encodedSequencePitchInInts,
                        encodedSequencePitchInInts,
                        candidateSequenceFwdData[i].data() + c * encodedSequencePitchInInts
                    );

                    std::copy_n(
                        candidateSequenceRevcData[i].data() + index * encodedSequencePitchInInts,
                        encodedSequencePitchInInts,
                        candidateSequenceRevcData[i].data() + c * encodedSequencePitchInInts
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
                candidateSequenceFwdData[i].erase(
                    candidateSequenceFwdData[i].begin() + numRemainingCandidates * encodedSequencePitchInInts, 
                    candidateSequenceFwdData[i].end()
                );
                candidateSequenceRevcData[i].erase(
                    candidateSequenceRevcData[i].begin() + numRemainingCandidates * encodedSequencePitchInInts, 
                    candidateSequenceRevcData[i].end()
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
            

            hashTimer.start();

            std::vector<read_number> candidateReadIds;

//            if(iter == 0){

                getCandidates(
                    candidateReadIds, 
                    currentAnchor.data(), 
                    currentAnchorLength,
                    currentAnchorReadId
                );

            // }else{
            //     //only hash the right end of current anchor
            //     getCandidates(
            //         candidateReadIds, 
            //         currentAnchor.data(), 
            //         currentAnchorLength,
            //         currentAnchorReadId,
            //         std::max(0, currentAnchorLength - maxextension - minhasher->getKmerSize() + 1)
            //     );
            // }

            // remove self from candidate list
            if(iter == 0){
                auto readIdPos = std::lower_bound(
                    candidateReadIds.begin(),                                            
                    candidateReadIds.end(),
                    currentAnchorReadId
                );

                if(readIdPos != candidateReadIds.end() && *readIdPos == currentAnchorReadId){
                    candidateReadIds.erase(readIdPos);
                }
            }

            //remove mate of input from candidate list if it is not possible that mate could be reached at the current iteration
            if(input.readLength2 + accumExtensionLengths < insertSize - insertSizeStddev){
                auto readIdPos = std::lower_bound(
                    candidateReadIds.begin(),                                            
                    candidateReadIds.end(),
                    input.readId2
                );

                if(readIdPos != candidateReadIds.end() && *readIdPos == input.readId2){
                    candidateReadIds.erase(readIdPos);
                }
            }

            if(input.verbose){    
                verboseStream << "initial candidate read ids for anchor 0:\n";
                std::copy(
                    candidateReadIds.begin(),
                    candidateReadIds.end(),
                    std::ostream_iterator<read_number>(verboseStream, ",")
                );
                verboseStream << "\n";
            }

            hashTimer.stop();

            collectTimer.start();

            /*
                Remove candidate pairs which have already been used for extension
            */
            {

                std::vector<read_number> tmp(candidateReadIds.size());

                auto end = std::set_difference(
                    candidateReadIds.begin(),
                    candidateReadIds.end(),
                    allUsedCandidateReadIdPairs.begin(),
                    allUsedCandidateReadIdPairs.end(),
                    tmp.begin()
                );

                tmp.erase(end, tmp.end());

                std::swap(candidateReadIds, tmp);

            }



            if(input.verbose){    
                verboseStream << "new candidate read ids for anchor 0:\n";
                std::copy(
                    candidateReadIds.begin(),
                    candidateReadIds.end(),
                    std::ostream_iterator<read_number>(verboseStream, ",")
                );
                verboseStream << "\n";
            }

            /*
                Load candidate sequences and compute reverse complements
            */

            cpu::ContiguousReadStorage::GatherHandle readStorageGatherHandle;

            std::vector<int> candidateSequenceLengths;
            std::vector<unsigned int> candidateSequenceFwdData;
            std::vector<unsigned int> candidateSequenceRevcData;

            {
                const int numCandidates = candidateReadIds.size();

                candidateSequenceLengths.resize(numCandidates);
                candidateSequenceFwdData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);
                candidateSequenceRevcData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);

                readStorage->gatherSequenceLengths(
                    readStorageGatherHandle,
                    candidateReadIds.data(),
                    candidateReadIds.size(),
                    candidateSequenceLengths.data()
                );

                readStorage->gatherSequenceData(
                    readStorageGatherHandle,
                    candidateReadIds.data(),
                    candidateReadIds.size(),
                    candidateSequenceFwdData.data(),
                    encodedSequencePitchInInts
                );

                for(int c = 0; c < numCandidates; c++){
                    const unsigned int* const seqPtr = candidateSequenceFwdData.data() 
                                                        + std::size_t(encodedSequencePitchInInts) * c;
                    unsigned int* const seqrevcPtr = candidateSequenceRevcData.data() 
                                                        + std::size_t(encodedSequencePitchInInts) * c;

                    reverseComplement2Bit(
                        seqrevcPtr,  
                        seqPtr,
                        candidateSequenceLengths[c]
                    );
                }
            }

            collectTimer.stop();


            /*
                Compute alignments
            */

            cpu::shd::CpuAlignmentHandle alignmentHandle;

            std::vector<care::cpu::SHDResult> alignments;
            std::vector<BestAlignment_t> alignmentFlags;

            alignmentTimer.start();

            {

                const int numCandidates = candidateReadIds.size();

                std::vector<care::cpu::SHDResult> forwardAlignments;
                std::vector<care::cpu::SHDResult> revcAlignments;

                forwardAlignments.resize(numCandidates);
                revcAlignments.resize(numCandidates);
                alignmentFlags.resize(numCandidates);
                alignments.resize(numCandidates);

                care::cpu::shd::cpuShiftedHammingDistancePopcount2Bit<care::cpu::shd::ShiftDirection::Right>(
                    alignmentHandle,
                    forwardAlignments.data(),
                    currentAnchor.data(),
                    currentAnchorLength,
                    candidateSequenceFwdData.data(),
                    encodedSequencePitchInInts,
                    candidateSequenceLengths.data(),
                    numCandidates,
                    goodAlignmentProperties.min_overlap,
                    //currentAnchorLength - maxextension,
                    goodAlignmentProperties.maxErrorRate,
                    goodAlignmentProperties.min_overlap_ratio
                );

                care::cpu::shd::cpuShiftedHammingDistancePopcount2Bit<care::cpu::shd::ShiftDirection::Right>(
                    alignmentHandle,
                    revcAlignments.data(),
                    currentAnchor.data(),
                    currentAnchorLength,
                    candidateSequenceRevcData.data(),
                    encodedSequencePitchInInts,
                    candidateSequenceLengths.data(),
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
                    const int candidateLength = candidateSequenceLengths[c];

                    alignmentFlags[c] = care::choose_best_alignment(
                        forwardAlignment,
                        revcAlignment,
                        currentAnchorLength,
                        candidateLength,
                        goodAlignmentProperties.min_overlap_ratio,
                        goodAlignmentProperties.min_overlap,
                        correctionOptions.estimatedErrorrate
                    );

                    if(alignmentFlags[c] == BestAlignment_t::Forward){
                        alignments[c] = forwardAlignment;
                    }else{
                        alignments[c] = revcAlignment;
                    }
                }

            }

            alignmentTimer.stop();

            alignmentFilterTimer.start();

            /*
                Remove bad alignments and the corresponding alignments of their mate
            */        

            const int size = alignments.size();

            std::vector<int> positionsOfCandidatesToKeep(size);
            std::vector<int> tmpPositionsOfCandidatesToKeep(size);

            int numRemainingCandidates = 0;

            //select candidates with good alignment and positive shift
            for(int c = 0; c < size; c++){
                const BestAlignment_t alignmentFlag0 = alignmentFlags[c];
                
                if(alignmentFlag0 != BestAlignment_t::None && alignments[c].shift >= 0){
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

            //perform binning of candidates. keep candidates in best bins 
            //such that total number of kept candidates passes a threshold

            {
                constexpr int numBins = 5;
                std::array<int, numBins> bins{}; //relative overlap
                std::array<float, numBins> boundaries{1.0f, 0.7f, 0.6f, 0.5f, 0.0f};

                /*
                    100%,
                    70% - 100%
                    60% - 70%
                    50% - 60%
                    < 50%
                */
                for(int i = 0; i < numRemainingCandidates; i++){
                    const int candidateIndex = positionsOfCandidatesToKeep[i];
                    const auto& alignment = alignments[candidateIndex];

                    const float relativeOverlap = float(alignment.overlap) / float(input.readLength1);
                    if(fgeq(relativeOverlap, boundaries[0])){
                        bins[0]++;
                    }else if(fgeq(relativeOverlap, boundaries[1])){
                        bins[1]++;
                    }else if(fgeq(relativeOverlap, boundaries[2])){
                        bins[2]++;
                    }else if(fgeq(relativeOverlap, boundaries[3])){
                        bins[3]++;
                    }else{
                        bins[4]++;
                    }
                }

                const int threshold = 5;

                //select bins. bin[0] does not count towards threshold, because reads with overlap 100% cannot be used for extension,
                //but only for calculating consensus in MSA
                int selectedBin = 1;
                int numSelectedInBins = 0;
                for(int i = 1; i < numBins; i++){
                    numSelectedInBins += bins[i];
                    selectedBin = i;
                    if(numSelectedInBins >= threshold){
                        break;
                    }
                }

                // for(int i = 0; i <numBins; i++){
                //     std::cerr << bins[i] << " ";
                // }
                // std::cerr << "\n";

                int numRemainingCandidatesTmp = 0;

                for(int i = 0; i < numRemainingCandidates; i++){
                    const int candidateIndex = positionsOfCandidatesToKeep[i];
                    const auto& alignment = alignments[candidateIndex];

                    const float relativeOverlap = float(alignment.overlap) / float(input.readLength1);

                    if(fgeq(relativeOverlap, boundaries[selectedBin])){
                        tmpPositionsOfCandidatesToKeep[numRemainingCandidatesTmp++] = candidateIndex;
                    }
                }

                std::swap(tmpPositionsOfCandidatesToKeep, positionsOfCandidatesToKeep);
                std::swap(numRemainingCandidatesTmp, numRemainingCandidates);
            }



            //compact selected candidates inplace

            std::vector<unsigned int> candidateSequenceData;

            {
                candidateSequenceData.resize(numRemainingCandidates * encodedSequencePitchInInts);

                for(int c = 0; c < numRemainingCandidates; c++){
                    const int index = positionsOfCandidatesToKeep[c];

                    alignments[c] = alignments[index];
                    alignmentFlags[c] = alignmentFlags[index];
                    candidateReadIds[c] = candidateReadIds[index];
                    candidateSequenceLengths[c] = candidateSequenceLengths[index];
                    
                    assert(alignmentFlags[index] != BestAlignment_t::None);

                    if(alignmentFlags[index] == BestAlignment_t::Forward){
                        std::copy_n(
                            candidateSequenceFwdData.data() + index * encodedSequencePitchInInts,
                            encodedSequencePitchInInts,
                            candidateSequenceData.data() + c * encodedSequencePitchInInts
                        );
                    }else{
                        //BestAlignment_t::ReverseComplement

                        std::copy_n(
                            candidateSequenceRevcData.data() + index * encodedSequencePitchInInts,
                            encodedSequencePitchInInts,
                            candidateSequenceData.data() + c * encodedSequencePitchInInts
                        );
                    }

                    // //not sure if these 2 arrays will be required further on
                    // std::copy_n(
                    //     candidateSequenceFwdData.data() + index * encodedSequencePitchInInts,
                    //     encodedSequencePitchInInts,
                    //     candidateSequenceFwdData.data() + c * encodedSequencePitchInInts
                    // );

                    // std::copy_n(
                    //     candidateSequenceRevcData.data() + index * encodedSequencePitchInInts,
                    //     encodedSequencePitchInInts,
                    //     candidateSequenceRevcData.data() + c * encodedSequencePitchInInts
                    // );
                    
                }

                //erase past-end elements
                alignments.erase(
                    alignments.begin() + numRemainingCandidates, 
                    alignments.end()
                );
                alignmentFlags.erase(
                    alignmentFlags.begin() + numRemainingCandidates, 
                    alignmentFlags.end()
                );
                candidateReadIds.erase(
                    candidateReadIds.begin() + numRemainingCandidates, 
                    candidateReadIds.end()
                );
                candidateSequenceLengths.erase(
                    candidateSequenceLengths.begin() + numRemainingCandidates, 
                    candidateSequenceLengths.end()
                );
                // //not sure if these 2 arrays will be required further on
                // candidateSequenceFwdData.erase(
                //     candidateSequenceFwdData.begin() + numRemainingCandidates * encodedSequencePitchInInts, 
                //     candidateSequenceFwdData.end()
                // );
                // candidateSequenceRevcData.erase(
                //     candidateSequenceRevcData.begin() + numRemainingCandidates * encodedSequencePitchInInts, 
                //     candidateSequenceRevcData.end()
                // );
                
            }

            

            if(input.verbose){    
                verboseStream << "new candidate read ids for anchor 0 after alignments:\n";
                std::copy(
                    candidateReadIds.begin(),
                    candidateReadIds.end(),
                    std::ostream_iterator<read_number>(verboseStream, ",")
                );
                verboseStream << "\n";
            }

            //check if mate has been reached
            mateIdLocationIter = std::lower_bound(
                candidateReadIds.begin(),
                candidateReadIds.end(),
                input.readId2
            );

            mateHasBeenFound = (mateIdLocationIter != candidateReadIds.end() && *mateIdLocationIter == input.readId2);

            if(input.verbose){    
                verboseStream << "mate has been found ? " << (mateHasBeenFound ? "yes":"no") << "\n";
            }

            //check that extending to mate does not leave fragment
            if(mateHasBeenFound){
                const int mateIndex = std::distance(candidateReadIds.begin(), mateIdLocationIter);
                const auto& mateAlignment = alignments[mateIndex];

                if(accumExtensionLengths + input.readLength2 + mateAlignment.shift > insertSize + insertSizeStddev){
                    mateHasBeenFound = false;

                    alignments.erase(alignments.begin() + mateIndex);
                    alignmentFlags.erase(alignmentFlags.begin() + mateIndex);
                    candidateReadIds.erase(candidateReadIds.begin() + mateIndex);
                    candidateSequenceLengths.erase(candidateSequenceLengths.begin() + mateIndex);

                    candidateSequenceData.erase(
                        candidateSequenceData.begin() + mateIndex * encodedSequencePitchInInts,
                        candidateSequenceData.begin() + (mateIndex + 1) * encodedSequencePitchInInts
                    );

                    if(input.verbose){    
                        verboseStream << "mate has been removed again because it would reach beyond fragment\n";
                    }
                }
            }

            alignmentFilterTimer.stop();

            msaTimer.start();

            

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
                    candidateShifts[c] = alignments[c].shift;

                    candidateOverlapWeights[c] = calculateOverlapWeight(
                        currentAnchorLength, 
                        alignments[c].nOps,
                        alignments[c].overlap
                    );
                }

                std::vector<char> candidateStrings(decodedSequencePitchInBytes * numRemainingCandidates, '\0');

                for(int c = 0; c < numRemainingCandidates; c++){
                    decode2BitSequence(
                        candidateStrings.data() + c * decodedSequencePitchInBytes,
                        candidateSequenceData.data() + c * encodedSequencePitchInInts,
                        candidateSequenceLengths[c]
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
                msaInput.candidateLengths = candidateSequenceLengths.data();
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
                        const int index = std::distance(candidateReadIds.begin(), mateIdLocationIter);
                        const int shift = alignments[index].shift;
                        const int clength = candidateSequenceLengths[index];
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

            msaTimer.stop();

            /*
                update book-keeping of used candidates
            */                        
            {
                std::vector<read_number> tmp(allUsedCandidateReadIdPairs.size() + candidateReadIds.size());
                auto tmp_end = std::merge(
                    allUsedCandidateReadIdPairs.begin(),
                    allUsedCandidateReadIdPairs.end(),
                    candidateReadIds.begin(),
                    candidateReadIds.end(),
                    tmp.begin()
                );

                tmp.erase(tmp_end, tmp.end());

                std::swap(allUsedCandidateReadIdPairs, tmp);
            }

            usedCandidateReadIdsPerIteration.emplace_back(std::move(candidateReadIds));

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

    void printTimers(){
        hashTimer.print();
        collectTimer.print();
        alignmentTimer.print();
        alignmentFilterTimer.print();
        msaTimer.print();
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
        read_number readId,
        int beginPos = 0 // only positions [beginPos, readLength] are hashed
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

    helpers::CpuTimer hashTimer;
    helpers::CpuTimer collectTimer;
    helpers::CpuTimer alignmentTimer;
    helpers::CpuTimer alignmentFilterTimer;
    helpers::CpuTimer msaTimer;
};







MemoryFileFixedSize<ExtendedRead> 
//std::vector<ExtendedRead>
extend_cpu(
    const GoodAlignmentProperties& goodAlignmentProperties,
    const CorrectionOptions& correctionOptions,
    const ExtensionOptions& extensionOptions,
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
    cpu::RangeGenerator<read_number> readIdGenerator(100000);

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

    
    const int insertSize = extensionOptions.insertSize;
    const int insertSizeStddev = extensionOptions.insertSizeStddev;
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
                    if(result1->mateHasBeenFound){
                        er.status1 = ExtendedReadStatus::FoundMate;
                    }else{
                        if(result1->aborted){
                            if(result1->abortReason == ReadExtender::AbortReason::NoPairedCandidates
                                    || result1->abortReason == ReadExtender::AbortReason::NoPairedCandidatesAfterAlignment){

                                er.status1 = ExtendedReadStatus::CandidateAbort;
                            }else if(result1->abortReason == ReadExtender::AbortReason::MsaNotExtended){
                                er.status1 = ExtendedReadStatus::MSANoExtension;
                            }
                        }else{
                            er.status1 = ExtendedReadStatus::LengthAbort;
                        }
                    }
                }
                if(result2 != nullptr){
                    er.extendedRead2 = result2->extendedReads.front().second;
                    if(result2->mateHasBeenFound){
                        er.status2 = ExtendedReadStatus::FoundMate;
                    }else{
                        if(result2->aborted){
                            if(result2->abortReason == ReadExtender::AbortReason::NoPairedCandidates
                                    || result2->abortReason == ReadExtender::AbortReason::NoPairedCandidatesAfterAlignment){

                                er.status2 = ExtendedReadStatus::CandidateAbort;
                            }else if(result2->abortReason == ReadExtender::AbortReason::MsaNotExtended){
                                er.status2 = ExtendedReadStatus::MSANoExtension;
                            }
                        }else{
                            er.status2 = ExtendedReadStatus::LengthAbort;
                        }
                    }
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

            const int tid = omp_get_thread_num();

            if(0 == tid){
                readExtender.printTimers();
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