
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




#include <readextension_cpu.hpp>
#include <correctionresultprocessing.hpp>
#include <rangegenerator.hpp>
#include <threadpool.hpp>
#include <memoryfile.hpp>
#include <util.hpp>
#include <filehelpers.hpp>

#include <omp.h>


namespace care{


struct ReadExtender{
public:
    struct ExtendResult{
        bool success = false;
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
        read_number readId1;
        const unsigned int* encodedRead1;
        int readLength1;
        int numInts1;
        read_number readId2;
        const unsigned int* encodedRead2;
        int readLength2;
        int numInts2;
    };

    ReadExtender(
        int insertSize,
        int maximumSequenceLength,
        const cpu::ContiguousReadStorage& rs, 
        const Minhasher& mh
        
    ) : insertSize(insertSize), maximumSequenceLength(maximumSequenceLength),
        minhasher(&mh), readStorage(&rs){

        encodedSequencePitchInInts = getEncodedNumInts2Bit(maximumSequenceLength);
        decodedSequencePitchInBytes = maximumSequenceLength;
        qualityPitchInBytes = maximumSequenceLength;

    }

    /*
        Assumes read1 is on the forward strand, read2 is on the reverse strand
    */
    ExtendResult extendPairedRead(
        ExtendInput& input
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
        std::array<std::vector<std::vector<read_number>>, 2> totalCandidateReadIdsPerIteration;

        //saves the read ids of used candidates for each strand over all iterations
        std::array<std::vector<read_number>, 2> totalCandidateReadIds; 

        bool abort = false;
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

        int iter = 0;
        while(iter < insertSize && !abort && !mateHasBeenFound){

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

            /*
                Remove new candidates whose mates are not candidates of the other read.
            */

            auto mateIdLessThan = [](auto id1, auto id2){
                //read pairs have consecutive read ids. (0,1) (2,3) ...
                //compare the smaller one. 

                const auto firstId1 = id1 - id1 % 2;
                const auto firstId2 = id2 - id2 % 2;

                return firstId1 < firstId2;
            };

            std::vector<read_number> mateIdsToKeep(
                std::min(newCandidateReadIds[0].size(), newCandidateReadIds[1].size())
            );

            auto mateIdsToKeep_end = std::set_intersection(
                newCandidateReadIds[0].begin(),
                newCandidateReadIds[0].end(),
                newCandidateReadIds[1].begin(),
                newCandidateReadIds[1].end(),
                mateIdsToKeep.begin(),
                [&](auto id1, auto id2){

                    if(id1 == id2){
                        return true; //this will remove equal read Ids from both sets
                    }else{
                        return mateIdLessThan(id1, id2);
                    }
                }
            );

            if(std::distance(mateIdsToKeep.begin(), mateIdsToKeep_end) == 0){
                abort = true;
                break; //terminate while loop
            }

            std::vector<read_number> tmp(std::min(newCandidateReadIds[0].size(), mateIdsToKeep.size()));
            assert(tmp.size() == std::min(newCandidateReadIds[1].size(), mateIdsToKeep.size()));

            //cppref: ... elements will be copied from the first range to the destination range
            auto tmp_end1 = std::set_intersection(
                newCandidateReadIds[0].begin(),
                newCandidateReadIds[0].end(),
                mateIdsToKeep.begin(),
                mateIdsToKeep_end,
                tmp.begin(),
                mateIdLessThan
            );

            newCandidateReadIds[0].erase(
                std::copy(
                    tmp.begin(),
                    tmp_end1,
                    newCandidateReadIds[0].begin()
                ),
                newCandidateReadIds[0].end()
            );

            //cppref: ... elements will be copied from the first range to the destination range
            auto tmp_end2 = std::set_intersection(
                newCandidateReadIds[1].begin(),
                newCandidateReadIds[1].end(),
                mateIdsToKeep.begin(),
                mateIdsToKeep_end,
                tmp.begin(),
                mateIdLessThan
            );

            newCandidateReadIds[1].erase(
                std::copy(
                    tmp.begin(),
                    tmp_end2,
                    newCandidateReadIds[1].begin()
                ),
                newCandidateReadIds[1].end()
            );

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

                care::cpu::shd::cpuShiftedHammingDistancePopcount2Bit(
                    alignmentHandle,
                    newForwardAlignments.data(),
                    currentAnchor[i].data(),
                    currentAnchorLength[i],
                    newCandidateSequenceFwdData[i].data(),
                    encodedSequencePitchInInts,
                    newCandidateSequenceLengths[i].data(),
                    numCandidates,
                    goodAlignmentProperties.min_overlap,
                    goodAlignmentProperties.maxErrorRate,
                    goodAlignmentProperties.min_overlap_ratio
                );

                care::cpu::shd::cpuShiftedHammingDistancePopcount2Bit(
                    alignmentHandle,
                    newRevcAlignments.data(),
                    currentAnchor[i].data(),
                    currentAnchorLength[i],
                    newCandidateSequenceRevcData[i].data(),
                    encodedSequencePitchInInts,
                    newCandidateSequenceLengths[i].data(),
                    numCandidates,
                    goodAlignmentProperties.min_overlap,
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

            numRemainingCandidatesTmp = 0;

            //remove candidates which have already been used in a previous iteration
            assert(std::is_sorted(totalCandidateReadIds[0].begin(), totalCandidateReadIds[0].end()));

            for(int c = 0; c < numRemainingCandidates; c++){
                const int index = positionsOfCandidatesToKeep[c];
                const read_number readId = newCandidateReadIds[0][index];
                //because of symmetry it is sufficient to check one strand

                auto it = std::lower_bound(
                    totalCandidateReadIds[0].begin(),
                    totalCandidateReadIds[0].end(),
                    readId
                );

                if(!(it != totalCandidateReadIds[0].end() && *it == readId)){
                    //readId not found, keep it
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
                break; //terminate while loop
            }

            std::array<std::vector<unsigned int>, 2> newCandidateSequenceData;

            //compact selected candidates inplace
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
                        //BestAlignment_T::ReverseComplement

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

            //update "total" arrays of candidates
            for(int i = 0; i < 2; i++){
                totalCandidateReadIdsPerIteration[i].emplace_back(newCandidateReadIds[i]);

                assert(std::is_sorted(newCandidateReadIds[i].begin(), newCandidateReadIds[i].end()));

                std::vector<read_number> tmp(totalCandidateReadIds[i].size() + newCandidateReadIds[i].size());
                auto tmp_end = std::merge(
                    totalCandidateReadIds[i].begin(),
                    totalCandidateReadIds[i].end(),
                    newCandidateReadIds[i].begin(),
                    newCandidateReadIds[i].end(),
                    tmp.begin()
                );
                assert(tmp_end == tmp.end()); // duplicates should have been removed
                //tmp.erase(tmp_end, tmp.end());

                std::swap(totalCandidateReadIds[i], tmp);
            }

            //check if mate has been reached
            mateIdLocationIter = std::lower_bound(
                newCandidateReadIds[0].begin(),
                newCandidateReadIds[0].end(),
                input.readId2
            );

            mateHasBeenFound = (mateIdLocationIter != newCandidateReadIds[0].end() && *mateIdLocationIter == input.readId2);

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


                if(!mateHasBeenFound){
                    //mate not found. prepare next while-loop iteration
                    int consensusLength = msa.consensus.size();

                    //the first currentAnchorLength[i] columns are occupied by anchor. try to extend read 
                    //by at most 10 bp. In case consensuslength == anchorlength, abort

                    if(consensusLength == currentAnchorLength[i]){
                        abort = true;
                        
                    }else{
                        assert(consensusLength > currentAnchorLength[i]);
                        constexpr int maxextension = 10;

                        const int extendBy = std::min(consensusLength - currentAnchorLength[i], maxextension);
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

                        std::string decodedAnchor(msa.consensus.data(), endcolumn);

                        totalDecodedAnchors[i].emplace_back(std::move(decodedAnchor));
                        totalAnchorBeginInExtendedRead[i].emplace_back(accumExtensionLengths[i]);
                    }
                }

            }



            iter++; //control outer while-loop
        }

        ExtendResult extendResult;


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

                const auto& shifts = totalAnchorBeginInExtendedRead[0];
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
                
                extendResult.success = true;

                std::string extendedRead(msa.consensus.begin(), msa.consensus.end());
                extendResult.extendedReads.emplace_back(input.readId1, std::move(extendedRead));
            }else{
                ; //no read extension possible
            }
        }

        return extendResult;
    }

private:

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


    int insertSize;
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







MemoryFileFixedSize<EncodedTempCorrectedSequence> 
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

    if(availableMemoryInBytes > 2*(std::size_t(1) << 30)){
        memoryForPartialResultsInBytes = availableMemoryInBytes - 2*(std::size_t(1) << 30);
    }

    const std::string tmpfilename{fileOptions.tempdirectory + "/" + "MemoryFileFixedSizetmp"};
    MemoryFileFixedSize<EncodedTempCorrectedSequence> partialResults(memoryForPartialResultsInBytes, tmpfilename);

    cpu::RangeGenerator<read_number> readIdGenerator(sequenceFileProperties.nReads);

    BackgroundThread outputThread(true);

    /*

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

    */
    const int insertSize = 300;
    const int maximumSequenceLength = sequenceFileProperties.maxSequenceLength;
    const std::size_t encodedSequencePitchInInts = getEncodedNumInts2Bit(maximumSequenceLength);

    omp_set_num_threads(1);

    //#pragma omp parallel
    {
        ReadExtender readExtender{
            insertSize,
            maximumSequenceLength,
            readStorage, 
            minhasher      
        };

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
                ReadExtender::ExtendInput input;

                input.readId1 = currentIds[order[0]];
                input.readId2 = currentIds[order[1]];
                input.encodedRead1 = currentEncodedReads.data() + order[0] * encodedSequencePitchInInts;
                input.encodedRead2 = currentEncodedReads.data() + order[1] * encodedSequencePitchInInts;
                input.readLength1 = currentReadLengths[order[0]];
                input.readLength2 = currentReadLengths[order[1]];
                input.numInts1 = getEncodedNumInts2Bit(currentReadLengths[order[0]]);
                input.numInts2 = getEncodedNumInts2Bit(currentReadLengths[order[1]]);

                auto extendResult = readExtender.extendPairedRead(input);

                if(extendResult.success){
                    const int numResults = extendResult.extendedReads.size();
                    auto encodeddata = std::make_unique<EncodedTempCorrectedSequence[]>(numResults);

                    for(int i = 0; i < numResults; i++){
                        auto& pair = extendResult.extendedReads[i];

                        TempCorrectedSequence tcs;
                        tcs.hq = false;
                        tcs.useEdits = false;
                        tcs.type = TempCorrectedSequence::Type::Anchor;
                        tcs.shift = 0;
                        tcs.readId = pair.first;
                        tcs.sequence = std::move(pair.second);

                        encodeddata[i] = tcs.encode();
                    }

                    auto func = [&, size = numResults, encodeddata = encodeddata.release()](){
                        for(int i = 0; i < size; i++){
                            partialResults.storeElement(std::move(encodeddata[i]));
                        }
                    };

                    outputThread.enqueue(
                        std::move(func)
                    );
                } 

                return extendResult;  
            };

            // it is not known which of both reads is on the forward strand / reverse strand. try both combinations
            auto extendResult0 = processReadOrder({0,1});

            auto extendResult1 = processReadOrder({1,0});

            //std::cerr << "success0 " << success0 << ", success1 " << success1 << "\n";
            if(extendResult0.success || extendResult1.success){
                std::cerr << "success0 " << extendResult0.success 
                    << ", success1 " << extendResult1.success << "\n";
            }
            
        }
        
    }

    //progressThread.finished();

    outputThread.stopThread(BackgroundThread::StopType::FinishAndStop);

    //outputstream.flush();
    partialResults.flush();



    return partialResults;
}



















} // namespace care