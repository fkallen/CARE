
#include <config.hpp>
#include <sequence.hpp>
#include <minhasher.hpp>
#include <readstorage.hpp>
#include <options.hpp>
#include <cpu_alignment.hpp>
#include <bestalignment.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>
#include <iostream>
#include <string>


namespace care{


struct ReadExtender{
public:
    struct ExtendResult{

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

    ExtendResult extendPairedRead(
        ExtendInput& input
    ){
        ws.decodedPairedRead[0].resize(input.readLength1);
        ws.decodedPairedRead[1].resize(input.readLength2);

        decode2BitSequence(
            &ws.decodedPairedRead[0][0],
            input.encodedRead1,
            input.readLength1
        );

        decode2BitSequence(
            &ws.decodedPairedRead[1][0],
            input.encodedRead2,
            input.readLength2
        );

        std::array<std::vector<unsigned int>, 2> currentAnchor;
        std::array<int, 2> currentAnchorLength;
        std::array<read_number, 2> currentAnchorReadId;

        int iter = 0;
        while(iter < insertSize){

            if(iter == 0){
                currentAnchor[0].resize(input.numInts1);
                std::copy_n(input.encodedRead1, input.numInts1, currentAnchor[0].begin());
                currentAnchor[1].resize(input.numInts2);
                std::copy_n(input.encodedRead2, input.numInts2, currentAnchor[1].begin());

                currentAnchorLength[0] = input.readLength1;
                currentAnchorLength[1] = input.readLength2;

                currentAnchorReadId[0] = input.readId1;
                currentAnchorReadId[1] = input.readId2;
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
                [](auto id1, auto id2){
                    //read pairs have consecutive read ids. (0,1) (2,3) ...
                    //if id1 == id2, return true, else compare the smaller one. 

                    if(id1 == id2){
                        return true;
                    }else{

                        const auto firstId1 = id1 - id1 % 2;
                        const auto firstId2 = id2 - id2 % 2;

                        return firstId1 < firstId2;
                    }
                }
            );

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
            std::array<std::vector<unsigned int>, 2> newCandidateSequenceData;
            std::array<std::vector<unsigned int>, 2> newCandidateSequenceRevcData;

            for(int i = 0; i < 2; i++){
                const int numCandidates = newCandidateReadIds[i].size();

                newCandidateSequenceLengths[i].resize(numCandidates);
                newCandidateSequenceData[i].resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);
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
                    newCandidateSequenceData[i].data(),
                    encodedSequencePitchInInts
                );

                for(int c = 0; c < numCandidates; c++){
                    const unsigned int* const seqPtr = newCandidateSequenceData[i].data() 
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
                    newCandidateSequenceData[i].data(),
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

            int numRemainingCandidates = 0;

            //select remaining candidates
            for(int c = 0; c < size; c++){
                const BestAlignment_t alignmentFlag0 = newAlignmentFlags[0][c];
                const BestAlignment_t alignmentFlag1 = newAlignmentFlags[1][c];

                //if any of the mates aligns badly, remove both of them
                if(!(alignmentFlag0 == BestAlignment_t::None || alignmentFlag1 == BestAlignment_t::None)){
                    //keep
                    positionsOfCandidatesToKeep[numRemainingCandidates] = c;
                    numRemainingCandidates++;
                }else{
                    ; //don't keep
                }
            }

            //compact selected candidates inplace
            for(int i = 0; i < 2; i++){
                for(int c = 0; c < numRemainingCandidates; c++){
                    const int index = positionsOfCandidatesToKeep[c];

                    newAlignments[i][c] = newAlignments[i][index];
                    newAlignmentFlags[i][c] = newAlignmentFlags[i][index];
                    newCandidateReadIds[i][c] = newCandidateReadIds[i][index];
                    newCandidateSequenceLengths[i][c] = newCandidateSequenceLengths[i][index];
                    //TODO only keep the sequence with matching alignment orientation
                    std::copy_n(
                        newCandidateSequenceData[i].data() + index * encodedSequencePitchInInts,
                        encodedSequencePitchInInts,
                        newCandidateSequenceData[i].data() + c * encodedSequencePitchInInts
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
                newCandidateSequenceData[i].erase(
                    newCandidateSequenceData[i].begin() + numRemainingCandidates * encodedSequencePitchInInts, 
                    newCandidateSequenceData[i].end()
                );
                newCandidateSequenceRevcData[i].erase(
                    newCandidateSequenceRevcData[i].begin() + numRemainingCandidates * encodedSequencePitchInInts, 
                    newCandidateSequenceRevcData[i].end()
                );
                
            }
        }
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

} // namespace care