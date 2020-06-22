
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
            

            std::array<std::vector<read_number>, 2> newCandidates;

            getCandidates(
                newCandidates[0], 
                currentAnchor[0].data(), 
                currentAnchorLength[0],
                currentAnchorReadId[0]
            );

            getCandidates(
                newCandidates[1], 
                currentAnchor[1].data(), 
                currentAnchorLength[1],
                currentAnchorReadId[1]
            );

            if(iter == 0){
                // remove self from candidate list
                for(int i = 0; i < 2; i++){
                    auto readIdPos = std::lower_bound(
                        newCandidates[i].begin(),                                            
                        newCandidates[i].end(),
                        currentAnchorReadId[i]
                    );

                    if(readIdPos != newCandidates[i].end() && *readIdPos == currentAnchorReadId[i]){
                        newCandidates[i].erase(readIdPos);
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
                std::min(newCandidates[0].size(), newCandidates[1].size())
            );

            auto mateIdsToKeep_end = std::set_intersection(
                newCandidates[0].begin(),
                newCandidates[0].end(),
                newCandidates[1].begin(),
                newCandidates[1].end(),
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

            std::vector<read_number> tmp(std::min(newCandidates[0].size(), mateIdsToKeep.size()));
            assert(tmp.size() == std::min(newCandidates[1].size(), mateIdsToKeep.size()));

            //cppref: ... elements will be copied from the first range to the destination range
            auto tmp_end1 = std::set_intersection(
                newCandidates[0].begin(),
                newCandidates[0].end(),
                mateIdsToKeep.begin(),
                mateIdsToKeep_end,
                tmp.begin(),
                mateIdLessThan
            );

            newCandidates[0].erase(
                std::copy(
                    tmp.begin(),
                    tmp_end1,
                    newCandidates[0].begin()
                ),
                newCandidates[0].end()
            );

            //cppref: ... elements will be copied from the first range to the destination range
            auto tmp_end2 = std::set_intersection(
                newCandidates[1].begin(),
                newCandidates[1].end(),
                mateIdsToKeep.begin(),
                mateIdsToKeep_end,
                tmp.begin(),
                mateIdLessThan
            );

            newCandidates[1].erase(
                std::copy(
                    tmp.begin(),
                    tmp_end2,
                    newCandidates[1].begin()
                ),
                newCandidates[1].end()
            );

            /*
                Load candidate sequences and compute reverse complements
            */

            cpu::ContiguousReadStorage::GatherHandle readStorageGatherHandle;

            std::array<std::vector<int>, 2> newCandidateSequenceLengths;
            std::array<std::vector<unsigned int>, 2> newCandidateSequencesData;
            std::array<std::vector<unsigned int>, 2> newCandidateSequencesRevcData;

            for(int i = 0; i < 2; i++){
                const int numCandidates = newCandidates[i].size();

                newCandidateSequenceLengths[i].resize(numCandidates);
                newCandidateSequencesData[i].resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);
                newCandidateSequencesRevcData[i].resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);

                readStorage->gatherSequenceLengths(
                    readStorageGatherHandle,
                    newCandidates[i].data(),
                    newCandidates[i].size(),
                    newCandidateSequenceLengths[i].data()
                );

                readStorage->gatherSequenceData(
                    readStorageGatherHandle,
                    newCandidates[i].data(),
                    newCandidates[i].size(),
                    newCandidateSequencesData[i].data(),
                    encodedSequencePitchInInts
                );

                for(int c = 0; c < numCandidates; c++){
                    const unsigned int* const seqPtr = newCandidateSequencesData[i].data() 
                                                        + std::size_t(encodedSequencePitchInInts) * c;
                    unsigned int* const seqrevcPtr = newCandidateSequencesRevcData[i].data() 
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

            std::array<std::vector<care::cpu::SHDResult>, 2> newForwardAlignments;
            std::array<std::vector<care::cpu::SHDResult>, 2> newRevcAlignments;
            std::array<std::vector<BestAlignment_t>, 2> newAlignmentFlags;

            for(int i = 0; i < 2; i++){

                const int numCandidates = newCandidates[i].size();

                newForwardAlignments[i].resize(numCandidates);
                newRevcAlignments[i].resize(numCandidates);
                newAlignmentFlags[i].resize(numCandidates);

                care::cpu::shd::cpuShiftedHammingDistancePopcount2Bit(
                    alignmentHandle,
                    newForwardAlignments[i].data(),
                    currentAnchor[i].data(),
                    currentAnchorLength[i],
                    newCandidateSequencesData[i].data(),
                    encodedSequencePitchInInts,
                    newCandidateSequenceLengths[i].data(),
                    numCandidates,
                    goodAlignmentProperties.min_overlap,
                    goodAlignmentProperties.maxErrorRate,
                    goodAlignmentProperties.min_overlap_ratio
                );

                care::cpu::shd::cpuShiftedHammingDistancePopcount2Bit(
                    alignmentHandle,
                    newRevcAlignments[i].data(),
                    currentAnchor[i].data(),
                    currentAnchorLength[i],
                    newCandidateSequencesRevcData[i].data(),
                    encodedSequencePitchInInts,
                    newCandidateSequenceLengths[i].data(),
                    numCandidates,
                    goodAlignmentProperties.min_overlap,
                    goodAlignmentProperties.maxErrorRate,
                    goodAlignmentProperties.min_overlap_ratio
                );

                //decide whether to keep forward or reverse complement

                for(int c = 0; c < numCandidates; c++){
                    const auto& forwardAlignment = newForwardAlignments[i][c];
                    const auto& revcAlignment = newRevcAlignments[i][c];
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
                }

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