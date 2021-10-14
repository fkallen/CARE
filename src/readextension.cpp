
#include <config.hpp>
#include <sequencehelpers.hpp>
#include <cpuminhasher.hpp>
#include <cpureadstorage.hpp>
#include <options.hpp>
#include <cpu_alignment.hpp>
#include <alignmentorientation.hpp>
#include <msa.hpp>
#include <readextension_cpu.hpp>
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


#include <readextender_cpu.hpp>
#include <extendedread.hpp>
#include <rangegenerator.hpp>
#include <threadpool.hpp>
#include <util.hpp>
#include <filehelpers.hpp>
#include <readextender_common.hpp>

#include <omp.h>

#include <thrust/iterator/counting_iterator.h>


namespace care{




template<class Callback>
void extend_cpu_pairedend(
    const GoodAlignmentProperties& goodAlignmentProperties,
    const CorrectionOptions& correctionOptions,
    const ExtensionOptions& extensionOptions,
    const RuntimeOptions& runtimeOptions,
    const FileOptions& /*fileOptions*/,
    const MemoryOptions& memoryOptions,
    const CpuMinhasher& minhasher,
    const CpuReadStorage& readStorage,
    Callback submitReadyResults
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


    const std::size_t numReadsToProcess = 100000;
    //const std::size_t numReadsToProcess = readStorage.getNumberOfReads();

    auto readIdGenerator = makeIteratorRangeTraversal(
        thrust::make_counting_iterator<read_number>(0),
        thrust::make_counting_iterator<read_number>(0) + numReadsToProcess
    );

    // IteratorRangeTraversal<thrust::counting_iterator<read_number>> readIdGenerator(
    //     thrust::make_counting_iterator<read_number>(0) + 0,
    //     thrust::make_counting_iterator<read_number>(0) + 16
    // );
   
    const std::uint64_t totalNumReadPairs = readStorage.getNumberOfReads() / 2;

    auto showProgress = [&](auto totalCount, auto seconds){
        if(runtimeOptions.showProgress){

            printf("Processed %10u of %10lu read pairs (Runtime: %03d:%02d:%02d)\r",
                    totalCount, totalNumReadPairs,
                    int(seconds / 3600),
                    int(seconds / 60) % 60,
                    int(seconds) % 60);
            std::cout.flush();
        }

        if(totalCount == totalNumReadPairs){
            std::cerr << '\n';
        }
    };

    auto updateShowProgressInterval = [](auto duration){
        return duration;
    };

    ProgressThread<read_number> progressThread(totalNumReadPairs, showProgress, updateShowProgressInterval);

    
    const int insertSize = extensionOptions.insertSize;
    const int insertSizeStddev = extensionOptions.insertSizeStddev;
    const int maximumSequenceLength = readStorage.getSequenceLengthUpperBound();
    const std::size_t encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength);
    //const std::size_t decodedSequencePitchInBytes = maximumSequenceLength;
    const std::size_t qualityPitchInBytes = 4 * SDIV(maximumSequenceLength, 4);

    std::mutex verboseMutex;

    std::int64_t totalNumSuccess0 = 0;
    std::int64_t totalNumSuccess1 = 0;
    std::int64_t totalNumSuccess01 = 0;
    std::int64_t totalNumSuccessRead = 0;

    std::map<int, int> totalExtensionLengthsMap;

    std::map<int, int> totalMismatchesBetweenMateExtensions;

    //omp_set_num_threads(1);

    std::atomic<std::size_t> totalNumToRepeat{0};

    #pragma omp parallel
    {
        GoodAlignmentProperties goodAlignmentProperties2 = goodAlignmentProperties;
        //goodAlignmentProperties2.maxErrorRate = 0.05;

        constexpr int maxextensionPerStep = 20;

        cpu::QualityScoreConversion qualityConversion{};

        ReadExtenderCpu readExtender{
            insertSize,
            insertSizeStddev,
            maxextensionPerStep,
            maximumSequenceLength,
            readStorage, 
            minhasher,
            correctionOptions,
            goodAlignmentProperties2,
            &qualityConversion
        };

        std::int64_t numSuccess0 = 0;
        std::int64_t numSuccess1 = 0;
        std::int64_t numSuccess01 = 0;
        std::int64_t numSuccessRead = 0;

        std::map<int, int> extensionLengthsMap;
        std::map<int, int> mismatchesBetweenMateExtensions;

        const int batchsizePairs = correctionOptions.batchsize;

        std::vector<read_number> currentIds(2 * batchsizePairs);
        std::vector<unsigned int> currentEncodedReads(2 * encodedSequencePitchInInts * batchsizePairs);
        std::vector<int> currentReadLengths(2 * batchsizePairs);
        std::vector<char> currentQualityScores(2 * qualityPitchInBytes * batchsizePairs);

        if(!correctionOptions.useQualityScores){
            std::fill(currentQualityScores.begin(), currentQualityScores.end(), 'I');
        }

        int minCoverageForExtension = 3;
        int fixedStepsize = 20;

        readExtender.setMaxExtensionPerStep(fixedStepsize);
        readExtender.setMinCoverageForExtension(minCoverageForExtension);

        std::vector<std::pair<read_number, read_number>> pairsWhichShouldBeRepeated;
        std::vector<std::pair<read_number, read_number>> pairsWhichShouldBeRepeatedTemp;
        bool isLastIteration = false;
        bool isRepeatedIteration = false;

        auto init = [&](){
            int numReadsInBatch = 0;
            readIdGenerator.process_next_n(
                batchsizePairs * 2, 
                [&](auto begin, auto end){
                    auto readIdsEnd = std::copy(begin, end, currentIds.begin());
                    numReadsInBatch = std::distance(currentIds.begin(), readIdsEnd);
                }
            );

            if(numReadsInBatch % 2 == 1){
                throw std::runtime_error("Input files not properly paired. Aborting read extension.");
            }
            
            if(numReadsInBatch == 0 && pairsWhichShouldBeRepeated.size() > 0){

                const int numPairsToCopy = std::min(batchsizePairs, int(pairsWhichShouldBeRepeated.size()));

                for(int i = 0; i < numPairsToCopy; i++){
                    currentIds[2*i + 0] = pairsWhichShouldBeRepeated[i].first;
                    currentIds[2*i + 1] = pairsWhichShouldBeRepeated[i].second;
                }

                for(int i = 0; i < numPairsToCopy; i++){
                    if(currentIds[2*i + 0] > currentIds[2*i + 1]){
                        std::swap(currentIds[2*i + 0], currentIds[2*i + 1]);
                    }
                    assert(currentIds[2*i + 1] == currentIds[2*i + 0] + 1);
                }

                pairsWhichShouldBeRepeated.erase(pairsWhichShouldBeRepeated.begin(), pairsWhichShouldBeRepeated.begin() + numPairsToCopy);

                numReadsInBatch = 2 * numPairsToCopy;
            }

            if(numReadsInBatch > 0){

                readStorage.gatherSequenceLengths(
                    currentReadLengths.data(),
                    currentIds.data(),
                    currentIds.size()
                );

                readStorage.gatherSequences(
                    currentEncodedReads.data(),
                    encodedSequencePitchInInts,
                    currentIds.data(),
                    currentIds.size()
                );

                if(correctionOptions.useQualityScores){
                    readStorage.gatherQualities(
                        currentQualityScores.data(),
                        qualityPitchInBytes,
                        currentIds.data(),
                        numReadsInBatch
                    );
                }
            }

            const int numReadPairsInBatch = numReadsInBatch / 2;

            std::vector<extension::ExtendInput> inputs(numReadPairsInBatch);

            for(int i = 0; i < numReadPairsInBatch; i++){
                auto& input = inputs[i];

                input.readId1 = currentIds[2*i];
                input.readId2 = currentIds[2*i+1];
                input.readLength1 = currentReadLengths[2*i];
                input.readLength2 = currentReadLengths[2*i+1];
                input.encodedRead1.resize(encodedSequencePitchInInts);
                input.encodedRead2.resize(encodedSequencePitchInInts);
                std::copy_n(currentEncodedReads.data() + (2*i) * encodedSequencePitchInInts, encodedSequencePitchInInts, input.encodedRead1.begin());
                std::copy_n(currentEncodedReads.data() + (2*i + 1) * encodedSequencePitchInInts, encodedSequencePitchInInts, input.encodedRead2.begin());

                input.qualityScores1.resize(input.readLength1);
                input.qualityScores2.resize(input.readLength2);
                std::copy_n(currentQualityScores.data() + (2*i) * qualityPitchInBytes, input.readLength1, input.qualityScores1.begin());
                std::copy_n(currentQualityScores.data() + (2*i + 1) * qualityPitchInBytes, input.readLength2, input.qualityScores2.begin());
            }

            return inputs;
        };

        auto output = [&](auto extensionResults, bool isRepeatedIteration){
            auto splittedExtOutput = splitExtensionOutput(extensionResults, isRepeatedIteration);

            for(std::size_t i = 0; i < splittedExtOutput.idsOfPartiallyExtendedReads.size(); i += 2){
                pairsWhichShouldBeRepeatedTemp.push_back(std::make_pair(splittedExtOutput.idsOfPartiallyExtendedReads[i], splittedExtOutput.idsOfPartiallyExtendedReads[i+1]));
            }

            // pairsWhichShouldBeRepeatedTemp.insert(
            //     pairsWhichShouldBeRepeatedTemp.end(), 
            //     splittedExtOutput.idsOfPartiallyExtendedReads.begin(), 
            //     splittedExtOutput.idsOfPartiallyExtendedReads.end()
            // );            

            const std::size_t numExtended = splittedExtOutput.extendedReads.size();

            if(!extensionOptions.allowOutwardExtension){
                for(auto& er : splittedExtOutput.extendedReads){
                    er.removeOutwardExtension();
                }
            }

            std::vector<EncodedExtendedRead> encvec(numExtended);
            for(std::size_t i = 0; i < numExtended; i++){
                splittedExtOutput.extendedReads[i].encodeInto(encvec[i]);
            }

            submitReadyResults(
                std::move(splittedExtOutput.extendedReads), 
                std::move(encvec),
                std::move(splittedExtOutput.idsOfNotExtendedReads)
            );

            progressThread.addProgress(extensionResults.size() - splittedExtOutput.idsOfPartiallyExtendedReads.size());
        };

        isRepeatedIteration = false;

        #pragma omp single
        {

            std::cerr << "First iteration. insertsizedev: " << extensionOptions.insertSizeStddev 
                << ", maxextensionPerStep: " << fixedStepsize
                << ", minCoverageForExtension: " << minCoverageForExtension
                << ", isLastIteration: " << isLastIteration 
                << ", extraHashing: " << true << "\n";

        }
        
        while(!(readIdGenerator.empty())){
            auto inputs = init();

            if(inputs.size() > 0){
                auto extensionResults = readExtender.extend(std::move(inputs), isRepeatedIteration);

                output(std::move(extensionResults), isRepeatedIteration);
            }
        }


        //fixedStepsize -= 4;
        //minCoverageForExtension += increment;
        std::swap(pairsWhichShouldBeRepeatedTemp, pairsWhichShouldBeRepeated);
        pairsWhichShouldBeRepeatedTemp.clear();

        totalNumToRepeat += pairsWhichShouldBeRepeated.size();

        #pragma omp barrier

        #pragma omp single
        {
            std::cerr << "Will repeat extension of " << totalNumToRepeat << " read pairs with fixedStepsize = " << fixedStepsize << "\n";

            std::cerr << "Second iteration. insertsizedev: " << extensionOptions.insertSizeStddev 
            << ", maxextensionPerStep: " << fixedStepsize
            << ", minCoverageForExtension: " << minCoverageForExtension
            << ", isLastIteration: " << isLastIteration 
            << ", extraHashing: " << true << "\n";
        }


        isRepeatedIteration = true;
        isLastIteration = false;

        //while(pairsWhichShouldBeRepeated.size() > 0 && (fixedStepsize > 0))
        {

            readExtender.setMaxExtensionPerStep(fixedStepsize);
            //std::cerr << "fixedStepsize = " << fixedStepsize << "\n";

            //isLastIteration = (fixedStepsize <= 4);

            while(pairsWhichShouldBeRepeated.size() > 0){
                auto inputs = init();

                if(inputs.size() > 0){
                    auto extensionResults = readExtender.extend(std::move(inputs), isRepeatedIteration);

                    output(std::move(extensionResults), isRepeatedIteration);
                }
            }

            //fixedStepsize -= 4;
            std::swap(pairsWhichShouldBeRepeatedTemp, pairsWhichShouldBeRepeated);
            pairsWhichShouldBeRepeatedTemp.clear();
        }

        std::vector<read_number> tmpvec;
        tmpvec.reserve(pairsWhichShouldBeRepeated.size() * 2);
        for(const auto& p : pairsWhichShouldBeRepeated){
            tmpvec.push_back(p.first);
            tmpvec.push_back(p.second);
        }

        submitReadyResults(
            {}, 
            {},
            std::move(tmpvec) //pairs which did not find mate after repetition will remain unextended
        );

        #pragma omp critical
        {
            //std::lock_guard<std::mutex> lg(ompCriticalMutex);

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

    // std::cout << "totalNumSuccess0: " << totalNumSuccess0 << std::endl;
    // std::cout << "totalNumSuccess1: " << totalNumSuccess1 << std::endl;
    // std::cout << "totalNumSuccess01: " << totalNumSuccess01 << std::endl;
    // std::cout << "totalNumSuccessRead: " << totalNumSuccessRead << std::endl;

    // std::cout << "Extension lengths:\n";

    // for(const auto& pair : totalExtensionLengthsMap){
    //     std::cout << pair.first << ": " << pair.second << "\n";
    // }

    // std::cout << "mismatches between mate extensions:\n";

    // for(const auto& pair : totalMismatchesBetweenMateExtensions){
    //     std::cout << pair.first << ": " << pair.second << "\n";
    // }

}


#if 0

MemoryFileFixedSize<ExtendedRead> 
//std::vector<ExtendedRead>
extend_cpu_singleend(
    const GoodAlignmentProperties& goodAlignmentProperties,
    const CorrectionOptions& correctionOptions,
    const ExtensionOptions& extensionOptions,
    const RuntimeOptions& runtimeOptions,
    const FileOptions& fileOptions,
    const MemoryOptions& memoryOptions,
    const CpuMinhasher& minhasher,
    const CpuReadStorage& readStorage
){
    
}



#endif


void extend_cpu(
    const GoodAlignmentProperties& goodAlignmentProperties,
    const CorrectionOptions& correctionOptions,
    const ExtensionOptions& extensionOptions,
    const RuntimeOptions& runtimeOptions,
    const FileOptions& fileOptions,
    const MemoryOptions& memoryOptions,
    const CpuMinhasher& minhasher,
    const CpuReadStorage& readStorage,
    SubmitReadyExtensionResultsCallback submitReadyResults
){
    if(fileOptions.pairType == SequencePairType::SingleEnd){
        throw std::runtime_error("single end extension not possible");
        // return extend_cpu_singleend(
        //     goodAlignmentProperties,
        //     correctionOptions,
        //     extensionOptions,
        //     runtimeOptions,
        //     fileOptions,
        //     memoryOptions,
        //     minhasher,
        //     readStorage
        // );
    }else{
        extend_cpu_pairedend(
            goodAlignmentProperties,
            correctionOptions,
            extensionOptions,
            runtimeOptions,
            fileOptions,
            memoryOptions,
            minhasher,
            readStorage,
            submitReadyResults
        );
    }
}














} // namespace care