
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


#include <readextender_cpu.hpp>
#include <extensionresultprocessing.hpp>
#include <correctionresultprocessing.hpp>
#include <rangegenerator.hpp>
#include <threadpool.hpp>
#include <memoryfile.hpp>
#include <util.hpp>
#include <filehelpers.hpp>
#include <readextender_common.hpp>

#include <omp.h>


namespace care{





MemoryFileFixedSize<ExtendedRead> 
//std::vector<ExtendedRead>
extend_cpu_pairedend(
    const GoodAlignmentProperties& goodAlignmentProperties,
    const CorrectionOptions& correctionOptions,
    const ExtensionOptions& extensionOptions,
    const RuntimeOptions& runtimeOptions,
    const FileOptions& fileOptions,
    const MemoryOptions& memoryOptions,
    const CpuMinhasher& minhasher,
    const CpuReadStorage& readStorage
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

    // {
    //     std::vector<std::string> queries{
    //         "AAAAAGTCGAGATTTTCGCACAAAAAGTTGAATTTTGAAAAACTCAAAACTTTTTCAGCGGTTTCGTTATGAAAATCAGGTAATTTCAGCATCTAAGCAT",
    //         "AAAAAGTCGAGATTTTCGCACAAAAAGTTGAATTTTGAAAAACTCAAAACTTTTTCAGCGGTCTCGTTATGAAAATCAGGTAATTTCAGCATCTAAGCAT"
    //     };

    //     auto minhashHandle = minhasher.makeMinhasherHandle();
    //     const int encodedSequencePitchInInts = 16;

    //     for(const auto& q : queries){

    //         std::vector<unsigned int> encodedRead(encodedSequencePitchInInts);
    //         SequenceHelpers::encodeSequence2Bit(
    //             encodedRead.data(), 
    //             q.data(), 
    //             q.size()
    //         );

    //         int numValuesPerSequence = 0;
    //         int totalNumValues = 0;
    //         int readLength = q.size();

    //         minhasher.determineNumValues(
    //             minhashHandle,
    //             encodedRead.data(),
    //             encodedSequencePitchInInts,
    //             &readLength,
    //             1,
    //             &numValuesPerSequence,
    //             totalNumValues
    //         );

    //         std::vector<read_number> result(totalNumValues);
    //         std::array<int, 2> offsets{};

    //         minhasher.retrieveValues(
    //             minhashHandle,
    //             nullptr, //do not remove selfid
    //             1,
    //             totalNumValues,
    //             result.data(),
    //             &numValuesPerSequence,
    //             offsets.data()
    //         );

    //         result.erase(result.begin() + numValuesPerSequence, result.end());

    //         std::cerr << "Query: " << q << "\n";
    //         std::cerr << numValuesPerSequence << " candidates\n";
    //         for(int k = 0; k < numValuesPerSequence; k++){
    //             std::cerr << result[k] << " ";
    //         }
    //         std::cerr << "\n";
    //     }

    //     minhasher.destroyHandle(minhashHandle);
    // }


    const std::size_t availableMemoryInBytes = memoryAvailableBytesHost; //getAvailableMemoryInKB() * 1024;
    std::size_t memoryForPartialResultsInBytes = 0;

    if(availableMemoryInBytes > 3*(std::size_t(1) << 30)){
        memoryForPartialResultsInBytes = availableMemoryInBytes - 3*(std::size_t(1) << 30);
    }

    const std::string tmpfilename{fileOptions.tempdirectory + "/" + "MemoryFileFixedSizetmp"};
    MemoryFileFixedSize<ExtendedRead> partialResults(memoryForPartialResultsInBytes, tmpfilename);

    std::vector<ExtendedRead> resultExtendedReads;

    //cpu::RangeGenerator<read_number> readIdGenerator(readStorage.getNumberOfReads());
    cpu::RangeGenerator<read_number> readIdGenerator(1000000);

    BackgroundThread outputThread(true);

    
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
    const std::size_t decodedSequencePitchInBytes = maximumSequenceLength;
    const std::size_t qualityPitchInBytes = 4 * SDIV(maximumSequenceLength, 4);

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
        bool isLastIteration = true;

        auto init = [&](){
            auto readIdsEnd = readIdGenerator.next_n_into_buffer(
                batchsizePairs * 2, 
                currentIds.begin()
            );

            int numReadsInBatch = std::distance(currentIds.begin(), readIdsEnd);

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

        auto output = [&](auto extensionResults){
            const int numresults = extensionResults.size();

            std::vector<ExtendedRead> extendedReads;
            extendedReads.reserve(numresults);

            int repeated = 0;

            for(int i = 0; i < numresults; i++){
                auto& extensionOutput = extensionResults[i];
                const int extendedReadLength = extensionOutput.extendedRead.size();
                //if(extendedReadLength == extensionOutput.originalLength){
                //if(!extensionOutput.mateHasBeenFound){
                if(extendedReadLength > extensionOutput.originalLength && !extensionOutput.mateHasBeenFound && !isLastIteration){
                    //do not insert directly into pairsWhichShouldBeRepeated. it causes an infinite loop
                    pairsWhichShouldBeRepeatedTemp.emplace_back(std::make_pair(extensionOutput.readId1, extensionOutput.readId2));
                    repeated++;
                }else{
                    //assert(extensionOutput.extendedRead.size() > extensionOutput.originalLength);

                    ExtendedRead er;

                    er.readId = extensionOutput.readId1;
                    er.mergedFromReadsWithoutMate = extensionOutput.mergedFromReadsWithoutMate;
                    er.extendedSequence = std::move(extensionOutput.extendedRead);
                    //er.qualityScores = std::move(extensionOutput.qualityScores);
                    er.read1begin = extensionOutput.read1begin;
                    er.read1end = extensionOutput.read1begin + extensionOutput.originalLength;
                    er.read2begin = extensionOutput.read2begin;
                    if(er.read2begin != -1){
                        er.read2end = extensionOutput.read2begin + extensionOutput.originalMateLength;
                    }else{
                        er.read2end = -1;
                    }

                    if(extensionOutput.mateHasBeenFound){
                        er.status = ExtendedReadStatus::FoundMate;
                    }else{
                        if(extensionOutput.aborted){
                            if(extensionOutput.abortReason == extension::AbortReason::NoPairedCandidates
                                    || extensionOutput.abortReason == extension::AbortReason::NoPairedCandidatesAfterAlignment){

                                er.status = ExtendedReadStatus::CandidateAbort;
                            }else if(extensionOutput.abortReason == extension::AbortReason::MsaNotExtended){
                                er.status = ExtendedReadStatus::MSANoExtension;
                            }
                        }else{
                            er.status = ExtendedReadStatus::LengthAbort;
                        }
                    }  
                    
                    extendedReads.emplace_back(std::move(er));

                }
                              
            }

            auto outputfunc = [&, vec = std::move(extendedReads)](){
                for(const auto& er : vec){
                    partialResults.storeElement(&er);
                }
            };

            outputThread.enqueue(
                std::move(outputfunc)
            );

            progressThread.addProgress(extensionResults.size() - repeated);
        };
        

        isLastIteration = false;
        while(!(readIdGenerator.empty())){
            auto inputs = init();

            if(inputs.size() > 0){
                auto extensionResults = readExtender.extend(std::move(inputs));

                output(std::move(extensionResults));
            }
        }

        fixedStepsize -= 2;
        //minCoverageForExtension += increment;
        std::swap(pairsWhichShouldBeRepeatedTemp, pairsWhichShouldBeRepeated);

        while(pairsWhichShouldBeRepeated.size() > 0 && (fixedStepsize > 0)){

            readExtender.setMaxExtensionPerStep(fixedStepsize);
            //std::cerr << "fixedStepsize = " << fixedStepsize << "\n";

            //std::cerr << "Will repeat extension of " << pairsWhichShouldBeRepeated.size() << " read pairs with fixedStepsize = " << fixedStepsize << "\n";
            isLastIteration = (fixedStepsize <= 2);

            while(pairsWhichShouldBeRepeated.size() > 0){
                auto inputs = init();

                if(inputs.size() > 0){
                    auto extensionResults = readExtender.extend(std::move(inputs));

                    output(std::move(extensionResults));
                }
            }

            fixedStepsize -= 2;
            std::swap(pairsWhichShouldBeRepeatedTemp, pairsWhichShouldBeRepeated);
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
    std::cerr << "extend_cpu_singleend\n";
    
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

    const std::size_t availableMemoryInBytes = memoryAvailableBytesHost; //getAvailableMemoryInKB() * 1024;
    std::size_t memoryForPartialResultsInBytes = 0;

    if(availableMemoryInBytes > 3*(std::size_t(1) << 30)){
        memoryForPartialResultsInBytes = availableMemoryInBytes - 3*(std::size_t(1) << 30);
    }

    const std::string tmpfilename{fileOptions.tempdirectory + "/" + "MemoryFileFixedSizetmp"};
    MemoryFileFixedSize<ExtendedRead> partialResults(memoryForPartialResultsInBytes, tmpfilename);

    std::vector<ExtendedRead> resultExtendedReads;

    cpu::RangeGenerator<read_number> readIdGenerator(readStorage.getNumberOfReads());
    //cpu::RangeGenerator<read_number> readIdGenerator(1000);

    BackgroundThread outputThread(true);

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

    ProgressThread<read_number> progressThread(readStorage.getNumberOfReads(), showProgress, updateShowProgressInterval);

    
    const int insertSize = extensionOptions.insertSize;
    const int insertSizeStddev = extensionOptions.insertSizeStddev;
    const int maximumSequenceLength = readStorage.getSequenceLengthUpperBound();
    const std::size_t encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength);

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

        constexpr int maxextensionPerStep = 20;

        ReadExtenderCpu readExtender{
            insertSize,
            insertSizeStddev,
            maxextensionPerStep,
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

        const int batchsize = correctionOptions.batchsize;

        std::vector<read_number> currentIds(batchsize);
        std::vector<unsigned int> currentEncodedReads(encodedSequencePitchInInts * batchsize);
        std::vector<int> currentReadLengths(batchsize);
        

        while(!(readIdGenerator.empty())){

            auto readIdsEnd = readIdGenerator.next_n_into_buffer(
                batchsize, 
                currentIds.begin()
            );

            const int numReadsInBatch = std::distance(currentIds.begin(), readIdsEnd);
            
            if(numReadsInBatch == 0){
                continue; //this should only happen if all reads have been processed
            }

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

            std::vector<extension::ExtendInput> inputs(numReadsInBatch);

            for(int i = 0; i < numReadsInBatch; i++){
                auto& input = inputs[i];

                input.readId1 = currentIds[i];
                input.readId2 = std::numeric_limits<read_number>::max();
                input.readLength1 = currentReadLengths[i];
                input.readLength2 = 0;
                input.encodedRead1.resize(encodedSequencePitchInInts);
                input.encodedRead2.resize(0);
                std::copy_n(currentEncodedReads.data() + (2*i) * encodedSequencePitchInInts, encodedSequencePitchInInts, input.encodedRead1.begin());
            }

            auto extensionResultsBatch = readExtender.extendSingleEndReadBatch(inputs);

            //convert results of ReadExtender
            std::vector<ExtendedRead> resultvector(extensionResultsBatch.size());

            for(int i = 0; i < numReadsInBatch; i++){
                auto& extensionOutput = extensionResultsBatch[i];
                ExtendedRead& er = resultvector[i];

                er.readId = extensionOutput.readId1;
                er.extendedSequence = std::move(extensionOutput.extendedRead);

                if(extensionOutput.mateHasBeenFound){
                    er.status = ExtendedReadStatus::FoundMate;
                }else{
                    if(extensionOutput.aborted){
                        if(extensionOutput.abortReason == extension::AbortReason::NoPairedCandidates
                                || extensionOutput.abortReason == extension::AbortReason::NoPairedCandidatesAfterAlignment){

                            er.status = ExtendedReadStatus::CandidateAbort;
                        }else if(extensionOutput.abortReason == extension::AbortReason::MsaNotExtended){
                            er.status = ExtendedReadStatus::MSANoExtension;
                        }
                    }else{
                        er.status = ExtendedReadStatus::LengthAbort;
                    }
                }  
                
                if(extensionOutput.success){
                    numSuccessRead++;
                }                
            }

            auto outputfunc = [&, vec = std::move(resultvector)](){
                for(const auto& er : vec){
                    partialResults.storeElement(&er);
                }
            };

            outputThread.enqueue(
                std::move(outputfunc)
            );

            progressThread.addProgress(numReadsInBatch);            
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



#endif


MemoryFileFixedSize<ExtendedRead> 
//std::vector<ExtendedRead>
extend_cpu(
    const GoodAlignmentProperties& goodAlignmentProperties,
    const CorrectionOptions& correctionOptions,
    const ExtensionOptions& extensionOptions,
    const RuntimeOptions& runtimeOptions,
    const FileOptions& fileOptions,
    const MemoryOptions& memoryOptions,
    const CpuMinhasher& minhasher,
    const CpuReadStorage& readStorage
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
        return extend_cpu_pairedend(
            goodAlignmentProperties,
            correctionOptions,
            extensionOptions,
            runtimeOptions,
            fileOptions,
            memoryOptions,
            minhasher,
            readStorage
        );
    }
}














} // namespace care