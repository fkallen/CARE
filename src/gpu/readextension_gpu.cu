
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

#include <gpu/readextender_gpu.hpp>
#include <extensionresultprocessing.hpp>
#include <rangegenerator.hpp>
#include <threadpool.hpp>
#include <memoryfile.hpp>
#include <util.hpp>
#include <filehelpers.hpp>

#include <omp.h>


namespace care{
namespace gpu{






MemoryFileFixedSize<ExtendedRead> 
//std::vector<ExtendedRead>
extend_gpu(
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

    cpu::RangeGenerator<read_number> readIdGenerator(sequenceFileProperties.nReads);
    //cpu::RangeGenerator<read_number> readIdGenerator(1000000);

    BackgroundThread outputThread(true);

    const std::uint64_t totalNumReadPairs = sequenceFileProperties.nReads / 2;

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
    const int maximumSequenceLength = sequenceFileProperties.maxSequenceLength;
    const std::size_t encodedSequencePitchInInts = getEncodedNumInts2Bit(maximumSequenceLength);

    std::mutex verboseMutex;
    std::mutex ompCriticalMutex;

    std::int64_t totalNumSuccess0 = 0;
    std::int64_t totalNumSuccess1 = 0;
    std::int64_t totalNumSuccess01 = 0;
    std::int64_t totalNumSuccessRead = 0;

    std::map<int, int> totalExtensionLengthsMap;

    std::map<int, int> totalMismatchesBetweenMateExtensions;

    //omp_set_num_threads(1);

    #pragma omp parallel
    {
        const int deviceId = runtimeOptions.deviceIds.at(0);
        cudaSetDevice(deviceId); CUERR;

        GoodAlignmentProperties goodAlignmentProperties2 = goodAlignmentProperties;
        //goodAlignmentProperties2.maxErrorRate = 0.05;

        ReadExtenderGpu readExtenderGpu{
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

        const int batchsizePairs = correctionOptions.batchsize;

        std::vector<read_number> currentIds(2 * batchsizePairs);
        std::vector<unsigned int> currentEncodedReads(2 * encodedSequencePitchInInts * batchsizePairs);
        std::vector<int> currentReadLengths(2 * batchsizePairs);
        

        while(!(readIdGenerator.empty())){

            auto readIdsEnd = readIdGenerator.next_n_into_buffer(
                batchsizePairs * 2, 
                currentIds.begin()
            );

            const int numReadsInBatch = std::distance(currentIds.begin(), readIdsEnd);

            if(numReadsInBatch % 2 == 1){
                throw std::runtime_error("Input files not properly paired. Aborting read extension.");
            }
            
            if(numReadsInBatch == 0){
                continue; //this should only happen if all reads have been processed
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

            const int numReadPairsInBatch = numReadsInBatch / 2;

            auto processReadOrder = [&](std::array<int, 2> order){                

                std::vector<ReadExtenderGpu::ExtendInput> inputs(numReadPairsInBatch);

                for(int i = 0; i < numReadPairsInBatch; i++){
                    auto& input = inputs[i];

                    input.readId1 = currentIds[2*i + order[0]];
                    input.readId2 = currentIds[2*i + order[1]];
                    input.encodedRead1 = currentEncodedReads.data() + (2*i + order[0]) * encodedSequencePitchInInts;
                    input.encodedRead2 = currentEncodedReads.data() + (2*i + order[1]) * encodedSequencePitchInInts;
                    input.readLength1 = currentReadLengths[2*i + order[0]];
                    input.readLength2 = currentReadLengths[2*i + order[1]];
                    input.numInts1 = getEncodedNumInts2Bit(currentReadLengths[2*i + order[0]]);
                    input.numInts2 = getEncodedNumInts2Bit(currentReadLengths[2*i + order[1]]);
                    input.verbose = false;
                    input.verboseMutex = &verboseMutex;
                }

                auto extendResult = readExtenderGpu.extendPairedReadBatch(inputs);

                return extendResult;  
            };

            auto handleMultiResult = [&](
                const ReadExtenderGpu::ExtendResult* result1, 
                const ReadExtenderGpu::ExtendResult* result2,
                read_number readId1,
                read_number readId2,
                int readLength1,
                int readLength2,
                const unsigned int* encodedRead1,
                const unsigned int* encodedRead2
            ){
                ExtendedReadDebug er{};

                if(result1 != nullptr){
                    er.extendedRead1 = result1->extendedReads.front().second;
                    if(result1->mateHasBeenFound){
                        er.status1 = ExtendedReadStatus::FoundMate;
                    }else{
                        if(result1->aborted){
                            if(result1->abortReason == ReadExtenderGpu::AbortReason::NoPairedCandidates
                                    || result1->abortReason == ReadExtenderGpu::AbortReason::NoPairedCandidatesAfterAlignment){

                                er.status1 = ExtendedReadStatus::CandidateAbort;
                            }else if(result1->abortReason == ReadExtenderGpu::AbortReason::MsaNotExtended){
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
                            if(result2->abortReason == ReadExtenderGpu::AbortReason::NoPairedCandidates
                                    || result2->abortReason == ReadExtenderGpu::AbortReason::NoPairedCandidatesAfterAlignment){

                                er.status2 = ExtendedReadStatus::CandidateAbort;
                            }else if(result2->abortReason == ReadExtenderGpu::AbortReason::MsaNotExtended){
                                er.status2 = ExtendedReadStatus::MSANoExtension;
                            }
                        }else{
                            er.status2 = ExtendedReadStatus::LengthAbort;
                        }
                    }
                }

                er.readId1 = readId1;
                er.readId2 = readId2;

                er.originalRead1.resize(readLength1, '\0');

                decode2BitSequence(
                    &er.originalRead1[0],
                    encodedRead1,
                    readLength1
                );

                er.originalRead2.resize(readLength2, '\0');

                decode2BitSequence(
                    &er.originalRead2[0],
                    encodedRead2,
                    readLength2
                );

                // if(readId1 == 90 || readId2 == 90){
                //     std::cerr << result1 << " " << result2 << "\n";
                //     std::cerr << er.extendedRead1 << " " << er.extendedRead2 << "\n";
                //     std::cerr << int(er.status1) << " " << int(er.status2) << "\n";
                //     std::cerr << er.originalRead1 << " " << er.originalRead2 << "\n";
                // }

                ExtendedRead result(er);

                return result;                
            };

            // it is not known which of both reads is on the forward strand / reverse complement strand. try both combinations
            auto extendResult0 = processReadOrder({0,1});

            auto extendResult1 = processReadOrder({1,0});

            std::vector<ExtendedRead> resultvector;

            for(int i = 0; i < numReadPairsInBatch; i++){
                const auto& result0 = extendResult0[i];
                const auto& result1 = extendResult1[i];

                if(result0.success || result1.success){
                    numSuccessRead++;
                }

                if(result0.success && !result1.success){
                    auto r = handleMultiResult(&result0, nullptr, 
                        currentIds[2*i],
                        currentIds[2*i+1],
                        currentReadLengths[2*i],
                        currentReadLengths[2*i+1],
                        currentEncodedReads.data() + (2*i) * encodedSequencePitchInInts,
                        currentEncodedReads.data() + (2*i+1) * encodedSequencePitchInInts
                    );
                    resultvector.emplace_back(std::move(r));
                    numSuccess0++;
                }

                if(!result0.success && result1.success){
                    auto r = handleMultiResult(nullptr, &result1,
                        currentIds[2*i],
                        currentIds[2*i+1],
                        currentReadLengths[2*i],
                        currentReadLengths[2*i+1],
                        currentEncodedReads.data() + (2*i) * encodedSequencePitchInInts,
                        currentEncodedReads.data() + (2*i+1) * encodedSequencePitchInInts
                    );
                    resultvector.emplace_back(std::move(r));
                    numSuccess1++;
                }

                if(result0.success && result1.success){

                    const auto& extendedString0 = result0.extendedReads.front().second;
                    const auto& extendedString1 = result1.extendedReads.front().second;

                    std::string mateExtendedReverseComplement = reverseComplementString(
                        extendedString1.c_str(), 
                        extendedString1.length()
                    );
                    const int mismatches = care::cpu::hammingDistance(
                        extendedString0.cbegin(),
                        extendedString0.cend(),
                        mateExtendedReverseComplement.cbegin(),
                        mateExtendedReverseComplement.cend()
                    );

                    mismatchesBetweenMateExtensions[mismatches]++;

                    auto r = handleMultiResult(&result0, &result1,
                        currentIds[2*i],
                        currentIds[2*i+1],
                        currentReadLengths[2*i],
                        currentReadLengths[2*i+1],
                        currentEncodedReads.data() + (2*i) * encodedSequencePitchInInts,
                        currentEncodedReads.data() + (2*i+1) * encodedSequencePitchInInts
                    );
                    resultvector.emplace_back(std::move(r));

                    numSuccess01++;
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

            progressThread.addProgress(numReadPairsInBatch);            
        }

        //#pragma omp critical
        {
            std::lock_guard<std::mutex> lg(ompCriticalMutex);

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
                readExtenderGpu.printTimers();
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





} // namespace gpu

} // namespace care