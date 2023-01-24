
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
#include <future>

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




struct ExtensionPipeline{
    using ReadyResultsCallback = std::function<void(std::vector<ExtendedRead>&&, std::vector<EncodedExtendedRead>&&, std::vector<read_number>&&)>;
    static constexpr bool isPairedEnd = true; 

    const ProgramOptions& programOptions;
    const CpuMinhasher& minhasher;
    const CpuReadStorage& readStorage;
    std::size_t encodedSequencePitchInInts;
    std::size_t decodedSequencePitchInBytes;
    std::size_t qualityPitchInBytes;
    std::size_t msaColumnPitchInElements;
    ReadyResultsCallback submitReadyResults;
    ProgressThread<read_number>* progressThread;

    ExtensionPipeline(
        const ProgramOptions& programOptions_,
        const CpuMinhasher& minhasher_,
        const CpuReadStorage& readStorage_,
        ReadyResultsCallback submitReadyResults_,
        ProgressThread<read_number>* progressThread_
    )
    : programOptions(programOptions_),
        minhasher(minhasher_),
        readStorage(readStorage_),
        submitReadyResults(submitReadyResults_),
        progressThread(progressThread_)
    {
        const int maximumSequenceLength = readStorage.getSequenceLengthUpperBound();
        encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength);
        decodedSequencePitchInBytes = maximumSequenceLength;
        qualityPitchInBytes = 4 * SDIV(maximumSequenceLength, 4);
    }


    std::vector<read_number> executeFirstPass(){
        constexpr bool extraHashing = false;
        constexpr bool isRepeatedIteration = false;

        int maxextensionPerStep = programOptions.fixedStepsize == 0 ? 20 : programOptions.fixedStepsize;
        int minCoverageForExtension = 3;

        const std::size_t numReadsToProcess = getNumReadsToProcess(&readStorage, programOptions);

        IteratorRangeTraversal<thrust::counting_iterator<read_number>> readIdGenerator(
            thrust::make_counting_iterator<read_number>(0),
            thrust::make_counting_iterator<read_number>(0) + numReadsToProcess
        );

        const int maxNumThreads = programOptions.threads;

        std::cerr << "First Pass\n";
        std::cerr << "use " << maxNumThreads << " threads\n";

        std::vector<std::future<std::vector<read_number>>> futures;
        for(int t = 0; t < maxNumThreads; t++){
            futures.emplace_back(
                std::async(
                    std::launch::async,
                    [&](){
                        return combinedHasherAndExtenderThreadFunc(
                            &readIdGenerator,
                            isRepeatedIteration,
                            extraHashing,
                            minCoverageForExtension,
                            maxextensionPerStep
                        );
                    }
                )
            );
        }

        std::vector<read_number> pairsWhichShouldBeRepeated{};

        for(auto& f : futures){
            auto vec = f.get();
            pairsWhichShouldBeRepeated.insert(pairsWhichShouldBeRepeated.end(), vec.begin(), vec.end());
        }

        std::sort(pairsWhichShouldBeRepeated.begin(), pairsWhichShouldBeRepeated.end());

        return pairsWhichShouldBeRepeated;
    }


    std::vector<read_number> executeExtraHashingPass(const std::vector<read_number>& pairsWhichShouldBeRepeated){
        constexpr bool extraHashing = true;
        constexpr bool isRepeatedIteration = true;

        int maxextensionPerStep = programOptions.fixedStepsize == 0 ? 20 : programOptions.fixedStepsize;
        int minCoverageForExtension = 3;


        const int numPairsToRepeat = pairsWhichShouldBeRepeated.size() / 2;
        std::cerr << "Extra hashing pass\n";
        std::cerr << "Will repeat extension of " << numPairsToRepeat << " read pairs\n";

        auto readIdGenerator = makeIteratorRangeTraversal(
            pairsWhichShouldBeRepeated.data(), 
            pairsWhichShouldBeRepeated.data() + pairsWhichShouldBeRepeated.size()
        );

        const int maxNumThreads = programOptions.threads;
        std::cerr << "use " << maxNumThreads << " threads\n";

        std::vector<std::future<std::vector<read_number>>> futures;
        for(int t = 0; t < maxNumThreads; t++){
            futures.emplace_back(
                std::async(
                    std::launch::async,
                    [&](){
                        return combinedHasherAndExtenderThreadFunc(
                            &readIdGenerator,
                            isRepeatedIteration,
                            extraHashing,
                            minCoverageForExtension,
                            maxextensionPerStep
                        );
                    }
                )
            );
        }

        std::vector<read_number> pairsWhichShouldBeRepeatedTmp;

        for(auto& f : futures){
            auto vec = f.get();
            pairsWhichShouldBeRepeatedTmp.insert(pairsWhichShouldBeRepeatedTmp.end(), vec.begin(), vec.end());
        }

        std::sort(pairsWhichShouldBeRepeatedTmp.begin(), pairsWhichShouldBeRepeatedTmp.end());

        return pairsWhichShouldBeRepeatedTmp;
    }

    template<class ReadIdGenerator>
    std::vector<read_number> combinedHasherAndExtenderThreadFunc(
        ReadIdGenerator* readIdGenerator,
        bool isRepeatedIteration,
        bool extraHashing,
        int minCoverageForExtension,
        int maxextensionPerStep
    ){
        cpu::QualityScoreConversion qualityConversion{};

        ReadExtenderCpu readExtender{
            programOptions.insertSize,
            (programOptions.fixedStddev == 0 ? programOptions.insertSizeStddev : programOptions.fixedStddev),
            maxextensionPerStep,
            readStorage.getSequenceLengthUpperBound(),
            readStorage, 
            minhasher,
            programOptions,
            &qualityConversion
        };

        const int batchsizePairs = programOptions.batchsize;

        std::vector<read_number> currentIds(2 * batchsizePairs);
        std::vector<unsigned int> currentEncodedReads(2 * encodedSequencePitchInInts * batchsizePairs);
        std::vector<int> currentReadLengths(2 * batchsizePairs);
        std::vector<char> currentQualityScores(2 * qualityPitchInBytes * batchsizePairs);

        if(!programOptions.useQualityScores){
            std::fill(currentQualityScores.begin(), currentQualityScores.end(), 'I');
        }

        std::vector<extension::ExtendInput> inputs;

        readExtender.setMinCoverageForExtension(minCoverageForExtension);

        std::vector<read_number> pairsWhichShouldBeRepeated;

        auto initInputs = [&](){
            int numReadsInBatch = 0;
            readIdGenerator->process_next_n(
                batchsizePairs * 2, 
                [&](auto begin, auto end){
                    auto readIdsEnd = std::copy(begin, end, currentIds.begin());
                    numReadsInBatch = std::distance(currentIds.begin(), readIdsEnd);
                }
            );

            if(numReadsInBatch % 2 == 1){
                throw std::runtime_error("Input files not properly paired. Aborting read extension.");
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

                if(programOptions.useQualityScores){
                    readStorage.gatherQualities(
                        currentQualityScores.data(),
                        qualityPitchInBytes,
                        currentIds.data(),
                        numReadsInBatch
                    );
                }
            }

            const int numReadPairsInBatch = numReadsInBatch / 2;

            inputs.resize(numReadPairsInBatch);

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

            pairsWhichShouldBeRepeated.insert(
                pairsWhichShouldBeRepeated.end(), 
                splittedExtOutput.idsOfPartiallyExtendedReads.begin(), 
                splittedExtOutput.idsOfPartiallyExtendedReads.end()
            );            

            const std::size_t numExtended = splittedExtOutput.extendedReads.size();

            if(!programOptions.allowOutwardExtension){
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

            progressThread->addProgress(extensionResults.size() - splittedExtOutput.idsOfPartiallyExtendedReads.size());
        };

        while(!(readIdGenerator->empty())){
            initInputs();

            if(inputs.size() > 0){
                auto extensionResults = readExtender.extend(inputs, extraHashing);

                output(std::move(extensionResults), isRepeatedIteration);
            }
        }

        return pairsWhichShouldBeRepeated;
    };
};



template<class Callback>
void extend_cpu_pairedend(
    const ProgramOptions& programOptions,
    const CpuMinhasher& minhasher,
    const CpuReadStorage& readStorage,
    Callback submitReadyResults
){
    const auto rsMemInfo = readStorage.getMemoryInfo();
    const auto mhMemInfo = minhasher.getMemoryInfo();

    std::size_t memoryAvailableBytesHost = programOptions.memoryTotalLimit;
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

    const std::size_t numReadsToProcess = getNumReadsToProcess(&readStorage, programOptions);
   
    const std::uint64_t totalNumReadPairs = numReadsToProcess / 2;

    auto showProgress = [&](auto totalCount, auto seconds){
        if(programOptions.showProgress){

            printf("Processed %10u of %10lu read pairs (Runtime: %03d:%02d:%02d)\r",
                    totalCount, totalNumReadPairs,
                    int(seconds / 3600),
                    int(seconds / 60) % 60,
                    int(seconds) % 60);
            std::cout.flush();
        }

        if(totalCount == totalNumReadPairs){
            std::cout << '\n';
        }
    };

    auto updateShowProgressInterval = [](auto duration){
        return duration;
    };

    ProgressThread<read_number> progressThread(totalNumReadPairs, showProgress, updateShowProgressInterval);

    ExtensionPipeline extensionPipeline(
        programOptions,
        minhasher,
        readStorage,
        submitReadyResults,
        &progressThread
    );

    std::vector<read_number> pairsWhichShouldBeRepeated = extensionPipeline.executeFirstPass();    
    pairsWhichShouldBeRepeated = extensionPipeline.executeExtraHashingPass(pairsWhichShouldBeRepeated);

    submitReadyResults(
        {}, 
        {},
        std::move(pairsWhichShouldBeRepeated) //pairs which did not find mate after repetition will remain unextended
    );

    progressThread.finished();

}



void extend_cpu(
    const ProgramOptions& programOptions,
    const CpuMinhasher& minhasher,
    const CpuReadStorage& readStorage,
    SubmitReadyExtensionResultsCallback submitReadyResults
){
    if(programOptions.pairType == SequencePairType::SingleEnd){
        throw std::runtime_error("single end extension not possible");
    }else{
        extend_cpu_pairedend(
            programOptions,
            minhasher,
            readStorage,
            submitReadyResults
        );
    }
}














} // namespace care