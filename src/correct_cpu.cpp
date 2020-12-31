
//#include <cpu_correction_thread.hpp>

#include <correctionresultprocessing.hpp>

#include <config.hpp>

#include "options.hpp"


#include <readstorage.hpp>
#include "cpu_alignment.hpp"
#include "bestalignment.hpp"
#include <msa.hpp>
#include "qualityscoreweights.hpp"
#include "rangegenerator.hpp"

#include <threadpool.hpp>
#include <memoryfile.hpp>
#include <util.hpp>
#include <filehelpers.hpp>
#include <hostdevicefunctions.cuh>


#define ENABLE_CPU_CORRECTOR_TIMING
#include <corrector.hpp>
#include <cpuminhasher.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include <vector>

#include <omp.h>




namespace care{
namespace cpu{


MemoryFileFixedSize<EncodedTempCorrectedSequence>
correct_cpu(
    const GoodAlignmentProperties& goodAlignmentProperties,
    const CorrectionOptions& correctionOptions,
    const RuntimeOptions& runtimeOptions,
    const FileOptions& fileOptions,
    const MemoryOptions& memoryOptions,
    const SequenceFileProperties& sequenceFileProperties,
    //Minhasher& minhasher,
    CpuMinhasher& minhasher,
    cpu::ContiguousReadStorage& readStorage
){

    omp_set_num_threads(runtimeOptions.threads);

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


    ReadCorrectionFlags correctionFlags(sequenceFileProperties.nReads);


    std::cerr << "correctionStatusFlagsPerRead bytes: " << correctionFlags.sizeInBytes() / 1024. / 1024. << " MB\n";

    if(memoryAvailableBytesHost > correctionFlags.sizeInBytes()){
        memoryAvailableBytesHost -= correctionFlags.sizeInBytes();
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
    //cpu::RangeGenerator<read_number> readIdGenerator(1000000); 
    
    auto saveCorrectedSequence = [&](TempCorrectedSequence tmp, EncodedTempCorrectedSequence encoded){
        //std::unique_lock<std::mutex> l(outputstreammutex);
        //std::cerr << tmp.readId  << " hq " << tmp.hq << " " << "useedits " << tmp.useEdits << " emptyedits " << tmp.edits.empty() << "\n";
        if(!(tmp.hq && tmp.useEdits && tmp.edits.empty())){
            //std::cerr << tmp.readId << " " << tmp << '\n';
            partialResults.storeElement(std::move(encoded));
        }
    };

    BackgroundThread outputThread(true);

    CpuErrorCorrector::TimeMeasurements timingsOfAllThreads;
    
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

    const int numThreads = runtimeOptions.threads;

    #pragma omp parallel
    {
        //const int threadId = omp_get_thread_num();

        const std::size_t encodedSequencePitchInInts2Bit = SequenceHelpers::getEncodedNumInts2Bit(sequenceFileProperties.maxSequenceLength);
        const std::size_t decodedSequencePitchInBytes = sequenceFileProperties.maxSequenceLength;
        const std::size_t qualityPitchInBytes = sequenceFileProperties.maxSequenceLength;

        std::unique_ptr<ReadProvider> readProvider = std::make_unique<CpuReadStorageReadProvider>(readStorage);
        //std::unique_ptr<CandidateIdsProvider> candidateIdsProvider = std::make_unique<CpuMinhasherCandidateIdsProvider>(minhasher);

        CpuErrorCorrector errorCorrector(
            encodedSequencePitchInInts2Bit,
            decodedSequencePitchInBytes,
            qualityPitchInBytes,
            correctionOptions,
            goodAlignmentProperties,
            //*candidateIdsProvider,
            &minhasher,
            *readProvider,
            correctionFlags
        );

        std::vector<read_number> batchReadIds(correctionOptions.batchsize);
        std::vector<unsigned int> batchEncodedData(correctionOptions.batchsize * encodedSequencePitchInInts2Bit);
        std::vector<char> batchQualities(correctionOptions.batchsize * qualityPitchInBytes);
        std::vector<int> batchReadLengths(correctionOptions.batchsize);    

        while(!(readIdGenerator.empty())){

            batchReadIds.resize(correctionOptions.batchsize);

            auto readIdsEnd = readIdGenerator.next_n_into_buffer(
                correctionOptions.batchsize, 
                batchReadIds.begin()
            );
            
            batchReadIds.erase(readIdsEnd, batchReadIds.end());

            if(batchReadIds.empty()){
                continue;
            }

            //collect input data of all reads in batch

            readProvider->gatherSequenceLengths(
                batchReadIds.data(),
                batchReadIds.size(),
                batchReadLengths.data()
            );

            readProvider->gatherSequenceData(
                batchReadIds.data(),
                batchReadIds.size(),
                batchEncodedData.data(),
                encodedSequencePitchInInts2Bit
            );

            if(correctionOptions.useQualityScores){
                readProvider->gatherSequenceQualities(
                    batchReadIds.data(),
                    batchReadIds.size(),
                    batchQualities.data(),
                    qualityPitchInBytes
                );
            }

            CorrectionOutput correctionOutput;

            for(size_t i = 0; i < batchReadIds.size(); i++){
                const read_number readId = batchReadIds[i];

                CpuErrorCorrector::CorrectionInput input;
                input.anchorReadId = readId;
                input.encodedAnchor = batchEncodedData.data() + i * encodedSequencePitchInInts2Bit;
                input.anchorQualityscores = batchQualities.data() + i * qualityPitchInBytes;
                input.anchorLength = batchReadLengths[i];

                auto output = errorCorrector.process(input);

                if(output.hasAnchorCorrection){
                    correctionOutput.encodedAnchorCorrections.emplace_back(output.anchorCorrection.encode());
                    correctionOutput.anchorCorrections.emplace_back(std::move(output.anchorCorrection));
                }

                for(auto& tmp : output.candidateCorrections){
                    correctionOutput.encodedCandidateCorrections.emplace_back(tmp.encode());
                    correctionOutput.candidateCorrections.emplace_back(std::move(tmp));
                }
            }

            auto outputfunction = [
                &, 
                correctionOutput = std::move(correctionOutput)
            ](){
                const int numA = correctionOutput.anchorCorrections.size();
                const int numC = correctionOutput.candidateCorrections.size();

                for(int i = 0; i < numA; i++){
                    saveCorrectedSequence(
                        std::move(correctionOutput.anchorCorrections[i]), 
                        std::move(correctionOutput.encodedAnchorCorrections[i])
                    );
                }

                for(int i = 0; i < numC; i++){
                    saveCorrectedSequence(
                        std::move(correctionOutput.candidateCorrections[i]), 
                        std::move(correctionOutput.encodedCandidateCorrections[i])
                    );
                }
            };

            outputThread.enqueue(std::move(outputfunction));

            progressThread.addProgress(batchReadIds.size()); 
            
        } //while unprocessed reads exist loop end   

        #pragma omp critical
        {
            timingsOfAllThreads += errorCorrector.getTimings();            
        }

    } // parallel end

    progressThread.finished();

    outputThread.stopThread(BackgroundThread::StopType::FinishAndStop);

    //outputstream.flush();
    partialResults.flush();

    #ifdef ENABLE_CPU_CORRECTOR_TIMING

    auto totalDurationOfThreads = timingsOfAllThreads.getSumOfDurations();

    auto printDuration = [&](const auto& name, const auto& duration){
        std::cout << "# average time per thread ("<< name << "): "
                  << duration.count() / numThreads  << " s. "
                  << (100.0 * duration.count() / totalDurationOfThreads.count()) << " %."<< std::endl;
    };

    #define printme(x) printDuration((#x), timingsOfAllThreads.x);

    printme(getSubjectSequenceDataTimeTotal);
    printme(getCandidatesTimeTotal);
    printme(copyCandidateDataToBufferTimeTotal);
    printme(getAlignmentsTimeTotal);
    printme(findBestAlignmentDirectionTimeTotal);
    printme(gatherBestAlignmentDataTimeTotal);
    printme(mismatchRatioFilteringTimeTotal);
    printme(compactBestAlignmentDataTimeTotal);
    printme(fetchQualitiesTimeTotal);
    printme(makeCandidateStringsTimeTotal);
    printme(msaAddSequencesTimeTotal);
    printme(msaFindConsensusTimeTotal);
    printme(msaMinimizationTimeTotal);
    printme(msaCorrectSubjectTimeTotal);
    printme(msaCorrectCandidatesTimeTotal);

    #undef printme

    #endif

    return partialResults;
}

} //namespace cpu

} //namespace care


