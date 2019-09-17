#include <build.hpp>

#include <config.hpp>

#include <options.hpp>

#include <hpc_helpers.cuh>

#include <minhasher.hpp>
#include <minhasher_transform.hpp>
#include "readstorage.hpp"
#include "sequencefileio.hpp"
#include "sequence.hpp"
#include "threadsafe_buffer.hpp"

#include <threadpool.hpp>


#include <map>
#include <stdexcept>
#include <iostream>
#include <limits>
#include <thread>
#include <future>
#include <mutex>
#include <iterator>
#include <random>
#include <omp.h>
#include <mutex>
#include <condition_variable>

#ifdef __NVCC__

namespace care{
namespace gpu{



    BuiltDataStructure<GpuReadStorageWithFlags> buildGpuReadStorage(const FileOptions& fileOptions,
                                                const RuntimeOptions& runtimeOptions,
                                                bool useQualityScores,
                                                read_number expectedNumberOfReads,
                                                int expectedMaximumReadLength){



        if(fileOptions.load_binary_reads_from != ""){
            BuiltDataStructure<GpuReadStorageWithFlags> result;
            auto& readStorage = result.data.readStorage;

            readStorage.loadFromFile(fileOptions.load_binary_reads_from, runtimeOptions.deviceIds);
            result.builtType = BuiltType::Loaded;

            if(useQualityScores && !readStorage.canUseQualityScores())
                throw std::runtime_error("Quality scores are required but not present in compressed sequence file!");
            if(!useQualityScores && readStorage.canUseQualityScores())
                std::cerr << "Warning. The loaded compressed read file contains quality scores, but program does not use them!\n";

            std::cout << "Loaded binary reads from " << fileOptions.load_binary_reads_from << std::endl;

            return result;
        }else{
            //int nThreads = std::max(1, std::min(runtimeOptions.threads, 2));
            const int nThreads = std::max(1, runtimeOptions.threads);

            constexpr std::array<char, 4> bases = {'A', 'C', 'G', 'T'};
            int Ncount = 0;
            std::map<int,int> nmap{};

            BuiltDataStructure<GpuReadStorageWithFlags> result;
            DistributedReadStorage& readstorage = result.data.readStorage;
            //auto& validFlags = result.data.readIsValidFlags;

            readstorage = std::move(DistributedReadStorage{runtimeOptions.deviceIds, expectedNumberOfReads, useQualityScores, expectedMaximumReadLength});
            //validFlags.resize(expectedNumberOfReads, false);
            result.builtType = BuiltType::Constructed;

            auto flushBuffers = [&](std::vector<read_number>* indicesBuffer, std::vector<Read>* readsBuffer){
                if(indicesBuffer->size() > 0){
                    //TIMERSTARTCPU(setReads);
                    readstorage.setReads(*indicesBuffer, *readsBuffer);
                    //TIMERSTOPCPU(setReads);

                    //std::cerr << "clearing " << indicesBuffer->size() << " indices\n";
                    //TIMERSTARTCPU(fbclearindices);
                    indicesBuffer->clear();
                    //TIMERSTOPCPU(fbclearindices);

                    //std::cerr << "clearing " << readsBuffer->size() << " reads\n";
                    //TIMERSTARTCPU(fbclearreads);
                    readsBuffer->clear();
                    //TIMERSTOPCPU(fbclearreads);
                }
            };

            auto handle_read = [&](std::uint64_t readIndex, Read& read, std::vector<read_number>* indicesBuffer, std::vector<Read>* readsBuffer){
                const int readLength = int(read.sequence.size());

                if(readIndex >= expectedNumberOfReads){
                    throw std::runtime_error("Error! Expected " + std::to_string(expectedNumberOfReads)
                                            + " reads, but file contains at least "
                                            + std::to_string(readIndex) + " reads.");
                }

                if(readLength > expectedMaximumReadLength){
                    throw std::runtime_error("Error! Expected maximum read length = "
                                            + std::to_string(expectedMaximumReadLength)
                                            + ", but read " + std::to_string(readIndex)
                                            + "has length " + std::to_string(readLength));
                }

                const int undeterminedBasesInRead = std::count_if(read.sequence.begin(), read.sequence.end(), [](char c){
                    return c == 'N' || c == 'n';
                });

                nmap[undeterminedBasesInRead]++;

                if(undeterminedBasesInRead > 0){
                    readstorage.setReadContainsN(readIndex, true);
                }

                //if(undeterminedBasesInRead > 10){
                //    validFlags[readIndex] = false;
                //}else{
                    for(auto& c : read.sequence){
                        if(c == 'a') c = 'A';
                        if(c == 'c') c = 'C';
                        if(c == 'g') c = 'G';
                        if(c == 't') c = 'T';
                        if(c == 'N' || c == 'n'){
                            c = bases[Ncount];
                            Ncount = (Ncount + 1) % 4;
                        }
                    }

                    indicesBuffer->emplace_back(readIndex);
                    readsBuffer->emplace_back(read);

                    //validFlags[readIndex] = true;
                //}

            };

            constexpr size_t maxbuffersize = 1000000;
            constexpr int numBuffers = 2;

            std::array<std::vector<read_number>, numBuffers> indicesBuffers;
            std::array<std::vector<Read>, numBuffers> readsBuffers;
            std::array<bool, numBuffers> canBeUsed;
            std::array<std::mutex, numBuffers> mutex;
            std::array<std::condition_variable, numBuffers> cv;

            for(int i = 0; i < numBuffers; i++){
                indicesBuffers[i].reserve(maxbuffersize);
                readsBuffers[i].reserve(maxbuffersize);
                canBeUsed[i] = true;
            }

            int bufferindex = 0;

            forEachReadInFile(fileOptions.inputfile,
                            fileOptions.format,
                            [&](auto readnum, auto& read){

                    if(!canBeUsed[bufferindex]){
                        std::unique_lock<std::mutex> ul(mutex[bufferindex]);
                        if(!canBeUsed[bufferindex]){
                            //std::cerr << "waiting for other buffer\n";
                            cv[bufferindex].wait(ul, [&](){ return canBeUsed[bufferindex]; });
                        }
                    }

                    auto indicesBufferPtr = &indicesBuffers[bufferindex];
                    auto readsBufferPtr = &readsBuffers[bufferindex];

                    handle_read(readnum, read, indicesBufferPtr, readsBufferPtr);

                    if(indicesBufferPtr->size() > maxbuffersize){
                        //TIMERSTARTCPU(flushBuffers);
                        //flushBuffers(indicesBufferPtr, readsBufferPtr);
                        //TIMERSTOPCPU(flushBuffers);

                        canBeUsed[bufferindex] = false;

                        //std::cerr << "launch other thread\n";
                        threadpool.enqueue([&, indicesBufferPtr, readsBufferPtr, bufferindex](){
                            //TIMERSTARTCPU(flushBufferswithcv);

                            TIMERSTARTCPU(flushBuffers);

                            flushBuffers(indicesBufferPtr, readsBufferPtr);

                            TIMERSTOPCPU(flushBuffers);

                            std::lock_guard<std::mutex> l(mutex[bufferindex]);
                            canBeUsed[bufferindex] = true;
                            cv[bufferindex].notify_one();

                            //TIMERSTOPCPU(flushBufferswithcv);

                            //std::cerr << "other thread done\n";
                        });

                        bufferindex = (bufferindex + 1) % numBuffers; //swap buffers
                        //std::cerr << "bufferindex is now " << bufferindex << "\n";



                    }
                }
            );

            auto indicesBufferPtr = &indicesBuffers[bufferindex];
            auto readsBufferPtr = &readsBuffers[bufferindex];

            if(indicesBufferPtr->size() > 0){
                //TIMERSTARTCPU(flushBuffers);
                flushBuffers(indicesBufferPtr, readsBufferPtr);
                //TIMERSTOPCPU(flushBuffers);
            }

            // std::cerr << "occurences of n/N:\n";
            // for(const auto& p : nmap){
            //     std::cerr << p.first << " " << p.second << '\n';
            // }

            return result;
        }

    }


    BuiltDataStructure<Minhasher> build_minhasher(const FileOptions& fileOptions,
                                			   const RuntimeOptions& runtimeOptions,
                                			   std::uint64_t nReads,
                                               const MinhashOptions& minhashOptions,
                                			   const GpuReadStorageWithFlags& readStoragewFlags){

        BuiltDataStructure<Minhasher> result;
        auto& minhasher = result.data;

        auto identity = [](auto i){return i;};

        minhasher = std::move(Minhasher{minhashOptions});

        minhasher.init(nReads);

        if(fileOptions.load_hashtables_from != ""){
            minhasher.loadFromFile(fileOptions.load_hashtables_from);
            result.builtType = BuiltType::Loaded;

            std::cout << "Loaded hash tables from " << fileOptions.load_hashtables_from << std::endl;
        }else{
            result.builtType = BuiltType::Constructed;

            int oldnumthreads = 1;
            #pragma omp parallel
            {
                #pragma omp single
                oldnumthreads = omp_get_num_threads();
            }

            omp_set_num_threads(runtimeOptions.threads);
            //std::cerr << "setReads omp_set_num_threads end " << runtimeOptions.threads << "\n";

            const auto& readStorage = readStoragewFlags.readStorage;
            //const auto& validFlags = readStoragewFlags.readIsValidFlags;

            constexpr read_number parallelReads = 10000000;

            const int numBatches = SDIV(minhashOptions.maps, minhasherConstructionNumMaps);

            for(int batch = 0; batch < numBatches; batch++){
                const int firstMap = batch * minhasherConstructionNumMaps;
                const int lastMap = std::min(minhashOptions.maps, (batch+1) * minhasherConstructionNumMaps);
                const int numMaps = lastMap - firstMap;
                std::vector<int> mapIds(numMaps);
                std::iota(mapIds.begin(), mapIds.end(), firstMap);

                for(auto mapId : mapIds){
                    minhasher.initMap(mapId);
                }

                read_number numReads = readStorage.getNumberOfReads();
                int numIters = SDIV(numReads, parallelReads);

                auto sequencehandle = readStorage.makeGatherHandleSequences();
                auto lengthhandle = readStorage.makeGatherHandleLengths();
                size_t sequencepitch = getEncodedNumInts2BitHiLo(readStorage.getSequenceLengthLimit()) * sizeof(int);

                //TIMERSTARTCPU(iter);
                for(int iter = 0; iter < numIters; iter++){
                    read_number readIdBegin = iter * parallelReads;
                    read_number readIdEnd = std::min((iter+1) * parallelReads, numReads);

                    std::vector<read_number> indices(readIdEnd - readIdBegin);
                    std::iota(indices.begin(), indices.end(), readIdBegin);

                    std::vector<char> sequenceData(indices.size() * sequencepitch);
                    std::vector<DistributedReadStorage::Length_t> lengths(indices.size());

                    //TIMERSTARTCPU(gather);

                    auto future1 = readStorage.gatherSequenceDataToHostBufferAsync(
                                                sequencehandle,
                                                sequenceData.data(),
                                                sequencepitch,
                                                indices.data(),
                                                indices.size(),
                                                1);
                    auto future2 = readStorage.gatherSequenceLengthsToHostBufferAsync(
                                                lengthhandle,
                                                lengths.data(),
                                                indices.data(),
                                                indices.size(),
                                                1);

                    future1.wait();
                    future2.wait();

                    //TIMERSTOPCPU(gather);

                    //TIMERSTARTCPU(insert);

                    #pragma omp parallel for
                    for(read_number readId = readIdBegin; readId < readIdEnd; readId++){
                        //if(validFlags[readId]){
                            read_number localId = readId - readIdBegin;
            				const char* encodedsequence = (const char*)&sequenceData[localId * sequencepitch];
            				const int sequencelength = lengths[localId];
            				std::string sequencestring = get2BitHiLoString((const unsigned int*)encodedsequence, sequencelength);
                            minhasher.insertSequence(sequencestring, readId, mapIds);
                        //}else{
                        //    ; //invalid reads are discarded
                        //}
                    }

                    //TIMERSTOPCPU(insert);
                }
                //TIMERSTOPCPU(iter);

                for(auto mapId : mapIds){
                    transform_minhasher_gpu(minhasher, mapId, runtimeOptions.deviceIds);
                }
            }
            omp_set_num_threads(oldnumthreads);

        }

        //TIMERSTARTCPU(finalize_hashtables);
        //minhasher.transform();
        //TIMERSTOPCPU(finalize_hashtables);

        return result;
    }



    BuiltGpuDataStructures buildGpuDataStructures(const MinhashOptions& minhashOptions,
                                			const CorrectionOptions& correctionOptions,
                                			const RuntimeOptions& runtimeOptions,
                                			const FileOptions& fileOptions){

        BuiltGpuDataStructures result;

        auto& sequenceFileProperties = result.sequenceFileProperties;

        if(fileOptions.load_binary_reads_from == "") {
            if(fileOptions.nReads == 0 || fileOptions.maximum_sequence_length == 0) {
                std::cout << "Scanning file to get number of reads and maximum sequence length." << std::endl;
                sequenceFileProperties = getSequenceFileProperties(fileOptions.inputfile, fileOptions.format);
            }else{
                sequenceFileProperties.maxSequenceLength = fileOptions.maximum_sequence_length;
                sequenceFileProperties.minSequenceLength = 0;
                sequenceFileProperties.nReads = fileOptions.nReads;
            }
        }

        TIMERSTARTCPU(build_readstorage);
        result.builtReadStorage = buildGpuReadStorage(fileOptions,
                                                  runtimeOptions,
                                                  correctionOptions.useQualityScores,
                                                  sequenceFileProperties.nReads,
                                                  sequenceFileProperties.maxSequenceLength);
        TIMERSTOPCPU(build_readstorage);

        const auto& readStorage = result.builtReadStorage.data.readStorage;

        //if(result.builtReadStorage.builtType == BuiltType::Loaded) {
            sequenceFileProperties.nReads = readStorage.getNumberOfReads();
            sequenceFileProperties.maxSequenceLength = readStorage.getStatistics().maximumSequenceLength;
            sequenceFileProperties.minSequenceLength = readStorage.getStatistics().minimumSequenceLength;
        //}

        TIMERSTARTCPU(build_minhasher);
        result.builtMinhasher = build_minhasher(fileOptions, runtimeOptions, sequenceFileProperties.nReads, minhashOptions, result.builtReadStorage.data);
        TIMERSTOPCPU(build_minhasher);

        return result;
    }

    BuiltGpuDataStructures buildAndSaveGpuDataStructures(const MinhashOptions& minhashOptions,
                                                        const CorrectionOptions& correctionOptions,
                                                        const RuntimeOptions& runtimeOptions,
                                                        const FileOptions& fileOptions){

        BuiltGpuDataStructures result;

        auto& sequenceFileProperties = result.sequenceFileProperties;

        if(fileOptions.load_binary_reads_from == "") {
            if(fileOptions.nReads == 0 || fileOptions.maximum_sequence_length == 0) {
                std::cout << "Scanning file to get number of reads and maximum sequence length." << std::endl;
                sequenceFileProperties = getSequenceFileProperties(fileOptions.inputfile, fileOptions.format);
            }else{
                sequenceFileProperties.maxSequenceLength = fileOptions.maximum_sequence_length;
                sequenceFileProperties.minSequenceLength = 0;
                sequenceFileProperties.nReads = fileOptions.nReads;
            }
        }

        TIMERSTARTCPU(build_readstorage);
        result.builtReadStorage = buildGpuReadStorage(fileOptions,
                                                  runtimeOptions,
                                                  correctionOptions.useQualityScores,
                                                  sequenceFileProperties.nReads,
                                                  sequenceFileProperties.maxSequenceLength);
        TIMERSTOPCPU(build_readstorage);

        const auto& readStorage = result.builtReadStorage.data.readStorage;

        if(fileOptions.save_binary_reads_to != "") {
            std::cout << "Saving reads to file " << fileOptions.save_binary_reads_to << std::endl;
    		readStorage.saveToFile(fileOptions.save_binary_reads_to);
    		std::cout << "Saved reads" << std::endl;
    	}

        //if(result.builtReadStorage.builtType == BuiltType::Loaded) {
            sequenceFileProperties.nReads = readStorage.getNumberOfReads();
            sequenceFileProperties.maxSequenceLength = readStorage.getStatistics().maximumSequenceLength;
            sequenceFileProperties.minSequenceLength = readStorage.getStatistics().minimumSequenceLength;
        //}

        TIMERSTARTCPU(build_minhasher);
        result.builtMinhasher = build_minhasher(fileOptions, runtimeOptions, sequenceFileProperties.nReads, minhashOptions, result.builtReadStorage.data);
        TIMERSTOPCPU(build_minhasher);

        if(fileOptions.save_hashtables_to != "") {
            std::cout << "Saving minhasher to file " << fileOptions.save_hashtables_to << std::endl;
    		result.builtMinhasher.data.saveToFile(fileOptions.save_hashtables_to);
    		std::cout << "Saved minhasher" << std::endl;
    	}

        return result;
    }

}
}


#endif
