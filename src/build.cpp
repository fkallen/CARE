#include <build.hpp>

#include <config.hpp>

#include <options.hpp>

#include <minhasher.hpp>
#include <minhasher_transform.hpp>
#include "readstorage.hpp"
#include "sequencefileio.hpp"
#include "sequence.hpp"
#include "threadsafe_buffer.hpp"

#include <stdexcept>
#include <iostream>
#include <limits>
#include <thread>
#include <future>
#include <mutex>
#include <iterator>
#include <random>
#include <omp.h>




namespace care{



    BuiltDataStructure<cpu::ContiguousReadStorage> build_readstorage(const FileOptions& fileOptions,
                                                const RuntimeOptions& runtimeOptions,
                                                bool useQualityScores,
                                                read_number expectedNumberOfReads,
                                                int expectedMinimumReadLength,
                                                int expectedMaximumReadLength){



        if(fileOptions.load_binary_reads_from != ""){
            BuiltDataStructure<cpu::ContiguousReadStorage> result;
            auto& readStorage = result.data;

            readStorage.loadFromFile(fileOptions.load_binary_reads_from);
            result.builtType = BuiltType::Loaded;

            if(useQualityScores && !readStorage.hasQualityScores())
                throw std::runtime_error("Quality scores are required but not present in compressed sequence file!");
            if(!useQualityScores && readStorage.hasQualityScores())
                std::cerr << "Warning. The loaded compressed read file contains quality scores, but program does not use them!\n";

            std::cout << "Loaded binary reads from " << fileOptions.load_binary_reads_from << std::endl;

            return result;
        }else{
            //int nThreads = std::max(1, std::min(runtimeOptions.threads, 4));

            constexpr std::array<char, 4> bases = {'A', 'C', 'G', 'T'};
            int Ncount = 0;

            BuiltDataStructure<cpu::ContiguousReadStorage> result;

            result.data = std::move(cpu::ContiguousReadStorage{expectedNumberOfReads, useQualityScores, expectedMaximumReadLength});
            result.builtType = BuiltType::Constructed;

            auto handle_read = [&](std::uint64_t readIndex, Read& read){
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

                result.data.insertRead(readIndex, read.sequence, read.quality);
#if 0
                const char* ptr = result.data.fetchSequenceData_ptr(readIndex);
                int length = result.data.fetchSequenceLength(readIndex);

                std::string s = get2BitHiLoString((const unsigned int*)ptr, length, [](auto i){return i;});
                assert(s == read.sequence);
#endif
            };

            forEachReadInFile(fileOptions.inputfile,
                            fileOptions.format,
                            [&](auto readnum, auto& read){
                                handle_read(readnum, read);
                            }
            );

            return result;
        }

    }


    BuiltDataStructure<Minhasher> build_minhasher(const FileOptions& fileOptions,
                                			   const RuntimeOptions& runtimeOptions,
                                			   std::uint64_t nReads,
                                               const MinhashOptions& minhashOptions,
                                			   cpu::ContiguousReadStorage& readStorage){

        BuiltDataStructure<Minhasher> result;
        auto& minhasher = result.data;

        minhasher = std::move(Minhasher{minhashOptions});

        minhasher.init(nReads);

        if(fileOptions.load_hashtables_from != ""){
            minhasher.loadFromFile(fileOptions.load_hashtables_from);
            result.builtType = BuiltType::Loaded;

            std::cout << "Loaded hash tables from " << fileOptions.load_hashtables_from << std::endl;
        }else{
            result.builtType = BuiltType::Constructed;

            const int oldnumthreads = omp_get_thread_num();

            omp_set_num_threads(runtimeOptions.threads);

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

                #pragma omp parallel for
                for(std::size_t readId = 0; readId < readStorage.getNumberOfSequences(); readId++){

    				const std::uint8_t* sequenceptr = (const std::uint8_t*)readStorage.fetchSequenceData_ptr(readId);
    				const int sequencelength = readStorage.fetchSequenceLength(readId);
    				std::string sequencestring = get2BitHiLoString((const unsigned int*)sequenceptr, sequencelength);

                    minhasher.insertSequence(sequencestring, readId, mapIds);
                }

                for(auto mapId : mapIds){
                    transform_minhasher(minhasher, mapId);
                }

            }

            omp_set_num_threads(oldnumthreads);
        }

        //TIMERSTARTCPU(finalize_hashtables);
        //minhasher.transform();
        //TIMERSTOPCPU(finalize_hashtables);

        return result;
    }

    BuiltDataStructures buildDataStructures(const MinhashOptions& minhashOptions,
                                			const CorrectionOptions& correctionOptions,
                                			const RuntimeOptions& runtimeOptions,
                                			const FileOptions& fileOptions){

        BuiltDataStructures result;

        auto& sequenceFileProperties = result.sequenceFileProperties;

        if(fileOptions.load_binary_reads_from == "") {
            sequenceFileProperties = detail::getSequenceFilePropertiesFromFileOptions(fileOptions);

            detail::printInputFileProperties(std::cout, fileOptions.inputfile, sequenceFileProperties);
        }

        TIMERSTARTCPU(build_readstorage);
        result.builtReadStorage = build_readstorage(fileOptions,
                                                  runtimeOptions,
                                                  correctionOptions.useQualityScores,
                                                  sequenceFileProperties.nReads,
                                                  sequenceFileProperties.minSequenceLength,
                                                  sequenceFileProperties.maxSequenceLength);
        TIMERSTOPCPU(build_readstorage);

        auto& readStorage = result.builtReadStorage.data;

        if(result.builtReadStorage.builtType == BuiltType::Loaded) {
            auto stats = readStorage.getSequenceStatistics(runtimeOptions.threads);
            sequenceFileProperties.nReads = readStorage.getNumberOfSequences();
            sequenceFileProperties.maxSequenceLength = stats.maxSequenceLength;
            sequenceFileProperties.minSequenceLength = stats.minSequenceLength;
        }

        TIMERSTARTCPU(build_minhasher);
        result.builtMinhasher = build_minhasher(fileOptions, runtimeOptions, sequenceFileProperties.nReads, minhashOptions, readStorage);
        TIMERSTOPCPU(build_minhasher);

        return result;
    }

    BuiltDataStructures buildAndSaveDataStructures(const MinhashOptions& minhashOptions,
                                            const CorrectionOptions& correctionOptions,
                                            const RuntimeOptions& runtimeOptions,
                                            const FileOptions& fileOptions){

        BuiltDataStructures result;

        auto& sequenceFileProperties = result.sequenceFileProperties;

        if(fileOptions.load_binary_reads_from == "") {
            sequenceFileProperties = detail::getSequenceFilePropertiesFromFileOptions(fileOptions);

            detail::printInputFileProperties(std::cout, fileOptions.inputfile, sequenceFileProperties);
        }

        TIMERSTARTCPU(build_readstorage);
        result.builtReadStorage = build_readstorage(fileOptions,
                                                  runtimeOptions,
                                                  correctionOptions.useQualityScores,
                                                  sequenceFileProperties.nReads,
                                                  sequenceFileProperties.minSequenceLength,
                                                  sequenceFileProperties.maxSequenceLength);
        TIMERSTOPCPU(build_readstorage);

        auto& readStorage = result.builtReadStorage.data;

        if(fileOptions.save_binary_reads_to != "") {
            std::cout << "Saving reads to file " << fileOptions.save_binary_reads_to << std::endl;
            readStorage.saveToFile(fileOptions.save_binary_reads_to);
            std::cout << "Saved reads" << std::endl;
        }

        if(result.builtReadStorage.builtType == BuiltType::Loaded) {
            auto stats = readStorage.getSequenceStatistics(runtimeOptions.threads);
            sequenceFileProperties.nReads = readStorage.getNumberOfSequences();
            sequenceFileProperties.maxSequenceLength = stats.maxSequenceLength;
            sequenceFileProperties.minSequenceLength = stats.minSequenceLength;
        }

        TIMERSTARTCPU(build_minhasher);
        result.builtMinhasher = build_minhasher(fileOptions, runtimeOptions, sequenceFileProperties.nReads, minhashOptions, readStorage);
        TIMERSTOPCPU(build_minhasher);

        if(fileOptions.save_hashtables_to != "") {
            std::cout << "Saving minhasher to file " << fileOptions.save_hashtables_to << std::endl;
            result.builtMinhasher.data.saveToFile(fileOptions.save_hashtables_to);
            std::cout << "Saved minhasher" << std::endl;
        }

        return result;
    }
}
