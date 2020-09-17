#include <minhasher.hpp>

#include <hpc_helpers.cuh>
#include <util.hpp>
#include <config.hpp>
#include <memorymanagement.hpp>
#include <options.hpp>
#include <readstorage.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <numeric>

namespace care{


    Minhasher::Minhasher(Minhasher&& rhs){
        *this = std::move(rhs);
    }

    Minhasher& Minhasher::operator=(Minhasher&& rhs){
        minhashTables = std::move(rhs.minhashTables);
        kmerSize = std::move(rhs.kmerSize);
        resultsPerMapThreshold = std::move(rhs.resultsPerMapThreshold);

        return *this;
    }

    bool Minhasher::operator==(const Minhasher& rhs) const{
        if(kmerSize != rhs.kmerSize)
            return false;
        if(resultsPerMapThreshold != rhs.resultsPerMapThreshold)
            return false;
        if(minhashTables.size() != rhs.minhashTables.size())
            return false;
        for(std::size_t i = 0; i < minhashTables.size(); i++){
            if(*minhashTables[i] != *rhs.minhashTables[i])
                return false;
        }
        return true;
    }

    bool Minhasher::operator!=(const Minhasher& rhs) const{
        return !(*this == rhs);
    }


    MemoryUsage Minhasher::getMemoryInfo() const{
        MemoryUsage memoryInfo{};

        for(const auto& m : minhashTables)
            memoryInfo += m->getMemoryInfo();
            
        return memoryInfo;
    }



    void Minhasher::writeToStream(std::ostream& outstream) const{

        outstream.write(reinterpret_cast<const char*>(&kmerSize), sizeof(int));
        outstream.write(reinterpret_cast<const char*>(&resultsPerMapThreshold), sizeof(int));

        const int numTables = getNumberOfMaps();
        outstream.write(reinterpret_cast<const char*>(&numTables), sizeof(int));
        
        for(const auto& tableptr : minhashTables)
            tableptr->writeToStream(outstream);
    }

    void Minhasher::loadFromStream(std::ifstream& instream){

        destroy();

        instream.read(reinterpret_cast<char*>(&kmerSize), sizeof(int));
        instream.read(reinterpret_cast<char*>(&resultsPerMapThreshold), sizeof(int));

        int numTables = 0;
        instream.read(reinterpret_cast<char*>(&numTables), sizeof(int));

        for(int i = 0; i < numTables; i++){
            try{
                auto tmptableptr = std::make_unique<Minhasher::Map_t>();
                tmptableptr->loadFromStream(instream);
                minhashTables.emplace_back(std::move(tmptableptr));
            }catch(const std::bad_alloc& e){
                throw std::runtime_error("Not enough memory to load minhasher. Abort!");
            }catch(...){
                throw std::runtime_error("Exception occurred while loading minhasher. Abort!");
            }
        }
    }


    void Minhasher::construct(
        const FileOptions& fileOptions,
        const RuntimeOptions& runtimeOptions,
        const MemoryOptions& memoryOptions,
        std::uint64_t nReads,
        const CorrectionOptions& correctionOptions,
        cpu::ContiguousReadStorage& readStorage
    ){

        const int requestedNumberOfMaps = correctionOptions.numHashFunctions;

        minhashTables.clear();

        ThreadPool threadPool(runtimeOptions.threads);
        //ThreadPool threadPool(1);
        ThreadPool::ParallelForHandle pforHandle;

        const std::string tmpmapsFilename = fileOptions.tempdirectory + "/tmpmaps";
        std::ofstream outstream(tmpmapsFilename, std::ios::binary);
        if(!outstream){
            throw std::runtime_error("Could not open temp file " + tmpmapsFilename + "!");
        }
        std::size_t writtenTableBytes = 0;

        std::size_t maxMemoryForTables = getAvailableMemoryInKB() * 1024;

        maxMemoryForTables = std::min(maxMemoryForTables, 
                                std::min(memoryOptions.memoryForHashtables, memoryOptions.memoryTotalLimit));

        std::cerr << "maxMemoryForTables = " << maxMemoryForTables << " bytes\n";
        std::size_t availableMemForTables = maxMemoryForTables;

        int numSavedTables = 0;
        int numConstructedTables = 0;

        while(numConstructedTables < requestedNumberOfMaps && maxMemoryForTables > writtenTableBytes){

            int maxNumTables = 0;

            {
                const std::size_t requiredMemPerTable = nReads * sizeof(Key_t)
                                                        + nReads * sizeof(Value_t)
                                                        + 4 * 1024;
                maxNumTables = availableMemForTables / requiredMemPerTable;
                maxNumTables -= 2; // need free memory of 2 tables to perform transformation 
                std::cerr << "requiredMemPerTable = " << requiredMemPerTable << "\n";
                std::cerr << "maxNumTables = " << maxNumTables << "\n";
            }

            if(maxNumTables <= 0){
                throw std::runtime_error("Not enough memory to construct 1 table");
            }

            int currentIterNumTables = std::min(requestedNumberOfMaps - numConstructedTables, maxNumTables);           

            std::vector<int> tableIds(currentIterNumTables);                
            std::vector<int> hashIds(currentIterNumTables);
            std::vector<int> globalTableIds(currentIterNumTables);
            
            std::iota(tableIds.begin(), tableIds.end(), 0);
            std::iota(hashIds.begin(), hashIds.end(), numConstructedTables);
            std::iota(globalTableIds.begin(), globalTableIds.end(), numConstructedTables);

            std::cout << "Constructing maps: ";
            std::copy(globalTableIds.begin(), globalTableIds.end(), std::ostream_iterator<int>(std::cout, " "));
            std::cout << "\n";

            const read_number readIdBegin = 0;
            const read_number readIdEnd = readStorage.getNumberOfReads();

            auto showProgress = [&](auto totalCount, auto seconds){
                if(runtimeOptions.showProgress){
                    std::cout << "Hashed " << totalCount << " / " << nReads << " reads. Elapsed time: " 
                            << seconds << " seconds.\n";
                }
            };

            auto updateShowProgressInterval = [](auto duration){
                return duration * 2;
            };

            std::vector<std::vector<kmer_type>> hashesPerTable(currentIterNumTables);
            std::vector<std::vector<read_number>> readIdsPerTable(currentIterNumTables);

            for(int i = 0; i < currentIterNumTables; i++){
                hashesPerTable[i].resize(readStorage.getNumberOfReads());
                readIdsPerTable[i].resize(readStorage.getNumberOfReads());
            }

            ProgressThread<std::uint64_t> progressThread(
                nReads, 
                showProgress,
                updateShowProgressInterval
            );

            auto lambda = [&, readIdBegin](auto begin, auto end, int threadId) {

                for (read_number readId = begin; readId < end; readId++){
                    const read_number localId = readId - readIdBegin;

                    const std::uint8_t* sequenceptr = (const std::uint8_t*)readStorage.fetchSequenceData_ptr(localId);
                    const int sequencelength = readStorage.fetchSequenceLength(readId);
                    std::string sequencestring = get2BitString((const unsigned int*)sequenceptr, sequencelength);

                    if(readId >= nReads)
                        throw std::runtime_error("Minhasher::insertSequence: read number too large. "
                                                + std::to_string(readId) + " > " + std::to_string(nReads));

                    
                    if(sequencelength >= getKmerSize()){
                        auto hashValues = minhashfunc(sequencestring, requestedNumberOfMaps);

                        for(int i = 0; i < int(hashIds.size()); i++){
                            const auto hashValue = hashValues[hashIds[i]];
                            const kmer_type key = hashValue & key_mask;

                            hashesPerTable[i][readId] = key;
                            readIdsPerTable[i][readId] = readId;
                        }
                    }else{
                        for(int i = 0; i < int(hashIds.size()); i++){
                            const kmer_type key = std::numeric_limits<kmer_type>::max();

                            hashesPerTable[i][readId] = key;
                            readIdsPerTable[i][readId] = readId;
                        }
                    }

                    progressThread.addProgress(1);
                }
            };

            threadPool.parallelFor(
                pforHandle,
                readIdBegin,
                readIdEnd,
                std::move(lambda));

            progressThread.finished();

            //if all tables could be constructed at once, no need to save them to temporary file
            if(requestedNumberOfMaps == currentIterNumTables){
                for(int i = 0; i < currentIterNumTables; i++){
                    int globalTableId = globalTableIds[i];
                    int maxValuesPerKey = getNumResultsPerMapThreshold();                    
                    if(runtimeOptions.showProgress){
                        std::cout << "Constructing hash table " << globalTableId << "." << std::endl;
                    }

                    auto hashTable = std::make_unique<Map_t>(
                        std::move(hashesPerTable[i]), 
                        std::move(readIdsPerTable[i]),
                        maxValuesPerKey
                    );

                    numConstructedTables++;
                    minhashTables.emplace_back(std::move(hashTable));
                }
            }else{
                for(int i = 0; i < currentIterNumTables; i++){
                    int globalTableId = globalTableIds[i];
                    int maxValuesPerKey = getNumResultsPerMapThreshold();                    
                    if(runtimeOptions.showProgress){
                        std::cout << "Constructing hash table " << globalTableId << "." << std::endl;
                    }

                    auto hashTable = std::make_unique<Map_t>(
                        std::move(hashesPerTable[i]), 
                        std::move(readIdsPerTable[i]),
                        maxValuesPerKey
                    );

                    numConstructedTables++;
                    
                    hashTable->writeToStream(outstream);
                    numSavedTables++;
                    writtenTableBytes = outstream.tellp();

                    MemoryUsage memInfo = hashTable->getMemoryInfo();

                    std::cerr << "tablesize = " << memInfo.host << "\n";
                    std::cerr << "written total of " << writtenTableBytes << " / " << maxMemoryForTables << "\n";
                    std::cerr << "numSavedTables = " << numSavedTables << "\n";

                    if(maxMemoryForTables <= writtenTableBytes){
                        break;
                    }
                }

                if(numConstructedTables >= requestedNumberOfMaps || maxMemoryForTables < writtenTableBytes){
                    outstream.flush();

                    std::cerr << "available before loading maps: " << (getAvailableMemoryInKB() * 1024) << "\n";
                    
                    int usableNumMaps = 0;

                    //load as many hash tables from file as possible and move them to minhasher
                    std::ifstream instream(tmpmapsFilename, std::ios::binary);
                    for(int i = 0; i < numSavedTables; i++){
                        try{
                            std::cerr << "try loading table " << i << "\n";
                            auto tableptr = std::make_unique<Map_t>();
                            tableptr->loadFromStream(instream);
                            minhashTables.emplace_back(std::move(tableptr));
                            
                            std::cerr << "available after loading table " << i << ": " << (getAvailableMemoryInKB() * 1024) << "\n";
                            usableNumMaps++;
                            std::cerr << "usable num maps = " << usableNumMaps << "\n";
                        }catch(...){
                            std::cerr << "Loading table " << i << " failed\n";
                            break;
                        }                        
                    }

                    filehelpers::removeFile(tmpmapsFilename);
                    std::cout << "Can use " << usableNumMaps << " out of specified " 
                        << requestedNumberOfMaps << " tables\n";
                }   
            }
        }

    }



	void Minhasher::clear(){
		minhashTables.clear();
	}

	void Minhasher::destroy(){
		clear();
		minhashTables.shrink_to_fit();
	}


    std::pair<const Minhasher::Value_t*, const Minhasher::Value_t*>
    Minhasher::queryMap(int mapid, Minhasher::Key_t key) const noexcept{
        assert(mapid < getNumberOfMaps());

        const int numResultsPerMapQueryThreshold = getNumResultsPerMapThreshold();

        const auto mapQueryResult = minhashTables[mapid]->query(key);

        if(mapQueryResult.numValues > numResultsPerMapQueryThreshold){
            return std::make_pair(nullptr, nullptr); //return empty range
        }

        return std::make_pair(mapQueryResult.valuesBegin, mapQueryResult.valuesBegin + mapQueryResult.numValues);
    }

    void Minhasher::getCandidates_any_map(
            Minhasher::Handle& handle,
            const std::string& sequence,
            std::uint64_t) const noexcept{
        getCandidates_any_map(handle, sequence.c_str(), sequence.length(), 0);
    }

    void Minhasher::getCandidates_any_map(
            Minhasher::Handle& handle,
            const char* sequence,
            int sequenceLength,
            std::uint64_t) const noexcept{

        // we do not consider reads which are shorter than k
        if(sequenceLength < getKmerSize()){
            handle.allUniqueResults.clear();
            return;
        }
   
        auto hashValues = minhashfunc(sequence, sequenceLength);

        handle.ranges.clear();

        int maximumResultSize = 0;

        for(int map = 0; map < getNumberOfMaps(); ++map){
            kmer_type key = hashValues[map] & key_mask;
            auto entries_range = queryMap(map, key);
            int n_entries = std::distance(entries_range.first, entries_range.second);
            if(n_entries > 0){
                maximumResultSize += n_entries;
                handle.ranges.emplace_back(entries_range);
            }
        }

        handle.allUniqueResults.resize(maximumResultSize);

        auto resultEnd = k_way_set_union<Value_t>(handle.suHandle, handle.allUniqueResults.begin(), handle.ranges.data(), handle.ranges.size());
        handle.allUniqueResults.erase(resultEnd, handle.allUniqueResults.end());
    }




    std::array<std::uint64_t, maximum_number_of_maps> 
    minhashfunction(const char* sequence, int sequenceLength, int kmerLength, int numHashFuncs) noexcept{

        const int length = sequenceLength;

        std::array<std::uint64_t, maximum_number_of_maps> minhashSignature;
        std::fill_n(minhashSignature.begin(), numHashFuncs, std::numeric_limits<std::uint64_t>::max());

        if(length < kmerLength) return minhashSignature;

        const kmer_type kmer_mask = std::numeric_limits<kmer_type>::max() >> ((Minhasher::maximum_kmer_length - kmerLength) * 2);
        const int rcshiftamount = (Minhasher::maximum_kmer_length - kmerLength) * 2;
        

        auto murmur3_fmix = [](std::uint64_t x) {
            x ^= x >> 33;
            x *= 0xff51afd7ed558ccd;
            x ^= x >> 33;
            x *= 0xc4ceb9fe1a85ec53;
            x ^= x >> 33;
            return x;
        };

        
#if 0
        auto handlekmer = [&](auto fwd, auto rc, int numhashfunc){
            const auto fwdhash = murmur3_fmix(fwd + numhashfunc);
            const auto rchash = murmur3_fmix(rc + numhashfunc);
            const auto smallest = std::min(fwdhash, rchash);
            // if(numhashfunc == 1){
            //     std::cerr << fwd << ' ' << rc << ' ' << fwdhash << ' ' << rchash << ' '  << minhashSignature[numhashfunc] << '\n';
            // }
            minhashSignature[numhashfunc] = std::min(minhashSignature[numhashfunc], smallest);
        };
#else 

        auto handlekmer = [&](auto fwd, auto rc, int numhashfunc){
            const auto smallest = std::min(fwd, rc);
            const auto hashvalue = murmur3_fmix(smallest + numhashfunc);
            minhashSignature[numhashfunc] = std::min(minhashSignature[numhashfunc], hashvalue);
        };


#endif
        kmer_type kmer_encoded = 0;
        kmer_type rc_kmer_encoded = std::numeric_limits<kmer_type>::max();

        auto addBase = [&](char c){
            kmer_encoded <<= 2;
            rc_kmer_encoded >>= 2;
            switch(c) {
            case 'A':
                kmer_encoded |= 0;
                rc_kmer_encoded |= kmer_type(3) << (sizeof(kmer_type) * 8 - 2);
                break;
            case 'C':
                kmer_encoded |= 1;
                rc_kmer_encoded |= kmer_type(2) << (sizeof(kmer_type) * 8 - 2);
                break;
            case 'G':
                kmer_encoded |= 2;
                rc_kmer_encoded |= kmer_type(1) << (sizeof(kmer_type) * 8 - 2);
                break;
            case 'T':
                kmer_encoded |= 3;
                rc_kmer_encoded |= kmer_type(0) << (sizeof(kmer_type) * 8 - 2);
                break;
            default:break;
            }
        };

        for(int i = 0; i < kmerLength - 1; i++){
            addBase(sequence[i]);
        }

        for(int i = kmerLength - 1; i < length; i++){
            addBase(sequence[i]);

            for(int m = 0; m < numHashFuncs; m++){
                handlekmer(kmer_encoded & kmer_mask, 
                            rc_kmer_encoded >> rcshiftamount, 
                            m);
            }
        }

        return minhashSignature;
	}


    std::array<std::uint64_t, maximum_number_of_maps> 
    Minhasher::minhashfunc(const std::string& sequence, int numHashfuncs) const noexcept{
        return minhashfunc(sequence.c_str(), sequence.length(), numHashfuncs);
	}

    std::array<std::uint64_t, maximum_number_of_maps> 
    Minhasher::minhashfunc(const char* sequence, int sequenceLength, int numHashfuncs) const noexcept{
        return minhashfunction(sequence, sequenceLength, getKmerSize(), numHashfuncs);
	}

    int calculateResultsPerMapThreshold(int coverage){
        int result = int(coverage * 2.5f);
        result = std::min(result, int(std::numeric_limits<BucketSize>::max()));
        result = std::max(10, result);
        return result;
    }


}
