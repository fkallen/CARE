#include <minhasher.hpp>

#include <hpc_helpers.cuh>
#include <util.hpp>
#include <config.hpp>
#include <memorymanagement.hpp>
#include <options.hpp>
#include <readstorage.hpp>
#include <minhasher_transform.hpp>


#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <numeric>

//#define NVTXTIMELINE

#ifdef NVTXTIMELINE
#include <gpu/nvtxtimelinemarkers.hpp>
#endif


namespace care{


    Minhasher::Minhasher(Minhasher&& rhs){
        *this = std::move(rhs);
    }

    Minhasher& Minhasher::operator=(Minhasher&& rhs){
        minhashTables = std::move(rhs.minhashTables);
        nReads = std::move(rhs.nReads);
        kmerSize = std::move(rhs.kmerSize);
        resultsPerMapThreshold = std::move(rhs.resultsPerMapThreshold);

        return *this;
    }

    bool Minhasher::operator==(const Minhasher& rhs) const{
        if(kmerSize != rhs.kmerSize)
            return false;
        if(resultsPerMapThreshold != rhs.resultsPerMapThreshold)
            return false;
        if(nReads != rhs.nReads)
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
        int bits_key_tosave = bits_key;
        std::uint64_t key_mask_tosave = key_mask;
        std::uint64_t max_read_num_tosave = max_read_num;
        int maximum_number_of_maps_tosave = maximum_number_of_maps;
        int maximum_kmer_length_tosave = maximum_kmer_length;

        outstream.write(reinterpret_cast<const char*>(&bits_key_tosave), sizeof(int));
        outstream.write(reinterpret_cast<const char*>(&key_mask_tosave), sizeof(std::uint64_t));
        outstream.write(reinterpret_cast<const char*>(&max_read_num_tosave), sizeof(std::uint64_t));
        outstream.write(reinterpret_cast<const char*>(&maximum_number_of_maps_tosave), sizeof(int));
        outstream.write(reinterpret_cast<const char*>(&maximum_kmer_length_tosave), sizeof(int));


        outstream.write(reinterpret_cast<const char*>(&kmerSize), sizeof(int));
        outstream.write(reinterpret_cast<const char*>(&resultsPerMapThreshold), sizeof(int));
        outstream.write(reinterpret_cast<const char*>(&nReads), sizeof(read_number));

        const int numTables = getNumberOfMaps();
        outstream.write(reinterpret_cast<const char*>(&numTables), sizeof(int));
        
        for(const auto& tableptr : minhashTables)
            tableptr->writeToStream(outstream);
    }

    void Minhasher::loadFromStream(std::ifstream& instream){

        destroy();

        int bits_key_loaded;
    	std::uint64_t key_mask_loaded;
        std::uint64_t max_read_num_loaded;
        int maximum_number_of_maps_loaded;
        int maximum_kmer_length_loaded;

        instream.read(reinterpret_cast<char*>(&bits_key_loaded), sizeof(int));
        instream.read(reinterpret_cast<char*>(&key_mask_loaded), sizeof(std::uint64_t));
        instream.read(reinterpret_cast<char*>(&max_read_num_loaded), sizeof(std::uint64_t));
        instream.read(reinterpret_cast<char*>(&maximum_number_of_maps_loaded), sizeof(int));
        instream.read(reinterpret_cast<char*>(&maximum_kmer_length_loaded), sizeof(int));

        assert(bits_key == bits_key_loaded);
        assert(key_mask == key_mask_loaded);
        assert(max_read_num == max_read_num_loaded);
        assert(maximum_number_of_maps == maximum_number_of_maps_loaded);
        assert(maximum_kmer_length == maximum_kmer_length_loaded);

        instream.read(reinterpret_cast<char*>(&kmerSize), sizeof(int));
        instream.read(reinterpret_cast<char*>(&resultsPerMapThreshold), sizeof(int));
        instream.read(reinterpret_cast<char*>(&nReads), sizeof(read_number));

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

	void Minhasher::init(std::uint64_t nReads_){
		if(nReads_ == 0) throw std::runtime_error("Minhasher::init cannnot be called with argument 0");
		if(nReads_-1 > max_read_num)
			throw std::runtime_error("Minhasher::init: Minhasher is configured for only"
                                    + std::to_string(max_read_num) + " reads, not " + std::to_string(nReads_) + "!!!");

		nReads = nReads_;

		minhashTables.clear();
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

        init(nReads);

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
		nReads = 0;
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

    std::vector<Minhasher::Result_t> Minhasher::getCandidates(const std::string& sequence,
                                        int num_hits,
                                        std::uint64_t max_number_candidates) const noexcept{

        if(num_hits == 1){
            return getCandidates_any_map(sequence, max_number_candidates);
        }else if(num_hits == getNumberOfMaps()){
            return getCandidates_all_maps(sequence, max_number_candidates);
        }else{
            return getCandidates_some_maps(sequence, num_hits, max_number_candidates);
        }
    }

    std::vector<Minhasher::Result_t> Minhasher::getCandidates_any_map(const std::string& sequence,
                                        std::uint64_t) const noexcept{
        Minhasher::Handle handle;
        getCandidates_any_map(handle, sequence, 0);

        std::vector<Result_t> result(std::move(handle.result()));
        return result;
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

        static_assert(std::is_same<Result_t, Value_t>::value, "Value_t != Result_t");
        // we do not consider reads which are shorter than k
        if(sequenceLength < getKmerSize()){
            handle.allUniqueResults.clear();
            return;
        }

        //TIMERSTARTCPU(minhashfunc);
#ifdef NVTXTIMELINE        
        nvtx::push_range("hashing", 3);
#endif        
        auto hashValues = minhashfunc(sequence, sequenceLength);
#ifdef NVTXTIMELINE        
        nvtx::pop_range("hashing");
#endif        
        //TIMERSTOPCPU(minhashfunc);

        handle.ranges.clear();

        int maximumResultSize = 0;

#if 0
        std::array<Index_t, 64> preparedIndices{};

        auto prepare_map = [&](int mapid, kmer_type key){
            assert(mapid < getNumberOfMaps());
            preparedIndices[mapid] = minhashTables[mapid]->prepare_get_ranged(key);
        };

        auto query_map = [&](int mapid, Index_t preparedIndex){
            auto entries_range = minhashTables[mapid]->execute_get_ranged(preparedIndex);
            std::size_t n_entries = std::distance(entries_range.first, entries_range.second);

            if(n_entries > numResultsPerMapQueryThreshold){
                return std::make_pair(entries_range.first, entries_range.first); //return empty range
            }

            return entries_range;
        };

        constexpr int preparedistance = 8;
        const int chunks = SDIV(getNumberOfMaps(), preparedistance);

        for(int map = 0; map < std::min(preparedistance, getNumberOfMaps()); map++){
            kmer_type key = hashValues[map] & key_mask;
            prepare_map(map, key);
        }

        for(int iteration = 0; iteration < chunks; iteration++){
            if(iteration < chunks - 1){
                const int nextIteration = iteration + 1;
                const int begin = nextIteration * preparedistance;
                const int end = std::min((nextIteration+1) * preparedistance, getNumberOfMaps());

                for(int map = begin; map < end; map++){
                    kmer_type key = hashValues[map] & key_mask;
                    prepare_map(map, key);
                }
            }

            const int begin = iteration * preparedistance;
            const int end = std::min((iteration+1) * preparedistance, getNumberOfMaps());
            for(int map = begin; map < end; map++){
                auto entries_range = query_map(map, preparedIndices[map]);
                int n_entries = std::distance(entries_range.first, entries_range.second);
                if(n_entries > 0){
                    maximumResultSize += n_entries;
                    ranges.emplace_back(entries_range);
                }
            }

        }

#else

        //TIMERSTARTCPU(query);

#ifdef NVTXTIMELINE
        nvtx::push_range("map queries", 4);
#endif

        for(int map = 0; map < getNumberOfMaps(); ++map){
            kmer_type key = hashValues[map] & key_mask;
            auto entries_range = queryMap(map, key);
            int n_entries = std::distance(entries_range.first, entries_range.second);
            if(n_entries > 0){
                maximumResultSize += n_entries;
                handle.ranges.emplace_back(entries_range);
            }
        }
#ifdef NVTXTIMELINE
        nvtx::pop_range("map queries");
#endif 

#endif
        //TIMERSTOPCPU(query);

        //TIMERSTARTCPU(setunion);     
#ifdef NVTXTIMELINE         
        nvtx::push_range("setunion", 5);
#endif

#if 1
        handle.allUniqueResults.resize(maximumResultSize);

        auto resultEnd = k_way_set_union<Value_t>(handle.suHandle, handle.allUniqueResults.begin(), handle.ranges.data(), handle.ranges.size());
        handle.allUniqueResults.erase(resultEnd, handle.allUniqueResults.end());
#else 
        std::unordered_set<Value_t> uniqueValues;
        for(const auto& range : ranges){
            uniqueValues.insert(range.first, range.second);
        }
        std::vector<Value_t> allUniqueResults(uniqueValues.size());
        std::copy(uniqueValues.cbegin(), uniqueValues.cend(), allUniqueResults.begin());

#endif
        // std::vector<Value_t> allUniqueResultsPQ(maximumResultSize);
        // auto resultEndPQ = k_way_set_union_with_priorityqueue(allUniqueResultsPQ.begin(), ranges);
        // allUniqueResultsPQ.erase(resultEndPQ, allUniqueResultsPQ.end()); 

        // assert(allUniqueResults.size() == allUniqueResultsPQ.size());
        // assert(allUniqueResults == allUniqueResultsPQ);



#ifdef NVTXTIMELINE        
        nvtx::pop_range("setunion");
#endif 

        //TIMERSTOPCPU(setunion);
    }



    void Minhasher::calculateMinhashSignatures(
            Minhasher::Handle& handle,
            const std::vector<std::string>& sequences) const{

        handle.multiminhashSignatures.resize(getNumberOfMaps() * sequences.size());

        const int numSequences = sequences.size();
        for(int i = 0; i < numSequences; i++){
            const char* sequence = sequences[i].c_str();
            const int length = sequences[i].length();
            if(length < getKmerSize()){
                // handle.allUniqueResults.clear();
                // return;
            }else{
                auto hashValues = minhashfunc(sequence, length);
                std::copy(
                    hashValues.begin(), 
                    hashValues.end(), 
                    handle.multiminhashSignatures.begin() + getNumberOfMaps() * i
                );
            }
        }
    }

    void Minhasher::calculateMinhashSignatures(
            Minhasher::Handle& handle,
            const char* sequences,
            int numSequences,
            const int* sequenceLengths,
            int sequencesPitch) const{

        handle.multiminhashSignatures.resize(getNumberOfMaps() * numSequences);

        for(int i = 0; i < numSequences; i++){
            const char* sequence = sequences + i * sequencesPitch;
            const int length = sequenceLengths[i];
            if(length < getKmerSize()){
                // handle.allUniqueResults.clear();
                // return;
            }else{
                auto hashValues = minhashfunc(sequence, length);
                std::copy(
                    hashValues.begin(), 
                    hashValues.end(), 
                    handle.multiminhashSignatures.begin() + getNumberOfMaps() * i
                );
            }
        }
    }

    void Minhasher::queryPrecalculatedSignatures(Minhasher::Handle& handle, int numSequences) const{

        handle.multiranges.clear();
        handle.multiranges.reserve(getNumberOfMaps() * numSequences);
        handle.numResultsPerSequence.clear();
        handle.numResultsPerSequence.resize(numSequences, 0);        

        for(int i = 0; i < numSequences; i++){
            const std::uint64_t* signature = &handle.multiminhashSignatures[i * getNumberOfMaps()];

            for(int map = 0; map < getNumberOfMaps(); ++map){
                kmer_type key = signature[map] & key_mask;
                auto entries_range = queryMap(map, key);
                int n_entries = std::distance(entries_range.first, entries_range.second);
                if(n_entries > 0){
                    handle.numResultsPerSequence[i] += n_entries;
                }
                handle.multiranges.emplace_back(entries_range);
            }
        }      
    }

    void Minhasher::queryPrecalculatedSignatures(
        const std::uint64_t* signatures, //getNumberOfMaps() elements per sequence
        Minhasher::Range_t* ranges, //getNumberOfMaps() elements per sequence
        int* totalNumResultsInRanges, 
        int numSequences) const{ 
        
        int numResults = 0;

        for(int i = 0; i < numSequences; i++){
            const std::uint64_t* const signature = &signatures[i * getNumberOfMaps()];
            Minhasher::Range_t* const range = &ranges[i * getNumberOfMaps()];            

            for(int map = 0; map < getNumberOfMaps(); ++map){
                kmer_type key = signature[map] & key_mask;
                auto entries_range = queryMap(map, key);
                numResults += std::distance(entries_range.first, entries_range.second);
                range[map] = entries_range;
            }
        }   

        *totalNumResultsInRanges = numResults;   
    }

    //static bool once = true;

    void Minhasher::makeUniqueQueryResults(Minhasher::Handle& handle, int numSequences) const{
        int maxNumResults = 0;
        for(int i = 0; i < numSequences; i++){
            maxNumResults += handle.numResultsPerSequence[i];
        }
        handle.multiallUniqueResults.resize(maxNumResults);
        handle.numResultsPerSequencePrefixSum.resize(numSequences + 1);

        handle.numResultsPerSequencePrefixSum[0] = 0;

        // if(once){
        //     once = false;

        //     std::ofstream stream("queryranges.txt");

        //     for(int i = 0; i < numSequences; i++){
        //         stream << handle.numResultsPerSequence[i] << "\n";
        //         auto myranges = &handle.multiranges[i * getNumberOfMaps()];
        //         const int numRangesForSequence = getNumberOfMaps();

        //         for(int k = 0; k < numRangesForSequence; k++){
        //             auto& range = myranges[k];
        //             stream << (range.second - range.first) << "\n";
                    
        //             for(auto it = range.first; it != range.second; it++){
        //                 stream << *it << " ";                     
        //             }
        //             stream<< "\n";
        //         } 
        //     }
        // }

        handle.contiguousDataOfRanges.resize(maxNumResults);

        // copy the data identified by the ranges, which may be scattered across the whole hash tables, 
        // into a contiguous chunk of memory
        {
            auto currentBegin = handle.contiguousDataOfRanges.begin();
            auto currentEnd = handle.contiguousDataOfRanges.begin();
            for(int i = 0; i < numSequences; i++){
                auto myranges = &handle.multiranges[i * getNumberOfMaps()];
                const int numRangesForSequence = getNumberOfMaps();
                for(int r = 0; r < numRangesForSequence; r++){
                    auto& range = myranges[r];
                    currentEnd = std::copy(range.first, range.second, currentBegin);
                    range.first = &(*currentBegin);
                    range.second = &(*currentEnd);
                    currentBegin = currentEnd;
                }
            }
        }

        //for each queried sequence, perform set union of all its ranges
        auto currentBegin = handle.multiallUniqueResults.begin();
        auto currentEnd = handle.multiallUniqueResults.begin();
        for(int i = 0; i < numSequences; i++){
            auto myranges = &handle.multiranges[i * getNumberOfMaps()];
            const int numRangesForSequence = getNumberOfMaps();
            currentBegin = currentEnd;
            currentEnd = k_way_set_union<Value_t>(handle.suHandle, currentBegin, myranges, numRangesForSequence);
            handle.numResultsPerSequence[i] = std::distance(currentBegin, currentEnd);

            handle.numResultsPerSequencePrefixSum[i+1] = handle.numResultsPerSequencePrefixSum[i] + handle.numResultsPerSequence[i];
        }

        handle.multiallUniqueResults.erase(currentEnd, handle.multiallUniqueResults.end());   
    }





    std::vector<Minhasher::Result_t> Minhasher::getCandidates_some_maps2(const std::string& sequence,
                                        int num_hits,
                                        std::uint64_t max_number_candidates) const noexcept{
        static_assert(std::is_same<Result_t, Value_t>::value, "Value_t != Result_t");
        // we do not consider reads which are shorter than k
        const int length = sequence.size();
        if(length < getKmerSize())
            return {};

        if(num_hits > getNumberOfMaps() || num_hits < 1)
            return {};

        //TIMERSTARTCPU(minhashfunc);
        auto hashValues = minhashfunc(sequence);
        //TIMERSTOPCPU(minhashfunc);

        std::size_t total_num_ids = 0;
        std::vector<const Value_t*> iters;
        iters.reserve(getNumberOfMaps()*2);
        for(int map = 0; map < getNumberOfMaps(); ++map){
            kmer_type key = hashValues[map] & key_mask;

            auto entries_range = queryMap(map, key);

            iters.emplace_back(entries_range.first);
            iters.emplace_back(entries_range.second);

            std::size_t n_entries = std::distance(entries_range.first, entries_range.second);
            //std::cout << "map " << map << ", ids " << n_entries << std::endl;
            total_num_ids += n_entries;
        }

        std::vector<Value_t> allCandidateIds(total_num_ids);

        //the following two function can probably be fused

        //merge ids from all maps into vector
        auto allCandidateIdsNewEnd = k_way_merge_naive_sortonce(allCandidateIds.begin(), iters);

        //remove all ids which occure less than num_hits times.
        allCandidateIdsNewEnd = remove_by_count_unique_with_limit(allCandidateIds.begin(),
                                                                        allCandidateIdsNewEnd,
                                                                        num_hits,
                                                                        max_number_candidates);

        std::vector<Value_t> result(allCandidateIds.begin(), allCandidateIdsNewEnd);

        return result;
    }

    std::vector<Minhasher::Result_t> Minhasher::getCandidates_some_maps(const std::string& sequence,
                                        int num_hits,
                                        std::uint64_t max_number_candidates) const noexcept{
        static_assert(std::is_same<Result_t, Value_t>::value, "Value_t != Result_t");
        // we do not consider reads which are shorter than k
        const int length = sequence.size();
        if(length < getKmerSize())
            return {};

        if(num_hits > getNumberOfMaps() || num_hits < 1)
            return {};

        const int numResultsPerMapQueryThreshold = getNumResultsPerMapThreshold();

        //TIMERSTARTCPU(minhashfunc);
        auto hashValues = minhashfunc(sequence);
        //TIMERSTOPCPU(minhashfunc);

        const size_t maximumResultSize = std::min(std::uint64_t(numResultsPerMapQueryThreshold) * getNumberOfMaps(), max_number_candidates);

        std::vector<Value_t> allUniqueResults;
        std::vector<Value_t> tmp;
        allUniqueResults.reserve(maximumResultSize);
        tmp.reserve(maximumResultSize);

        //TIMERSTARTCPU(getcandrest);
        for(int map = 0; map < getNumberOfMaps() && allUniqueResults.size() < max_number_candidates; ++map) {
            kmer_type key = hashValues[map] & key_mask;

            auto entries_range = queryMap(map, key);
            std::size_t n_entries = std::distance(entries_range.first, entries_range.second);

            if(n_entries == 0){
                continue;
            }

            tmp.resize(allUniqueResults.size() + n_entries);
            auto merge_end = merge_with_count_theshold(entries_range.first,
                                                entries_range.second,
                                                allUniqueResults.begin(),
                                                allUniqueResults.end(),
                                                num_hits,
                                                max_number_candidates,
                                                tmp.begin());
            if(tmp.begin() == merge_end){
                return {};
            }else{
                tmp.resize(std::distance(tmp.begin(), merge_end));
                std::swap(tmp, allUniqueResults);
            }
        }

        //std::copy(allUniqueResults.begin(), allUniqueResults.end(), std::ostream_iterator<Value_t>(std::cout, " "));
	    //std::cout << std::endl;

        auto resultEnd = remove_by_count_unique_with_limit(allUniqueResults.begin(),
                                                            allUniqueResults.end(),
                                                            num_hits,
                                                            max_number_candidates);

        allUniqueResults.erase(resultEnd, allUniqueResults.end());

        //std::copy(allUniqueResults.begin(), allUniqueResults.end(), std::ostream_iterator<Value_t>(std::cout, " "));
	    //std::cout << std::endl;

        //char a;
        //std::cin >> a;

        return allUniqueResults;
    }


    std::vector<Minhasher::Result_t> Minhasher::getCandidates_all_maps(const std::string& sequence,
                                        std::uint64_t max_number_candidates) const noexcept{
        static_assert(std::is_same<Result_t, Value_t>::value, "Value_t != Result_t");
        // we do not consider reads which are shorter than k
        const int length = sequence.size();
        if(length < getKmerSize())
            return {};

        const int numResultsPerMapQueryThreshold = getNumResultsPerMapThreshold();

        //TIMERSTARTCPU(minhashfunc);
        auto hashValues = minhashfunc(sequence);
        //TIMERSTOPCPU(minhashfunc);

        const size_t maximumResultSize = std::min(std::uint64_t(numResultsPerMapQueryThreshold) * getNumberOfMaps(), max_number_candidates);

        std::vector<Value_t> allUniqueResults;
        std::vector<Value_t> tmp;
        allUniqueResults.reserve(maximumResultSize);
        tmp.reserve(maximumResultSize);

        //TIMERSTARTCPU(getcandrest);
        for(int map = 0; map < getNumberOfMaps() && allUniqueResults.size() < max_number_candidates; ++map) {
            kmer_type key = hashValues[map] & key_mask;

            auto entries_range = queryMap(map, key);
            std::size_t n_entries = std::distance(entries_range.first, entries_range.second);

            tmp.resize(allUniqueResults.size() + n_entries);
            auto intersection_end = set_intersection_n_or_empty(entries_range.first,
                                                entries_range.second,
                                                allUniqueResults.begin(),
                                                allUniqueResults.end(),
                                                max_number_candidates,
                                                tmp.begin());
            if(tmp.begin() == intersection_end){
                return {};
            }else{
                tmp.resize(std::distance(tmp.begin(), intersection_end));
                std::swap(tmp, allUniqueResults);
            }
        }

        return allUniqueResults;
    }


#if 0
std::vector<Minhasher::Result_t> 
Minhasher::getCandidates_fromHashvalues_any_map(
    const std::string& sequence,
    std::uint64_t* hashValues) const noexcept{
        static_assert(std::is_same<Result_t, Value_t>::value, "Value_t != Result_t");
        // we do not consider reads which are shorter than k
        const int length = sequence.size();
        if(length < getKmerSize())
            return {};

        //TIMERSTARTCPU(minhashfunc);     
        auto hashValuesOriginal = minhashfunc(sequence);
 
        //TIMERSTOPCPU(minhashfunc);

        using Range_t = std::pair<const Value_t*, const Value_t*>;
        std::vector<Range_t> ranges;
        ranges.reserve(getNumberOfMaps());

        int maximumResultSize = 0;


        //TIMERSTARTCPU(query);

#ifdef NVTXTIMELINE
        nvtx::push_range("map queries", 4);
#endif

        for(int map = 0; map < getNumberOfMaps(); ++map){
            kmer_type key = hashValues[map] & key_mask;
            auto entries_range = queryMap(map, key);
            int n_entries = std::distance(entries_range.first, entries_range.second);
            if(n_entries > 0){
                maximumResultSize += n_entries;
                ranges.emplace_back(entries_range);
            }
        }
#ifdef NVTXTIMELINE
        nvtx::pop_range("map queries");
#endif 

        //TIMERSTOPCPU(query);

        //TIMERSTARTCPU(setunion);     
#ifdef NVTXTIMELINE         
        nvtx::push_range("setunion", 5);
#endif

#if 1
        std::vector<Value_t> allUniqueResults(maximumResultSize);

        auto resultEnd = k_way_set_union(allUniqueResults.begin(), ranges);
        allUniqueResults.erase(resultEnd, allUniqueResults.end());
#else 
        std::unordered_set<Value_t> uniqueValues;
        for(const auto& range : ranges){
            uniqueValues.insert(range.first, range.second);
        }
        std::vector<Value_t> allUniqueResults(uniqueValues.size());
        std::copy(uniqueValues.cbegin(), uniqueValues.cend(), allUniqueResults.begin());

#endif
        // std::vector<Value_t> allUniqueResultsPQ(maximumResultSize);
        // auto resultEndPQ = k_way_set_union_with_priorityqueue(allUniqueResultsPQ.begin(), ranges);
        // allUniqueResultsPQ.erase(resultEndPQ, allUniqueResultsPQ.end()); 

        // assert(allUniqueResults.size() == allUniqueResultsPQ.size());
        // assert(allUniqueResults == allUniqueResultsPQ);



#ifdef NVTXTIMELINE        
        nvtx::pop_range("setunion");
#endif 

        //TIMERSTOPCPU(setunion);

        return allUniqueResults;
    }
#endif




    

// #############################

/*
    Query number of candidates
*/

    // std::int64_t Minhasher::getNumberOfCandidates(const std::string& sequence,
    //                                     int num_hits) const noexcept{

    //     const std::uint64_t max_number_candidates = std::numeric_limits<std::uint64_t>::max();

    //     std::vector<Result_t> result;

    //     if(num_hits == 1){
    //         result = getCandidates_any_map(sequence, max_number_candidates);
    //     }else if(num_hits == getNumberOfMaps()){
    //         result = getCandidates_all_maps(sequence, max_number_candidates);
    //     }else{
    //         result = getCandidates_some_maps(sequence, num_hits, max_number_candidates);
    //     }

    //     assert(result.size() <= std::numeric_limits<std::int64_t>::max());

    //     return std::int64_t(result.size());
    // }

    // std::int64_t Minhasher::getNumberOfCandidatesUpperBound(const std::string& sequence) const noexcept{
	// 	static_assert(std::is_same<Result_t, Value_t>::value, "Value_t != Result_t");
	// 	// we do not consider reads which are shorter than k
	// 	const int length = sequence.size();
    //     if(length < getKmerSize())
	// 		return 0;

	// 	//TIMERSTARTCPU(minhashfunc);
	// 	auto hashValues = minhashfunc(sequence);
	// 	//TIMERSTOPCPU(minhashfunc);

    //     std::size_t result = 0;

    //     for(int map = 0; map < getNumberOfMaps(); ++map) {
    //         kmer_type key = hashValues[map] & key_mask;

	// 		//TIMERSTARTCPU(get_ranged);
    //         const auto entries_range = minhashTables[map]->get_ranged(key);
    //         const std::size_t n_entries = std::distance(entries_range.first, entries_range.second);
    //         result += n_entries;
	// 		//TIMERSTOPCPU(get_ranged);
    //     }

    //     assert(result >= std::size_t(getNumberOfMaps()));
    //     result -= getNumberOfMaps(); //remove self from each map result

	// 	return std::int64_t(result);

	// }

//###################################################
   

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
