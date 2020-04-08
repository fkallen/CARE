#include <minhasher.hpp>
#include <options.hpp>
#include <hpc_helpers.cuh>
#include <util.hpp>
#include <config.hpp>
#include <memorymanagement.hpp>

#include <ntHash/nthash.hpp>

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

    Minhasher::Minhasher() : Minhasher(MinhashOptions{2,16}){}

	Minhasher::Minhasher(const MinhashOptions& parameters)
		: minparams(parameters), nReads(0)
	{
		if(maximum_number_of_maps < minparams.maps)
			throw std::runtime_error("Minhasher: Maximum number of maps is "
									+ std::to_string(maximum_number_of_maps) + "!");

		if(maximum_kmer_length < minparams.k){
			throw std::runtime_error("Minhasher is configured for maximum kmer length of "
									+ std::to_string(maximum_kmer_length) + "!");
		}
	}

    Minhasher::Minhasher(Minhasher&& rhs){
        *this = std::move(rhs);
    }

    Minhasher& Minhasher::operator=(Minhasher&& rhs){
        minhashTables = std::move(rhs.minhashTables);
        minparams = std::move(rhs.minparams);
        nReads = std::move(rhs.nReads);
        canUseGpu = std::move(rhs.canUseGpu);
        allowUVM = std::move(rhs.allowUVM);

        return *this;
    }

    bool Minhasher::operator==(const Minhasher& rhs) const{
        if(minparams != rhs.minparams)
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



    std::size_t Minhasher::numBytes() const{
        //return minhashTables[0]->numBytes() * minhashTables.size();
        std::size_t result = 0;
        for(const auto& m : minhashTables)
            result += m->numBytes();
        return result;
    }

    MemoryUsage Minhasher::getMemoryInfo() const{
        MemoryUsage memoryInfo;

        memoryInfo.host = 0;
        for(const auto& m : minhashTables)
            memoryInfo.host += m->numBytes();
            
        return memoryInfo;
    }



    void Minhasher::saveToFile(const std::string& filename) const{
        std::ofstream outstream(filename, std::ios::binary);

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

        outstream.write(reinterpret_cast<const char*>(&minparams), sizeof(MinhashOptions));
        outstream.write(reinterpret_cast<const char*>(&nReads), sizeof(read_number));
        outstream.write(reinterpret_cast<const char*>(&canUseGpu), sizeof(bool));
        for(const auto& tableptr : minhashTables)
            tableptr->writeToStream(outstream);
    }

    void Minhasher::loadFromFile(const std::string& filename){
        std::ifstream instream(filename, std::ios::binary);
        if(!instream)
            throw std::runtime_error("Cannot load hashtable from file " + filename);

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

        MinhashOptions minparams_loaded;
        read_number nReads_loaded;
        bool canUseGpu_loaded;

        instream.read(reinterpret_cast<char*>(&minparams_loaded), sizeof(MinhashOptions));
        instream.read(reinterpret_cast<char*>(&nReads_loaded), sizeof(read_number));
        instream.read(reinterpret_cast<char*>(&canUseGpu_loaded), sizeof(bool));

        // assert(minparams == minparams_loaded);
        // assert(nReads == nReads_loaded);

        minparams = minparams_loaded;
        nReads = nReads_loaded;


        //minhashTables.resize(minparams.maps);

		// for (int i = 0; i < minparams.maps; ++i) {
		// 	minhashTables[i].reset();
		// }

        // for(auto& tableptr : minhashTables)
        //     tableptr->readFromStream(instream);

        minhashTables.clear();

        for(int i = 0; i < minparams.maps; i++){
            try{
                auto tmptableptr = std::make_unique<Minhasher::Map_t>();
                tmptableptr->readFromStream(instream);
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

		minhashTables.resize(minparams.maps);

		for (int i = 0; i < minparams.maps; ++i) {
			minhashTables[i].reset();
		}
	}

    void Minhasher::initMap(int mapId){
        assert(mapId < minparams.maps);
        minhashTables[mapId].reset();
    }

    void Minhasher::moveassignMap(int mapId, Minhasher::Map_t&& newMap){
        assert(mapId < minparams.maps);
        minhashTables[mapId].reset(new Map_t(std::move(newMap)));
    }

	void Minhasher::clear(){
		minhashTables.clear();
		nReads = 0;
	}

	void Minhasher::destroy(){
		clear();
		minhashTables.shrink_to_fit();
	}

    std::map<int, std::int64_t> Minhasher::getBinSizeHistogramOfMap(const Minhasher::Map_t& table) const{
        std::map<int, std::int64_t> histogram;
        if(table.countsPrefixSum.size() > 0){
            for(size_t i = 0; i < table.countsPrefixSum.size() - 1; i++){
                const auto count = table.countsPrefixSum[i+1] - table.countsPrefixSum[i];
                histogram[count]++;
            }
        }
        return histogram;
    }

    std::map<int, std::int64_t> Minhasher::getBinSizeHistogramOfMap(int tableId) const{
        assert(tableId < minparams.maps);
        return getBinSizeHistogramOfMap(*minhashTables[tableId]);
    }

    std::vector<std::map<int, std::int64_t>> Minhasher::getBinSizeHistogramsOfMaps() const{
        std::vector<std::map<int, std::int64_t>> result;
        result.reserve(minparams.maps);

        for(int i = 0; i < minparams.maps; i++){
            result.emplace_back(getBinSizeHistogramOfMap(i));
        }

        return result;
    }

    void Minhasher::insertIntoExternalTable(Minhasher::Map_t& table, std::uint64_t hashValue, read_number readnum) const{

        kmer_type key = hashValue & key_mask;
        Value_t value(readnum);

        if (!table.add(key, value, readnum)) {
            throw std::runtime_error(("error adding key to map. key "
                                        + std::to_string(key) + " "
                                        + std::to_string(value) + " "
                                        + std::to_string(readnum)));
        }
    }

    void Minhasher::insertIntoMap(int map, std::uint64_t hashValue, read_number readnum){
        assert(map < minparams.maps);

        kmer_type key = hashValue & key_mask;
        Value_t value(readnum);

        if (!minhashTables[map]->add(key, value, readnum)) {
            throw std::runtime_error(("error adding key to map. key "
                                        + std::to_string(key) + " "
                                        + std::to_string(value) + " "
                                        + std::to_string(readnum)));
        }
    }

	void Minhasher::insertSequence(const std::string& sequence, read_number readnum){
		if(readnum >= nReads)
			throw std::runtime_error("Minhasher::insertSequence: read number too large. "
                                    + std::to_string(readnum) + " > " + std::to_string(nReads));

		// we do not consider reads which are shorter than k
		if(sequence.size() < unsigned(minparams.k))
			return;

		//get hash values
        //TIMERSTARTCPU(minhashfunc);
		auto hashValues = minhashfunc(sequence);
        //TIMERSTOPCPU(minhashfunc);

		// insert
        //TIMERSTARTCPU(insertTupleIntoMap);
		for (int map = 0; map < minparams.maps; ++map) {
            insertIntoMap(map, hashValues[map], readnum);
		}
        //TIMERSTOPCPU(insertTupleIntoMap);
	}

    //Calculate hash values of sequence. Insert hashvalue[hashIds[i]] into tables[tableIds[i]]
    void Minhasher::insertSequenceIntoExternalTables(const std::string& sequence, 
                                                    read_number readnum,                                                     
                                                    const std::vector<int>& tableIds,
                                                    std::vector<Minhasher::Map_t>& tables,
                                                    const std::vector<int>& hashIds) const{
		if(readnum >= nReads)
			throw std::runtime_error("Minhasher::insertSequence: read number too large. "
                                    + std::to_string(readnum) + " > " + std::to_string(nReads));
        assert(tableIds.size() == hashIds.size());
        assert(std::all_of(tableIds.begin(), tableIds.end(), [&](auto id){return id < int(tables.size());}));
        assert(std::all_of(hashIds.begin(), hashIds.end(), [&](auto id){return id < minparams.maps;}));

		// we do not consider reads which are shorter than k
		if(sequence.size() < unsigned(minparams.k))
			return; //TODO this breaks querying

		auto hashValues = minhashfunc(sequence);

        for(int i = 0; i < int(hashIds.size()); i++){
            auto hashValue = hashValues[hashIds[i]];
            auto tableId = tableIds[i];
            auto& table = tables[tableId];
            insertIntoExternalTable(table, hashValue, readnum);
        }
	}

    void Minhasher::insertSequenceIntoExternalTables(const std::uint64_t* hashValues, 
                                                    int numHashValues,
                                                    read_number readnum,                                                     
                                                    const std::vector<int>& tableIds,
                                                    std::vector<Minhasher::Map_t>& tables) const{
		if(readnum >= nReads)
			throw std::runtime_error("Minhasher::insertSequence: read number too large. "
                                    + std::to_string(readnum) + " > " + std::to_string(nReads));
        assert(tableIds.size() == std::size_t(numHashValues));
        assert(std::all_of(tableIds.begin(), tableIds.end(), [&](auto id){return id < int(tables.size());}));

        for(int i = 0; i < numHashValues; i++){
            auto hashValue = hashValues[i];
            auto tableId = tableIds[i];
            auto& table = tables[tableId];
            insertIntoExternalTable(table, hashValue, readnum);
        }
	}

    void Minhasher::insertSequence(const std::string& sequence, read_number readnum, std::vector<int> mapIds){
        assert(int(mapIds.size()) <= minparams.maps);

		if(readnum >= nReads)
			throw std::runtime_error("Minhasher::insertSequence: read number too large. "
                                    + std::to_string(readnum) + " > " + std::to_string(nReads));

		// we do not consider reads which are shorter than k
		if(sequence.size() < unsigned(minparams.k))
			return;

		//get hash values
        //TIMERSTARTCPU(minhashfunc);
		auto hashValues = minhashfunc(sequence);
        //TIMERSTOPCPU(minhashfunc);

		// insert
        //TIMERSTARTCPU(insertTupleIntoMap);
        for(auto mapId : mapIds){
            assert(mapId < minparams.maps);
            insertIntoMap(mapId, hashValues[mapId], readnum);

        }
        //TIMERSTOPCPU(insertTupleIntoMap);

	}

    std::pair<const Minhasher::Value_t*, const Minhasher::Value_t*>
    Minhasher::queryMap(int mapid, Minhasher::Map_t::Key_t key) const noexcept{
        assert(mapid < minparams.maps);

        const int numResultsPerMapQueryThreshold = getResultsPerMapThreshold();

        auto entries_range = minhashTables[mapid]->get_ranged(key);
        int n_entries = std::distance(entries_range.first, entries_range.second);

        if(n_entries > numResultsPerMapQueryThreshold){
            return std::make_pair(entries_range.first, entries_range.first); //return empty range
        }

        return entries_range;
    }

    std::vector<Minhasher::Result_t> Minhasher::getCandidates(const std::string& sequence,
                                        int num_hits,
                                        std::uint64_t max_number_candidates) const noexcept{

        if(num_hits == 1){
            return getCandidates_any_map(sequence, max_number_candidates);
        }else if(num_hits == minparams.maps){
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
        if(sequenceLength < minparams.k){
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
            assert(mapid < minparams.maps);
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
        const int chunks = SDIV(minparams.maps, preparedistance);

        for(int map = 0; map < std::min(preparedistance, minparams.maps); map++){
            kmer_type key = hashValues[map] & key_mask;
            prepare_map(map, key);
        }

        for(int iteration = 0; iteration < chunks; iteration++){
            if(iteration < chunks - 1){
                const int nextIteration = iteration + 1;
                const int begin = nextIteration * preparedistance;
                const int end = std::min((nextIteration+1) * preparedistance, minparams.maps);

                for(int map = begin; map < end; map++){
                    kmer_type key = hashValues[map] & key_mask;
                    prepare_map(map, key);
                }
            }

            const int begin = iteration * preparedistance;
            const int end = std::min((iteration+1) * preparedistance, minparams.maps);
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

        for(int map = 0; map < minparams.maps; ++map){
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
            if(length < minparams.k){
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
            if(length < minparams.k){
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
        handle.multiranges.reserve(minparams.maps * numSequences);
        handle.numResultsPerSequence.clear();
        handle.numResultsPerSequence.resize(numSequences, 0);        

        for(int i = 0; i < numSequences; i++){
            const std::uint64_t* signature = &handle.multiminhashSignatures[i * getNumberOfMaps()];

            for(int map = 0; map < minparams.maps; ++map){
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

            for(int map = 0; map < minparams.maps; ++map){
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
        //         auto myranges = &handle.multiranges[i * minparams.maps];
        //         const int numRangesForSequence = minparams.maps;

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
                auto myranges = &handle.multiranges[i * minparams.maps];
                const int numRangesForSequence = minparams.maps;
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
            auto myranges = &handle.multiranges[i * minparams.maps];
            const int numRangesForSequence = minparams.maps;
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
        if(sequence.size() < unsigned(minparams.k))
            return {};

        if(num_hits > minparams.maps || num_hits < 1)
            return {};

        //TIMERSTARTCPU(minhashfunc);
        auto hashValues = minhashfunc(sequence);
        //TIMERSTOPCPU(minhashfunc);

        std::size_t total_num_ids = 0;
        std::vector<const Value_t*> iters;
        iters.reserve(minparams.maps*2);
        for(int map = 0; map < minparams.maps; ++map){
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
        if(sequence.size() < unsigned(minparams.k))
            return {};

        if(num_hits > minparams.maps || num_hits < 1)
            return {};

        const int numResultsPerMapQueryThreshold = getResultsPerMapThreshold();

        //TIMERSTARTCPU(minhashfunc);
        auto hashValues = minhashfunc(sequence);
        //TIMERSTOPCPU(minhashfunc);

        const size_t maximumResultSize = std::min(std::uint64_t(numResultsPerMapQueryThreshold) * minparams.maps, max_number_candidates);

        std::vector<Value_t> allUniqueResults;
        std::vector<Value_t> tmp;
        allUniqueResults.reserve(maximumResultSize);
        tmp.reserve(maximumResultSize);

        //TIMERSTARTCPU(getcandrest);
        for(int map = 0; map < minparams.maps && allUniqueResults.size() < max_number_candidates; ++map) {
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
        if(sequence.size() < unsigned(minparams.k))
            return {};

        const int numResultsPerMapQueryThreshold = getResultsPerMapThreshold();

        //TIMERSTARTCPU(minhashfunc);
        auto hashValues = minhashfunc(sequence);
        //TIMERSTOPCPU(minhashfunc);

        const size_t maximumResultSize = std::min(std::uint64_t(numResultsPerMapQueryThreshold) * minparams.maps, max_number_candidates);

        std::vector<Value_t> allUniqueResults;
        std::vector<Value_t> tmp;
        allUniqueResults.reserve(maximumResultSize);
        tmp.reserve(maximumResultSize);

        //TIMERSTARTCPU(getcandrest);
        for(int map = 0; map < minparams.maps && allUniqueResults.size() < max_number_candidates; ++map) {
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
        if(sequence.size() < unsigned(minparams.k))
            return {};

        //TIMERSTARTCPU(minhashfunc);     
        auto hashValuesOriginal = minhashfunc(sequence);
 
        //TIMERSTOPCPU(minhashfunc);

        using Range_t = std::pair<const Value_t*, const Value_t*>;
        std::vector<Range_t> ranges;
        ranges.reserve(minparams.maps);

        int maximumResultSize = 0;


        //TIMERSTARTCPU(query);

#ifdef NVTXTIMELINE
        nvtx::push_range("map queries", 4);
#endif

        for(int map = 0; map < minparams.maps; ++map){
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

    std::int64_t Minhasher::getNumberOfCandidates(const std::string& sequence,
                                        int num_hits) const noexcept{

        const std::uint64_t max_number_candidates = std::numeric_limits<std::uint64_t>::max();

        std::vector<Result_t> result;

        if(num_hits == 1){
            result = getCandidates_any_map(sequence, max_number_candidates);
        }else if(num_hits == minparams.maps){
            result = getCandidates_all_maps(sequence, max_number_candidates);
        }else{
            result = getCandidates_some_maps(sequence, num_hits, max_number_candidates);
        }

        assert(result.size() <= std::numeric_limits<std::int64_t>::max());

        return std::int64_t(result.size());
    }

    std::int64_t Minhasher::getNumberOfCandidatesUpperBound(const std::string& sequence) const noexcept{
		static_assert(std::is_same<Result_t, Value_t>::value, "Value_t != Result_t");
		// we do not consider reads which are shorter than k
		if(sequence.size() < unsigned(minparams.k))
			return 0;

		//TIMERSTARTCPU(minhashfunc);
		auto hashValues = minhashfunc(sequence);
		//TIMERSTOPCPU(minhashfunc);

        std::size_t result = 0;

        for(int map = 0; map < minparams.maps; ++map) {
            kmer_type key = hashValues[map] & key_mask;

			//TIMERSTARTCPU(get_ranged);
            const auto entries_range = minhashTables[map]->get_ranged(key);
            const std::size_t n_entries = std::distance(entries_range.first, entries_range.second);
            result += n_entries;
			//TIMERSTOPCPU(get_ranged);
        }

        assert(result >= std::size_t(minparams.maps));
        result -= minparams.maps; //remove self from each map result

		return std::int64_t(result);

	}

//###################################################

	void Minhasher::resize(std::uint64_t nReads_){
		if(nReads_ == 0) throw std::runtime_error("Minhasher::init cannnot be called with argument 0");
		if(nReads_-1 > max_read_num)
			throw std::runtime_error("Minhasher::init: Minhasher is configured for only" + std::to_string(max_read_num) + " reads, not " + std::to_string(nReads_) + "!!!");

		nReads = nReads_;

		for (std::size_t i = 0; i < minhashTables.size(); ++i){
			auto& table = minhashTables[i];
			table->resize(nReads_);
		}
	}


    std::array<std::uint64_t, maximum_number_of_maps> 
    minhashfunc1(const char* sequence, int sequenceLength, int kmerLength, int numHashFuncs) noexcept{
        std::array<std::uint64_t, maximum_number_of_maps> kmerHashValues{0};
        std::array<std::uint64_t, maximum_number_of_maps> minhashSignature{0};

		std::uint64_t fhVal = 0;
        std::uint64_t rhVal = 0;
		bool isForward = false;
		// calc hash values of first canonical kmer
		NTMC64(sequence, kmerLength, numHashFuncs, minhashSignature.data(), fhVal, rhVal, isForward);

		//calc hash values of remaining canonical kmers
		for (int i = 0; i < sequenceLength - kmerLength; ++i) {
			NTMC64(fhVal, rhVal, sequence[i], sequence[i + kmerLength], kmerLength, numHashFuncs, 
                    kmerHashValues.data(), isForward);

			for (int j = 0; j < numHashFuncs; ++j) {
				if (minhashSignature[j] > kmerHashValues[j]){
					minhashSignature[j] = kmerHashValues[j];
				}
			}
		}

        return minhashSignature;
	}


    

    std::array<std::uint64_t, maximum_number_of_maps> 
    minhashfunc2(const char* sequence, int sequenceLength, int kmerLength, int numHashFuncs) noexcept{

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
    Minhasher::minhashfunc(const std::string& sequence) const noexcept{
        return minhashfunc(sequence.c_str(), sequence.length());
	}

    std::array<std::uint64_t, maximum_number_of_maps> 
    Minhasher::minhashfunc(const char* sequence, int sequenceLength) const noexcept{
        return minhashfunc2(sequence, sequenceLength, minparams.k, minparams.maps);
	}

    int Minhasher::getResultsPerMapThreshold() const{
        return minparams.numResultsPerMapQueryThreshold;
    };


    int calculateResultsPerMapThreshold(int coverage){
        int result = int(coverage * 2.5f);
        result = std::min(result, int(std::numeric_limits<BucketSize>::max()));
        result = std::max(10, result);
        return result;
    }


}
