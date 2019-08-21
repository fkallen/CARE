#include <minhasher.hpp>
#include <options.hpp>
#include <hpc_helpers.cuh>
#include <util.hpp>
#include <config.hpp>

#include <ntHash/nthash.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gpu/nvtxtimelinemarkers.hpp>


namespace care{

    Minhasher::Minhasher() : Minhasher(MinhashOptions{2,16}){}

    Minhasher::Minhasher(const MinhashOptions& parameters)
		: Minhasher(parameters, {})
	{
	}

	Minhasher::Minhasher(const MinhashOptions& parameters, const std::vector<int>& deviceIds_)
		: minparams(parameters), nReads(0), deviceIds(deviceIds_)
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
        deviceIds = std::move(rhs.deviceIds);
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

        assert(minparams == minparams_loaded);
        assert(nReads == nReads_loaded);

        minhashTables.resize(minparams.maps);

		for (int i = 0; i < minparams.maps; ++i) {
			minhashTables[i].reset();
			minhashTables[i].reset(new Map_t(nReads, deviceIds));
		}

        for(auto& tableptr : minhashTables)
            tableptr->readFromStream(instream);
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
			//minhashTables[i].reset(new Map_t(nReads, deviceIds));
		}
	}

    void Minhasher::initMap(int map){
        assert(map < minparams.maps);
        minhashTables[map].reset();
        minhashTables[map].reset(new Map_t(nReads, deviceIds));
    }

	void Minhasher::clear(){
		minhashTables.clear();
		nReads = 0;
	}

	void Minhasher::destroy(){
		clear();
		minhashTables.shrink_to_fit();
	}

    void Minhasher::insertTupleIntoMap(int map, const std::uint64_t* hashValues, read_number readnum){
        assert(map < minparams.maps);

        kmer_type key = hashValues[map] & key_mask;
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

		std::uint64_t hashValues[maximum_number_of_maps]{0};

		bool isForwardStrand[maximum_number_of_maps]{0};

		//get hash values
		minhashfunc(sequence, hashValues, isForwardStrand);

		// insert
		for (int map = 0; map < minparams.maps; ++map) {
            insertTupleIntoMap(map, &hashValues[0], readnum);
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

		std::uint64_t hashValues[maximum_number_of_maps]{0};

		bool isForwardStrand[maximum_number_of_maps]{0};

		//get hash values
		minhashfunc(sequence, hashValues, isForwardStrand);

		// insert
        for(auto mapId : mapIds){
            assert(mapId < minparams.maps);
            insertTupleIntoMap(mapId, &hashValues[0], readnum);
        }

	}

    std::pair<const Minhasher::Value_t*, const Minhasher::Value_t*>
    Minhasher::queryMap(int mapid, Minhasher::Map_t::Key_t key, size_t numResultsPerMapQueryThreshold) const noexcept{
        assert(mapid < minparams.maps);

        auto entries_range = minhashTables[mapid]->get_ranged(key);
        std::size_t n_entries = std::distance(entries_range.first, entries_range.second);

        if(n_entries > numResultsPerMapQueryThreshold){
            return std::make_pair(entries_range.first, entries_range.first); //return empty range
        }

        return entries_range;
    }

    std::vector<Minhasher::Result_t> Minhasher::getCandidates(const std::string& sequence,
                                        int num_hits,
                                        std::uint64_t max_number_candidates,
                                        size_t numResultsPerMapQueryThreshold) const noexcept{
        std::vector<Result_t> result;

        if(num_hits == 1){
            result = getCandidates_any_map(sequence, max_number_candidates, numResultsPerMapQueryThreshold);
        }else if(num_hits == minparams.maps){
            result = getCandidates_all_maps(sequence, max_number_candidates, numResultsPerMapQueryThreshold);
        }else{
            result = getCandidates_some_maps(sequence, num_hits, max_number_candidates, numResultsPerMapQueryThreshold);
        }

        return result;
    }

    std::vector<Minhasher::Result_t> Minhasher::getCandidates_any_map(const std::string& sequence,
                                        std::uint64_t max_number_candidates,
                                        size_t numResultsPerMapQueryThreshold) const noexcept{
        static_assert(std::is_same<Result_t, Value_t>::value, "Value_t != Result_t");
        // we do not consider reads which are shorter than k
        if(sequence.size() < unsigned(minparams.k))
            return {};

        std::uint64_t hashValues[maximum_number_of_maps]{0};

        bool isForwardStrand[maximum_number_of_maps]{0};
        //TIMERSTARTCPU(minhashfunc);
        minhashfunc(sequence, hashValues, isForwardStrand);
        //TIMERSTOPCPU(minhashfunc);

        const size_t maximumResultSize = std::min(numResultsPerMapQueryThreshold * minparams.maps, max_number_candidates);

        std::vector<Value_t> allUniqueResults;
        std::vector<Value_t> tmp;
        allUniqueResults.reserve(maximumResultSize);
        tmp.reserve(maximumResultSize);



#if 1
        allUniqueResults.resize(maximumResultSize);
        tmp.resize(maximumResultSize);

        size_t allUniqueResults_size = 0;
        size_t tmp_size = 0;

        //TIMERSTARTCPU(getcandrest);
        for(int map = 0; map < minparams.maps && allUniqueResults_size < max_number_candidates; ++map) {
            kmer_type key = hashValues[map] & key_mask;

            //nvtx::push_range("querymap", 6);
            auto entries_range = queryMap(map, key, numResultsPerMapQueryThreshold);
            //nvtx::pop_range();
            std::size_t n_entries = std::distance(entries_range.first, entries_range.second);
            //std::vector<Value_t> backup(allUniqueResults);

            if(n_entries == 0){
                continue;
            }

            //nvtx::push_range("union", 4);
            // auto union_end = set_union_n_or_empty(entries_range.first,
            //                                     entries_range.second,
            //                                     allUniqueResults.begin(),
            //                                     allUniqueResults.end(),
            //                                     max_number_candidates,
            //                                     tmp.begin());
            auto union_end = std::set_union(entries_range.first,
                                        entries_range.second,
                                        allUniqueResults.begin(),
                                        allUniqueResults.begin() + allUniqueResults_size,
                                        tmp.begin());
            //nvtx::pop_range();
            tmp_size = std::distance(tmp.begin(), union_end);

            std::swap(allUniqueResults, tmp);
            std::swap(allUniqueResults_size, tmp_size);
        }

        allUniqueResults.erase(allUniqueResults.begin() + allUniqueResults_size, allUniqueResults.end());

#else


        //TIMERSTARTCPU(getcandrest);
        for(int map = 0; map < minparams.maps && allUniqueResults.size() < max_number_candidates; ++map) {
            kmer_type key = hashValues[map] & key_mask;

            //nvtx::push_range("querymap", 6);
            auto entries_range = queryMap(map, key, numResultsPerMapQueryThreshold);
            //nvtx::pop_range();
            std::size_t n_entries = std::distance(entries_range.first, entries_range.second);
            //std::vector<Value_t> backup(allUniqueResults);

            if(n_entries == 0){
                continue;
            }

            tmp.resize(allUniqueResults.size() + n_entries);
            //nvtx::push_range("union", 4);
            // auto union_end = set_union_n_or_empty(entries_range.first,
            //                                     entries_range.second,
            //                                     allUniqueResults.begin(),
            //                                     allUniqueResults.end(),
            //                                     max_number_candidates,
            //                                     tmp.begin());
            auto union_end = std::set_union(entries_range.first,
                                                entries_range.second,
                                                allUniqueResults.begin(),
                                                allUniqueResults.end(),
                                                tmp.begin());
            //nvtx::pop_range();
            //if(tmp.begin() == union_end){
            //    //assert(n_entries != 0 || (allUniqueResults.size() == 0 && n_entries == 0));
            //    return {};
            //}else{
                //tmp.resize(std::distance(tmp.begin(), union_end));
                //nvtx::push_range("erase", 3);
                tmp.erase(union_end, tmp.end());
                //nvtx::pop_range();
                std::swap(tmp, allUniqueResults);
            //}
            //if(n_entries == 0){
            //    assert(backup == allUniqueResults);
            //}
        }


#endif

        return allUniqueResults;
    }



    std::vector<Minhasher::Result_t> Minhasher::getCandidates_some_maps2(const std::string& sequence,
                                        int num_hits,
                                        std::uint64_t max_number_candidates,
                                        size_t numResultsPerMapQueryThreshold) const noexcept{
        static_assert(std::is_same<Result_t, Value_t>::value, "Value_t != Result_t");
        // we do not consider reads which are shorter than k
        if(sequence.size() < unsigned(minparams.k))
            return {};

        if(num_hits > minparams.maps || num_hits < 1)
            return {};

        std::uint64_t hashValues[maximum_number_of_maps]{0};

        bool isForwardStrand[maximum_number_of_maps]{0};
        //TIMERSTARTCPU(minhashfunc);
        minhashfunc(sequence, hashValues, isForwardStrand);
        //TIMERSTOPCPU(minhashfunc);

        std::size_t total_num_ids = 0;
        std::vector<const Value_t*> iters;
        iters.reserve(minparams.maps*2);
        for(int map = 0; map < minparams.maps; ++map){
            kmer_type key = hashValues[map] & key_mask;

            auto entries_range = queryMap(map, key, numResultsPerMapQueryThreshold);

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
                                        std::uint64_t max_number_candidates,
                                        size_t numResultsPerMapQueryThreshold) const noexcept{
        static_assert(std::is_same<Result_t, Value_t>::value, "Value_t != Result_t");
        // we do not consider reads which are shorter than k
        if(sequence.size() < unsigned(minparams.k))
            return {};

        if(num_hits > minparams.maps || num_hits < 1)
            return {};

        std::uint64_t hashValues[maximum_number_of_maps]{0};

        bool isForwardStrand[maximum_number_of_maps]{0};
        //TIMERSTARTCPU(minhashfunc);
        minhashfunc(sequence, hashValues, isForwardStrand);
        //TIMERSTOPCPU(minhashfunc);

        const size_t maximumResultSize = std::min(numResultsPerMapQueryThreshold * minparams.maps, max_number_candidates);

        std::vector<Value_t> allUniqueResults;
        std::vector<Value_t> tmp;
        allUniqueResults.reserve(maximumResultSize);
        tmp.reserve(maximumResultSize);

        //TIMERSTARTCPU(getcandrest);
        for(int map = 0; map < minparams.maps && allUniqueResults.size() < max_number_candidates; ++map) {
            kmer_type key = hashValues[map] & key_mask;

            auto entries_range = queryMap(map, key, numResultsPerMapQueryThreshold);
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
                                        std::uint64_t max_number_candidates,
                                        size_t numResultsPerMapQueryThreshold) const noexcept{
        static_assert(std::is_same<Result_t, Value_t>::value, "Value_t != Result_t");
        // we do not consider reads which are shorter than k
        if(sequence.size() < unsigned(minparams.k))
            return {};

        std::uint64_t hashValues[maximum_number_of_maps]{0};

        bool isForwardStrand[maximum_number_of_maps]{0};
        //TIMERSTARTCPU(minhashfunc);
        minhashfunc(sequence, hashValues, isForwardStrand);
        //TIMERSTOPCPU(minhashfunc);

        const size_t maximumResultSize = std::min(numResultsPerMapQueryThreshold * minparams.maps, max_number_candidates);

        std::vector<Value_t> allUniqueResults;
        std::vector<Value_t> tmp;
        allUniqueResults.reserve(maximumResultSize);
        tmp.reserve(maximumResultSize);

        //TIMERSTARTCPU(getcandrest);
        for(int map = 0; map < minparams.maps && allUniqueResults.size() < max_number_candidates; ++map) {
            kmer_type key = hashValues[map] & key_mask;

            auto entries_range = queryMap(map, key, numResultsPerMapQueryThreshold);
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

// #############################

/*
    Query number of candidates
*/

    std::int64_t Minhasher::getNumberOfCandidates(const std::string& sequence,
                                        int num_hits) const noexcept{

        const std::uint64_t max_number_candidates = std::numeric_limits<std::uint64_t>::max();

        std::vector<Result_t> result;

        if(num_hits == 1){
            result = getCandidates_any_map(sequence, max_number_candidates, max_number_candidates);
        }else if(num_hits == minparams.maps){
            result = getCandidates_all_maps(sequence, max_number_candidates, max_number_candidates);
        }else{
            result = getCandidates_some_maps(sequence, num_hits, max_number_candidates, max_number_candidates);
        }

        assert(result.size() <= std::numeric_limits<std::int64_t>::max());

        return std::int64_t(result.size());
    }

    std::int64_t Minhasher::getNumberOfCandidatesUpperBound(const std::string& sequence) const noexcept{
		static_assert(std::is_same<Result_t, Value_t>::value, "Value_t != Result_t");
		// we do not consider reads which are shorter than k
		if(sequence.size() < unsigned(minparams.k))
			return 0;

		std::uint64_t hashValues[maximum_number_of_maps]{0};

		bool isForwardStrand[maximum_number_of_maps]{0};
		//TIMERSTARTCPU(minhashfunc);
		minhashfunc(sequence, hashValues, isForwardStrand);
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


	void Minhasher::minhashfunc(const std::string& sequence, std::uint64_t* minhashSignature, bool* isForwardStrand) const noexcept{
        std::uint64_t kmerHashValues[maximum_number_of_maps]{0};

		std::uint64_t fhVal = 0;
        std::uint64_t rhVal = 0;
		bool isForward = false;
		// calc hash values of first canonical kmer
		NTMC64(sequence.c_str(), minparams.k, minparams.maps, minhashSignature, fhVal, rhVal, isForward);

		for (int j = 0; j < minparams.maps; ++j) {
			isForwardStrand[j] = isForward;
		}

		//calc hash values of remaining canonical kmers
		for (size_t i = 0; i < sequence.size() - minparams.k; ++i) {
			NTMC64(fhVal, rhVal, sequence[i], sequence[i + minparams.k], minparams.k, minparams.maps, kmerHashValues, isForward);

			for (int j = 0; j < minparams.maps; ++j) {
				if (minhashSignature[j] > kmerHashValues[j]){
					minhashSignature[j] = kmerHashValues[j];
					isForwardStrand[j] = isForward;
				}
			}
		}
	}

    void Minhasher::minhashfunc_other(const std::string& sequence, std::uint64_t* minhashSignature, bool* isForwardStrand) const noexcept{
        assert(minparams.k <= maximum_kmer_length);

        auto make_kmer_encoded = [](const std::string& sequence, int k){

            constexpr int basesPerInt = sizeof(std::uint32_t) * 8 / 2;
            constexpr std::uint32_t BASE_A = 0x00000000;
            constexpr std::uint32_t BASE_C = 0x00000001;
            constexpr std::uint32_t BASE_G = 0x00000002;
            constexpr std::uint32_t BASE_T = 0x00000003;

            const int l = std::min(k, int(sequence.length()));

            std::uint32_t kmer_enc = 0;

            for(int i = 0; i < l; i++){
                const int pos = i % basesPerInt;
                switch(sequence[i]) {
                case 'A':
                        kmer_enc |= BASE_A << (2*((basesPerInt - 1)-pos));
                        break;
                case 'C':
                        kmer_enc |= BASE_C << (2*((basesPerInt - 1)-pos));
                        break;
                case 'G':
                        kmer_enc |= BASE_G << (2*((basesPerInt - 1)-pos));
                        break;
                case 'T':
                        kmer_enc |= BASE_T << (2*((basesPerInt - 1)-pos));
                        break;
                default:
                        kmer_enc |= BASE_A << (2*((basesPerInt - 1)-pos));
                        break;
                }
            }

            return kmer_enc;
        };

        auto make_next_kmer_enc = [](std::uint32_t kmer_enc, int k, const char nextbase){
            constexpr int basesPerInt = sizeof(std::uint32_t) * 8 / 2;
            constexpr std::uint32_t BASE_A = 0x00000000;
            constexpr std::uint32_t BASE_C = 0x00000001;
            constexpr std::uint32_t BASE_G = 0x00000002;
            constexpr std::uint32_t BASE_T = 0x00000003;

            kmer_enc <<= 2;

            const int pos = (k-1) % basesPerInt;
            switch(nextbase) {
            case 'A':
                    kmer_enc |= BASE_A << (2*((basesPerInt - 1)-pos));
                    break;
            case 'C':
                    kmer_enc |= BASE_C << (2*((basesPerInt - 1)-pos));
                    break;
            case 'G':
                    kmer_enc |= BASE_G << (2*((basesPerInt - 1)-pos));
                    break;
            case 'T':
                    kmer_enc |= BASE_T << (2*((basesPerInt - 1)-pos));
                    break;
            default:
                    kmer_enc |= BASE_A << (2*((basesPerInt - 1)-pos));
                    break;
            }
            return kmer_enc;
        };

        auto make_reverse_complement_int = [](std::uint32_t s){
            s = ((s >> 2)  & 0x33333333u) | ((s & 0x33333333u) << 2);
            s = ((s >> 4)  & 0x0F0F0F0Fu) | ((s & 0x0F0F0F0Fu) << 4);
            s = ((s >> 8)  & 0x00FF00FFu) | ((s & 0x00FF00FFu) << 8);
            s = ((s >> 16) & 0x0000FFFFu) | ((s & 0x0000FFFFu) << 16);
            return (std::uint32_t(-1) - s) >> (8 * sizeof(s) - (16 << 1));
        };

        auto thomas_mueller_hash = [](std::uint32_t x){
            x = ((x >> 16) ^ x) * 0x45d9f3b;
            x = ((x >> 16) ^ x) * 0x45d9f3b;
            x = ((x >> 16) ^ x);
            return x;
        };

        auto nvidia_hash = [](std::uint32_t x) {
            x = (x + 0x7ed55d16) + (x << 12);
            x = (x ^ 0xc761c23c) ^ (x >> 19);
            x = (x + 0x165667b1) + (x <<  5);
            x = (x + 0xd3a2646c) ^ (x <<  9);
            x = (x + 0xfd7046c5) + (x <<  3);
            x = (x ^ 0xb55a4f09) ^ (x >> 16);
            return x;
        };

        auto hashfunc = [&](std::uint32_t x, int i){
            if(i % 2 == 0){
                return thomas_mueller_hash(x);
            }else{
                return nvidia_hash(x);
            }
        };

        std::uint32_t kmer = make_kmer_encoded(sequence, minparams.k);
        std::uint32_t revcompl = make_reverse_complement_int(kmer);

        //std::cout << "kmer " << kmer << std::endl;
        //std::cout << "revcompl " << revcompl << std::endl;

        auto updatehashes = [&](bool first){
            std::uint32_t hashfwd = hashfunc(kmer,0);
            std::uint32_t hashrc = hashfunc(revcompl,0);

            std::uint32_t hash = hashfwd;
            bool isForward = true;
            if(hashrc < hashfwd){
                hash = hashrc;
                isForward = false;
            }

            if(first){
                minhashSignature[0] = hash;
                isForwardStrand[0] = isForward;
            }else{
                if (minhashSignature[0] > hash){
                    minhashSignature[0] = hash;
                    isForwardStrand[0] = isForward;
                }
            }

            for (int j = 1; j < minparams.maps; ++j) {
                hashfwd = hashfunc(hashfwd,j);
                hashrc = hashfunc(hashrc,j);
                hash = hashfwd;
                bool isForward = true;
                if(hashrc < hashfwd){
                    hash = hashrc;
                    isForward = false;
                }

                if(first){
                    minhashSignature[j] = hash;
                    isForwardStrand[j] = isForward;
                }else{
                    if (minhashSignature[j] > hash){
                        minhashSignature[j] = hash;
                        isForwardStrand[j] = isForward;
                    }
                }
            }
        };

        updatehashes(true);

        for (size_t i = 0; i < sequence.size() - minparams.k; ++i) {

			kmer = make_next_kmer_enc(kmer, minparams.k, sequence[i + minparams.k]);
            revcompl = make_reverse_complement_int(kmer);

            updatehashes((false));
		}

	}


}
