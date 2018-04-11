#include "../inc/minhasher.hpp"
#include "../inc/binarysequencehelpers.hpp"
#include "../inc/ntHash/nthash.hpp"
#include "../inc/hpc_helpers.cuh"

#include <stdexcept>
#include <chrono>

#ifdef __NVCC__
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#endif

namespace care{

Minhasher::Minhasher() : Minhasher(MinhashOptions{2,16})
{
}


Minhasher::Minhasher(const MinhashOptions& parameters)
		: minparams(parameters), nReads(0), minhashtime(0), maptime(0)
{
    if(maximum_number_of_maps < minparams.maps)
        throw std::runtime_error("Minhasher: Maximum number of maps is "
                                + std::to_string(maximum_number_of_maps) + "!");

    if(maximum_kmer_length < minparams.k){
        throw std::runtime_error("Minhasher is configured for maximum kmer length of "
                                + std::to_string(maximum_kmer_length) + "!");
    }
}

void Minhasher::init(ReadId_t nReads_){
    if(nReads_-1 > max_read_num)
		throw std::runtime_error("Minhasher::init: Minhasher is configured for only" + std::to_string(max_read_num) + " reads!!!");

	nReads = nReads_;

	minhashTables.resize(minparams.maps);

	for (int i = 0; i < minparams.maps; ++i) {
		minhashTables[i].reset();
		minhashTables[i].reset(new KVMapFixed<Key_t, Value_t, Index_t>(nReads));
	}
}


void Minhasher::clear(){
	for (int i = 0; i < minparams.maps; ++i) {
		minhashTables[i]->clear();
	}
}


void Minhasher::insertSequence(const std::string& sequence, const ReadId_t readnum)
{
	if(readnum > max_read_num)
		throw std::runtime_error("Minhasher::insertSequence: Index_t cannot represent readnum. "
                                + std::to_string(readnum) + " > " + std::to_string(max_read_num));
    if(readnum >= nReads)
		throw std::runtime_error("Minhasher::insertSequence: read number too large. " + std::to_string(readnum) + " > " + std::to_string(nReads));

	// we do not consider reads which are shorter than k
	if(sequence.size() < unsigned(minparams.k))
		return;

	std::uint64_t hashValues[maximum_number_of_maps]{0};

	bool isForwardStrand[maximum_number_of_maps]{0};

	//get hash values
	minhashfunc(sequence, hashValues, isForwardStrand);

	// insert
	for (int map = 0; map < minparams.maps; ++map) {
		Key_t key = hashValues[map] & key_mask;
		Value_t value(readnum);

		if (!minhashTables[map]->add(key, value, readnum)) {
			std::cout << "error adding key to map " << map
				<< ". key = " << key
				<< " , readnum = " << readnum << std::endl;
			throw std::runtime_error(("error adding key to map. key " + key));
		}
	}
}

std::vector<Minhasher::Result_t> Minhasher::getCandidates(const std::string& sequence,
                                                          std::uint64_t max_number_candidates) const{
    static_assert(std::is_same<Result_t, Value_t>::value, "Value_t != Result_t");
	// we do not consider reads which are shorter than k
	if(sequence.size() < unsigned(minparams.k))
		return {};

	std::uint64_t hashValues[maximum_number_of_maps]{0};

	bool isForwardStrand[maximum_number_of_maps]{0};

	minhashfunc(sequence, hashValues, isForwardStrand);


    std::vector<Value_t> allUniqueResults;
    Index_t n_unique_elements = 0;
	for(int map = 0; map < minparams.maps && allUniqueResults.size() <= max_number_candidates; ++map) {
		Key_t key = hashValues[map] & key_mask;

		std::vector<Value_t> entries = minhashTables[map]->get(key);

		/*allMinhashResults.insert(allMinhashResults.end(), entries.begin(), entries.end());
        std::sort(allMinhashResults.begin(), allMinhashResults.end());
    	auto uniqueEnd = std::unique(allMinhashResults.begin(), allMinhashResults.end());
    	n_unique_elements = std::distance(allMinhashResults.begin(), uniqueEnd);
    	allMinhashResults.resize(n_unique_elements);*/

        std::sort(entries.begin(), entries.end());
        auto uniqueEnd = std::unique(entries.begin(), entries.end());
        entries.resize(std::distance(entries.begin(), uniqueEnd));

        std::vector<Value_t> tmp(allUniqueResults);
        allUniqueResults.resize(tmp.size() + entries.size());
        std::merge(entries.begin(), entries.end(), tmp.begin(), tmp.end(), allUniqueResults.begin());
        auto uniqueEnd2 = std::unique(allUniqueResults.begin(), allUniqueResults.end());
        allUniqueResults.resize(std::distance(allUniqueResults.begin(), uniqueEnd2));
	}



	/*std::sort(allMinhashResults.begin(), allMinhashResults.end());
	auto uniqueEnd = std::unique(allMinhashResults.begin(), allMinhashResults.end());
	n_unique_elements = std::distance(allMinhashResults.begin(), uniqueEnd);
	allMinhashResults.resize(n_unique_elements);*/

	return allUniqueResults;
}

#if 0
std::vector<Minhasher::Result_t> Minhasher::getCandidates(MinhasherBuffers& buffers, const std::string& sequence) const{
    static_assert(std::is_same<Result_t, Value_t>::value, "Value_t != Result_t");
	// we do not consider reads which are shorter than k
	if(sequence.size() < unsigned(minparams.k))
		return {};

	std::uint64_t hashValues[maximum_number_of_maps]{0};

	bool isForwardStrand[maximum_number_of_maps]{0};

	minhashfunc(sequence, hashValues, isForwardStrand);

	std::vector<Value_t> allMinhashResults;

	for(int map = 0; map < minparams.maps; ++map) {
		Key_t key = hashValues[map] & key_mask;

		std::vector<Value_t> entries = minhashTables[map]->get(key);

		allMinhashResults.insert(allMinhashResults.end(), entries.begin(), entries.end());
	}

	Index_t n_unique_elements = 0;

#if 0
/*
    THIS DOES NOT COUNT HOW MANY MAPS WHERE HIT PER READ
*/
	cudaSetDevice(buffers.deviceId);

	buffers.grow(allMinhashResults.size());
	thrust::device_ptr<std::uint64_t> dev_ptr = thrust::device_pointer_cast(buffers.d_allMinhashResults);
	//thrust::copy(thrust::cuda::par.on(buffers.stream), allMinhashResults.begin(), allMinhashResults.end(), dev_ptr);
	cudaMemcpyAsync(dev_ptr.get(), allMinhashResults.data(), sizeof(std::uint64_t) * n_initial_candidates, cudaMemcpyHostToDevice, buffers.stream); CUERR;
	thrust::sort(thrust::cuda::par.on(buffers.stream), dev_ptr, dev_ptr + n_initial_candidates);
	auto d_unique_end = thrust::unique(thrust::cuda::par.on(buffers.stream), dev_ptr, dev_ptr + n_initial_candidates);

	cudaStreamSynchronize(buffers.stream); CUERR;

	n_unique_elements = thrust::distance(dev_ptr, d_unique_end);
	allMinhashResults.resize(n_unique_elements);
	//thrust::copy(thrust::cuda::par.on(buffers.stream), dev_ptr, d_unique_end, allMinhashResults.begin());
	cudaMemcpyAsync(allMinhashResults.data(), dev_ptr.get(), sizeof(std::uint64_t) * n_unique_elements, cudaMemcpyDeviceToHost, buffers.stream); CUERR;
	cudaStreamSynchronize(buffers.stream); CUERR;
#else

	std::sort(allMinhashResults.begin(), allMinhashResults.end());
	auto uniqueEnd = std::unique(allMinhashResults.begin(), allMinhashResults.end());
	n_unique_elements = std::distance(allMinhashResults.begin(), uniqueEnd);
	allMinhashResults.resize(n_unique_elements);

    /*
        make allMinhashResults unique and identical elements
    */
#if 0
    n_unique_elements = 1;
    std::vector<std::uint8_t> counts(allMinhashResults.size(), std::uint8_t(0));
    counts[0]++;

    Value_t prev = allMinhashResults[0];

    for (size_t k = 1; k < allMinhashResults.size(); k++) {
        Value_t cur = allMinhashResults[k];
        if (prev == cur) {
            counts[n_unique_elements-1]++;
        }else {
            allMinhashResults[n_unique_elements] = cur;
            counts[n_unique_elements]++;
            n_unique_elements++;
        }
        prev = cur;
    }

    /*
        only keep results with count geq threshold
    */
    double threshold = double(minparams.maps) * minparams.min_hits_per_candidate;
    Index_t valid_elements = 0;
    for(Index_t k = 0; k < n_unique_elements; k++){
        if(counts[k] >= threshold){
            allMinhashResults[valid_elements] = allMinhashResults[k];
            valid_elements++;
        }
    }
    allMinhashResults.resize(valid_elements);
#endif


#endif

	return allMinhashResults;
}
#endif

void Minhasher::minhashfunc(const std::string& sequence, std::uint64_t* minhashSignature, bool* isForwardStrand) const{
	std::uint64_t fhVal = 0; std::uint64_t rhVal = 0;

	// bitmask for kmer, k_max = 32
	const int kmerbits = (2*unsigned(minparams.k) <= bits_key ? 2*minparams.k : bits_key);

	const std::uint64_t kmerbitmask = (kmerbits < 64 ? (1ULL << kmerbits) - 1 : 1ULL - 2);

	std::uint64_t kmerHashValues[maximum_number_of_maps]{0};

	bool isForward = false;
	// calc hash values of first canonical kmer
	NTMC64(sequence.c_str(), minparams.k, minparams.maps, minhashSignature, fhVal, rhVal, isForward);

	for (int j = 0; j < minparams.maps; ++j) {
		minhashSignature[j] &= kmerbitmask;
		isForwardStrand[j] = isForward;
	}

	//calc hash values of remaining canonical kmers
	for (size_t i = 0; i < sequence.size() - minparams.k; ++i) {
		NTMC64(fhVal, rhVal, sequence[i], sequence[i + minparams.k], minparams.k, minparams.maps, kmerHashValues, isForward);

		for (int j = 0; j < minparams.maps; ++j) {
			std::uint64_t tmp = kmerHashValues[j] & kmerbitmask;
			if (minhashSignature[j] > tmp){
				minhashSignature[j] = tmp;
				isForwardStrand[j] = isForward;
			}
		}
	}
}

void Minhasher::saveTablesToFile(std::string filename) const{
	for (int i = 0; i < minparams.maps; ++i) {

	}
}

bool Minhasher::loadTablesFromFile(std::string filename){
	bool success = true;
	for (int i = 0; i < minparams.maps && success; ++i) {
		//success &= ;
	}
	return false;
}

void Minhasher::transform(){
	for (int i = 0; i < minparams.maps; ++i) {
		minhashTables[i]->freeze();
	}
}

}
