#include "../inc/minhasher.hpp"
#include "../inc/binarysequencehelpers.hpp"
#include "../inc/ntHash/nthash.hpp"
#include "../inc/ganja/hpc_helpers.cuh"

#include <stdexcept>
#include <chrono>

#ifdef __NVCC__
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#endif

namespace care{

MinhasherBuffers::MinhasherBuffers(int id){
	deviceId = id;
#ifdef __NVCC__
	cudaSetDevice(deviceId); CUERR;
	cudaStreamCreate(&stream); CUERR;
#endif
}

void MinhasherBuffers::grow(size_t newcapacity){
#ifdef __NVCC__
	cudaSetDevice(deviceId);
	if(newcapacity > capacity){
		cudaFree(d_allMinhashResults); CUERR;
		cudaMalloc(&d_allMinhashResults, sizeof(std::uint64_t) * newcapacity); CUERR;
	}
#endif
	size = 0;
	capacity = newcapacity;
}

void cuda_cleanup_MinhasherBuffers(MinhasherBuffers& buffer){
#ifdef __NVCC__
	cudaSetDevice(buffer.deviceId);
	cudaFree(buffer.d_allMinhashResults); CUERR;
	cudaStreamDestroy(buffer.stream); CUERR;
#endif
}

Minhasher::Minhasher() : Minhasher(MinhashOptions{2,16})
{
}


Minhasher::Minhasher(const MinhashOptions& parameters)
		: minparams(parameters), nReads(0), minhashtime(0), maptime(0)
{
}

void Minhasher::init(std::uint64_t nReads_){

	nReads = nReads_;

	minhashTables.resize(minparams.maps);

	for (int i = 0; i < minparams.maps; ++i) {
		minhashTables[i].reset();
		minhashTables[i].reset(new KVMapFixed<key_t>(nReads));
	}
}


void Minhasher::clear(){
	for (int i = 0; i < minparams.maps; ++i) {
		minhashTables[i]->clear();
	}
}


int Minhasher::insertSequence(const std::string& sequence, const std::uint64_t readnum)
{
	if(readnum >= nReads)
		throw std::runtime_error("Minhasher, too many reads, read number too large");

	// we do not consider reads which are shorter than k
	if(sequence.size() < unsigned(minparams.k))
		return false;

	std::uint64_t bandHashValues[minparams.maps];
	std::fill(bandHashValues, bandHashValues + minparams.maps, 0);

	const unsigned int hvPerMap = 1;
	const unsigned int numberOfHashvalues = minparams.maps * hvPerMap;

	std::uint32_t isForwardStrand[numberOfHashvalues];
	std::fill(isForwardStrand, isForwardStrand + numberOfHashvalues, 0);

	//get hash values
	make_minhash_band_hashes(sequence, bandHashValues, isForwardStrand);

	// insert
	for (int map = 0; map < minparams.maps; ++map) {
		std::uint64_t key = bandHashValues[map] & key_mask;
		// last bit of value is 1 if the hash value comes from the forward strand

		//std::uint64_t value = ((readnum << 1) | isForwardStrand[map]);

		const std::uint64_t value = readnum;
		if (!minhashTables[map]->add(key, value)) {
			std::cout << "error adding key to map " << map
				<< ". key = " << key
				<< " , readnum = " << readnum << std::endl;
			throw std::runtime_error(("error adding key to map. key " + key));
		}
	}

	return 1;
}

#if 0
std::vector<std::pair<std::uint64_t, int>> Minhasher::getCandidatesWithFlag(const std::string& sequence) const{

	// we do not consider reads which are shorter than k
	if(sequence.size() < unsigned(minparams.k))
		return {};

	std::uint64_t bandHashValues[minparams.maps];
	std::fill(bandHashValues, bandHashValues + minparams.maps, 0);

	const unsigned int numberOfHashvalues = minparams.maps;
	std::uint32_t isForwardStrand[numberOfHashvalues];
	std::fill(isForwardStrand, isForwardStrand + numberOfHashvalues, 0);

	make_minhash_band_hashes(sequence, bandHashValues, isForwardStrand);

	std::vector<std::pair<std::uint64_t, int>> result;

	std::map<std::uint64_t, int> fwdrevmapping;

	for(int map = 0; map < minparams.maps; ++map) {
		std::uint64_t key = bandHashValues[map] & hv_bitmask;

		std::vector<uint64_t> entries = minhashTables[map]->get(key);
		for(const auto x : entries){
			int increment = (x & 1) ? 1 : -1;
			std::uint64_t readnum = (x >> 1);
			fwdrevmapping[readnum] += increment;
		}
	}

	result.insert(result.cend(), fwdrevmapping.cbegin(), fwdrevmapping.cend());

	return result;
}
#endif

std::vector<std::uint64_t> Minhasher::getCandidates(MinhasherBuffers& buffers, const std::string& sequence) const{

	// we do not consider reads which are shorter than k
	if(sequence.size() < unsigned(minparams.k))
		return {};

	std::uint64_t bandHashValues[minparams.maps];
	std::fill(bandHashValues, bandHashValues + minparams.maps, 0);

	const unsigned int numberOfHashvalues = minparams.maps;
	std::uint32_t isForwardStrand[numberOfHashvalues];
	std::fill(isForwardStrand, isForwardStrand + numberOfHashvalues, 0);

	make_minhash_band_hashes(sequence, bandHashValues, isForwardStrand);

	std::vector<std::uint64_t> allMinhashResults;

	for(int map = 0; map < minparams.maps; ++map) {
		std::uint64_t key = bandHashValues[map] & key_mask;

		std::vector<uint64_t> entries2 = minhashTables[map]->get(key);

		allMinhashResults.insert(allMinhashResults.end(), entries2.begin(), entries2.end());
	}

    if(allMinhashResults.size() == 0)
        return allMinhashResults;

	std::uint64_t n_initial_candidates = allMinhashResults.size();


	std::uint64_t n_unique_elements = 0;

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
	/*auto uniqueEnd = std::unique(allMinhashResults.begin(), allMinhashResults.end());
	n_unique_elements = std::distance(allMinhashResults.begin(), uniqueEnd);
	allMinhashResults.resize(n_unique_elements);*/

    /*
        make allMinhashResults unique and identical elements
    */
    n_unique_elements = 1;
    std::vector<std::uint8_t> counts(allMinhashResults.size(), std::uint8_t(0));
    counts[0]++;

    std::uint64_t prev = allMinhashResults[0];

    for (size_t k = 1; k < allMinhashResults.size(); k++) {
        std::uint64_t cur = allMinhashResults[k];
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
    std::uint64_t valid_elements = 0;
    for(std::uint64_t k = 0; k < n_unique_elements; k++){
        if(counts[k] >= threshold){
            allMinhashResults[valid_elements] = allMinhashResults[k];
            valid_elements++;
        }
    }

    allMinhashResults.resize(valid_elements);

#endif

	//assert(n_initial_candidates - n_unique_elements >= minparams.maps - 1); //make sure we deduplicated at least the id of the query

	/*if(d != n_unique_elements){
		std::cout << "#unique elements wrong. normal " << d << ", thrust " << n_unique_elements << std::endl;
	}
	for(int i = 0; i < d; i++){
		if(tmp[i] != allMinhashResults[i]){
			std::cout << "unique element wrong. normal " << allMinhashResults[i] << ", thrust " << tmp[i] << std::endl;
		}
	}*/

	return allMinhashResults;
}


int Minhasher::make_minhash_band_hashes(const std::string& sequence, std::uint64_t* bandHashValues, std::uint32_t* isForwardStrand) const
{
	int val = minhashfunc(sequence, bandHashValues, isForwardStrand);

	return val;
}

int Minhasher::minhashfunc(const std::string& sequence, std::uint64_t* minhashSignature, std::uint32_t* isForwardStrand) const{
	std::uint64_t fhVal = 0; std::uint64_t rhVal = 0;

	// bitmask for kmer, k_max = 32
	const int kmerbits = (2*unsigned(minparams.k) <= bits_key ? 2*minparams.k : bits_key);

	const std::uint64_t kmerbitmask = (kmerbits < 64 ? (1ULL << kmerbits) - 1 : 1ULL - 2);

	const int numberOfHashvalues = minparams.maps;
	std::uint64_t kmerHashValues[numberOfHashvalues];

	std::fill(kmerHashValues, kmerHashValues + numberOfHashvalues, 0);

	std::uint32_t isForward = 0;
	// calc hash values of first canonical kmer
	NTMC64(sequence.c_str(), minparams.k, numberOfHashvalues, minhashSignature, fhVal, rhVal, isForward);

	for (int j = 0; j < numberOfHashvalues; ++j) {
		minhashSignature[j] &= kmerbitmask;
		isForwardStrand[j] = isForward;
	}

	//calc hash values of remaining canonical kmers
	for (size_t i = 0; i < sequence.size() - minparams.k; ++i) {
		NTMC64(fhVal, rhVal, sequence[i], sequence[i + minparams.k], minparams.k, numberOfHashvalues, kmerHashValues, isForward);

		for (int j = 0; j < numberOfHashvalues; ++j) {
			std::uint64_t tmp = kmerHashValues[j] & kmerbitmask;
			if (minhashSignature[j] > tmp){
				minhashSignature[j] = tmp;
				isForwardStrand[j] = isForward;
			}
		}
	}

	return 0;
}

void Minhasher::bandhashfunc(const std::uint64_t* minhashSignature, std::uint64_t* bandHashValues) const{
	// xor the minhash hash values that belong to the same band
	for (int map = 0; map < minparams.maps; map++) {
		bandHashValues[map] ^= minhashSignature[map];
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
