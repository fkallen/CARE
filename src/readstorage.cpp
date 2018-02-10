#include "../inc/readstorage.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include <cassert>
#include <map>

	ReadStorage::ReadStorage() : isReadOnly(false){}

	ReadStorage& ReadStorage::operator=(const ReadStorage&& other){
		headers = std::move(other.headers);
		qualityscores = std::move(other.qualityscores);
		reverseComplqualityscores = std::move(other.reverseComplqualityscores);
		sequences = std::move(other.sequences);

		sequencepointers = std::move(other.sequencepointers);
		reverseComplSequencepointers = std::move(other.reverseComplSequencepointers);
		all_unique_sequences = std::move(other.all_unique_sequences);

		return *this;
	}

	void ReadStorage::init(size_t nReads){
		clear();

		headers.resize(nReads);
		qualityscores.resize(nReads);
		reverseComplqualityscores.resize(nReads);
		sequences.resize(nReads);		
	}

	void ReadStorage::clear(){
		headers.clear();
		qualityscores.clear();
		reverseComplqualityscores.clear();
		sequences.clear();

		sequencepointers.clear();
		reverseComplSequencepointers.clear();
		all_unique_sequences.clear();

		isReadOnly = false;
	}

	void ReadStorage::insertRead(size_t readNumber, const Read& read){
		if(isReadOnly) throw std::runtime_error("cannot insert read into ReadStorage after calling noMoreInserts()");

		Sequence seq(read.sequence);  
		std::string q(read.quality);
		std::reverse(q.begin(),q.end());

		headers.at(readNumber) = std::move(read.header);
		qualityscores.at(readNumber) = std::move(read.quality);
		reverseComplqualityscores.at(readNumber) = std::move(q);
		sequences.at(readNumber) = std::move(seq);
	}

	void ReadStorage::noMoreInserts(){
		isReadOnly = true;

		size_t nSequences = sequences.size();

		if(nSequences == 0) return;

		std::vector<Sequence> sequencesflat;
		sequencesflat.reserve(2*nSequences);

		for(size_t readnum = 0; readnum < nSequences; readnum++){
			sequencesflat.push_back(std::move(sequences.at(readnum)));
		}

		sequences.clear();

		std::vector<Sequence> tmp(sequencesflat);

TIMERSTARTCPU(READ_STORAGE_SORT);
		std::sort(sequencesflat.begin(), sequencesflat.end());
		sequencesflat.erase(std::unique(sequencesflat.begin(), sequencesflat.end()), sequencesflat.end());
TIMERSTOPCPU(READ_STORAGE_SORT);
		std::map<const Sequence, int> seqToSortedIndex;

TIMERSTARTCPU(READ_STORAGE_MAKE_MAP);
		for(const auto& s : sequencesflat){
			seqToSortedIndex[s] = &s - sequencesflat.data();
		}
TIMERSTOPCPU(READ_STORAGE_MAKE_MAP);
		assert(sequencesflat.size() == seqToSortedIndex.size());

		size_t n_unique_forward_sequences = sequencesflat.size();
		std::cout << "ReadStorage: found " << (nSequences - n_unique_forward_sequences) << " duplicates\n";

TIMERSTARTCPU(READ_STORAGE_MAKE_FWD_POINTERS);
		sequencepointers.resize(nSequences);
		for(size_t i = 0; i < nSequences; i++)
			sequencepointers[i] = &sequencesflat[seqToSortedIndex[tmp[i]]];
TIMERSTOPCPU(READ_STORAGE_MAKE_FWD_POINTERS);

TIMERSTARTCPU(READ_STORAGE_MAKE_REVCOMPL_POINTERS);
		reverseComplSequencepointers.resize(nSequences);
		for(size_t i = 0; i < nSequences; i++){
			Sequence revcompl = sequencepointers[i]->reverseComplement();
			auto it = seqToSortedIndex.find(revcompl);
			if(it == seqToSortedIndex.end()){
				//sequence does not exist yet, insert into map and save pointer in list

				sequencesflat.push_back(std::move(revcompl));
				reverseComplSequencepointers[i] = &(sequencesflat.back());
				seqToSortedIndex[*reverseComplSequencepointers[i]] = reverseComplSequencepointers[i] - sequencesflat.data();
				
				
			}else{
				//sequence is a duplicate, read position from map
				reverseComplSequencepointers[i] = &(sequencesflat[it->second]);
			}
		}
TIMERSTOPCPU(READ_STORAGE_MAKE_REVCOMPL_POINTERS);

		std::cout << "ReadStorage: holding a total of " << seqToSortedIndex.size() << " unique sequences\n";

		seqToSortedIndex.clear();

#if 1	
TIMERSTARTCPU(READ_STORAGE_CHECK);
		//check
		for(size_t i = 0; i < nSequences; i++){
			assert(*sequencepointers[i] == tmp[i] && "readstorage wrong sequence after dedup");
		}
		for(size_t i = 0; i < nSequences; i++){
			assert(*reverseComplSequencepointers[i] == sequencepointers[i]->reverseComplement() && "readstorage wrong reverse complement after dedup");
		}
TIMERSTOPCPU(READ_STORAGE_CHECK);
#endif

		all_unique_sequences = std::move(sequencesflat);

/*

		//make combined quality weights for identical forward sequences

		std::map<const Sequence*, std::vector<size_t>> seqToIds;
		for(size_t i = 0; i < nSequences; i++){
			seqToIds[sequencepointers[i]].push_back(i);
		}

		quality_weights.reserve(n_unique_forward_sequences);

		for(auto pair : seqToIds){
			const int len = pair.first->getNbases();
			std::vector<float> weights(0.0f, len);
			for(size_t id : pair.second){
				const std::string* qptr = fetchQuality_ptr(id);
				for(int i = 0; i < len; i++){
					weights[i] += 1.0f; // TODO lookup weight
				}
			}
			quality_weights.push_back(std::move(weights));
			for(size_t id : pair.second){
				quality_weights_pointers[id] = &(quality_weights.back());
			}
		}
*/
	}

	Read ReadStorage::fetchRead(size_t readNumber) const{
		Read returnvalue;

		returnvalue.header = *fetchHeader_ptr(readNumber);
		returnvalue.quality = *fetchQuality_ptr(readNumber);
		returnvalue.sequence = fetchSequence_ptr(readNumber)->toString();

		return returnvalue;
	}

	const std::string* ReadStorage::fetchHeader_ptr(size_t readNumber) const{
		return &(headers.at(readNumber));
	}

    	const std::string* ReadStorage::fetchQuality_ptr(size_t readNumber) const{
		return &(qualityscores.at(readNumber));
	}

    	const std::string* ReadStorage::fetchReverseComplementQuality_ptr(size_t readNumber) const{
		return &(reverseComplqualityscores.at(readNumber));
	}

    	const Sequence* ReadStorage::fetchSequence_ptr(size_t readNumber) const{
		if(isReadOnly){
			return sequencepointers.at(readNumber);
		}else{
			return &(sequences.at(readNumber));
		}
	}

	const Sequence* ReadStorage::fetchReverseComplementSequence_ptr(size_t readNumber) const{
		if(isReadOnly){
			return reverseComplSequencepointers.at(readNumber);
		}else{
			throw std::runtime_error("fetchReverseComplementSequence_ptr");
		}
	}

	double ReadStorage::getMemUsageMB() const{
		size_t bytes = 0;

		for(const auto& s : sequences){
			bytes += s.getNumBytes();
		}

		for(const auto& s : all_unique_sequences){
			bytes += s.getNumBytes();
		}

		for(const auto& s : headers){
			bytes += s.size();
		}

		for(const auto& s : qualityscores){
			bytes += s.size();
		}

		for(const auto& s : reverseComplqualityscores){
			bytes += s.size();
		}

		return bytes / 1024. / 1024.;
	}

