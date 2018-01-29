#include "../inc/readstorage.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include <cassert>
#include <map>

#define SAFE_ACCESS


	ReadStorage::ReadStorage() : ReadStorage(1){}

	ReadStorage::ReadStorage(int threads) : nThreads(threads), isReadOnly(false){
		headers.resize(nThreads);
		qualityscores.resize(nThreads);
		reverseComplqualityscores.resize(nThreads);
		sequences.resize(nThreads);
	}

	ReadStorage::ReadStorage(const ReadStorage& other){
		*this = other;
	}

	ReadStorage& ReadStorage::operator=(const ReadStorage& other){
		headers = other.headers;
		qualityscores = other.qualityscores;
		reverseComplqualityscores = other.reverseComplqualityscores;
		nThreads = other.nThreads;
		sequences = other.sequences;

		sequencepointers = other.sequencepointers;
		reverseComplSequencepointers = other.reverseComplSequencepointers;
		all_unique_sequences = other.all_unique_sequences;

		return *this;
	}

	ReadStorage& ReadStorage::operator=(const ReadStorage&& other){
		headers = std::move(other.headers);
		qualityscores = std::move(other.qualityscores);
		reverseComplqualityscores = std::move(other.reverseComplqualityscores);
		sequences = std::move(other.sequences);
		nThreads = other.nThreads;

		sequencepointers = std::move(other.sequencepointers);
		reverseComplSequencepointers = std::move(other.reverseComplSequencepointers);
		all_unique_sequences = std::move(other.all_unique_sequences);

		return *this;
	}

	void ReadStorage::clear(){
		for(auto& a : headers)
		    a.clear();
		for(auto& a : qualityscores)
		    a.clear();
		for(auto& a : sequences)
		    a.clear();
		for(auto& a : reverseComplqualityscores)
		    a.clear();

		sequencepointers.clear();
		reverseComplSequencepointers.clear();
		all_unique_sequences.clear();

		isReadOnly = false;
	}

	void ReadStorage::insertRead(std::uint32_t readNumber, const Read& read){
		if(isReadOnly) throw std::runtime_error("cannot insert read into ReadStorage after calling noMoreInserts()");

		std::uint32_t id = readNumber % nThreads;

		size_t indexInVector = readNumber / nThreads;

		headers[id].resize(indexInVector+1);
		qualityscores[id].resize(indexInVector+1);
		reverseComplqualityscores[id].resize(indexInVector+1);
		sequences[id].resize(indexInVector+1);

		Sequence seq(read.sequence);  
		std::string q(read.quality);

		headers[id][indexInVector] = std::move(read.header);
		qualityscores[id][indexInVector] = std::move(read.quality);

		std::reverse(q.begin(),q.end());
		reverseComplqualityscores[id][indexInVector] = std::move(q);

		sequences[id][indexInVector] = (seq);
	}

	void ReadStorage::noMoreInserts(){
		isReadOnly = true;

		int nSequences = 0;
		for(const auto& v : sequences)
			nSequences += v.size();

		if(nSequences == 0) return;

		std::vector<Sequence> sequencesflat;
		sequencesflat.reserve(2*nSequences);

		for(int readnum = 0; readnum < nSequences; readnum++){
			std::uint32_t vecnum = readnum % nThreads;
			int indexInVector = readnum / nThreads;

			sequencesflat.push_back(std::move(sequences[vecnum].at(indexInVector)));
		}

		for(auto& vec : sequences)
			vec.clear();

		std::vector<Sequence> tmp(sequencesflat);

		std::sort(sequencesflat.begin(), sequencesflat.end());
		sequencesflat.erase(std::unique(sequencesflat.begin(), sequencesflat.end()), sequencesflat.end());

		std::map<const Sequence, int> seqToSortedIndex;

		for(const auto& s : sequencesflat){
			seqToSortedIndex[s] = &s - sequencesflat.data();
		}

		assert(sequencesflat.size() == seqToSortedIndex.size());

		std::cout << "ReadStorage: found " << (nSequences - sequencesflat.size()) << " duplicates\n";

		sequencepointers.resize(nSequences);
		for(int i = 0; i < nSequences; i++)
			sequencepointers[i] = &sequencesflat[seqToSortedIndex[tmp[i]]];

		reverseComplSequencepointers.resize(nSequences);
		for(int i = 0; i < nSequences; i++){
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

#if 1	
		//check
		for(int i = 0; i < nSequences; i++){
			assert(*sequencepointers[i] == tmp[i] && "readstorage wrong sequence after dedup");
		}
		for(int i = 0; i < nSequences; i++){
			assert(*reverseComplSequencepointers[i] == sequencepointers[i]->reverseComplement() && "readstorage wrong reverse complement after dedup");
		}
#endif

		all_unique_sequences = std::move(sequencesflat);
	}

	Read ReadStorage::fetchRead(std::uint32_t readNumber) const{
		Read returnvalue;

		returnvalue.header = *fetchHeader_ptr(readNumber);
		returnvalue.quality = *fetchQuality_ptr(readNumber);
		returnvalue.sequence = fetchSequence_ptr(readNumber)->toString();

		return returnvalue;
	}

	const std::string* ReadStorage::fetchHeader_ptr(std::uint32_t readNumber) const{
		std::uint32_t id = readNumber % nThreads;

		size_t indexInVector = readNumber / nThreads;
#ifdef SAFE_ACCESS
		return &(headers[id].at(indexInVector));
#else
		return &(headers[id][indexInVector]);		
#endif
	}

    	const std::string* ReadStorage::fetchQuality_ptr(std::uint32_t readNumber) const{
		std::uint32_t id = readNumber % nThreads;

		size_t indexInVector = readNumber / nThreads;

#ifdef SAFE_ACCESS
		return &(qualityscores[id].at(indexInVector));
#else
		return &(qualityscores[id][indexInVector]);		
#endif
	}

    	const std::string* ReadStorage::fetchReverseComplementQuality_ptr(std::uint32_t readNumber) const{
		std::uint32_t id = readNumber % nThreads;

		size_t indexInVector = readNumber / nThreads;

#ifdef SAFE_ACCESS
		return &(reverseComplqualityscores[id].at(indexInVector));
#else
		return &(reverseComplqualityscores[id][indexInVector]);		
#endif
	}

    	const Sequence* ReadStorage::fetchSequence_ptr(std::uint32_t readNumber) const{
		if(isReadOnly){
			return sequencepointers.at(readNumber);
		}else{
			std::uint32_t id = readNumber % nThreads;

			size_t indexInVector = readNumber / nThreads;

	#ifdef SAFE_ACCESS
			return &(sequences[id].at(indexInVector));
	#else
			return &(sequences[id][indexInVector]);		
	#endif
		}
	}

	const Sequence* ReadStorage::fetchReverseComplementSequence_ptr(std::uint32_t readNumber) const{
		if(isReadOnly){
			return reverseComplSequencepointers.at(readNumber);
		}else{
			throw std::runtime_error("fetchReverseComplementSequence_ptr");
		}
	}

	double ReadStorage::getMemUsageMB(){
		size_t bytes = 0;

		for(const auto& vec : sequences){
			for(const auto& s : vec){
				bytes += s.getNumBytes();
			}
		}

		/*for(const auto& vec : reverseComplSequences){
			for(const auto& s : vec){
				bytes += s.getNumBytes();
			}
		}*/

		for(const auto& s : all_unique_sequences){
			bytes += s.getNumBytes();
		}

		for(const auto& vec : headers){
			for(const auto& s : vec){
				bytes += s.size();
			}
		}

		for(const auto& vec : qualityscores){
			for(const auto& s : vec){
				bytes += s.size();
			}
		}

		for(const auto& vec : reverseComplqualityscores){
			for(const auto& s : vec){
				bytes += s.size();
			}
		}

		return bytes / 1024. / 1024.;
	}

