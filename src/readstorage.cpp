#include "../inc/readstorage.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <unordered_set>
#include <vector>

#define SAFE_ACCESS


	ReadStorage::ReadStorage() : ReadStorage(1){}

	ReadStorage::ReadStorage(int threads) : nThreads(threads){
		headers.resize(nThreads);
		qualityscores.resize(nThreads);
		reverseComplqualityscores.resize(nThreads);
		sequences.resize(nThreads);
		reverseComplSequences.resize(nThreads);
		bytecounts.resize(nThreads);
		//dedupqscores.resize(nThreads);
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
		reverseComplSequences = other.reverseComplSequences;
		bytecounts = other.bytecounts;
		return *this;
	}

	ReadStorage& ReadStorage::operator=(const ReadStorage&& other){
		headers = std::move(other.headers);
		qualityscores = std::move(other.qualityscores);
		reverseComplqualityscores = std::move(other.reverseComplqualityscores);
		sequences = std::move(other.sequences);
		reverseComplSequences = std::move(other.reverseComplSequences);
		nThreads = other.nThreads;
		bytecounts = other.bytecounts;
		return *this;
	}

	void ReadStorage::clear(){
		for(auto& a : headers)
		    a.clear();
		for(auto& a : qualityscores)
		    a.clear();
		for(auto& a : sequences)
		    a.clear();
		for(auto& a : reverseComplSequences)
		    a.clear();
		for(auto& a : reverseComplqualityscores)
		    a.clear();
		bytecounts.clear();
	}

	void ReadStorage::insertRead(std::uint32_t readNumber, const Read& read){
		std::uint32_t id = readNumber % nThreads;

		size_t indexInVector = readNumber / nThreads;

		headers[id].resize(indexInVector+1);
		qualityscores[id].resize(indexInVector+1);
		reverseComplqualityscores[id].resize(indexInVector+1);
		sequences[id].resize(indexInVector+1);
		reverseComplSequences[id].resize(indexInVector+1);

		//if sequence does not contain 'N' , it can be stored in a compressed format using two bits per base
		bool hasN = std::any_of(read.sequence.begin(), read.sequence.end(), [](char c){return c == 'N';});

		Sequence seq{read.sequence, !hasN}; 
		std::string q(read.quality);

		bytecounts[id] += 2*seq.getNumBytes(); //sequence + reverse complement
		bytecounts[id] += 2*read.quality.size(); //quality scores + reverse complement qscores 

		headers[id][indexInVector] = std::move(read.header);
		qualityscores[id][indexInVector] = std::move(read.quality);

		std::reverse(q.begin(),q.end());
		reverseComplqualityscores[id][indexInVector] = std::move(q);
		
		
		reverseComplSequences[id][indexInVector] = (seq.reverseComplement());
		sequences[id][indexInVector] = (seq);

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
		std::uint32_t id = readNumber % nThreads;

		size_t indexInVector = readNumber / nThreads;

#ifdef SAFE_ACCESS
		return &(sequences[id].at(indexInVector));
#else
		return &(sequences[id][indexInVector]);		
#endif
	}

	const Sequence* ReadStorage::fetchReverseComplementSequence_ptr(std::uint32_t readNumber) const{
		std::uint32_t id = readNumber % nThreads;

		size_t indexInVector = readNumber / nThreads;

#ifdef SAFE_ACCESS
		return &(reverseComplSequences[id].at(indexInVector));
#else
		return &(reverseComplSequences[id][indexInVector]);		
#endif
	}

