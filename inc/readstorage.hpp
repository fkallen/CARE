#ifndef READ_STORAGE_HPP
#define READ_STORAGE_HPP

#include <algorithm>
#include <limits>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>
#include <omp.h>
#include <map>

namespace care{

/*
    Data structure to store sequences and their quality scores
*/
template<class sequence_t,
		 class readId_t>
struct ReadStorage{

	using Sequence_t = sequence_t;
	using ReadId_t = readId_t;

    static constexpr bool has_reverse_complement = true;

	bool useQualityScores = false;
    bool isReadOnly = false;

    std::vector<std::string> qualityscores;
    std::vector<std::string> reverseComplqualityscores;
    std::vector<Sequence_t> sequences;
    std::vector<Sequence_t*> sequencepointers;
    std::vector<Sequence_t*> reverseComplSequencepointers;

    ReadStorage() : ReadStorage(false){}
    ReadStorage(bool b) : useQualityScores(b){}

	void init(ReadId_t nReads){
		clear();

		sequences.resize(nReads);
        if(useQualityScores){
            qualityscores.resize(nReads);
    		reverseComplqualityscores.resize(nReads);
        }
	}

	void clear(){
		qualityscores.clear();
		reverseComplqualityscores.clear();
		sequences.clear();
		sequencepointers.clear();
		reverseComplSequencepointers.clear();

		isReadOnly = false;
	}

	void destroy(){
		clear();
		qualityscores.shrink_to_fit();
		reverseComplqualityscores.shrink_to_fit();
		sequencepointers.shrink_to_fit();
		reverseComplSequencepointers.shrink_to_fit();
	}

    void insertRead(ReadId_t readNumber, const std::string& sequence){
		if(useQualityScores){
			insertRead(readNumber, sequence, std::string(sequence.length(), 'A'));
		}else{
			Sequence_t seq(sequence);
			sequences[readNumber] = std::move(seq);
		}
	}

    void insertRead(ReadId_t readNumber, const std::string& sequence, const std::string& quality){
		if(isReadOnly) throw std::runtime_error("cannot insert read into ReadStorage after calling transform()");

		Sequence_t seq(sequence);
		std::string q(quality);
		std::reverse(q.begin(),q.end());

		sequences[readNumber] = std::move(seq);
		if(useQualityScores){
			qualityscores[readNumber] = std::move(quality);
			reverseComplqualityscores[readNumber] = std::move(q);
		}
	}

	const std::string* fetchQuality_ptr(ReadId_t readNumber) const{
		if(useQualityScores){
			return &(qualityscores[readNumber]);
		}else{
			return nullptr;
		}
	}

   	const std::string* fetchReverseComplementQuality_ptr(ReadId_t readNumber) const{
		if(useQualityScores){
			return &(reverseComplqualityscores[readNumber]);
		}else{
			return nullptr;
		}
	}

	//Must call transform() beforehand
	const Sequence_t* fetchSequence_ptr(ReadId_t readNumber) const{
		return sequencepointers[readNumber];
	}

	//Must call transform() beforehand
	const Sequence_t* fetchReverseComplementSequence_ptr(ReadId_t readNumber) const{
		return reverseComplSequencepointers[readNumber];
	}

	void transform(){
        if(isReadOnly)
            return;

		isReadOnly = true;

		std::size_t nSequences = sequences.size();

		if(nSequences == 0) return;

        std::vector<Sequence_t> tmp(sequences);
        sequences.reserve(2*nSequences);

//TIMERSTARTCPU(READ_STORAGE_SORT);
        std::sort(sequences.begin(), sequences.end());
        sequences.erase(std::unique(sequences.begin(), sequences.end()), sequences.end());
//TIMERSTOPCPU(READ_STORAGE_SORT);

        std::map<const Sequence_t, int> seqToSortedIndex;
		std::vector<std::map<const Sequence_t, int>> seqToSortedIndextmpvec;

//TIMERSTARTCPU(READ_STORAGE_MAKE_MAP);
        #pragma omp parallel
        {
            int threadId = omp_get_thread_num();
            #pragma omp single
            seqToSortedIndextmpvec.resize(omp_get_num_threads()-1);

            #pragma omp barrier

            auto& mymap = threadId == 0 ? seqToSortedIndex : seqToSortedIndextmpvec[threadId-1];
            #pragma omp for
            for(std::size_t i = 0; i < sequences.size(); i++){
                const auto& sequence = sequences[i];
                mymap[sequence] = &sequence - sequences.data();
            }
        }

        for(auto& tmpmap : seqToSortedIndextmpvec){
            seqToSortedIndex.insert(tmpmap.begin(), tmpmap.end());
            tmpmap.clear();
        }
        seqToSortedIndextmpvec.clear();

//TIMERSTOPCPU(READ_STORAGE_MAKE_MAP);

        assert(sequences.size() == seqToSortedIndex.size());

//TIMERSTARTCPU(READ_STORAGE_MAKE_FWD_POINTERS);
		sequencepointers.resize(nSequences);
        #pragma omp parallel for
		for(size_t i = 0; i < nSequences; i++)
            sequencepointers[i] = &sequences[seqToSortedIndex[tmp[i]]];
//TIMERSTOPCPU(READ_STORAGE_MAKE_FWD_POINTERS);

//TIMERSTARTCPU(READ_STORAGE_MAKE_REVCOMPL_POINTERS);
		reverseComplSequencepointers.resize(nSequences);
		for(size_t i = 0; i < nSequences; i++){
			Sequence_t revcompl = sequencepointers[i]->reverseComplement();
			auto it = seqToSortedIndex.find(revcompl);
			if(it == seqToSortedIndex.end()){
				//sequence does not exist yet, insert into map and save pointer in list
                sequences.push_back(std::move(revcompl));
            	reverseComplSequencepointers[i] = &(sequences.back());
                seqToSortedIndex[*reverseComplSequencepointers[i]] = reverseComplSequencepointers[i] - sequences.data();
			}else{
                reverseComplSequencepointers[i] = &(sequences[it->second]);
			}
		}
//TIMERSTOPCPU(READ_STORAGE_MAKE_REVCOMPL_POINTERS);

		seqToSortedIndex.clear();

#if 1
//TIMERSTARTCPU(READ_STORAGE_CHECK);
		//check
		#pragma omp parallel
		{
			#pragma omp for
			for(size_t i = 0; i < nSequences; i++){
				assert(*sequencepointers[i] == tmp[i] && "readstorage wrong sequence after dedup");
			}
			#pragma omp for
			for(size_t i = 0; i < nSequences; i++){
				assert(*reverseComplSequencepointers[i] == sequencepointers[i]->reverseComplement() && "readstorage wrong reverse complement after dedup");
			}
		}
//TIMERSTOPCPU(READ_STORAGE_CHECK);
#endif

	}

};





template<class sequence_t,
		 class readId_t>
struct ReadStorageNoPointer{

	using Sequence_t = sequence_t;
	using ReadId_t = readId_t;
    using Index_t = std::uint32_t;

    static constexpr bool has_reverse_complement = true;

	bool useQualityScores = false;
    bool isReadOnly = false;

    std::vector<std::string> qualityscores;
    std::vector<std::string> reverseComplqualityscores;
    std::vector<Sequence_t> sequences;
    std::vector<Index_t> sequenceIndices;
    std::vector<Index_t> reverseComplSequenceIndices;

    std::size_t size() const{
        std::size_t result = 0;

        for(const auto& s : qualityscores){
            result += sizeof(std::string) + s.capacity();
        }

        for(const auto& s : reverseComplqualityscores){
            result += sizeof(std::string) + s.capacity();
        }
        for(const auto& s : sequences){
            result += sizeof(Sequence_t) + s.getNumBytes();
        }

        result += sizeof(Index_t) * sequenceIndices.capacity();

        result += sizeof(Index_t) * reverseComplSequenceIndices.capacity();

        return result;
    }

    std::size_t sizereal() const{
        std::size_t result = 0;

        for(std::size_t i = 0; i < qualityscores.capacity(); i++){
            result += sizeof(std::string) + qualityscores[i].capacity();
        }
        for(std::size_t i = 0; i < reverseComplqualityscores.capacity(); i++){
            result += sizeof(std::string) + reverseComplqualityscores[i].capacity();
        }
        for(std::size_t i = 0; i < sequences.capacity(); i++){
            result += sizeof(Sequence_t) + sequences[i].getNumBytes();
        }
        result += sizeof(Index_t) * sequenceIndices.capacity();

        result += sizeof(Index_t) * reverseComplSequenceIndices.capacity();

        return result;
    }

    ReadStorageNoPointer() : ReadStorageNoPointer(false){}
    ReadStorageNoPointer(bool b) : useQualityScores(b){}

	void init(ReadId_t nReads){
        if(nReads > std::numeric_limits<Index_t>::max() / 2)
            throw std::runtime_error("ReadStorageNoPointer::init : nReads too large.");

		clear();

		sequences.resize(nReads);
        if(useQualityScores){
            qualityscores.resize(nReads);
    		reverseComplqualityscores.resize(nReads);
        }
	}

	void clear(){
		qualityscores.clear();
		reverseComplqualityscores.clear();
		sequences.clear();
		sequenceIndices.clear();
		reverseComplSequenceIndices.clear();

		isReadOnly = false;
	}

	void destroy(){
		clear();
		qualityscores.shrink_to_fit();
		reverseComplqualityscores.shrink_to_fit();
		sequenceIndices.shrink_to_fit();
		reverseComplSequenceIndices.shrink_to_fit();
	}

    void insertRead(ReadId_t readNumber, const std::string& sequence){
		if(useQualityScores){
			insertRead(readNumber, sequence, std::string(sequence.length(), 'A'));
		}else{
			Sequence_t seq(sequence);
			sequences[readNumber] = std::move(seq);
		}
	}

    void insertRead(ReadId_t readNumber, const std::string& sequence, const std::string& quality){
		if(isReadOnly) throw std::runtime_error("cannot insert read into ReadStorage after calling transform()");

		Sequence_t seq(sequence);
		std::string q(quality);
		std::reverse(q.begin(),q.end());

		sequences[readNumber] = std::move(seq);
		if(useQualityScores){
			qualityscores[readNumber] = std::move(quality);
			reverseComplqualityscores[readNumber] = std::move(q);
		}
	}

	const std::string* fetchQuality_ptr(ReadId_t readNumber) const{
		if(useQualityScores){
			return &(qualityscores[readNumber]);
		}else{
			return nullptr;
		}
	}

   	const std::string* fetchReverseComplementQuality_ptr(ReadId_t readNumber) const{
		if(useQualityScores){
			return &(reverseComplqualityscores[readNumber]);
		}else{
			return nullptr;
		}
	}

	//Must call transform() beforehand
	const Sequence_t* fetchSequence_ptr(ReadId_t readNumber) const{
		return &sequences[sequenceIndices[readNumber]];
	}

	//Must call transform() beforehand
	const Sequence_t* fetchReverseComplementSequence_ptr(ReadId_t readNumber) const{
        return &sequences[reverseComplSequenceIndices[readNumber]];
	}

	void transform(){
        if(isReadOnly)
            return;

		isReadOnly = true;

		std::size_t nSequences = sequences.size();

		if(nSequences == 0) return;

        std::vector<Sequence_t> tmp(sequences);

        sequences.reserve(2*nSequences);

//TIMERSTARTCPU(READ_STORAGE_SORT);
        std::sort(sequences.begin(), sequences.end());
        sequences.erase(std::unique(sequences.begin(), sequences.end()), sequences.end());
//TIMERSTOPCPU(READ_STORAGE_SORT);

        std::map<const Sequence_t, Index_t> seqToSortedIndex;
		std::vector<std::map<const Sequence_t, Index_t>> seqToSortedIndextmpvec;

//TIMERSTARTCPU(READ_STORAGE_MAKE_MAP);
        #pragma omp parallel
        {
            int threadId = omp_get_thread_num();
            #pragma omp single
            seqToSortedIndextmpvec.resize(omp_get_num_threads()-1);

            #pragma omp barrier

            auto& mymap = threadId == 0 ? seqToSortedIndex : seqToSortedIndextmpvec[threadId-1];
            #pragma omp for
            for(std::size_t i = 0; i < sequences.size(); i++){
                const auto& sequence = sequences[i];
                mymap[sequence] = Index_t(i);
            }
        }

        for(auto& tmpmap : seqToSortedIndextmpvec){
            seqToSortedIndex.insert(tmpmap.begin(), tmpmap.end());
            tmpmap.clear();
        }

        seqToSortedIndextmpvec.clear();

//TIMERSTOPCPU(READ_STORAGE_MAKE_MAP);

        assert(sequences.size() == seqToSortedIndex.size());

//TIMERSTARTCPU(READ_STORAGE_MAKE_FWD_POINTERS);
		sequenceIndices.resize(nSequences);

        #pragma omp parallel for
		for(size_t i = 0; i < nSequences; i++){
            sequenceIndices[i] = seqToSortedIndex[tmp[i]];
        }
//TIMERSTOPCPU(READ_STORAGE_MAKE_FWD_POINTERS);

//TIMERSTARTCPU(READ_STORAGE_MAKE_REVCOMPL_POINTERS);
		reverseComplSequenceIndices.resize(nSequences);

		for(size_t i = 0; i < nSequences; i++){
			Sequence_t revcompl = sequences[sequenceIndices[i]].reverseComplement();
			auto it = seqToSortedIndex.find(revcompl);
			if(it == seqToSortedIndex.end()){
				//sequence does not exist yet, insert into map and save pointer in list
                sequences.push_back(std::move(revcompl));
            	reverseComplSequenceIndices[i] = Index_t(sequences.size()) - 1;
                seqToSortedIndex[sequences.back()] = reverseComplSequenceIndices[i];
			}else{
                reverseComplSequenceIndices[i] = it->second;
			}
		}

//TIMERSTOPCPU(READ_STORAGE_MAKE_REVCOMPL_POINTERS);

		seqToSortedIndex.clear();

#if 1
//TIMERSTARTCPU(READ_STORAGE_CHECK);
		//check
		#pragma omp parallel
		{
			#pragma omp for
			for(size_t i = 0; i < nSequences; i++){
				assert(sequences[sequenceIndices[i]] == tmp[i] && "readstorage wrong sequence after dedup");
			}
			#pragma omp for
			for(size_t i = 0; i < nSequences; i++){
				assert(sequences[reverseComplSequenceIndices[i]] == sequences[sequenceIndices[i]].reverseComplement() && "readstorage wrong reverse complement after dedup");
			}
		}
//TIMERSTOPCPU(READ_STORAGE_CHECK);
#endif

	}

};










/*
    Data structure to store sequences and their quality scores
*/
template<class sequence_t,
		 class readId_t>
struct ReadStorageMinMemory{

	using Sequence_t = sequence_t;
	using ReadId_t = readId_t;

    static constexpr bool has_reverse_complement = false;

	bool useQualityScores = false;

    std::vector<std::string> qualityscores;
    std::vector<Sequence_t> sequences;

    ReadStorageMinMemory() : ReadStorageMinMemory(false){}
    ReadStorageMinMemory(bool b) : useQualityScores(b){}

    ReadStorageMinMemory(const ReadStorageMinMemory& other)
        : useQualityScores(other.useQualityScores),
        qualityscores(other.qualityscores),
        sequences(sequences){

    }

    ReadStorageMinMemory(ReadStorageMinMemory&& other)
        : useQualityScores(other.useQualityScores),
        qualityscores(std::move(other.qualityscores)),
        sequences(std::move(other.sequences)){

    }

    ReadStorageMinMemory& operator=(const ReadStorageMinMemory& other){
        useQualityScores = other.useQualityScores;
        qualityscores = other.qualityscores;
        sequences = sequences;
        return *this;
    }

    ReadStorageMinMemory& operator=(ReadStorageMinMemory&& other){
        useQualityScores = std::move(other.useQualityScores);
        qualityscores = std::move(other.qualityscores);
        sequences = std::move(sequences);
        return *this;
    }

    std::size_t size() const{
        std::size_t result = 0;

        for(const auto& s : qualityscores){
            result += sizeof(std::string) + s.capacity();
        }

        for(const auto& s : sequences){
            result += sizeof(Sequence_t) + s.getNumBytes();
        }

        return result;
    }

    std::size_t sizereal() const{
        std::size_t result = 0;

        for(std::size_t i = 0; i < qualityscores.capacity(); i++){
            result += sizeof(std::string) + qualityscores[i].capacity();
        }

        for(std::size_t i = 0; i < sequences.capacity(); i++){
            result += sizeof(Sequence_t) + sequences[i].getNumBytes();
        }

        return result;
    }

	void init(ReadId_t nReads){
		clear();

        //std::string stmp;
        //std::cout << "resize sequences" << std::endl;
        //std::cin >> stmp;

		sequences.resize(nReads);
        if(useQualityScores){

            //std::cout << "resize qualityscores" << std::endl;
            //std::cin >> stmp;

            qualityscores.resize(nReads);
        }
	}

	void clear(){
		qualityscores.clear();
		sequences.clear();
	}

	void destroy(){
		clear();
		qualityscores.shrink_to_fit();
		sequences.shrink_to_fit();
	}

    void insertRead(ReadId_t readNumber, const std::string& sequence){
		if(useQualityScores){
			insertRead(readNumber, sequence, std::string(sequence.length(), 'A'));
		}else{
			Sequence_t seq(sequence);
			//sequences[readNumber] = std::move(seq);
            sequences.at(readNumber) = std::move(seq);
		}
	}

    void insertRead(ReadId_t readNumber, const std::string& sequence, const std::string& quality){
		Sequence_t seq(sequence);

		//sequences[readNumber] = std::move(seq);
        sequences.at(readNumber) = std::move(seq);
		if(useQualityScores){
			//qualityscores[readNumber] = std::move(quality);
            qualityscores.at(readNumber) = std::move(quality);
		}
        //std::string stmp;
        //std::cout << "inserted sequence" << std::endl;
        //std::cin >> stmp;
	}

	const std::string* fetchQuality_ptr(ReadId_t readNumber) const{
		if(useQualityScores){
			return &(qualityscores[readNumber]);
		}else{
			return nullptr;
		}
	}

	const Sequence_t* fetchSequence_ptr(ReadId_t readNumber) const{
		return &sequences[readNumber];
	}

   //not supported
   const std::string* fetchReverseComplementQuality_ptr(ReadId_t readNumber) const{
       throw std::runtime_error("not supported");
   }

   //not supported
   const Sequence_t* fetchReverseComplementSequence_ptr(ReadId_t readNumber) const{
       throw std::runtime_error("not supported");
   }

   void transform(){}

};

}

#endif
