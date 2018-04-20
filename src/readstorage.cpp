#include "../inc/readstorage.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include <cassert>
#include <map>
#include <omp.h>

namespace care{

	ReadStorage::ReadStorage() : isReadOnly(false){}

	ReadStorage& ReadStorage::operator=(const ReadStorage&& other){
		headers = std::move(other.headers);
		qualityscores = std::move(other.qualityscores);
		reverseComplqualityscores = std::move(other.reverseComplqualityscores);
		sequences = std::move(other.sequences);

		sequencepointers = std::move(other.sequencepointers);
		reverseComplSequencepointers = std::move(other.reverseComplSequencepointers);
		//all_unique_sequences = std::move(other.all_unique_sequences);

		return *this;
	}

	void ReadStorage::setUseQualityScores(bool use){
		useQualityScores = use;
		if(!useQualityScores){
			qualityscores.clear();
			reverseComplqualityscores.clear();
			qualityscores.shrink_to_fit();
			reverseComplqualityscores.shrink_to_fit();
		}
	}

	void ReadStorage::init(ReadId_t nReads){
		clear();

		//headers.resize(nReads);
		sequences.resize(nReads);
        if(useQualityScores){
            qualityscores.resize(nReads);
    		reverseComplqualityscores.resize(nReads);
        }
	}

	void ReadStorage::clear(){
		headers.clear();
		qualityscores.clear();
		reverseComplqualityscores.clear();
		sequences.clear();

		sequencepointers.clear();
		reverseComplSequencepointers.clear();
		//all_unique_sequences.clear();

		isReadOnly = false;
	}

	void ReadStorage::destroy(){
		clear();
		headers.shrink_to_fit();
		qualityscores.shrink_to_fit();
		reverseComplqualityscores.shrink_to_fit();
		sequencepointers.shrink_to_fit();
		reverseComplSequencepointers.shrink_to_fit();
		//all_unique_sequences.shrink_to_fit();
	}

	void ReadStorage::insertRead(ReadId_t readNumber, const Read& read){
		if(isReadOnly) throw std::runtime_error("cannot insert read into ReadStorage after calling noMoreInserts()");

		Sequence_t seq(read.sequence);
		std::string q(read.quality);
		std::reverse(q.begin(),q.end());

		//headers[readNumber] = std::move(read.header);
		sequences[readNumber] = std::move(seq);
		if(useQualityScores){
			qualityscores[readNumber] = std::move(read.quality);
			reverseComplqualityscores[readNumber] = std::move(q);
		}
	}

	void ReadStorage::noMoreInserts(){
        if(isReadOnly)
            return;

		isReadOnly = true;

		size_t nSequences = sequences.size();

		if(nSequences == 0) return;

        std::vector<Sequence_t> tmp(sequences);
        sequences.reserve(2*nSequences);
		//std::vector<Sequence_t> sequencesflat;
		//sequencesflat.reserve(2*nSequences);

		//for(size_t index = 0; index < nSequences; index++){
		//	sequencesflat.push_back(std::move(sequences[index]));
		//}

		//sequences.clear();
		//sequences.shrink_to_fit();

		//std::vector<Sequence_t> tmp(sequencesflat);

//TIMERSTARTCPU(READ_STORAGE_SORT);
		//std::sort(sequencesflat.begin(), sequencesflat.end());
		//sequencesflat.erase(std::unique(sequencesflat.begin(), sequencesflat.end()), sequencesflat.end());
        std::sort(sequences.begin(), sequences.end());
        sequences.erase(std::unique(sequences.begin(), sequences.end()), sequences.end());
//TIMERSTOPCPU(READ_STORAGE_SORT);

        std::map<const Sequence_t, int> seqToSortedIndex;
		std::vector<std::map<const Sequence_t, int>> seqToSortedIndextmpvec;

        #pragma omp parallel
        {
            int threadId = omp_get_thread_num();
            #pragma omp single
            seqToSortedIndextmpvec.resize(omp_get_num_threads()-1);

            #pragma omp barrier

            auto& mymap = threadId == 0 ? seqToSortedIndex : seqToSortedIndextmpvec[threadId-1];
            #pragma omp for
            //for(std::size_t i = 0; i < sequencesflat.size(); i++){
            //    const auto& sequence = sequencesflat[i];
            //    mymap[sequence] = &sequence - sequencesflat.data();
            //}
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

//TIMERSTARTCPU(READ_STORAGE_MAKE_MAP);
	/*	for(const auto& s : sequencesflat){
			seqToSortedIndex[s] = &s - sequencesflat.data();
		}*/
        /*	for(const auto& s : sequences){
    			seqToSortedIndex[s] = &s - sequences.data();
    		}*/
//TIMERSTOPCPU(READ_STORAGE_MAKE_MAP);
		//assert(sequencesflat.size() == seqToSortedIndex.size());
        assert(sequences.size() == seqToSortedIndex.size());

		//size_t n_unique_forward_sequences = sequencesflat.size();
		//std::cout << "ReadStorage: found " << (nSequences - n_unique_forward_sequences) << " duplicates\n";

//TIMERSTARTCPU(READ_STORAGE_MAKE_FWD_POINTERS);
		sequencepointers.resize(nSequences);
        #pragma omp parallel for
		for(size_t i = 0; i < nSequences; i++)
			//sequencepointers[i] = &sequencesflat[seqToSortedIndex[tmp[i]]];
            sequencepointers[i] = &sequences[seqToSortedIndex[tmp[i]]];
//TIMERSTOPCPU(READ_STORAGE_MAKE_FWD_POINTERS);

//TIMERSTARTCPU(READ_STORAGE_MAKE_REVCOMPL_POINTERS);
		reverseComplSequencepointers.resize(nSequences);
		for(size_t i = 0; i < nSequences; i++){
			Sequence_t revcompl = sequencepointers[i]->reverseComplement();
			auto it = seqToSortedIndex.find(revcompl);
			if(it == seqToSortedIndex.end()){
				//sequence does not exist yet, insert into map and save pointer in list

				//sequencesflat.push_back(std::move(revcompl));
                sequences.push_back(std::move(revcompl));
			//	reverseComplSequencepointers[i] = &(sequencesflat.back());
            	reverseComplSequencepointers[i] = &(sequences.back());
				//seqToSortedIndex[*reverseComplSequencepointers[i]] = reverseComplSequencepointers[i] - sequencesflat.data();
                seqToSortedIndex[*reverseComplSequencepointers[i]] = reverseComplSequencepointers[i] - sequences.data();


			}else{
				//sequence is a duplicate, read position from map
				//reverseComplSequencepointers[i] = &(sequencesflat[it->second]);
                reverseComplSequencepointers[i] = &(sequences[it->second]);
			}
		}
//TIMERSTOPCPU(READ_STORAGE_MAKE_REVCOMPL_POINTERS);

		//std::cout << "ReadStorage: holding a total of " << seqToSortedIndex.size() << " unique sequences\n";

		seqToSortedIndex.clear();

#if 1
//TIMERSTARTCPU(READ_STORAGE_CHECK);
		//check
		for(size_t i = 0; i < nSequences; i++){
			assert(*sequencepointers[i] == tmp[i] && "readstorage wrong sequence after dedup");
		}
		for(size_t i = 0; i < nSequences; i++){
			assert(*reverseComplSequencepointers[i] == sequencepointers[i]->reverseComplement() && "readstorage wrong reverse complement after dedup");
		}
//TIMERSTOPCPU(READ_STORAGE_CHECK);
#endif

		//all_unique_sequences = std::move(sequencesflat);
	}

	Read ReadStorage::fetchRead(ReadId_t readNumber) const{
		Read returnvalue;

		returnvalue.header = *fetchHeader_ptr(readNumber);
		returnvalue.sequence = fetchSequence_ptr(readNumber)->toString();

		if(useQualityScores){
			returnvalue.quality = *fetchQuality_ptr(readNumber);
		}

		return returnvalue;
	}

	const std::string* ReadStorage::fetchHeader_ptr(ReadId_t readNumber) const{
		//return &(headers[readNumber]);
        return nullptr;
	}

    	const std::string* ReadStorage::fetchQuality_ptr(ReadId_t readNumber) const{
		if(useQualityScores){
			return &(qualityscores[readNumber]);
		}else{
			return nullptr;
		}
	}

    	const std::string* ReadStorage::fetchReverseComplementQuality_ptr(ReadId_t readNumber) const{
		if(useQualityScores){
			return &(reverseComplqualityscores[readNumber]);
		}else{
			return nullptr;
		}
	}

    	const Sequence_t* ReadStorage::fetchSequence_ptr(ReadId_t readNumber) const{
		//if(isReadOnly){
			return sequencepointers[readNumber];
		//}else{
		//	return &(sequences[readNumber]);
		//}
	}

	const Sequence_t* ReadStorage::fetchReverseComplementSequence_ptr(ReadId_t readNumber) const{
		//if(isReadOnly){
			return reverseComplSequencepointers[readNumber];
		//}else{
		//	throw std::runtime_error("fetchReverseComplementSequence_ptr");
		//}
	}

	double ReadStorage::getMemUsageMB() const{
		size_t bytes = 0;

		for(const auto& s : sequences){
			bytes += s.getNumBytes();
		}

		//for(const auto& s : all_unique_sequences){
		//	bytes += s.getNumBytes();
		//}

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

}
