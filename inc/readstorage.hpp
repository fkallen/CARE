#ifndef READ_STORAGE_HPP
#define READ_STORAGE_HPP

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>
#include <omp.h>

namespace care{

/*
    Data structure to store sequences and their quality scores
*/
template<class sequence_t,
		 class readId_t,
		 bool useQualityScores>
struct ReadStorage{
	
	using Sequence_t = sequence_t;
	using ReadId_t = readId_t;
	static constexpr bool hasQualityScores = useQualityScores;

    bool isReadOnly;

    std::vector<std::string> qualityscores;
    std::vector<std::string> reverseComplqualityscores;
    std::vector<Sequence_t> sequences;
    std::vector<Sequence_t*> sequencepointers;
    std::vector<Sequence_t*> reverseComplSequencepointers;
	
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

}

#endif
