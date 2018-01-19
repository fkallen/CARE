#ifndef ERRORCORRECTOR_HPP
#define ERRORCORRECTOR_HPP

#include "minhasher.hpp"

#include "read.hpp"
#include "readstorage.hpp"
#include "threadsafe_buffer.hpp"

//#include "countminsketch.hpp"

#include "errorgraph.hpp"
#include "alignment.hpp"
#include "alignment_semi_global.hpp"
#include "../inc/aligner.hpp"

#include <cstdint>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <tuple>

#define USE_QUALITY_SCORES

class Barrier
{

 public:
    Barrier(int count): counter(0), thread_count(count){}

    void wait()
    {
        //fence mechanism
        std::unique_lock<std::mutex> lk(m);
        ++counter;
	if(counter < thread_count)
	        cv.wait(lk);
	else{
		counter = 0;
		cv.notify_all();
	}
    }

 private:
      std::mutex m;
      std::condition_variable cv;
      int counter;
      int thread_count;
};

#if 0
struct EC_Thread_Data{
	int ALIGNMENTSCORE_MATCH = 1;
	int ALIGNMENTSCORE_SUB = -1;
	int ALIGNMENTSCORE_INS = -100;
	int ALIGNMENTSCORE_DEL = -100;

	double MAX_MISMATCH_RATIO = 0.2;
	int MIN_OVERLAP = 35;
	double MIN_OVERLAP_RATIO = 0.35;

	// if there are at least CANDIDATES_CORRECTION_THRESHOLD candidates
	// this read will be corrected using these candidates
	static constexpr int CANDIDATES_CORRECTION_THRESHOLD = 3;

	// the query to the hashmap needs to return at least this many candidates to start
	// the correction process, filtering these candidates, etc.
	static constexpr int MINIMUM_POSSIBLE_CANDIDATES = 3;

	// how many correction results should be buffered before writing the correction results to file
	static constexpr int bufferedResultsThreshold = 1000;

	int threadId;
	int deviceId;
	bool useGPUalignment;

	std::uint32_t readIdBeginInc;
	std::uint32_t readIdEndExcl;
	std::uint32_t maxbatchsize;
	std::uint32_t nReads;

	double graphx;
	double graphalpha;

	const ReadStorage* readStorage;
	const Minhasher* minhasher;
	std::mutex* writelock;

	std::string outputPath;

	std::uint32_t nProcessedReads;
	double progress;
	bool done;

#ifdef __NVCC__
	AlignerDataArrays alignerData;
#endif

	// the query sequences fetched from ReadStorage
	std::vector<const Sequence*> queries;
	std::vector<std::string> queryStrings;
	// the readnums of candidates for each query
	std::vector<std::vector<std::pair<std::uint64_t, int>>> candidateIds;

	// the initial number of candidates per query returned by minhasher
	std::vector<int> initialNumberOfCandidates;
	// the corrected reads
	std::vector<CorrectedRead> correctedQueries;
	// save if correction was performed
	std::vector<bool> corrected;
	// save if correction process changed the query
	std::vector<bool> correctedAndChanged;
	std::vector<int> numberOfCorrectionCandidates;
	// the candidate sequences from candidateReadsWithFrequency
	std::vector<std::vector<const Sequence*>> candidateReads;
	std::vector<std::vector<const Sequence*>> revComplcandidateReads;
	std::vector<std::vector<int>> frequencies;
	std::vector<std::vector<AlignResult>> alignmentResults;
	std::vector<std::vector<const Sequence*>> candidateReadsAndRevcompls;
	std::vector<const std::string*> queryQualities;
	std::vector<std::vector<const std::string*>> candidateQualities;
	std::vector<std::vector<const std::string*>> revcomplcandidateQualities;

	EC_Thread_Data(int tid, int did, bool useGPUalignment_, const ReadStorage* readStorageptr, const Minhasher* minhasherptr,
			double graphx_, double graphalpha_, 
			int ALIGNMENTSCORE_MATCH_, int ALIGNMENTSCORE_SUB_, int ALIGNMENTSCORE_INS_, int ALIGNMENTSCORE_DEL_,
			double MAX_MISMATCH_RATIO_, int MIN_OVERLAP, double MIN_OVERLAP_RATIO,
			std::uint32_t readIdBeginInc_, std::uint32_t readIdEndExcl_, std::uint32_t maxbatchsize_,
			const std::string& outputPath_, std::mutex* mutex) 
		: threadId(tid), deviceId(did), useGPUalignment(useGPUalignment_), readStorage(readStorageptr),
			minhasher(minhasherptr), graphx(graphx_), graphalpha(graphalpha_),
			ALIGNMENTSCORE_MATCH(ALIGNMENTSCORE_MATCH_), ALIGNMENTSCORE_SUB(ALIGNMENTSCORE_SUB_), 
			ALIGNMENTSCORE_INS(ALIGNMENTSCORE_INS_), ALIGNMENTSCORE_DEL(ALIGNMENTSCORE_DEL_),
			MAX_MISMATCH_RATIO(MAX_MISMATCH_RATIO_), MIN_OVERLAP(MIN_OVERLAP_), MIN_OVERLAP_RATIO(MIN_OVERLAP_RATIO_),
			readIdBeginInc(readIdBeginInc_), readIdEndExcl(readIdEndExcl_), maxbatchsize(maxbatchsize_),
			outputPath(outputPath_), writelock(mutex),
			done(false), progress(0.0), nReads(readIdEndExcl_ - readIdBeginInc_){
#ifdef __NVCC__
		if(useGPUalignment)
			alignerData = std::move(AlignerDataArrays(did));
#endif		
	}

	void correct(){
		done = false;
	// the output file of this thread
		std::ofstream outputfile(outputPath + "/" + std::to_string(threadId));

		// number of buffered correction results. nBufferedResults < bufferedResultsThreshold	
		int nBufferedResults = 0; 

		// buffer of correction results
		std::stringstream resultstringstream;

		nProcessedReads = 0;

		// total number of candidates returned from minhasher
		std::uint64_t processedCandidates = 0;

		for(std::uint32_t readnum = readIdBeginInc; readnum < readIdEndExcl; readnum += maxbatchsize){
			const std::uint32_t actualBatchSize = std::min(maxbatchsize, readIdEndExcl - readnum);

			//fit vector size to actual batch size
			if(actualBatchSize < batchsize){
				queries.resize(actualBatchSize);
				queryStrings.resize(actualBatchSize);
				candidateIds.resize(actualBatchSize);
				initialNumberOfCandidates.resize(actualBatchSize);
				correctedQueries.resize(actualBatchSize);
				corrected.resize(actualBatchSize);
				correctedAndChanged.resize(actualBatchSize);
				numberOfCorrectionCandidates.resize(actualBatchSize);
				candidateReads.resize(actualBatchSize);
				revComplcandidateReads.resize(actualBatchSize);
				frequencies.resize(actualBatchSize);
				alignmentResults.resize(actualBatchSize);
				candidateReadsAndRevcompls.resize(actualBatchSize);
				queryQualities.resize(actualBatchSize);
				candidateQualities.resize(actualBatchSize);
				revcomplcandidateQualities.resize(actualBatchSize);
			}

			// fetch the reads of current batch from the readstorage
			for(int i = 0; i < actualBatchSize; i++){
				queries[i] = readStorage->fetchSequence_ptr(readnum + i);
				queryQualities[i] = readStorage->fetchQuality_ptr(readnum + i);
			}

			// for each read of the current batch, find their correction candidates
			for(int i = 0; i < actualBatchSize; i++){
				queryStrings[i] = queries[i]->toString();
				candidateIds[i] = minhasher->getCandidates(queryStrings[i]);
				initialNumberOfCandidates[i] = candidateIds[i].size();
				processedCandidates += initialNumberOfCandidates[i];
			}

			// map minhash ids to sequences
			mapMinhashResultsToSequences();

			/*for(int i = 0; i < candidateReads.size(); i++){
				for(int j = 0; j < candidateReads[i].size(); j++)
				std::cout << candidateReads[i][j]->toString() << " " << frequencies[i][j] << std::endl;
			}*/

			for(int i = 0; i < actualBatchSize; i++){
				numberOfAlignmentCandidates[i] = frequencies[i].size();
			}

			getMultipleAlignments();

			// perform error correction
			for(int i = 0; i < actualBatchSize; i++){
				if(numberOfCorrectionCandidates[i] >= CANDIDATES_CORRECTION_THRESHOLD){

					assert(alignmentResults[i].size() == frequencies[i].size());

					const std::string& seq = queryStrings[i];
					ErrorGraph errorgraph(seq.c_str(), seq.length(), queryQualities[i]->c_str(), true);

					int qualindex = 0;
					for(int j = 0; j < alignmentResults[i].size() / 2; j++){
						auto& res = alignmentResults[i][2*j];	
						auto& revcomplres = alignmentResults[i][2*j+1];
						int best, overlap, nMismatch;
					
						std::tie(best, overlap, nMismatch) = get_best_alignment(res, revcomplres, false, MAX_MISMATCH_RATIO, MIN_OVERLAP);

						if(best == 0){
							split_subs(res, seq.c_str());

							for(int f = 0; f < frequencies[i][2*j]; f++){
								auto qual = candidateQualities[i][qualindex + f + 0 * frequencies[i][2*j]];
								errorgraph.insertAlignment(res, qual->c_str(), nMismatch, overlap, MAX_MISMATCH_RATIO, 1);
							}
						
						}else if(best == 1){
							split_subs(revcomplres, seq.c_str());

							for(int f = 0; f < frequencies[i][2*j]; f++){
								auto qual = candidateQualities[i][qualindex + f + 1 * frequencies[i][2*j]];
								errorgraph.insertAlignment(revcomplres, qual->c_str(), nMismatch, overlap, MAX_MISMATCH_RATIO, 1);
							}
						}else
							; //both alignments are bad	

						qualindex += 2*frequencies[i][2*j];
					}

					errorgraph.readid = readnum + i;

					// let the graph to its work
					correctedQueries[i] = errorgraph.getCorrectedRead(graphalpha, graphx);

					if(false){
						errorgraph.dumpGraph("graph"+std::to_string(readnum + i));
					}

					corrected[i] = true;
					correctedAndChanged[i] = (queries[i]->operator!=(correctedQueries[i].sequence));

					/*std::cout << "correction " << readnum + i  
						<< ". insertCalls : " << errorgraph.insertCalls 
						<< ". usedCandidates : " <<  errorgraph.totalInsertedAlignments 
						<< ", prob : " << correctedQueries[i].probability << std::endl;*/

				}
			}

			// write result to output buffer
			for(int i = 0; i < actualBatchSize; i++){
				std::string header = *(readStorage->fetchHeader_ptr(readnum));
				std::string quality = *(readStorage->fetchQuality_ptr(readnum));
				if (corrected[i]) {
					resultstringstream	<< header << " corrected, changed : " << correctedAndChanged[i]
					  << ", " << numberOfCorrectionCandidates[i] << " candidates, "
					  << correctedQueries[i].probability << " probability" << '\n'
					  << correctedQueries[i].sequence << '\n' << '+' << '\n'
					  << quality << '\n';
				}else{
					resultstringstream	<< header << '\n'
					  << queryStrings[i] << '\n' << '+' << '\n'
					  << quality << '\n';
				}

				nBufferedResults++;
			}

			// write result to output file if output buffer is full
			if(nBufferedResults >= bufferedResultsThreshold){

				std::lock_guard<std::mutex> lg(*writelock);
				outputfile << resultstringstream.rdbuf();
				nBufferedResults = 0;
				resultstringstream.str(std::string());
				resultstringstream.clear();
			}

			nProcessedReads += actualBatchSize;
			progress = double(nProcessedReads) / double(nReads);
			//printf("t%d %d, %d, %d, %f\n", threadId, actualBatchSize, nProcessedReads, nReads, progress);
		}

		done = true;

		{
			std::lock_guard<std::mutex> lg(*writelock);
			std::cout << "thread " << threadId << " processed candidates " << processedCandidates << std::endl;
		}
	}

	void set_batch_size(int size){
		queries.clear();
		queryStrings.clear();
		candidateIds.clear();
		initialNumberOfCandidates.clear();
		correctedQueries.clear();
		corrected.clear();
		correctedAndChanged.clear();
		numberOfCorrectionCandidates.clear();
		numberOfAlignmentCandidates.clear();
		candidateReads.clear();
		frequencies.clear();
		alignmentResults.clear();
		queryQualities.clear();
		candidateQualities.clear();

		queries.resize(size);
		queryStrings.resize(size);
		candidateIds.resize(size);
		initialNumberOfCandidates.resize(size);
		correctedQueries.resize(size);
		corrected.resize(size);
		correctedAndChanged.resize(size);
		numberOfCorrectionCandidates.resize(size);
		numberOfAlignmentCandidates.resize(size);
		candidateReads.resize(size);
		frequencies.resize(size);
		alignmentResults.resize(size);
		queryQualities.resize(size);
		candidateQualities.resize(size);

		/*for(auto& e : candidateIds)
			e.clear();
		for(auto& e : frequencies)
			e.clear();
		for(auto& e : alignmentResults)
			e.clear();
		for(auto& e : candidateQualities)
			e.clear();
		candidateReads.clear();*/

	}

	void mapMinhashResultsToSequences(){

		for(int i = 0; i < queries.size(); i++){

			int numberOfCandidates = 0;
			std::map<const Sequence*, int, SequencePtrLess> candidateSequencesWithFrequency;
			std::map<const Sequence*, const Sequence*, SequencePtrLess> revCompls;
			std::map<const Sequence*, std::vector<const std::string*>, SequencePtrLess> qscoremap;
			std::map<const Sequence*, std::vector<const std::string*>, SequencePtrLess> revComplqscoremap;

			for (const auto r : candidateIds[i]) {
				const auto sequence = readStorage->fetchSequence_ptr(r.first);
				const auto qscore = readStorage->fetchQuality_ptr(r.first);
				const auto revComplqscore = readStorage->fetchReverseComplementQuality_ptr(r.first);

				candidateSequencesWithFrequency[sequence]++;
				qscoremap[sequence].push_back(qscore);
				revComplqscoremap[sequence].push_back(revComplqscore);

				if(candidateSequencesWithFrequency[sequence] < 2)
					revCompls[sequence] = readStorage->fetchReverseComplementSequence_ptr(r.first);

				numberOfCandidates++;
			}

			for(const auto p : candidateSequencesWithFrequency){
				candidateReads[i].push_back(p.first);
				candidateReads[i].push_back(revCompls[p.first]);
				frequencies[i].push_back(p.second);
				frequencies[i].push_back(p.second);

				candidateQualities[i].insert(candidateQualities[i].end(), qscoremap[p.first].begin(), qscoremap[p.first].end());
				candidateQualities[i].insert(candidateQualities[i].end(), revComplqscoremap[p.first].begin(), revComplqscoremap[p.first].end());
			}

			numberOfCorrectionCandidates[i] = numberOfCandidates;
		}	
	}

	void getMultipleAlignments(){	

		int totalNumberOfAlignments = 0;
		for(const auto& v : candidateReads) totalNumberOfAlignments += v.size();

		// check for empty input
		if(queries.size() == 0 || totalNumberOfAlignments == 0 || numberOfAlignmentCandidates.size() == 0){
			for(auto& a : alignmentResults)
				a.clear();
			return;
		}

		// check for correct input dimensions
		int expectedCandidates = 0;
		for(auto i : numberOfAlignmentCandidates) expectedCandidates += i;

		if(queries.size() != numberOfAlignmentCandidates.size() || totalNumberOfAlignments != expectedCandidates){

			std::cout << "foo" << std::endl;

			throw std::runtime_error("getMultipleAlignments incorrect input dimensions");
		}


	#ifdef __CUDACC__

		if(useGPUalignment){

			cudaSetDevice(alignerData.deviceId); CUERR;

			std::vector<int> numberOfAlignmentCandidatesPrefixSum{0};
			for(int i = 0; i < numberOfAlignmentCandidates.size(); i++){
				numberOfAlignmentCandidatesPrefixSum.push_back(numberOfAlignmentCandidatesPrefixSum.back() + numberOfAlignmentCandidates[i]);
			}

			int maximumCandidateLength = 0;
			int maximumQueryLength = 0;

			std::vector<int> candidateLengths(totalNumberOfAlignments);
			std::vector<int> candidateBytes(totalNumberOfAlignments + 1);
			std::vector<int> candidateIsEncoded(totalNumberOfAlignments);
			std::vector<char> candidateData;

			std::vector<int> queryLengths(queries.size());
			std::vector<int> queryBytes(queries.size() + 1);
			std::vector<int> queryIsEncoded(queries.size());
			std::vector<char> queryData;

			candidateBytes[0] = 0;
			queryBytes[0] = 0;

			// copy candidate data into consecutive memory.
			int globalCandidateIndex = 0;
			for(const auto& v : candidateReads){
				for(const auto& candidate : v){
					int b = candidate->getNumBytes();		
					int l = candidate->getNbases();

					maximumCandidateLength = maximumCandidateLength < l ? l : maximumCandidateLength;

					candidateLengths[globalCandidateIndex] = l;
					candidateBytes[globalCandidateIndex+1] = b + candidateBytes[globalCandidateIndex];
					candidateIsEncoded[globalCandidateIndex] = candidate->isCompressed();

					candidateData.insert(candidateData.end(), candidate->begin(), candidate->end());

					globalCandidateIndex++;
				}
			}

			for(int i = 0; i < queries.size(); i++){
				const auto& query = queries[i];

				int b = query->getNumBytes();		
				int l = query->getNbases();

				maximumQueryLength = maximumQueryLength < l ? l : maximumQueryLength;

				queryLengths[i] = l;
				queryBytes[i+1] = b + queryBytes[i];
				queryIsEncoded[i] = query->isCompressed();

				queryData.insert(queryData.end(), query->begin(), query->end());
			}

			// resize gpu storage
			int ml = maximumCandidateLength;
			int max_ops_per_alignment = 2 * (ml + 1);
			int max_ops = max_ops_per_alignment * totalNumberOfAlignments;
			alignerData.resize(totalNumberOfAlignments, // number of alignments
							max_ops, // maximum number of align ops 
							queryBytes.back(), // number of bytes for queries
							candidateBytes.back()); // number of bytes for candidates
			CUERR;

			// copy data to pinned host memory
			memcpy(alignerData.h_r.get(), queryData.data(), sizeof(char) * queryData.size());
			memcpy(alignerData.h_c.get(), candidateData.data(), sizeof(char) * candidateData.size());
			memcpy(alignerData.h_rBytesPrefixSum.get(), queryBytes.data(), sizeof(int) * queryBytes.size());
			memcpy(alignerData.h_rLengths.get(), queryLengths.data(), sizeof(int) * queryLengths.size());
			memcpy(alignerData.h_rIsEncoded.get(), queryIsEncoded.data(), sizeof(int) * queryIsEncoded.size());
			memcpy(alignerData.h_cBytesPrefixSum.get(), candidateBytes.data(), sizeof(int) * candidateBytes.size());
			memcpy(alignerData.h_cLengths.get(), candidateLengths.data(), sizeof(int) * candidateLengths.size());
			memcpy(alignerData.h_cIsEncoded.get(), candidateIsEncoded.data(), sizeof(int) * candidateIsEncoded.size());
			memcpy(alignerData.h_r2PerR1.get(), numberOfAlignmentCandidatesPrefixSum.data(), sizeof(int) * numberOfAlignmentCandidatesPrefixSum.size());

			// copy data to gpu
			cudaMemcpyAsync(alignerData.d_r.get(), 
					alignerData.h_r.get(), 
					sizeof(char) * queryData.size(), 
					H2D, 
					alignerData.stream); CUERR;
			cudaMemcpyAsync(alignerData.d_c.get(),
					alignerData.h_c.get(), 
					sizeof(char) * candidateData.size(), 
					H2D, 
					alignerData.stream); CUERR;
			cudaMemcpyAsync(alignerData.d_cBytesPrefixSum.get(),
					alignerData.h_cBytesPrefixSum.get(), 
					sizeof(int) * candidateBytes.size(), 
					H2D, 
					alignerData.stream); CUERR;
			cudaMemcpyAsync(alignerData.d_cLengths.get(),
					alignerData.h_cLengths.get(), 
					sizeof(int) * candidateLengths.size(), 
					H2D, 
					alignerData.stream); CUERR;
			cudaMemcpyAsync(alignerData.d_cIsEncoded.get(),
					alignerData.h_cIsEncoded.get(), 
					sizeof(int) * candidateIsEncoded.size(), 
					H2D, 
					alignerData.stream); CUERR;
			cudaMemcpyAsync(alignerData.d_rBytesPrefixSum.get(),
					alignerData.h_rBytesPrefixSum.get(), 
					sizeof(int) * queryBytes.size(), 
					H2D, 
					alignerData.stream); CUERR;
			cudaMemcpyAsync(alignerData.d_rLengths.get(),
					alignerData.h_rLengths.get(), 
					sizeof(int) * queryLengths.size(), 
					H2D, 
					alignerData.stream); CUERR;
			cudaMemcpyAsync(alignerData.d_rIsEncoded.get(),
					alignerData.h_rIsEncoded.get(), 
					sizeof(int) * queryIsEncoded.size(), 
					H2D, 
					alignerData.stream); CUERR;
			cudaMemcpyAsync(alignerData.d_r2PerR1.get(),
					alignerData.h_r2PerR1.get(), 
					sizeof(int) * numberOfAlignmentCandidatesPrefixSum.size(), 
					H2D, 
					alignerData.stream); CUERR;

			size_t smem = cuda_semi_global_align_getSharedMemSize(maximumQueryLength, maximumCandidateLength);

			dim3 block(std::min(512, 32 * SDIV(maximumCandidateLength+1, 32)), 1, 1);
			dim3 grid(totalNumberOfAlignments, 1, 1);

			// start kernel

			cuda_semi_global_align_multiplesubjects
				<<<grid, block, smem, alignerData.stream>>>(
					alignerData.d_results.get(),
					alignerData.d_ops.get(),
					max_ops_per_alignment, 
					queries.size(), 
					totalNumberOfAlignments,
					alignerData.d_r.get(), 
					alignerData.d_c.get(),
					alignerData.d_rBytesPrefixSum.get(),
					alignerData.d_rLengths.get(),
					alignerData.d_rIsEncoded.get(),
					alignerData.d_cBytesPrefixSum.get(),
					alignerData.d_cLengths.get(),
					alignerData.d_cIsEncoded.get(),
					alignerData.d_r2PerR1.get(),
					maximumQueryLength, maximumCandidateLength,
					ALIGNMENTSCORE_MATCH, ALIGNMENTSCORE_SUB, ALIGNMENTSCORE_INS, ALIGNMENTSCORE_DEL);
			CUERR;

			cudaStreamSynchronize(alignerData.stream); CUERR;

			// copy result to host
			cudaMemcpyAsync(alignerData.h_results.get(), 
					alignerData.d_results.get(), 
					sizeof(CudaAlignResult) * totalNumberOfAlignments, 
					D2H, 
					alignerData.stream); CUERR;
			cudaMemcpyAsync(alignerData.h_ops.get(), 
					alignerData.d_ops.get(), 
					sizeof(AlignOp) * max_ops, 
					D2H, 
					alignerData.stream); CUERR;

			cudaStreamSynchronize(alignerData.stream); CUERR;

			// store alignments in result vector

			int previousCandidates = 0;

			for(int i = 0; i < queries.size(); i++){

				alignmentResults[i].resize(numberOfAlignmentCandidates[i]);

				for(int j = 0; j < numberOfAlignmentCandidates[i]; j++){
					const int alignmentIndex = previousCandidates + j;

					if(alignerData.h_results.get()[alignmentIndex].flag != 0){
						throw std::runtime_error(("alignment backtrack error."));
					}

					// set alignment score
					alignmentResults[i][j].score = alignerData.h_results.get()[alignmentIndex].score;

					// set overlap region
					alignmentResults[i][j].subject_begin_incl = alignerData.h_results.get()[alignmentIndex].subject_begin_incl;
					alignmentResults[i][j].subject_end_excl = alignerData.h_results.get()[alignmentIndex].subject_end_excl;
					alignmentResults[i][j].query_begin_incl = alignerData.h_results.get()[alignmentIndex].query_begin_incl;
					alignmentResults[i][j].query_end_excl = alignerData.h_results.get()[alignmentIndex].query_end_excl;

					// reserve space for operations
					alignmentResults[i][j].operations.resize(alignerData.h_results.get()[alignmentIndex].nOps);
					// set operations
					std::reverse_copy(alignerData.h_ops.get() + alignmentIndex * max_ops_per_alignment,
							  alignerData.h_ops.get() + alignmentIndex * max_ops_per_alignment 
										      + alignerData.h_results.get()[alignmentIndex].nOps,
							  alignmentResults[i][j].operations.begin());
				}

				previousCandidates += numberOfAlignmentCandidates[i];
			}

		}else{ // use cpu for alignment



	#endif // __CUDACC_

			for(int i = 0; i < queries.size(); i++){
				const auto& query = queries[i];

				alignmentResults[i].reserve(numberOfAlignmentCandidates[i]);
				alignmentResults[i].clear();
			

				if (query->isCompressed()) {
					const char* qdata = (const char*) query->begin();
					int qbases = query->getNbases();
					EncodedAccessor ea(qdata, qbases);

					for(const auto& c : candidateReads[i]){
						if(c->isCompressed()){
							const char* cdata = (const char*)c->begin();
							int cbases = c->getNbases();

							alignmentResults[i].push_back(
								cpu_semi_global_align<EncodedAccessor, EncodedAccessor>(qdata, cdata, qbases, cbases, 
												ALIGNMENTSCORE_MATCH, ALIGNMENTSCORE_SUB,ALIGNMENTSCORE_INS, ALIGNMENTSCORE_DEL)
							);
						}else{
							const char* cdata = (const char*)c->begin();
							int cbases = c->getNbases();

							alignmentResults[i].push_back(
								cpu_semi_global_align<EncodedAccessor, OrdinaryAccessor>(qdata, cdata, qbases, cbases, 
												ALIGNMENTSCORE_MATCH, ALIGNMENTSCORE_SUB,ALIGNMENTSCORE_INS, ALIGNMENTSCORE_DEL)
							);
						}
					}
				}else{
					const char* qdata = (const char*) query->begin();
					int qbases = query->getNbases();
					for(const auto& c : candidateReads[i]){
						if(c->isCompressed()){
							const char* cdata = (const char*)c->begin();
							int cbases = c->getNbases();

							alignmentResults[i].push_back(
								cpu_semi_global_align<OrdinaryAccessor, EncodedAccessor>(qdata, cdata, qbases, cbases, 
												ALIGNMENTSCORE_MATCH, ALIGNMENTSCORE_SUB,ALIGNMENTSCORE_INS, ALIGNMENTSCORE_DEL)
							);
						}else{
							const char* cdata = (const char*)c->begin();
							int cbases = c->getNbases();

							alignmentResults[i].push_back(
								cpu_semi_global_align<OrdinaryAccessor, OrdinaryAccessor>(qdata, cdata, qbases, cbases, 
												ALIGNMENTSCORE_MATCH, ALIGNMENTSCORE_SUB,ALIGNMENTSCORE_INS, ALIGNMENTSCORE_DEL)
							);
						}
					}
				}
			}
	#ifdef __CUDACC__
		}
	#endif // __CUDACC__
	}


};
#endif

/*
    Corrects a single fastq file
*/
struct ErrorCorrector {

	ErrorCorrector();

	ErrorCorrector(const MinhashParameters& minhashparameters, int nInserterThreads, int nCorrectorThreads);

	void correct(const std::string& filename);

	void setOutputPath(const std::string& path);

	void setGraphSettings(double alpha, double x);

	void setOutputFilename(const std::string& filename);

	void setBatchsize(int n);

	void setAlignmentScores(int matchscore, int subscore, int insertscore, int delscore);

	void setMaxMismatchRatio(double ratio);
	void setMinimumAlignmentOverlap(int overlap);
	void setMinimumAlignmentOverlapRatio(double ratio);
	void setFileFormat(const std::string& format);
	void setUseQualityScores(bool val);
	void setEstimatedCoverage(int cov);
	void setEstimatedErrorRate(double rate);
	void setM(double m);

	

private:



	enum class Fileformat {FASTA, FASTQ};

	void insertFile(const std::string& filename, bool buildHashmap);

	void errorcorrectFile(const std::string& filename);

	void errorcorrectWork(int threadId, int nThreads, const std::string& fileToCorrect);

#if 0
	std::map<const Sequence*, std::vector<int>, SequencePtrLess> mapMinhashResultsToSequences(const std::vector<std::pair<std::uint64_t, int>>& minhashresults,
					std::vector<const Sequence*>& candidates,
					std::vector<const Sequence*>& revcomplcandidates,
					std::vector<const std::string*>& qualityscores,
					std::vector<const std::string*>& revcomplqualityscores,
					std::vector<int>& frequencies, 
					std::chrono::duration<double>& a,
					std::chrono::duration<double>& b) const;
#else


	std::map<const Sequence*, std::vector<int>, SequencePtrLess> mapMinhashResultsToSequences(const std::vector<std::uint64_t>& minhashresults,
					std::vector<const Sequence*>& candidates,
					std::vector<const Sequence*>& revcomplcandidates,
					std::vector<const std::string*>& qualityscores,
					std::vector<const std::string*>& revcomplqualityscores,
					std::vector<int>& frequencies, 
					std::chrono::duration<double>& a,
					std::chrono::duration<double>& b) const;

#endif

	void getMultipleAlignments(int threadId, const std::vector<const Sequence*>& queries,
				   const std::vector<std::vector<const Sequence*>>& candidates,
				   std::vector<std::vector<AlignResult>>& alignments,
				   std::vector<bool> activeBatches,
				   std::chrono::duration<double>& h2dtimetotal,
				   std::chrono::duration<double>& d2htimetotal,
				   std::chrono::duration<double>& kerneltimetotal);

	void mergeThreadResults(const std::string& filename) const;

	void mergeUnorderedThreadResults(const std::string& filename) const;

	std::uint64_t getReadPos(const std::string& readheader) const;

	void updateGlobalProgress(std::uint64_t increment, std::uint64_t maxglobalprogress);

	Minhasher minhasher;

	MinhashParameters minhashparams;

	mutable ReadStorage readStorage;

	std::vector<ThreadsafeBuffer<
			    std::pair<Read, std::uint32_t>,
			    30000> > buffers;

	std::unique_ptr<Aligner> aligner;
	std::unique_ptr<Aligner> shdaligner;
	std::unique_ptr<Aligner> semiglobalaligner;

	std::vector<int> deviceIds;

#ifdef __CUDACC__	
	
	std::vector<AlignerDataArrays> alignerData;
#endif	

	bool useMultithreading;

	int nInserterThreads;
	int nCorrectorThreads;

	std::string outputPath;
	std::string outputFilename = "";

	std::uint32_t batchsize = 20;

	int ALIGNMENTSCORE_MATCH = 1;
	int ALIGNMENTSCORE_SUB = -1;
	int ALIGNMENTSCORE_INS = -100;
	int ALIGNMENTSCORE_DEL = -100;

	double MAX_MISMATCH_RATIO = 0.2;
	int MIN_OVERLAP = 35;
	double MIN_OVERLAP_RATIO = 0.35;

	bool useQualityScores = false;

	Fileformat inputfileformat = Fileformat::FASTQ;

	std::map<std::string, std::uint64_t> readsPerFile;

	std::mutex writelock;

	// settings for error graph
	double graphx = 2.0;
	double graphalpha = 1.0;

	std::mutex progresslock;
	std::uint64_t progress = 0;

	std::vector<char> readIsProcessedVector;
	std::unique_ptr<std::mutex[]> locksForProcessedFlags;
	size_t nLocksForProcessedFlags = 0;

	int estimatedCoverage;
	double errorrate;
	double m_coverage;

};

#endif
