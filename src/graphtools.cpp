#include "../inc/graphtools.hpp"
#include "../inc/alignment.hpp"
#include "../inc/sga.hpp"
#include "../inc/graph.hpp"

#include <cassert>
#include <chrono>
#include <cstring>
#include <vector>

namespace graphtools{

	AlignerDataArrays::AlignerDataArrays(int deviceId_, int maxseqlength, int scorematch, int scoresub, int scoreins, int scoredel) 
			: deviceId(deviceId_), ALIGNMENTSCORE_MATCH(scorematch), ALIGNMENTSCORE_SUB(scoresub), 
						ALIGNMENTSCORE_INS(scoreins), ALIGNMENTSCORE_DEL(scoredel), 
						max_ops_per_alignment(2 * (maxseqlength + 1)),
						max_sequence_length(maxseqlength), max_sequence_bytes(SDIV(maxseqlength,4)){
		#ifdef __NVCC__
		cudaSetDevice(deviceId); CUERR;
		cudaStreamCreate(&stream); CUERR;
		#endif
	};

	void AlignerDataArrays::resize_new(int n_sub, int n_quer){
	#ifdef __NVCC__
		cudaSetDevice(deviceId); CUERR;

		bool resizeResult = false;

		if(n_sub > max_n_subjects){
			size_t oldpitch = sequencepitch;
			cudaFree(d_subjectsdata); CUERR;
			cudaMallocPitch(&d_subjectsdata, &sequencepitch, max_sequence_bytes, n_sub); CUERR;
			assert(!oldpitch || oldpitch == sequencepitch);

			cudaFree(d_queriesPerSubject); CUERR;
			cudaMalloc(&d_queriesPerSubject, sizeof(int) * n_sub); CUERR;

			cudaFreeHost(h_subjectsdata); CUERR;
			cudaMallocHost(&h_subjectsdata, sequencepitch * n_sub); CUERR;

			cudaFreeHost(h_queriesPerSubject); CUERR;
			cudaMallocHost(&h_queriesPerSubject, sizeof(int) * n_sub); CUERR;

			max_n_subjects = n_sub;

			resizeResult = true;
		}


		if(n_quer > max_n_queries){
			size_t oldpitch = sequencepitch;
			cudaFree(d_queriesdata); CUERR;
			cudaMallocPitch(&d_queriesdata, &sequencepitch, max_sequence_bytes, n_quer); CUERR;
			assert(!oldpitch || oldpitch == sequencepitch);

			cudaFreeHost(h_queriesdata); CUERR;
			cudaMallocHost(&h_queriesdata, sequencepitch * n_quer); CUERR;

			max_n_queries = n_quer;

			resizeResult = true;
		}

		if(resizeResult){
			cudaFree(d_results); CUERR;
			cudaMalloc(&d_results, sizeof(AlignResultCompact) * max_n_subjects * max_n_queries); CUERR;

			cudaFreeHost(h_results); CUERR;
			cudaMallocHost(&h_results, sizeof(AlignResultCompact) * max_n_subjects * max_n_queries); CUERR;

			cudaFree(d_lengths); CUERR;
			cudaMalloc(&d_lengths, sizeof(int) * (max_n_subjects + max_n_queries)); CUERR;	

			cudaFreeHost(h_lengths); CUERR;
			cudaMallocHost(&h_lengths, sizeof(int) * (max_n_subjects + max_n_queries)); CUERR;

			cudaFree(d_ops); CUERR;
			cudaFreeHost(h_ops); CUERR;

			cudaMalloc(&d_ops, sizeof(AlignOp) * max_n_queries * max_ops_per_alignment); CUERR;
			cudaMallocHost(&h_ops, sizeof(AlignOp) * max_n_queries * max_ops_per_alignment); CUERR;
		}
	#endif
		n_subjects = n_sub;
		n_queries = n_quer;
	}


	/*
		res = number of alignments
		ops = maximum total number of alignment operations for res alignments
		r = number of bytes to store all subjects
		c = number of bytes to store all candidates
	*/
	void AlignerDataArrays::resize(size_t res, size_t ops, size_t r, size_t c){
		#ifdef __NVCC__
		cudaSetDevice(deviceId); 

		if(res > results_size){
			cudaMalloc(&d_results, sizeof(AlignResultCompact) * res); CUERR;
			cudaMalloc(&d_cBytesPrefixSum, sizeof(int) * (res+1)); CUERR;
			cudaMalloc(&d_cLengths, sizeof(int) * res); CUERR;
			cudaMalloc(&d_cIsEncoded, sizeof(int) * res); CUERR;
			cudaMalloc(&d_rBytesPrefixSum, sizeof(int) * (res+1)); CUERR;
			cudaMalloc(&d_rLengths, sizeof(int) * res); CUERR;
			cudaMalloc(&d_rIsEncoded, sizeof(int) * res); CUERR;
			cudaMalloc(&d_r2PerR1, sizeof(int) * (res+1)); CUERR;

			cudaMallocHost(&h_results, sizeof(AlignResultCompact) * res); CUERR;
			cudaMallocHost(&h_cBytesPrefixSum, sizeof(int) * (res+1)); CUERR;
			cudaMallocHost(&h_cLengths, sizeof(int) * res); CUERR;
			cudaMallocHost(&h_cIsEncoded, sizeof(int) * res); CUERR;
			cudaMallocHost(&h_rBytesPrefixSum, sizeof(int) * (res+1)); CUERR;
			cudaMallocHost(&h_rLengths, sizeof(int) * res); CUERR;
			cudaMallocHost(&h_rIsEncoded, sizeof(int) * res); CUERR;
			cudaMallocHost(&h_r2PerR1, sizeof(int) * (res+1)); CUERR;

			results_size = res;
		}

		if(ops > ops_size){
			cudaMalloc(&d_ops, sizeof(AlignOp) * ops); CUERR;
			cudaMallocHost(&h_ops, sizeof(AlignOp) * ops); CUERR;

			ops_size = ops;
		}

		if(r > r_size){
			cudaMalloc(&d_subjectsdata, sizeof(char) * r); CUERR;
			cudaMallocHost(&h_subjectsdata, sizeof(char) * r); CUERR;

			r_size = r;
		}

		if(c > c_size){
			cudaMalloc(&d_queriesdata, sizeof(char) * c); CUERR;
			cudaMallocHost(&h_queriesdata, sizeof(char) * c); CUERR;

			c_size = c;
		}
		#endif
	}


	void cuda_cleanup_AlignerDataArrays(AlignerDataArrays& data){
		#ifdef __NVCC__
			cudaSetDevice(data.deviceId); CUERR;

			cudaFree(data.d_results); CUERR;
			cudaFree(data.d_ops); CUERR;
			cudaFree(data.d_subjectsdata); CUERR;
			cudaFree(data.d_queriesdata); CUERR;
			cudaFree(data.d_rBytesPrefixSum); CUERR;
			cudaFree(data.d_rLengths); CUERR;
			cudaFree(data.d_rIsEncoded); CUERR;
			cudaFree(data.d_cBytesPrefixSum); CUERR;
			cudaFree(data.d_cLengths); CUERR;
			cudaFree(data.d_cIsEncoded); CUERR;
			cudaFree(data.d_r2PerR1); CUERR;

			cudaFreeHost(data.h_results); CUERR;
			cudaFreeHost(data.h_ops); CUERR;
			cudaFreeHost(data.h_subjectsdata); CUERR;
			cudaFreeHost(data.h_queriesdata); CUERR;
			cudaFreeHost(data.h_rBytesPrefixSum); CUERR;
			cudaFreeHost(data.h_rLengths); CUERR;
			cudaFreeHost(data.h_rIsEncoded); CUERR;
			cudaFreeHost(data.h_cBytesPrefixSum); CUERR;
			cudaFreeHost(data.h_cLengths); CUERR;
			cudaFreeHost(data.h_cIsEncoded); CUERR;
			cudaFreeHost(data.h_r2PerR1); CUERR;

			cudaFree(data.d_queriesPerSubject); CUERR;
			cudaFree(data.d_lengths); CUERR;

			cudaFreeHost(data.h_queriesPerSubject); CUERR;
			cudaFreeHost(data.h_lengths); CUERR;

			cudaStreamDestroy(data.stream); CUERR;
		#endif
	}

	void init_once(){
		correction::init_once();
	}

	std::vector<std::vector<AlignResult>> 
	getMultipleAlignments(AlignerDataArrays& mybuffers, const std::vector<const Sequence*>& subjects,
			   const std::vector<std::vector<const Sequence*>>& queries,
			   std::vector<bool> activeBatches, bool useGpu){	

		std::chrono::time_point<std::chrono::system_clock> tpa;
		std::chrono::time_point<std::chrono::system_clock> tpb;

		if(subjects.size() != queries.size()){
			throw std::runtime_error("graphtools::getMultipleAlignments incorrect input dimensions. queries.size() != candidates.size()");
		}

		int numberOfRealSubjects = 0;
		int totalNumberOfAlignments = 0;

		for(size_t i = 0; i < queries.size(); i++){
			if(activeBatches[i]){
				numberOfRealSubjects++;
				totalNumberOfAlignments += queries[i].size();
			}
		}

		std::vector<std::vector<AlignResult>> alignments(subjects.size());

		// check for empty input
		if(totalNumberOfAlignments == 0){
			return alignments;
		}

	#ifdef __NVCC__

		if(useGpu){ // use gpu for alignment

#if 1

			cudaSetDevice(mybuffers.deviceId); CUERR;

			//tpa = std::chrono::system_clock::now();

			int maximumCandidateLength = 0;
			int maximumQueryLength = 0;
			int totalCandidateBytes = 0;
			int totalQueryBytes = 0;

			/*determine required buffer sizes
			  resize buffers
			  collect data and write to buffers
			  copy data from CPU buffers to GPU buffers
			  run alignment kernel
			  copy data from GPU buffers to CPU buffers
			  make alignment results*/
		

			//determine buffer size

			for(size_t i = 0; i < subjects.size(); i++){
				if(activeBatches[i]){
					const auto& query = subjects[i];
					int bs = query->getNumBytes();		
					int ls = query->getNbases();
					maximumQueryLength = maximumQueryLength < ls ? ls : maximumQueryLength;
					totalQueryBytes += bs;

					assert(query->isCompressed());

					for(const auto& candidate : queries[i]){
						int b = candidate->getNumBytes();		
						int l = candidate->getNbases();
						maximumCandidateLength = maximumCandidateLength < l ? l : maximumCandidateLength;
						totalCandidateBytes += b;

						assert(candidate->isCompressed());
					}
				}
			}

			// resize buffers
			int ml = maximumCandidateLength;
			int max_ops_per_alignment = 2 * (ml + 1);
			int max_ops = max_ops_per_alignment * totalNumberOfAlignments;

			mybuffers.max_ops_per_alignment = max_ops_per_alignment;
			mybuffers.n_subjects = numberOfRealSubjects;
			mybuffers.n_queries = totalNumberOfAlignments;

			mybuffers.resize(totalNumberOfAlignments, // number of alignments
							max_ops, // maximum number of align ops 
							totalQueryBytes, // bytes of subjects
							totalCandidateBytes); // total number of candidate candidateBytes

			// write to buffers
			mybuffers.h_cBytesPrefixSum[0] = 0;
			mybuffers.h_rBytesPrefixSum[0] = 0;

			int candidateIndex = 0;
			size_t alignmentSubjectIndex = 0;
			for(size_t i = 0; i < subjects.size(); i++){
				if(activeBatches[i]){
					const auto& query = subjects[i];

					int bs = query->getNumBytes();		
					int ls = query->getNbases();

					maximumQueryLength = maximumQueryLength < ls ? ls : maximumQueryLength;

					mybuffers.h_rLengths[alignmentSubjectIndex] = ls;
					mybuffers.h_rBytesPrefixSum[alignmentSubjectIndex+1] = bs + mybuffers.h_rBytesPrefixSum[alignmentSubjectIndex];
					mybuffers.h_rIsEncoded[alignmentSubjectIndex] = query->isCompressed();

					memcpy(mybuffers.h_subjectsdata + mybuffers.h_rBytesPrefixSum[alignmentSubjectIndex], query->begin(), bs);	

					for(const auto& candidate : queries[i]){
						int b = candidate->getNumBytes();		
						int l = candidate->getNbases();

						mybuffers.h_cLengths[candidateIndex] = l;
						mybuffers.h_cBytesPrefixSum[candidateIndex+1] = b + mybuffers.h_cBytesPrefixSum[candidateIndex];
						mybuffers.h_cIsEncoded[candidateIndex] = candidate->isCompressed();

						memcpy(mybuffers.h_queriesdata + mybuffers.h_cBytesPrefixSum[candidateIndex], candidate->begin(), b);				
						candidateIndex++;
					}
					alignmentSubjectIndex++;
				}
			}

			mybuffers.maximumQueryLength = maximumQueryLength;
			mybuffers.maximumCandidateLength = maximumCandidateLength;

			mybuffers.h_r2PerR1[0] = 0;

			int r2perr1index = 1;
			for(size_t i = 0; i < queries.size(); i++){
				if(activeBatches[i]){
					mybuffers.h_r2PerR1[r2perr1index] = mybuffers.h_r2PerR1[r2perr1index-1] + queries[i].size();
					r2perr1index++;
				}
			}

			// copy data to gpu
			cudaMemcpyAsync(mybuffers.d_subjectsdata, 
					mybuffers.h_subjectsdata, 
					sizeof(char) * totalQueryBytes, 
					H2D, 
					mybuffers.stream); CUERR;
			cudaMemcpyAsync(mybuffers.d_queriesdata,
					mybuffers.h_queriesdata, 
					sizeof(char) * totalCandidateBytes, 
					H2D, 
					mybuffers.stream); CUERR;
			cudaMemcpyAsync(mybuffers.d_cBytesPrefixSum,
					mybuffers.h_cBytesPrefixSum, 
					sizeof(int) * (totalNumberOfAlignments + 1), 
					H2D, 
					mybuffers.stream); CUERR;
			cudaMemcpyAsync(mybuffers.d_cLengths,
					mybuffers.h_cLengths, 
					sizeof(int) * totalNumberOfAlignments, 
					H2D, 
					mybuffers.stream); CUERR;
			cudaMemcpyAsync(mybuffers.d_cIsEncoded,
					mybuffers.h_cIsEncoded, 
					sizeof(int) * totalNumberOfAlignments, 
					H2D, 
					mybuffers.stream); CUERR;
			cudaMemcpyAsync(mybuffers.d_rBytesPrefixSum,
					mybuffers.h_rBytesPrefixSum, 
					sizeof(int) * (numberOfRealSubjects + 1), 
					H2D, 
					mybuffers.stream); CUERR;
			cudaMemcpyAsync(mybuffers.d_rLengths,
					mybuffers.h_rLengths, 
					sizeof(int) * numberOfRealSubjects, 
					H2D, 
					mybuffers.stream); CUERR;
			cudaMemcpyAsync(mybuffers.d_rIsEncoded,
					mybuffers.h_rIsEncoded, 
					sizeof(int) * numberOfRealSubjects, 
					H2D, 
					mybuffers.stream); CUERR;
			cudaMemcpyAsync(mybuffers.d_r2PerR1,
					mybuffers.h_r2PerR1, 
					sizeof(int) * (numberOfRealSubjects + 1), 
					H2D, 
					mybuffers.stream); CUERR;

			alignment::call_cuda_semi_global_align_kernel(mybuffers);

			// copy result to host
			cudaMemcpyAsync(mybuffers.h_results, 
					mybuffers.d_results, 
					sizeof(AlignResultCompact) * mybuffers.n_queries, 
					D2H, 
					mybuffers.stream); CUERR;
			cudaMemcpyAsync(mybuffers.h_ops, 
					mybuffers.d_ops, 
					sizeof(AlignOp) * mybuffers.max_ops_per_alignment * mybuffers.n_queries, 
					D2H, 
					mybuffers.stream); CUERR;

			cudaStreamSynchronize(mybuffers.stream); CUERR;

			// store alignments in result vector

			candidateIndex = 0;
			int previousOutputIndex = -1;

			for(int i = 0; i < mybuffers.n_subjects; i++){

				int outputindex = previousOutputIndex + 1;
				while(!activeBatches[outputindex]) outputindex++;
				previousOutputIndex = outputindex;

				int nqueriesForThisSubject = mybuffers.h_r2PerR1[i+1] - mybuffers.h_r2PerR1[i];

				alignments[outputindex].resize(nqueriesForThisSubject);

				for(int j = 0; j < nqueriesForThisSubject; j++){
					const auto& currentResult = mybuffers.h_results[candidateIndex];
					alignments[outputindex][j].setOpsAndDataFromAlignResultCompact(currentResult, 
												mybuffers.h_ops + candidateIndex * mybuffers.max_ops_per_alignment, 
												true);
					candidateIndex++; 
				}			
			}

#else

			tpa = std::chrono::system_clock::now();

			cudaSetDevice(mybuffers.deviceId); CUERR;

			mybuffers.resize_new(numberOfRealSubjects, totalNumberOfAlignments);

			mybuffers.n_subjects = numberOfRealSubjects;
			mybuffers.n_queries = totalNumberOfAlignments;

			tpb = std::chrono::system_clock::now();
			
			mybuffers.resizetime += tpb - tpa;

			tpa = std::chrono::system_clock::now();

			int subjectindex = 0;
			int queryindex = 0;
			for(size_t i = 0; i < subjects.size(); i++){
				if(activeBatches[i]){
					assert(subjects[i]->getNbases() <= mybuffers.max_sequence_length);
					assert(subjects[i]->getNumBytes() <= mybuffers.max_sequence_bytes);

					std::memcpy(mybuffers.h_subjectsdata + i * mybuffers.sequencepitch,
						    subjects[i]->begin(), 
						    subjects[i]->getNumBytes());

					mybuffers.h_queriesPerSubject[subjectindex] = queries[i].size();
					mybuffers.h_lengths[subjectindex] = subjects[i]->getNbases();

					for(size_t j = 0; j < queries[i].size(); j++){
						assert(queries[i][j]->getNbases() <= mybuffers.max_sequence_length);
						assert(queries[i][j]->getNumBytes() <= mybuffers.max_sequence_bytes);

						std::memcpy(mybuffers.h_queriesdata + queryindex * mybuffers.sequencepitch,
							    queries[i][j]->begin(), 
							    queries[i][j]->getNumBytes());

						mybuffers.h_lengths[numberOfRealSubjects + queryindex] = queries[i][j]->getNbases();

						queryindex++;
					}

					subjectindex++;
				}
			}	

			tpb = std::chrono::system_clock::now();
		
			mybuffers.preprocessingtime += tpb - tpa;

			assert(subjectindex == numberOfRealSubjects);
			assert(queryindex == totalNumberOfAlignments);

			tpa = std::chrono::system_clock::now();

			// copy data to gpu
			cudaMemcpyAsync(mybuffers.d_subjectsdata, 
					mybuffers.h_subjectsdata, 
					mybuffers.sequencepitch * mybuffers.n_subjects, 
					H2D, 
					mybuffers.stream); CUERR;
			cudaMemcpyAsync(mybuffers.d_queriesdata,
					mybuffers.h_queriesdata, 
					mybuffers.sequencepitch * mybuffers.n_queries, 
					H2D, 
					mybuffers.stream); CUERR;
			cudaMemcpyAsync(mybuffers.d_queriesPerSubject,
					mybuffers.h_queriesPerSubject, 
					sizeof(int) * mybuffers.n_subjects, 
					H2D, 
					mybuffers.stream); CUERR;
			cudaMemcpyAsync(mybuffers.d_lengths,
					mybuffers.h_lengths, 
					sizeof(int) * (numberOfRealSubjects + totalNumberOfAlignments), 
					H2D, 
					mybuffers.stream); CUERR;

			cudaStreamSynchronize(mybuffers.stream);

			tpb = std::chrono::system_clock::now();
		
			mybuffers.h2dtime += tpb - tpa;

			tpa = std::chrono::system_clock::now();

			// start kernel
			alignment::call_cuda_semi_global_align_kernel_new(mybuffers);

			tpb = std::chrono::system_clock::now();
		
			mybuffers.alignmenttime += tpb - tpa;

			tpa = std::chrono::system_clock::now();

			// copy result to host
			cudaMemcpyAsync(mybuffers.h_results, 
					mybuffers.d_results, 
					sizeof(AlignResultCompact) * mybuffers.n_queries, 
					D2H, 
					mybuffers.stream); CUERR;
			cudaMemcpyAsync(mybuffers.h_ops, 
					mybuffers.d_ops, 
					sizeof(AlignOp) * mybuffers.max_ops_per_alignment * mybuffers.n_queries, 
					D2H, 
					mybuffers.stream); CUERR;

			cudaStreamSynchronize(mybuffers.stream); CUERR;

			tpb = std::chrono::system_clock::now();
		
			mybuffers.d2htime += tpb - tpa;

			tpa = std::chrono::system_clock::now();

			// store alignments in result vector

			int candidateIndex = 0;
			int previousOutputIndex = -1;

			for(int i = 0; i < mybuffers.n_subjects; i++){

				int outputindex = previousOutputIndex + 1;
				while(!activeBatches[outputindex]) outputindex++;
				previousOutputIndex = outputindex;

				int nqueriesForThisSubject = mybuffers.h_queriesPerSubject[i];

				alignments[outputindex].resize(nqueriesForThisSubject);

				for(int j = 0; j < nqueriesForThisSubject; j++){
					const auto& currentResult = mybuffers.h_results[candidateIndex];
					alignments[outputindex][j].setOpsAndDataFromAlignResultCompact(currentResult, 
												mybuffers.h_ops + candidateIndex * mybuffers.max_ops_per_alignment, 
												true);
					candidateIndex++; 
				}			
			}

			tpb = std::chrono::system_clock::now();
		
			mybuffers.postprocessingtime += tpb - tpa;



#endif

		}else{ // use cpu for alignment

	#endif 

			for(size_t i = 0; i < subjects.size(); i++){
				alignments[i].resize(queries[i].size());

				if(activeBatches[i]){
					const auto& query = subjects[i];
					const char* qdata = (const char*) query->begin();
					int qbases = query->getNbases();

					for(size_t j = 0; j < queries[i].size(); j++){
						const auto& c = queries[i][j];
						const char* cdata = (const char*)c->begin();
						int cbases = c->getNbases();

						alignments[i][j] = alignment::cpu_semi_global_alignment(mybuffers, qdata, cdata, qbases, cbases);
					}
				}
			}
	#ifdef __NVCC__
		}
	#endif

		return alignments;
	}


	std::tuple<std::chrono::duration<double>,std::chrono::duration<double>> performCorrection(std::string& subject,
				std::vector<AlignResult>& alignments,
				const std::string& subjectqualityScores, 
				const std::vector<const std::string*>& queryqualityScores,
				const std::vector<int>& frequenciesPrefixSum,
				bool useQScores,
				double MAX_MISMATCH_RATIO,
				double graphalpha,
				double graphx){

		return correction::correct_cpu(subject, alignments, subjectqualityScores, queryqualityScores, frequenciesPrefixSum,
				useQScores, MAX_MISMATCH_RATIO, graphalpha, graphx);

	}


} //namespace end
