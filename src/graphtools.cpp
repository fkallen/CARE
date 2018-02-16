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
						max_sequence_length(32 * SDIV(maxseqlength, 32)), //round up to multiple of 32
						max_sequence_bytes(SDIV(max_sequence_length,4)),
						max_ops_per_alignment(2 * (max_sequence_length + 1)){
		#ifdef __NVCC__
		cudaSetDevice(deviceId); CUERR;
		for(int i = 0; i < 8; i++)
			cudaStreamCreate(&streams[i]); CUERR;
		cudaStreamCreate(&stream); CUERR;
		cudaMalloc(&d_this, sizeof(AlignerDataArrays)); CUERR;
		#endif
	};

	void AlignerDataArrays::resize(int n_sub, int n_quer){
	#ifdef __NVCC__
		cudaSetDevice(deviceId); CUERR;

		bool resizeResult = false;

		if(n_sub > max_n_subjects){
			const int newmax = 1.5 * n_sub;
			size_t oldpitch = sequencepitch;
			cudaFree(d_subjectsdata); CUERR;
			cudaMallocPitch(&d_subjectsdata, &sequencepitch, max_sequence_bytes, newmax); CUERR;
			assert(!oldpitch || oldpitch == sequencepitch);

			cudaFree(d_queriesPerSubject); CUERR;
			cudaMalloc(&d_queriesPerSubject, sizeof(int) * newmax); CUERR;

			cudaFreeHost(h_subjectsdata); CUERR;
			cudaMallocHost(&h_subjectsdata, sequencepitch * newmax); CUERR;

			cudaFreeHost(h_queriesPerSubject); CUERR;
			cudaMallocHost(&h_queriesPerSubject, sizeof(int) * newmax); CUERR;
			
			cudaFree(d_subjectlengths); CUERR;
			cudaMalloc(&d_subjectlengths, sizeof(int) * newmax); CUERR;	
			
			cudaFreeHost(h_subjectlengths); CUERR;
			cudaMallocHost(&h_subjectlengths, sizeof(int) * newmax); CUERR;		

			max_n_subjects = newmax;

			resizeResult = true;
		}


		if(n_quer > max_n_queries){
			const int newmax = 1.5 * n_quer;
			size_t oldpitch = sequencepitch;
			cudaFree(d_queriesdata); CUERR;
			cudaMallocPitch(&d_queriesdata, &sequencepitch, max_sequence_bytes, newmax); CUERR;
			assert(!oldpitch || oldpitch == sequencepitch);

			cudaFreeHost(h_queriesdata); CUERR;
			cudaMallocHost(&h_queriesdata, sequencepitch * newmax); CUERR;
			
			cudaFree(d_querylengths); CUERR;
			cudaMalloc(&d_querylengths, sizeof(int) * newmax); CUERR;
			
			cudaFreeHost(h_querylengths); CUERR;
			cudaMallocHost(&h_querylengths, sizeof(int) * newmax); CUERR;			

			max_n_queries = newmax;

			resizeResult = true;
		}

		if(resizeResult){
			cudaFree(d_results); CUERR;
			cudaMalloc(&d_results, sizeof(AlignResultCompact) * max_n_subjects * max_n_queries); CUERR;

			cudaFreeHost(h_results); CUERR;
			cudaMallocHost(&h_results, sizeof(AlignResultCompact) * max_n_subjects * max_n_queries); CUERR;

			cudaFree(d_ops); CUERR;
			cudaFreeHost(h_ops); CUERR;

			cudaMalloc(&d_ops, sizeof(AlignOp) * max_n_queries * max_ops_per_alignment); CUERR;
			cudaMallocHost(&h_ops, sizeof(AlignOp) * max_n_queries * max_ops_per_alignment); CUERR;
			
			cudaFree(d_lengths); CUERR;
			cudaMalloc(&d_lengths, sizeof(int) * (max_n_subjects + max_n_queries)); CUERR;				
			cudaFreeHost(h_lengths); CUERR;
			cudaMallocHost(&h_lengths, sizeof(int) * (max_n_subjects + max_n_queries)); CUERR;

			cudaFree(d_newresults); CUERR;
			cudaMalloc(&d_newresults, sizeof(alignment::sgaresult) * max_n_subjects * max_n_queries); CUERR;

			cudaFreeHost(h_newresults); CUERR;
			cudaMallocHost(&h_newresults, sizeof(alignment::sgaresult) * max_n_subjects * max_n_queries); CUERR;

			cudaFree(d_newops); CUERR;
			cudaMalloc(&d_newops, sizeof(alignment::sgaop) * max_n_queries * max_ops_per_alignment); CUERR;

			cudaFreeHost(h_newops); CUERR;			
			cudaMallocHost(&h_newops, sizeof(alignment::sgaop) * max_n_queries * max_ops_per_alignment); CUERR;				
		}
	#endif
		n_subjects = n_sub;
		n_queries = n_quer;
	}


	void cuda_cleanup_AlignerDataArrays(AlignerDataArrays& data){
		#ifdef __NVCC__
			cudaSetDevice(data.deviceId); CUERR;

			cudaFree(data.d_results); CUERR;
			cudaFree(data.d_ops); CUERR;
			cudaFree(data.d_subjectsdata); CUERR;
			cudaFree(data.d_queriesdata); CUERR;
			cudaFree(data.d_queriesPerSubject); CUERR;
			cudaFree(data.d_subjectlengths); CUERR;
			cudaFree(data.d_querylengths); CUERR;

			cudaFreeHost(data.h_results); CUERR;
			cudaFreeHost(data.h_ops); CUERR;
			cudaFreeHost(data.h_subjectsdata); CUERR;
			cudaFreeHost(data.h_queriesdata); CUERR;		
			cudaFreeHost(data.h_queriesPerSubject); CUERR;
			cudaFreeHost(data.h_subjectlengths); CUERR;
			cudaFreeHost(data.h_querylengths); CUERR;

			cudaFree(data.d_newops); CUERR;
			cudaFree(data.d_newresults); CUERR;

			cudaFreeHost(data.h_newops); CUERR;
			cudaFreeHost(data.h_newresults); CUERR;

			cudaFree(data.d_this);
			
			for(int i = 0; i < 8; i++)
				cudaStreamDestroy(data.streams[i]); CUERR;
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
			
			tpa = std::chrono::system_clock::now();

			cudaSetDevice(mybuffers.deviceId); CUERR;

			mybuffers.resize(numberOfRealSubjects, totalNumberOfAlignments);

			mybuffers.n_subjects = numberOfRealSubjects;
			mybuffers.n_queries = totalNumberOfAlignments;

			tpb = std::chrono::system_clock::now();
			
			mybuffers.resizetime += tpb - tpa;

			tpa = std::chrono::system_clock::now();

			int subjectindex = 0;
			int querysum = 0;
			int batchid = 0;
			std::vector<alignment::sgaparams> params(subjects.size());
			for(size_t i = 0; i < subjects.size(); i++){
				if(activeBatches[i]){
					tpa = std::chrono::system_clock::now();
					batchid = subjectindex;
					
					params[batchid].max_sequence_length = mybuffers.max_sequence_length;
					params[batchid].max_ops_per_alignment = mybuffers.max_ops_per_alignment;
					params[batchid].sequencepitch = mybuffers.sequencepitch;
					params[batchid].subjectlength = subjects[i]->getNbases();
					params[batchid].n_queries = queries[i].size();
					params[batchid].querylengths = mybuffers.d_querylengths + querysum;
					params[batchid].subjectdata = mybuffers.d_subjectsdata + mybuffers.sequencepitch * subjectindex;
					params[batchid].queriesdata = mybuffers.d_queriesdata + mybuffers.sequencepitch * querysum;
					params[batchid].results = mybuffers.d_results + querysum;
					params[batchid].ops = mybuffers.d_ops + querysum * mybuffers.max_ops_per_alignment;					
					params[batchid].ALIGNMENTSCORE_MATCH = mybuffers.ALIGNMENTSCORE_MATCH;
					params[batchid].ALIGNMENTSCORE_SUB = mybuffers.ALIGNMENTSCORE_SUB;
					params[batchid].ALIGNMENTSCORE_INS = mybuffers.ALIGNMENTSCORE_INS;
					params[batchid].ALIGNMENTSCORE_DEL = mybuffers.ALIGNMENTSCORE_DEL;					
					
					int* querylengths = mybuffers.h_querylengths + querysum;
					char* subjectdata = mybuffers.h_subjectsdata + mybuffers.sequencepitch * subjectindex;
					char* queriesdata = mybuffers.h_queriesdata + mybuffers.sequencepitch * querysum;
					
					assert(subjects[i]->getNbases() <= mybuffers.max_sequence_length);
					
					std::memcpy(subjectdata, subjects[i]->begin(), subjects[i]->getNumBytes());

					for(size_t j = 0; j < queries[i].size(); j++){
						assert(queries[i][j]->getNbases() <= mybuffers.max_sequence_length);

						std::memcpy(queriesdata + j * mybuffers.sequencepitch,
							    queries[i][j]->begin(), 
							    queries[i][j]->getNumBytes());

						querylengths[j] = queries[i][j]->getNbases();
					}
					
					tpb = std::chrono::system_clock::now();
		
					mybuffers.preprocessingtime += tpb - tpa;
					// copy data to gpu
					cudaMemcpyAsync(const_cast<char*>(params[batchid].subjectdata), 
							subjectdata, 
							mybuffers.sequencepitch, 
							H2D, 
							mybuffers.streams[batchid]); CUERR;						
					cudaMemcpyAsync(const_cast<char*>(params[batchid].queriesdata),
							queriesdata, 
							mybuffers.sequencepitch * params[batchid].n_queries, 
							H2D, 
							mybuffers.streams[batchid]); CUERR;										
					cudaMemcpyAsync(const_cast<int*>(params[batchid].querylengths),
							querylengths, 
							sizeof(int) * params[batchid].n_queries, 
							H2D, 
							mybuffers.streams[batchid]); CUERR;						
					alignment::call_cuda_semi_global_alignment_kernel_async(params[batchid], mybuffers.streams[batchid]);
					CUERR;			
					querysum += queries[i].size();
					subjectindex++;					
				}
			}	
			
			subjectindex = 0;
			querysum = 0;
			for(size_t i = 0; i < subjects.size(); i++){
				if(activeBatches[i]){
					batchid = subjectindex;
					//copy results to host
					AlignResultCompact* results = mybuffers.h_results + querysum;
					AlignOp* ops = mybuffers.h_ops + querysum * mybuffers.max_ops_per_alignment;
					const int n_queries = params[batchid].n_queries;
					
					cudaMemcpyAsync(results, 
						params[batchid].results, 
						sizeof(AlignResultCompact) * n_queries, 
						D2H, 
						mybuffers.streams[batchid]); CUERR;
					cudaMemcpyAsync(ops, 
						params[batchid].ops, 
						sizeof(AlignOp) * n_queries * mybuffers.max_ops_per_alignment, 
						D2H, 
						mybuffers.streams[batchid]); CUERR;						
						
					cudaStreamSynchronize(mybuffers.streams[batchid]);
					
					tpa = std::chrono::system_clock::now();

					alignments[i].resize(n_queries);

					for(int j = 0; j < n_queries; j++){
						const auto& currentResult = results[j];
						alignments[i][j].setOpsAndDataFromAlignResultCompact(currentResult, 
												ops + j * mybuffers.max_ops_per_alignment, 
												true);						
					}							
	
					querysum += n_queries;
					subjectindex++;
					
					tpb = std::chrono::system_clock::now();
					
					mybuffers.postprocessingtime += tpb - tpa;				
				}
			}	

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

						alignments[i][j] = alignment::cpu_semi_global_alignment(&mybuffers, qdata, cdata, qbases, cbases);
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
