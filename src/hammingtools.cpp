#include "../inc/hammingtools.hpp"
#include "../inc/ganja/hpc_helpers.cuh"
#include "../inc/cudareduce.cuh"

#include "../inc/shd.hpp"
#include "../inc/pileup.hpp"

#include <stdexcept>
#include <cstdio>

#ifdef __NVCC__
#include <cublas_v2.h>
#endif

namespace hammingtools{

	int reserved_SMs = 1;



	SHDdata::SHDdata(int deviceId_, int cpuThreadsOnDevice, int maxseqlength) 
		: deviceId(deviceId_), max_sequence_length(maxseqlength), max_sequence_bytes(SDIV(maxseqlength,4)){
	#ifdef __NVCC__
		cudaSetDevice(deviceId); CUERR;
		for(int i = 0; i < 8; i++)
			cudaStreamCreate(&streams[i]); CUERR;
		cudaMalloc(&d_this, sizeof(SHDdata));

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, deviceId); CUERR;

		int numBlocksPerSM = 8;
		//cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, alignment::cuda_shifted_hamming_distance<256>, 256,0); CUERR;

		int mySMs = std::max(1, (prop.multiProcessorCount-1) / cpuThreadsOnDevice);
		shd_max_blocks = mySMs * numBlocksPerSM;
		//printf("shd_max_blocks = %d\n", shd_max_blocks);

	#endif

	};

	void SHDdata::resize(int n_sub, int n_quer){
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
			
			cudaFree(d_subjectlengths); CUERR;
			cudaMalloc(&d_subjectlengths, sizeof(int) * n_sub); CUERR;	
			
			cudaFreeHost(h_subjectlengths); CUERR;
			cudaMallocHost(&h_subjectlengths, sizeof(int) * n_sub); CUERR;
			
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
			
			cudaFree(d_querylengths); CUERR;
			cudaMalloc(&d_querylengths, sizeof(int) * n_quer); CUERR;
			
			cudaFreeHost(h_querylengths); CUERR;
			cudaMallocHost(&h_querylengths, sizeof(int) * n_quer); CUERR;

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
		}
	#endif
		n_subjects = n_sub;
		n_queries = n_quer;
	}

	void cuda_cleanup_SHDdata(SHDdata& data){
	#ifdef __NVCC__
		cudaSetDevice(data.deviceId); CUERR;

		cudaFree(data.d_results); CUERR;
		cudaFree(data.d_subjectsdata); CUERR;
		cudaFree(data.d_queriesdata); CUERR;
		cudaFree(data.d_queriesPerSubject); CUERR;
		cudaFree(data.d_subjectlengths); CUERR;
		cudaFree(data.d_querylengths); CUERR;
		cudaFree(data.d_this); CUERR;

		cudaFreeHost(data.h_results); CUERR;
		cudaFreeHost(data.h_subjectsdata); CUERR;
		cudaFreeHost(data.h_queriesdata); CUERR;
		cudaFreeHost(data.h_queriesPerSubject); CUERR;
		cudaFreeHost(data.h_subjectlengths); CUERR;
		cudaFreeHost(data.h_querylengths); CUERR;
		
		for(int i = 0; i < 8; i++)
			cudaStreamDestroy(data.streams[i]); CUERR;
	#endif
	}

	void print_SHDdata(const SHDdata& mybuffers){
		printf("d_results %p\n", mybuffers.d_results);
		printf("d_subjectsdata %p\n", mybuffers.d_subjectsdata);
		printf("d_queriesdata %p\n", mybuffers.d_queriesdata);
		printf("d_queriesPerSubject %p\n", mybuffers.d_queriesPerSubject);
		printf("h_results %p\n", mybuffers.h_results);
		printf("h_subjectsdata %p\n", mybuffers.h_subjectsdata);
		printf("h_queriesdata %p\n", mybuffers.h_queriesdata);
		printf("h_queriesPerSubject %p\n", mybuffers.h_queriesPerSubject);
	#ifdef __NVCC__
		printf("stream %p\n", mybuffers.streams[0]);
	#endif
		printf("deviceId %d\n", mybuffers.deviceId);
		printf("sequencepitch %lu\n", mybuffers.sequencepitch);
		printf("max_sequence_length %d\n", mybuffers.max_sequence_length);
		printf("max_sequence_bytes %d\n", mybuffers.max_sequence_bytes);
		printf("n_subjects %d\n", mybuffers.n_subjects);
		printf("n_queries %d\n", mybuffers.n_queries);
		printf("max_n_subjects %d\n", mybuffers.max_n_subjects);
		printf("max_n_queries %d\n", mybuffers.max_n_queries);
	}
	
	void init_once(){
        hammingtools::correction::init_once();
    }

	std::vector<std::vector<AlignResultCompact>> 
	getMultipleAlignments(SHDdata& mybuffers, 
					const std::vector<const Sequence*>& subjects,
					const std::vector<std::vector<const Sequence*>>& queries,
					std::vector<bool> activeBatches, bool useGpu){

		std::chrono::time_point<std::chrono::system_clock> tpa = std::chrono::system_clock::now();
		std::chrono::time_point<std::chrono::system_clock> tpb = std::chrono::system_clock::now();

		if(subjects.size() != queries.size()){
			throw std::runtime_error("hammingtools::getMultipleAlignments incorrect input dimensions. queries.size() != candidates.size()");
		}

		int numberOfRealSubjects = 0;
		int totalNumberOfAlignments = 0;

		for(size_t i = 0; i < queries.size(); i++){
			if(activeBatches[i]){
				numberOfRealSubjects++;
				totalNumberOfAlignments += queries[i].size();
			}
		}

		std::vector<std::vector<AlignResultCompact>> alignments(subjects.size());

		// check for empty input
		if(totalNumberOfAlignments == 0){
			return alignments;
		}

	#ifdef __NVCC__

		if(useGpu){ // use gpu for alignment
			
#if 0

			tpa = std::chrono::system_clock::now();

			cudaSetDevice(mybuffers.deviceId); CUERR;

			mybuffers.resize(numberOfRealSubjects, totalNumberOfAlignments);

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
					mybuffers.h_subjectlengths[subjectindex] = subjects[i]->getNbases();		

					for(size_t j = 0; j < queries[i].size(); j++){
						assert(queries[i][j]->getNbases() <= mybuffers.max_sequence_length);
						assert(queries[i][j]->getNumBytes() <= mybuffers.max_sequence_bytes);

						std::memcpy(mybuffers.h_queriesdata + queryindex * mybuffers.sequencepitch,
							    queries[i][j]->begin(), 
							    queries[i][j]->getNumBytes());

						mybuffers.h_querylengths[queryindex] = queries[i][j]->getNbases();

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
					mybuffers.streams[0]); CUERR;
			cudaMemcpyAsync(mybuffers.d_queriesdata,
					mybuffers.h_queriesdata, 
					mybuffers.sequencepitch * mybuffers.n_queries, 
					H2D, 
					mybuffers.streams[0]); CUERR;
			cudaMemcpyAsync(mybuffers.d_queriesPerSubject,
					mybuffers.h_queriesPerSubject, 
					sizeof(int) * mybuffers.n_subjects, 
					H2D, 
					mybuffers.streams[0]); CUERR;
			cudaMemcpyAsync(mybuffers.d_subjectlengths,
					mybuffers.h_subjectlengths, 
					sizeof(int) * (numberOfRealSubjects), 
					H2D, 
					mybuffers.streams[0]); CUERR;
					
			cudaMemcpyAsync(mybuffers.d_querylengths,
					mybuffers.h_querylengths, 
					sizeof(int) * (totalNumberOfAlignments), 
					H2D, 
					mybuffers.streams[0]); CUERR;					

			cudaStreamSynchronize(mybuffers.streams[0]);

			tpb = std::chrono::system_clock::now();
		
			mybuffers.h2dtime += tpb - tpa;

			tpa = std::chrono::system_clock::now();

			// start kernel
			for(int batchid = 0; batchid < numberOfRealSubjects; batchid++)
				alignment::call_shd_kernel_async(mybuffers, batchid);
			//alignment::call_shd_kernel(mybuffers, 0);
			
			for(int batchid = 0; batchid < numberOfRealSubjects; batchid++){
					cudaStreamSynchronize(mybuffers.streams[batchid]);
			}

			tpb = std::chrono::system_clock::now();
		
			mybuffers.alignmenttime += tpb - tpa;

			tpa = std::chrono::system_clock::now();

			//copy results to host
			cudaMemcpyAsync(mybuffers.h_results, 
				mybuffers.d_results, 
				sizeof(AlignResultCompact) * totalNumberOfAlignments, 
				D2H, 
				mybuffers.streams[0]); CUERR;

			cudaStreamSynchronize(mybuffers.streams[0]); CUERR;

			tpb = std::chrono::system_clock::now();
		
			mybuffers.d2htime += tpb - tpa;

			tpa = std::chrono::system_clock::now();

			int previousOutputIndex = -1;
			int candidateIndex = 0;

			//make function result
			for(int i = 0; i < numberOfRealSubjects; i++){

				int outputindex = previousOutputIndex + 1;
				while(!activeBatches[outputindex]) outputindex++;
				previousOutputIndex = outputindex;

				int nqueriesForThisSubject = mybuffers.h_queriesPerSubject[i];

				alignments[outputindex].resize(nqueriesForThisSubject);

				for(int j = 0; j < nqueriesForThisSubject; j++){
					alignments[outputindex][j] = mybuffers.h_results[candidateIndex];
					candidateIndex++; 
				}			
			}

			tpb = std::chrono::system_clock::now();
		
			mybuffers.postprocessingtime += tpb - tpa;
			
#else
			
			

			tpa = std::chrono::system_clock::now();

			cudaSetDevice(mybuffers.deviceId); CUERR;

			mybuffers.resize(numberOfRealSubjects, totalNumberOfAlignments);

			tpb = std::chrono::system_clock::now();
			
			mybuffers.resizetime += tpb - tpa;

			int querysum = 0;
			int subjectindex = 0;
			int batchid = 0;
			std::vector<alignment::shdparams> params(subjects.size());
			for(size_t i = 0; i < subjects.size(); i++){
				if(activeBatches[i]){
					tpa = std::chrono::system_clock::now();
					batchid = subjectindex;
					
					params[batchid].max_sequence_bytes = mybuffers.max_sequence_bytes;
					params[batchid].sequencepitch = mybuffers.sequencepitch;
					params[batchid].subjectlength = subjects[i]->getNbases();
					params[batchid].n_queries = queries[i].size();
					params[batchid].querylengths = mybuffers.d_querylengths + querysum;
					params[batchid].subjectdata = mybuffers.d_subjectsdata + mybuffers.sequencepitch * subjectindex;
					params[batchid].queriesdata = mybuffers.d_queriesdata + mybuffers.sequencepitch * querysum;
					params[batchid].results = mybuffers.d_results + querysum;
					
					int* querylengths = mybuffers.h_querylengths + querysum;
					char* subjectdata = mybuffers.h_subjectsdata + mybuffers.sequencepitch * subjectindex;
					char* queriesdata = mybuffers.h_queriesdata + mybuffers.sequencepitch * querysum;
					
					assert(subjects[i]->getNbases() <= mybuffers.max_sequence_length);
					assert(subjects[i]->getNumBytes() <= mybuffers.max_sequence_bytes);
					
					std::memcpy(subjectdata, subjects[i]->begin(), subjects[i]->getNumBytes());

					for(size_t j = 0; j < queries[i].size(); j++){
						assert(queries[i][j]->getNbases() <= mybuffers.max_sequence_length);
						assert(queries[i][j]->getNumBytes() <= mybuffers.max_sequence_bytes);

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
							
					alignment::call_shd_kernel_async(params[batchid], mybuffers.streams[batchid]);
								
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
					const int n_queries = params[batchid].n_queries;
					
					cudaMemcpyAsync(results, 
						params[batchid].results, 
						sizeof(AlignResultCompact) * n_queries, 
						D2H, 
						mybuffers.streams[batchid]); CUERR;
						
					cudaStreamSynchronize(mybuffers.streams[batchid]);
					
					tpa = std::chrono::system_clock::now();

					alignments[i].resize(n_queries);

					for(int j = 0; j < n_queries; j++){
						alignments[i][j] = results[j];
					}							
	
					querysum += n_queries;
					subjectindex++;
					
					tpb = std::chrono::system_clock::now();
					
					mybuffers.postprocessingtime += tpb - tpa;				
				}
			}			
			
#endif			

		}else{ // use cpu for alignment



	#endif // __NVCC__

			tpa = std::chrono::system_clock::now();

			for(size_t i = 0; i < subjects.size(); i++){
				alignments[i].resize(queries[i].size());

				if(activeBatches[i]){
					const char* subject = (const char*)subjects[i]->begin();
					int ns = subjects[i]->getNbases();
					for(size_t j = 0; j < queries[i].size(); j++){
						const char* query =  (const char*)queries[i][j]->begin();
						int nq = queries[i][j]->getNbases();

						alignments[i][j] = alignment::cpu_shifted_hamming_distance(subject, query, ns, nq);
					}
				}
			}

			tpb = std::chrono::system_clock::now();
		
			mybuffers.alignmenttime += tpb - tpa;
	#ifdef __NVCC__
		}
	#endif // __NVCC__

		return alignments;
	}

#if 1



	CorrectionBuffers::CorrectionBuffers(int id, int maxseqlength){
		deviceId = id;
		max_seq_length = maxseqlength;
		//printf("CorrectionBuffers id %d maxseqlength %d\n", id, maxseqlength);
	#ifdef __NVCC__
		cudaSetDevice(deviceId); CUERR;
		cudaStreamCreate(&stream); CUERR;
		cudaMalloc(&d_this, sizeof(CorrectionBuffers)); CUERR;
		//cublasCreate(&handle);
	#endif
	}

	void CorrectionBuffers::resize_host_cols(int cols){
		if(cols > max_n_columns){
			const int newmaxcols = 1.5 * cols;
#ifdef __NVCC__
			cudaFreeHost(h_consensus); CUERR;
			cudaFreeHost(h_support); CUERR;
			cudaFreeHost(h_coverage); CUERR;
			cudaFreeHost(h_origWeights); CUERR;
			cudaFreeHost(h_origCoverage); CUERR;
			cudaFreeHost(h_As); CUERR;
			cudaFreeHost(h_Cs); CUERR;
			cudaFreeHost(h_Gs); CUERR;
			cudaFreeHost(h_Ts); CUERR;
			cudaFreeHost(h_Aweights); CUERR;
			cudaFreeHost(h_Cweights); CUERR;
			cudaFreeHost(h_Gweights); CUERR;
			cudaFreeHost(h_Tweights); CUERR;

			cudaMallocHost(&h_consensus, sizeof(char) * newmaxcols); CUERR;
			cudaMallocHost(&h_support, sizeof(double) * newmaxcols); CUERR;
			cudaMallocHost(&h_coverage, sizeof(int) * newmaxcols); CUERR;
			cudaMallocHost(&h_origWeights, sizeof(double) * newmaxcols); CUERR;
			cudaMallocHost(&h_origCoverage, sizeof(int) * newmaxcols); CUERR;
			cudaMallocHost(&h_As, sizeof(int) * newmaxcols); CUERR;
			cudaMallocHost(&h_Cs, sizeof(int) * newmaxcols); CUERR;
			cudaMallocHost(&h_Gs, sizeof(int) * newmaxcols); CUERR;
			cudaMallocHost(&h_Ts, sizeof(int) * newmaxcols); CUERR;
			cudaMallocHost(&h_Aweights, sizeof(double) * newmaxcols); CUERR;
			cudaMallocHost(&h_Cweights, sizeof(double) * newmaxcols); CUERR;
			cudaMallocHost(&h_Gweights, sizeof(double) * newmaxcols); CUERR;
			cudaMallocHost(&h_Tweights, sizeof(double) * newmaxcols); CUERR;
#else

			delete [] h_consensus;
			delete [] h_support;
			delete [] h_coverage;
			delete [] h_origWeights;
			delete [] h_origCoverage;
			delete [] h_As;
			delete [] h_Cs;
			delete [] h_Gs;
			delete [] h_Ts;
			delete [] h_Aweights;
			delete [] h_Cweights;
			delete [] h_Gweights;
			delete [] h_Tweights;

			h_consensus = new char[newmaxcols];
			h_support = new double[newmaxcols];
			h_coverage = new int[newmaxcols];
			h_origWeights = new double[newmaxcols];
			h_origCoverage = new int[newmaxcols];
			h_As = new int[newmaxcols];
			h_Cs = new int[newmaxcols];
			h_Gs = new int[newmaxcols];
			h_Ts = new int[newmaxcols];
			h_Aweights = new double[newmaxcols];
			h_Cweights = new double[newmaxcols];
			h_Gweights = new double[newmaxcols];
			h_Tweights = new double[newmaxcols];
#endif
			max_n_columns = newmaxcols;
		}

		n_columns = cols;
	}

	void CorrectionBuffers::resize(int cols, int nsequences, int nqualityscores){
#ifdef __NVCC__
		cudaSetDevice(deviceId); CUERR;
#endif
		bool resizepileup = false;
		bool resizequalpileup;
		if(cols > max_n_columns){
			const int newmaxcols = 1.5 * cols;
#ifdef __NVCC__
			cudaFree(d_consensus); CUERR;
			cudaFree(d_support); CUERR;
			cudaFree(d_coverage); CUERR;
			cudaFree(d_origWeights); CUERR;
			cudaFree(d_origCoverage); CUERR;
			cudaFree(d_As); CUERR;
			cudaFree(d_Cs); CUERR;
			cudaFree(d_Gs); CUERR;
			cudaFree(d_Ts); CUERR;
			cudaFree(d_Aweights); CUERR;
			cudaFree(d_Cweights); CUERR;
			cudaFree(d_Gweights); CUERR;
			cudaFree(d_Tweights); CUERR;

			cudaFreeHost(h_consensus); CUERR;
			cudaFreeHost(h_support); CUERR;
			cudaFreeHost(h_coverage); CUERR;
			cudaFreeHost(h_origWeights); CUERR;
			cudaFreeHost(h_origCoverage); CUERR;
			cudaFreeHost(h_As); CUERR;
			cudaFreeHost(h_Cs); CUERR;
			cudaFreeHost(h_Gs); CUERR;
			cudaFreeHost(h_Ts); CUERR;
			cudaFreeHost(h_Aweights); CUERR;
			cudaFreeHost(h_Cweights); CUERR;
			cudaFreeHost(h_Gweights); CUERR;
			cudaFreeHost(h_Tweights); CUERR;

			cudaMalloc(&d_consensus, sizeof(char) * newmaxcols); CUERR;
			cudaMalloc(&d_support, sizeof(double) * newmaxcols); CUERR;
			cudaMalloc(&d_coverage, sizeof(int) * newmaxcols); CUERR;
			cudaMalloc(&d_origWeights, sizeof(double) * newmaxcols); CUERR;
			cudaMalloc(&d_origCoverage, sizeof(int) * newmaxcols); CUERR;
			cudaMalloc(&d_As, sizeof(int) * newmaxcols); CUERR;
			cudaMalloc(&d_Cs, sizeof(int) * newmaxcols); CUERR;
			cudaMalloc(&d_Gs, sizeof(int) * newmaxcols); CUERR;
			cudaMalloc(&d_Ts, sizeof(int) * newmaxcols); CUERR;
			cudaMalloc(&d_Aweights, sizeof(double) * newmaxcols); CUERR;
			cudaMalloc(&d_Cweights, sizeof(double) * newmaxcols); CUERR;
			cudaMalloc(&d_Gweights, sizeof(double) * newmaxcols); CUERR;
			cudaMalloc(&d_Tweights, sizeof(double) * newmaxcols); CUERR;

			cudaMallocHost(&h_consensus, sizeof(char) * newmaxcols); CUERR;
			cudaMallocHost(&h_support, sizeof(double) * newmaxcols); CUERR;
			cudaMallocHost(&h_coverage, sizeof(int) * newmaxcols); CUERR;
			cudaMallocHost(&h_origWeights, sizeof(double) * newmaxcols); CUERR;
			cudaMallocHost(&h_origCoverage, sizeof(int) * newmaxcols); CUERR;
			cudaMallocHost(&h_As, sizeof(int) * newmaxcols); CUERR;
			cudaMallocHost(&h_Cs, sizeof(int) * newmaxcols); CUERR;
			cudaMallocHost(&h_Gs, sizeof(int) * newmaxcols); CUERR;
			cudaMallocHost(&h_Ts, sizeof(int) * newmaxcols); CUERR;
			cudaMallocHost(&h_Aweights, sizeof(double) * newmaxcols); CUERR;
			cudaMallocHost(&h_Cweights, sizeof(double) * newmaxcols); CUERR;
			cudaMallocHost(&h_Gweights, sizeof(double) * newmaxcols); CUERR;
			cudaMallocHost(&h_Tweights, sizeof(double) * newmaxcols); CUERR;
#else

			delete [] h_consensus;
			delete [] h_support;
			delete [] h_coverage;
			delete [] h_origWeights;
			delete [] h_origCoverage;
			delete [] h_As;
			delete [] h_Cs;
			delete [] h_Gs;
			delete [] h_Ts;
			delete [] h_Aweights;
			delete [] h_Cweights;
			delete [] h_Gweights;
			delete [] h_Tweights;

			h_consensus = new char[newmaxcols];
			h_support = new double[newmaxcols];
			h_coverage = new int[newmaxcols];
			h_origWeights = new double[newmaxcols];
			h_origCoverage = new int[newmaxcols];
			h_As = new int[newmaxcols];
			h_Cs = new int[newmaxcols];
			h_Gs = new int[newmaxcols];
			h_Ts = new int[newmaxcols];
			h_Aweights = new double[newmaxcols];
			h_Cweights = new double[newmaxcols];
			h_Gweights = new double[newmaxcols];
			h_Tweights = new double[newmaxcols];
#endif
			max_n_columns = newmaxcols;
			resizepileup = true;
			resizequalpileup = true;
		}

		if(nsequences > max_n_sequences){
			const int newmaxseqs = 1.5 * nsequences;

#ifdef __NVCC__
			cudaFree(d_lengths); CUERR;
			cudaFree(d_alignments); CUERR;
			cudaFree(d_frequencies_prefix_sum); CUERR;
			cudaFreeHost(h_lengths); CUERR;
			cudaFreeHost(h_alignments); CUERR;
			cudaFreeHost(h_frequencies_prefix_sum); CUERR;

			cudaMalloc(&d_lengths, sizeof(int) * newmaxseqs); CUERR;
			cudaMalloc(&d_alignments, sizeof(AlignResultCompact) * newmaxseqs); CUERR;
			cudaMalloc(&d_frequencies_prefix_sum, sizeof(int) * newmaxseqs); CUERR;
			cudaMallocHost(&h_lengths, sizeof(int) * newmaxseqs); CUERR;
			cudaMallocHost(&h_alignments, sizeof(AlignResultCompact) * newmaxseqs); CUERR;
			cudaMallocHost(&h_frequencies_prefix_sum, sizeof(int) * (newmaxseqs+1)); CUERR;
#else
			delete [] h_lengths;
			h_lengths = new int[newmaxseqs];
			


#endif
			max_n_sequences = newmaxseqs;
			resizepileup = true;
		}

		if(nqualityscores > max_n_qualityscores){
			const int newmaxqscores = 1.5 * nqualityscores;

			max_n_qualityscores = newmaxqscores;

			resizequalpileup = true;
		}

		if(resizepileup){
#ifdef __NVCC__
			cudaFree(d_pileup); CUERR;
			cudaFree(d_pileup_transposed); CUERR;
			cudaFreeHost(h_pileup); CUERR;

			cudaMalloc(&d_pileup, sizeof(char) * max_n_sequences * max_n_columns); CUERR;
			cudaMalloc(&d_pileup_transposed, sizeof(char) * max_n_sequences * max_n_columns); CUERR;
			cudaMallocHost(&h_pileup, sizeof(char) * max_n_sequences * max_n_columns); CUERR;
#else
			delete [] h_pileup;
			h_pileup = new char[max_n_sequences * max_n_columns];

#endif
		}

		if(resizequalpileup){
#ifdef __NVCC__
			cudaFree(d_qual_pileup); CUERR;
			cudaFree(d_qual_pileup_transposed); CUERR;
			cudaFreeHost(h_qual_pileup); CUERR;

			cudaMalloc(&d_qual_pileup, sizeof(char) * max_n_qualityscores * max_n_columns); CUERR;
			cudaMalloc(&d_qual_pileup_transposed, sizeof(char) * max_n_qualityscores * max_n_columns); CUERR;
			cudaMallocHost(&h_qual_pileup, sizeof(char) * max_n_qualityscores * max_n_columns); CUERR;
#else
			delete [] h_qual_pileup;
			h_qual_pileup = new char[max_n_sequences * max_n_columns];

#endif
		}


		n_columns = cols;
		n_sequences = nsequences;
		n_qualityscores = nqualityscores;
	}

	void cuda_cleanup_CorrectionBuffers(CorrectionBuffers& buffers){
	#ifdef __NVCC__
			cudaSetDevice(buffers.deviceId); CUERR;

			cudaFree(buffers.d_consensus); CUERR;
			cudaFree(buffers.d_support); CUERR;
			cudaFree(buffers.d_coverage); CUERR;
			cudaFree(buffers.d_origWeights); CUERR;
			cudaFree(buffers.d_origCoverage); CUERR;
			cudaFree(buffers.d_As); CUERR;
			cudaFree(buffers.d_Cs); CUERR;
			cudaFree(buffers.d_Gs); CUERR;
			cudaFree(buffers.d_Ts); CUERR;
			cudaFree(buffers.d_Aweights); CUERR;
			cudaFree(buffers.d_Cweights); CUERR;
			cudaFree(buffers.d_Gweights); CUERR;
			cudaFree(buffers.d_Tweights); CUERR;
			cudaFree(buffers.d_pileup); CUERR;
			cudaFree(buffers.d_pileup_transposed); CUERR;
			cudaFree(buffers.d_qual_pileup); CUERR;
			cudaFree(buffers.d_qual_pileup_transposed); CUERR;

			cudaFree(buffers.d_lengths); CUERR;
			cudaFree(buffers.d_alignments); CUERR;
			cudaFree(buffers.d_frequencies_prefix_sum); CUERR;
			cudaFreeHost(buffers.h_lengths); CUERR;
			cudaFreeHost(buffers.h_alignments); CUERR;
			cudaFreeHost(buffers.h_frequencies_prefix_sum); CUERR;
			cudaFreeHost(buffers.h_pileup); CUERR;
			cudaFreeHost(buffers.h_qual_pileup); CUERR;

			cudaFree(buffers.d_this); CUERR;

			cudaStreamDestroy(buffers.stream); CUERR;
	#else
			delete [] buffers.h_pileup;
			delete [] buffers.h_qual_pileup;
			delete [] buffers.h_consensus;
			delete [] buffers.h_support;
			delete [] buffers.h_coverage;
			delete [] buffers.h_origWeights;
			delete [] buffers.h_origCoverage;
			delete [] buffers.h_As;
			delete [] buffers.h_Cs;
			delete [] buffers.h_Gs;
			delete [] buffers.h_Ts;
			delete [] buffers.h_Aweights;
			delete [] buffers.h_Cweights;
			delete [] buffers.h_Gweights;
			delete [] buffers.h_Tweights;
			delete [] buffers.h_lengths;

	#endif

			buffers.max_n_columns = 0;
			buffers.max_n_sequences = 0;
			buffers.max_n_qualityscores = 0;

	}

#endif

	std::tuple<int,std::chrono::duration<double>,std::chrono::duration<double>>
	performCorrection(CorrectionBuffers& buffers, std::string& subject,
				int nQueries, 
				std::vector<std::string>& queries,
				const std::vector<AlignResultCompact>& alignments,
				const std::string& subjectqualityScores, 
				const std::vector<const std::string*>& queryqualityScores,
				const std::vector<int>& frequenciesPrefixSum,
				double maxErrorRate,
				bool useQScores,
				std::vector<bool>& correctedQueries,
				bool correctQueries_,
				int estimatedCoverage,
				double errorrate,
				double m,
				int kmerlength,
				bool useGpu){

			std::chrono::duration<double> majorityvotetime(0);
			std::chrono::duration<double> basecorrectiontime(0);
			std::chrono::time_point<std::chrono::system_clock> tpa, tpb, tpc, tpd;


//#ifdef __NVCC__
#if 0
		if(useGpu){

			assert(buffers.max_seq_length  > 0);



			tpc = std::chrono::system_clock::now();

			int startindex = 0;
			int endindex = subject.length();
			for(int i = 0; i < nQueries; i++){
				startindex = alignments[i].shift < startindex ? alignments[i].shift : startindex;
				const int queryEndsAt = queryqualityScores[i]->length() + alignments[i].shift;
				endindex = queryEndsAt > endindex ? queryEndsAt : endindex;
				correctedQueries[i] = false;
			}

			const int columnsToCheck = endindex - startindex;
			const int subjectColumnsBegin_incl = std::max(-startindex,0);
			const int subjectColumnsEnd_excl = subjectColumnsBegin_incl + subject.length();

			tpa = std::chrono::system_clock::now();

			

			buffers.resize(columnsToCheck, frequenciesPrefixSum.back()+1, frequenciesPrefixSum.back()+1);

			tpd = std::chrono::system_clock::now();
			buffers.resizetime += tpd - tpc;
			tpc = std::chrono::system_clock::now();

			buffers.h_lengths[0] = subject.length();
			for(int i = 0; i < nQueries; i++){
				buffers.h_lengths[i+1] = queries[i].length();		
			}

			std::memcpy(buffers.h_alignments, alignments.data(), sizeof(AlignResultCompact) * nQueries);
			std::memcpy(buffers.h_frequencies_prefix_sum, frequenciesPrefixSum.data(), sizeof(int) * (nQueries+1));

			correction::cpu_pileup_create(&buffers, subject,
						queries,
						subjectqualityScores, 
						queryqualityScores,
						alignments,
						frequenciesPrefixSum,
						subjectColumnsBegin_incl);

			correction::gpu_pileup_transpose(&buffers);
			correction::gpu_qual_pileup_transpose(&buffers);

			buffers.avg_support = 0;
			buffers.min_support = 0;
			buffers.max_coverage = 0;
			buffers.min_coverage = 0;

			tpd = std::chrono::system_clock::now();
			buffers.preprocessingtime += tpd - tpc;
			tpc = std::chrono::system_clock::now();

			// copy buffers to gpu
			cudaMemcpyAsync(buffers.d_lengths, 
					buffers.h_lengths, 
					sizeof(int) * (nQueries+1), 
					H2D, buffers.stream); CUERR;
			cudaMemcpyAsync(buffers.d_alignments, 
					buffers.h_alignments, 
					sizeof(AlignResultCompact) * nQueries, 
					H2D, buffers.stream); CUERR;
			cudaMemcpyAsync(buffers.d_frequencies_prefix_sum, 
					buffers.h_frequencies_prefix_sum, 
					sizeof(int) * (nQueries+1), 
					H2D, buffers.stream); CUERR;

			cudaStreamSynchronize(buffers.stream);

			tpd = std::chrono::system_clock::now();
			buffers.h2dtime += tpd - tpc;
			tpc = std::chrono::system_clock::now();

			printf("%d %d %d %d %d %d %d\n", buffers.max_n_columns, buffers.n_columns, buffers.max_n_sequences, buffers.n_sequences, buffers.max_n_qualityscores, buffers.n_qualityscores, buffers.max_seq_length);

			correction::call_cuda_pileup_vote_transposed_kernel(&buffers, nQueries+1, frequenciesPrefixSum.back()+1, columnsToCheck, 
						subjectColumnsBegin_incl, subjectColumnsEnd_excl,
						maxErrorRate, 
						useQScores);

			cudaStreamSynchronize(buffers.stream); CUERR;

			tpd = std::chrono::system_clock::now();
			buffers.alignmenttime += tpd - tpc;
			tpc = std::chrono::system_clock::now();

			cudaMemcpyAsync(&buffers, buffers.d_this, sizeof(CorrectionBuffers), D2H, buffers.stream); CUERR;
			cudaMemcpyAsync(buffers.h_consensus, buffers.d_consensus, sizeof(char) * buffers.n_columns, D2H, buffers.stream); CUERR;
			cudaMemcpyAsync(buffers.h_support, buffers.d_support, sizeof(double) * buffers.n_columns, D2H, buffers.stream); CUERR;
			cudaMemcpyAsync(buffers.h_coverage, buffers.d_coverage, sizeof(int) * buffers.n_columns, D2H, buffers.stream); CUERR;
			cudaMemcpyAsync(buffers.h_origWeights, buffers.d_origWeights, sizeof(double) * buffers.n_columns, D2H, buffers.stream); CUERR;
			cudaMemcpyAsync(buffers.h_origCoverage, buffers.d_origCoverage, sizeof(int) * buffers.n_columns, D2H, buffers.stream); CUERR;

			cudaStreamSynchronize(buffers.stream); CUERR;

			tpd = std::chrono::system_clock::now();
			buffers.d2htime += tpd - tpc;

			tpb = std::chrono::system_clock::now();

			majorityvotetime += tpb - tpa;

			tpa = std::chrono::system_clock::now();

			int status = correction::cpu_pileup_correct(&buffers, nQueries, columnsToCheck, 
						subjectColumnsBegin_incl, subjectColumnsEnd_excl,
						startindex, endindex,
						errorrate, estimatedCoverage, m, 
						correctQueries_, kmerlength, correctedQueries);

			tpb = std::chrono::system_clock::now();
			basecorrectiontime += tpb - tpa;

			tpc = std::chrono::system_clock::now();

			subject.replace(0, buffers.h_lengths[0], buffers.h_pileup + subjectColumnsBegin_incl, buffers.h_lengths[0]);
			for(int i = 0; i < nQueries; i++){
				if(correctedQueries[i]){
					queries[i].replace(0, buffers.h_lengths[1+i], 
								buffers.h_pileup + (1+i) * buffers.max_n_columns 
									+ subjectColumnsBegin_incl + alignments[i].shift, 
								buffers.h_lengths[1+i]);
				}
			}

			tpd = std::chrono::system_clock::now();
			buffers.postprocessingtime += tpd - tpc;

			return std::tie(status, majorityvotetime, basecorrectiontime);

			//std::string gpuresult(buffers.h_sequences, buffers.h_lengths[0]);

			/*std::cout << "avgsup gpu " << buffers.avg_support << " >= " << (1-errorrate) << '\n';
			std::cout << "minsup gpu " << buffers.min_support << " >= " << (1-2*errorrate) << '\n';
			std::cout << "mincov gpu " << buffers.min_coverage << " >= " << (m) << '\n';
			std::cout << "maxcov gpu " << buffers.max_coverage << " <= " << (3*m) << '\n';*/

		}else{
#endif	

//#else




#if 0

			//determine number of columns in pileup image
			int startindex = 0;
			int endindex = subject.length();
			for(int i = 0; i < nQueries; i++){
				startindex = alignments[i].shift < startindex ? alignments[i].shift : startindex;
				const int queryEndsAt = queryqualityScores[i]->length() + alignments[i].shift;
				endindex = queryEndsAt > endindex ? queryEndsAt : endindex;
				correctedQueries[i] = false;
			}

			const int columnsToCheck = endindex - startindex;
			const int subjectColumnsBegin_incl = std::max(-startindex,0);
			const int subjectColumnsEnd_excl = subjectColumnsBegin_incl + subject.length();

			//tpa = std::chrono::system_clock::now();

			buffers.resize(columnsToCheck, frequenciesPrefixSum.back()+1, frequenciesPrefixSum.back()+1);

			std::memset(buffers.h_consensus, 0, sizeof(char) * buffers.n_columns);
			std::memset(buffers.h_support, 0, sizeof(double) * buffers.n_columns);
			std::memset(buffers.h_coverage, 0, sizeof(int) * buffers.n_columns);
			std::memset(buffers.h_origWeights, 0, sizeof(double) * buffers.n_columns);
			std::memset(buffers.h_origCoverage, 0, sizeof(int) * buffers.n_columns);
			std::memset(buffers.h_As, 0, sizeof(int) * buffers.n_columns);
			std::memset(buffers.h_Cs, 0, sizeof(int) * buffers.n_columns);
			std::memset(buffers.h_Gs, 0, sizeof(int) * buffers.n_columns);
			std::memset(buffers.h_Ts, 0, sizeof(int) * buffers.n_columns);
			std::memset(buffers.h_Aweights, 0, sizeof(double) * buffers.n_columns);
			std::memset(buffers.h_Cweights, 0, sizeof(double) * buffers.n_columns);
			std::memset(buffers.h_Gweights, 0, sizeof(double) * buffers.n_columns);
			std::memset(buffers.h_Tweights, 0, sizeof(double) * buffers.n_columns);

			buffers.h_lengths[0] = subject.length();
			for(int i = 0; i < nQueries; i++){
				const int freqSum = frequenciesPrefixSum[i];
				const int freqSumNext = frequenciesPrefixSum[i+1];
				const int freq = freqSumNext - freqSum;

				for(int f = 0; f < freq; f++){
					const int row = 1 + freqSum + f;
					buffers.h_lengths[row] = queries[i].length();
				}
			}



			tpa = std::chrono::system_clock::now();

			correction::cpu_pileup_create(&buffers, subject,
						queries,
						subjectqualityScores, 
						queryqualityScores,
						alignments,
						frequenciesPrefixSum,
						subjectColumnsBegin_incl);

			correction::cpu_pileup_vote(&buffers, alignments,
						frequenciesPrefixSum,
						maxErrorRate, 
						useQScores,
						subjectColumnsBegin_incl, subjectColumnsEnd_excl);

			tpb = std::chrono::system_clock::now();
			majorityvotetime = tpb - tpa;

			tpa = std::chrono::system_clock::now();

			int status = correction::cpu_pileup_correct(&buffers, alignments, frequenciesPrefixSum,
						columnsToCheck, 
						subjectColumnsBegin_incl, subjectColumnsEnd_excl,
						startindex, endindex,
						errorrate, estimatedCoverage,m, 
						correctQueries_, kmerlength, correctedQueries);

			tpb = std::chrono::system_clock::now();
			basecorrectiontime = tpb - tpa;

			tpc = std::chrono::system_clock::now();

			subject.replace(0, buffers.h_lengths[0], buffers.h_pileup + subjectColumnsBegin_incl, buffers.h_lengths[0]);
			for(int i = 0; i < nQueries; i++){
				if(correctedQueries[i]){
					const int row = 1 + frequenciesPrefixSum[i];
					queries[i].replace(0, buffers.h_lengths[row], 
								buffers.h_pileup + (row) * buffers.max_n_columns 
									+ subjectColumnsBegin_incl + alignments[i].shift, 
								buffers.h_lengths[row]);
				}
			}

			return std::tie(status, majorityvotetime, basecorrectiontime);
#else

			tpc = std::chrono::system_clock::now();

			//determine number of columns in pileup image
			int startindex = 0;
			int endindex = subject.length();
			for(int i = 0; i < nQueries; i++){
				startindex = alignments[i].shift < startindex ? alignments[i].shift : startindex;
				const int queryEndsAt = queries[i].length() + alignments[i].shift;
				endindex = queryEndsAt > endindex ? queryEndsAt : endindex;
				correctedQueries[i] = false;
			}

			const int columnsToCheck = endindex - startindex;
			const int subjectColumnsBegin_incl = std::max(-startindex,0);
			const int subjectColumnsEnd_excl = subjectColumnsBegin_incl + subject.length();

			tpd = std::chrono::system_clock::now();
			buffers.preprocessingtime += tpd - tpc;
			tpc = std::chrono::system_clock::now();

			buffers.resize_host_cols(columnsToCheck);

			tpd = std::chrono::system_clock::now();
			buffers.resizetime += tpd - tpc;
			tpc = std::chrono::system_clock::now();

			std::memset(buffers.h_consensus, 0, sizeof(char) * buffers.n_columns);
			std::memset(buffers.h_support, 0, sizeof(double) * buffers.n_columns);
			std::memset(buffers.h_coverage, 0, sizeof(int) * buffers.n_columns);
			std::memset(buffers.h_origWeights, 0, sizeof(double) * buffers.n_columns);
			std::memset(buffers.h_origCoverage, 0, sizeof(int) * buffers.n_columns);
			std::memset(buffers.h_As, 0, sizeof(int) * buffers.n_columns);
			std::memset(buffers.h_Cs, 0, sizeof(int) * buffers.n_columns);
			std::memset(buffers.h_Gs, 0, sizeof(int) * buffers.n_columns);
			std::memset(buffers.h_Ts, 0, sizeof(int) * buffers.n_columns);
			std::memset(buffers.h_Aweights, 0, sizeof(double) * buffers.n_columns);
			std::memset(buffers.h_Cweights, 0, sizeof(double) * buffers.n_columns);
			std::memset(buffers.h_Gweights, 0, sizeof(double) * buffers.n_columns);
			std::memset(buffers.h_Tweights, 0, sizeof(double) * buffers.n_columns);

			tpd = std::chrono::system_clock::now();
			buffers.preprocessingtime += tpd - tpc;
			tpc = std::chrono::system_clock::now();

			const auto res = correction::cpu_pileup_all_in_one(&buffers, subject, nQueries, queries, alignments, 
						subjectqualityScores, queryqualityScores, frequenciesPrefixSum,
						startindex, endindex,
						columnsToCheck, subjectColumnsBegin_incl, subjectColumnsEnd_excl,
						maxErrorRate, useQScores, correctedQueries, correctQueries_, estimatedCoverage, errorrate, m, kmerlength);

			tpd = std::chrono::system_clock::now();
			buffers.correctiontime += tpd - tpc;

			return res;

#endif

#ifdef __NVCC__
		//}
#endif

	}

	

}// end namespace hammingtools

