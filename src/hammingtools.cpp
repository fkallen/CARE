#include "../inc/hammingtools.hpp"
#include "../inc/ganja/hpc_helpers.cuh"
#include "../inc/cudareduce.cuh"

#include "../inc/shd.hpp"
#include "../inc/pileup.hpp"

#include <stdexcept>
#include <cstdio>

namespace hammingtools{

	int reserved_SMs = 1;



	SHDdata::SHDdata(int deviceId_, int cpuThreadsOnDevice, int maxseqlength) 
		: deviceId(deviceId_), max_sequence_length(maxseqlength), max_sequence_bytes(SDIV(maxseqlength,4)){
	#ifdef __NVCC__
		cudaSetDevice(deviceId); CUERR;
		cudaStreamCreate(&stream); CUERR;

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
		cudaFree(data.d_lengths); CUERR;

		cudaFreeHost(data.h_results); CUERR;
		cudaFreeHost(data.h_subjectsdata); CUERR;
		cudaFreeHost(data.h_queriesdata); CUERR;
		cudaFreeHost(data.h_queriesPerSubject); CUERR;
		cudaFreeHost(data.h_lengths); CUERR;

		cudaStreamDestroy(data.stream); CUERR;
	#endif
	}

	void print_SHDdata(const SHDdata& mybuffers){
		printf("d_results %p\n", mybuffers.d_results);
		printf("d_subjectsdata %p\n", mybuffers.d_subjectsdata);
		printf("d_queriesdata %p\n", mybuffers.d_queriesdata);
		printf("d_queriesPerSubject %p\n", mybuffers.d_queriesPerSubject);
		printf("d_lengths %p\n", mybuffers.d_lengths);
		printf("h_results %p\n", mybuffers.h_results);
		printf("h_subjectsdata %p\n", mybuffers.h_subjectsdata);
		printf("h_queriesdata %p\n", mybuffers.h_queriesdata);
		printf("h_queriesPerSubject %p\n", mybuffers.h_queriesPerSubject);
		printf("h_lengths %p\n", mybuffers.h_lengths);
	#ifdef __NVCC__
		printf("stream %p\n", mybuffers.stream);
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
			throw std::runtime_error("SHDAligner::getMultipleAlignments incorrect input dimensions. queries.size() != candidates.size()");
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
			alignment::call_shd_kernel(mybuffers);

			tpb = std::chrono::system_clock::now();
		
			mybuffers.alignmenttime += tpb - tpa;

			tpa = std::chrono::system_clock::now();

			//copy results to host
			cudaMemcpyAsync(mybuffers.h_results, 
				mybuffers.d_results, 
				sizeof(AlignResultCompact) * totalNumberOfAlignments, 
				D2H, 
				mybuffers.stream); CUERR;

			cudaStreamSynchronize(mybuffers.stream); CUERR;

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



	int performCorrection(std::string& subject,
				int nQueries, 
				std::vector<std::string>& queries,
				const std::vector<AlignResultCompact>& alignments,
				const std::string& subjectqualityScores, 
				const std::vector<std::string>& queryqualityScores,
				const std::vector<int>& frequenciesPrefixSum,
				double maxErrorRate,
				bool useQScores,
				std::vector<bool>& correctedQueries,
				bool correctQueries_,
				int estimatedCoverage,
				double errorrate,
				double m,
				int kmerlength){

		return correction::correct_cpu(subject, nQueries, queries, alignments, subjectqualityScores, queryqualityScores, frequenciesPrefixSum,
				maxErrorRate, useQScores, correctedQueries, correctQueries_, estimatedCoverage, errorrate, m, kmerlength);

	}

	

}// end namespace hammingtools

