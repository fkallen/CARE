#include "../inc/hammingtools.hpp"
#include "../inc/ganja/hpc_helpers.cuh"
#include "../inc/cudareduce.cuh"

#include "../inc/shd.hpp"

#include <stdexcept>

#if 1

namespace hammingtools{



	SHDdata::SHDdata(int deviceId_, int seqlength, int seqbytes) 
		: deviceId(deviceId_), sequencelength(seqlength), sequencebytes(seqbytes){
	#ifdef __CUDACC__
		cudaSetDevice(deviceId); CUERR;
		cudaStreamCreate(&stream); CUERR;
	#endif
	};

	void SHDdata::resize(int n_sub, int n_quer){
	#ifdef __CUDACC__
		cudaSetDevice(deviceId); CUERR;

		if(n_sub > max_n_subjects){
			cudaFree(d_subjectsdata); CUERR;
			cudaMallocPitch(&d_subjectsdata, &sequencepitch, sequencebytes, n_sub); CUERR;
			cudaFree(d_queriesPerSubject); CUERR;
			cudaMalloc(&d_queriesPerSubject, sizeof(int) * n_sub); CUERR;

			cudaFreeHost(h_subjectsdata); CUERR;
			cudaMallocHost(&h_subjectsdata, sequencepitch * n_sub); CUERR;
			cudaFreeHost(h_queriesPerSubject); CUERR;
			cudaMallocHost(&h_queriesPerSubject, sizeof(int) * n_sub); CUERR;

			max_n_subjects = n_sub;
		}

		if(n_quer > max_n_queries){
			cudaFree(d_queriesdata); CUERR;
			cudaMallocPitch(&d_queriesdata, &sequencepitch, sequencebytes, n_sub); CUERR;

			cudaFreeHost(h_queriesdata); CUERR;
			cudaMallocHost(&h_queriesdata, sequencepitch * n_sub); CUERR;

			max_n_queries = n_quer;
		}

		if(n_sub > n_subjects || n_quer > n_queries){
			cudaFree(d_results); CUERR;
			cudaMalloc(&d_results, sizeof(AlignResultCompact) * n_sub * n_quer); CUERR;
			cudaFree(h_results); CUERR;
			cudaMallocHost(&h_results, sizeof(AlignResultCompact) * n_sub * n_quer); CUERR;		
		}
	#endif
		n_subjects = n_sub;
		n_queries = n_quer;
	}

	void cuda_cleanup_SHDdata(SHDdata& data){
	#ifdef __CUDACC__
		cudaSetDevice(data.deviceId); CUERR;

		cudaFree(data.d_results); CUERR;
		cudaFree(data.d_subjectsdata); CUERR;
		cudaFree(data.d_queriesdata); CUERR;
		cudaFree(data.d_queriesPerSubject); CUERR;

		cudaFreeHost(data.h_results); CUERR;
		cudaFreeHost(data.h_subjectsdata); CUERR;
		cudaFreeHost(data.h_queriesdata); CUERR;
		cudaFreeHost(data.h_queriesPerSubject); CUERR;

		cudaStreamDestroy(data.stream); CUERR;
	#endif
	}

	std::vector<std::vector<AlignResult>> 
	SHDAligner::getMultipleAlignments(SHDdata& mybuffers, 
					const std::vector<const Sequence*>& subjects,
					const std::vector<std::vector<const Sequence*>>& queries,
					std::vector<bool> activeBatches, bool useGpu) const{

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

		std::vector<std::vector<AlignResult>> alignments(subjects.size());

		// check for empty input
		if(totalNumberOfAlignments == 0){
			return alignments;
		}

	#ifdef __CUDACC__

		if(useGpu){ // use gpu for alignment

			cudaSetDevice(mybuffers.deviceId); CUERR;

			mybuffers.resize(numberOfRealSubjects, totalNumberOfAlignments);

			int subjectindex = 0;
			int queryindex = 0;
			for(size_t i = 0; i < subjects.size(); i++){
				if(activeBatches[i]){
					std::memcpy(mybuffers.h_subjectsdata + i * mybuffers.sequencepitch,
						    subjects[i]->begin(), 
						    mybuffers.sequencebytes);

					mybuffers.h_queriesPerSubject[subjectindex] = queries.size();

					for(size_t j = 0; j < queries.size(); j++){
						std::memcpy(mybuffers.h_queriesdata + queryindex * mybuffers.sequencepitch,
							    queries[i][j]->begin(), 
							    mybuffers.sequencebytes);
						queryindex++;
					}
				}
			}

			// copy data to gpu
			cudaMemcpyAsync(mybuffers.d_subjectsdata, 
					mybuffers.h_subjectsdata, 
					mybuffers.sequencebytes * mybuffers.n_subjects, 
					H2D, 
					mybuffers.stream); CUERR;
			cudaMemcpyAsync(mybuffers.d_queriesdata,
					mybuffers.h_queriesdata, 
					mybuffers.sequencebytes * mybuffers.n_queries, 
					H2D, 
					mybuffers.stream);
			cudaMemcpyAsync(mybuffers.d_queriesPerSubject,
					mybuffers.h_queriesPerSubject, 
					sizeof(int) * mybuffers.n_subjects, 
					H2D, 
					mybuffers.stream);


			// start kernel
			alignment::call_shd_kernel(mybuffers);

			//copy results to host
			cudaMemcpyAsync(mybuffers.h_results, 
				mybuffers.d_results, 
				sizeof(AlignResultCompact) * totalNumberOfAlignments, 
				D2H, 
				mybuffers.stream); CUERR;

			cudaStreamSynchronize(mybuffers.stream); CUERR;

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
					alignments[outputindex][j].arc = mybuffers.h_results[candidateIndex];
					candidateIndex++; 
				}			
			}

		}else{ // use cpu for alignment



	#endif // __CUDACC__

			for(size_t i = 0; i < subjects.size(); i++){
				alignments[i].resize(queries[i].size());

				if(activeBatches[i]){
					const char* subject = (const char*)subjects[i]->begin();
					int ns = subjects[i]->getNbases();
					for(size_t j = 0; j < queries[i].size(); j++){
						const char* query =  (const char*)queries[i][j]->begin();
						int nq = queries[i][j]->getNbases();

						alignments[i][j].arc = alignment::cpu_shifted_hamming_distance(subject, query, ns, nq);
					}
				}
			}
	#ifdef __CUDACC__
		}
	#endif // __CUDACC__

		return alignments;
	}

	

}// end namespace hammingtools

#endif
