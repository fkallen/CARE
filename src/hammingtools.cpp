#include "../inc/hammingtools.hpp"
#include "../inc/ganja/hpc_helpers.cuh"
#include "../inc/cudareduce.cuh"

#include "../inc/shd.hpp"
#include "../inc/pileup.hpp"

#include "../inc/batchelem.hpp"

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


	void CorrectionBuffers::resize(int cols){

		if(cols > max_n_columns){
			const int newmaxcols = 1.5 * cols;

			h_consensus.reset();
			h_support.reset();
			h_coverage.reset();
			h_origWeights.reset();
			h_origCoverage.reset();
			h_As.reset();
			h_Cs.reset();
			h_Gs.reset();
			h_Ts.reset();
			h_Aweights.reset();
			h_Cweights.reset();
			h_Gweights.reset();
			h_Tweights.reset();
			
			h_consensus.reset(new char[newmaxcols]);
			h_support.reset(new double[newmaxcols]);
			h_coverage.reset(new int[newmaxcols]);
			h_origWeights.reset(new double[newmaxcols]);
			h_origCoverage.reset(new int[newmaxcols]);
			h_As.reset(new int[newmaxcols]);
			h_Cs.reset(new int[newmaxcols]);
			h_Gs.reset(new int[newmaxcols]);
			h_Ts.reset(new int[newmaxcols]);
			h_Aweights.reset(new double[newmaxcols]);
			h_Cweights.reset(new double[newmaxcols]);
			h_Gweights.reset(new double[newmaxcols]);
			h_Tweights.reset(new double[newmaxcols]);

			max_n_columns = newmaxcols;
		}

		n_columns = cols;
	}
	
	void CorrectionBuffers::reset(){
			std::memset(h_consensus.get(), 0, sizeof(char) * n_columns);
			std::memset(h_support.get(), 0, sizeof(double) * n_columns);
			std::memset(h_coverage.get(), 0, sizeof(int) * n_columns);
			std::memset(h_origWeights.get(), 0, sizeof(double) * n_columns);
			std::memset(h_origCoverage.get(), 0, sizeof(int) * n_columns);
			std::memset(h_As.get(), 0, sizeof(int) * n_columns);
			std::memset(h_Cs.get(), 0, sizeof(int) * n_columns);
			std::memset(h_Gs.get(), 0, sizeof(int) * n_columns);
			std::memset(h_Ts.get(), 0, sizeof(int) * n_columns);
			std::memset(h_Aweights.get(), 0, sizeof(double) * n_columns);
			std::memset(h_Cweights.get(), 0, sizeof(double) * n_columns);
			std::memset(h_Gweights.get(), 0, sizeof(double) * n_columns);
			std::memset(h_Tweights.get(), 0, sizeof(double) * n_columns);		
	}

    void getMultipleAlignments(SHDdata& mybuffers, std::vector<BatchElem>& batch, bool useGpu){

        std::chrono::time_point<std::chrono::system_clock> tpa = std::chrono::system_clock::now();
        std::chrono::time_point<std::chrono::system_clock> tpb = std::chrono::system_clock::now();

        int numberOfRealSubjects = 0;
        int totalNumberOfAlignments = 0;

        for(auto& b : batch){
            if(b.active){
                numberOfRealSubjects++;
                totalNumberOfAlignments += b.fwdSequences.size();
                totalNumberOfAlignments += b.revcomplSequences.size();
            }
        }

        // check for empty input
        if(totalNumberOfAlignments == 0){
            return;
        }

    #ifdef __NVCC__

        if(useGpu){ // use gpu for alignment

            tpa = std::chrono::system_clock::now();

            cudaSetDevice(mybuffers.deviceId); CUERR;

            mybuffers.resize(numberOfRealSubjects, totalNumberOfAlignments);

            tpb = std::chrono::system_clock::now();

            mybuffers.resizetime += tpb - tpa;

            int querysum = 0;
            int subjectindex = 0;
            int batchid = 0;
            std::vector<alignment::shdparams> params(batch.size());

            for(auto& b : batch){
                if(b.active){
                    tpa = std::chrono::system_clock::now();
                    batchid = subjectindex;

                    params[batchid].max_sequence_bytes = mybuffers.max_sequence_bytes;
                    params[batchid].sequencepitch = mybuffers.sequencepitch;
                    params[batchid].subjectlength = b.fwdSequence->length();
                    params[batchid].n_queries = b.fwdSequences.size() + b.revcomplSequences.size();
                    params[batchid].querylengths = mybuffers.d_querylengths + querysum;
                    params[batchid].subjectdata = mybuffers.d_subjectsdata + mybuffers.sequencepitch * subjectindex;
                    params[batchid].queriesdata = mybuffers.d_queriesdata + mybuffers.sequencepitch * querysum;
                    params[batchid].results = mybuffers.d_results + querysum;

                    int* querylengths = mybuffers.h_querylengths + querysum;
                    char* subjectdata = mybuffers.h_subjectsdata + mybuffers.sequencepitch * subjectindex;
                    char* queriesdata = mybuffers.h_queriesdata + mybuffers.sequencepitch * querysum;

                    assert(b.fwdSequence->length() <= mybuffers.max_sequence_length);
                    assert(b.fwdSequence->getNumBytes() <= mybuffers.max_sequence_bytes);

                    std::memcpy(subjectdata, b.fwdSequence->begin(), b.fwdSequence->getNumBytes());

                    int count = 0;
                    for(const auto seq : b.fwdSequences){
                        assert(seq->length() <= mybuffers.max_sequence_length);
                        assert(seq->getNumBytes() <= mybuffers.max_sequence_bytes);

                        std::memcpy(queriesdata + count * mybuffers.sequencepitch,
                                seq->begin(),
                                seq->getNumBytes());

                        querylengths[count] = seq->length();
                        count++;
                    }
                    for(const auto seq : b.revcomplSequences){
                        assert(seq->length() <= mybuffers.max_sequence_length);
                        assert(seq->getNumBytes() <= mybuffers.max_sequence_bytes);

                        std::memcpy(queriesdata + count * mybuffers.sequencepitch,
                                seq->begin(),
                                seq->getNumBytes());

                        querylengths[count] = seq->length();
                        count++;
                    }
                    assert(params[batchid].n_queries == count);

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

                    // start kernel
                    alignment::call_shd_kernel_async(params[batchid], mybuffers.streams[batchid]);

                    querysum += count;
                    subjectindex++;
                }
            }
            
            subjectindex = 0;
            querysum = 0;
			//initialize transfer d2h
            for(auto& b : batch){
                if(b.active){
                    batchid = subjectindex;
                    AlignResultCompact* results = mybuffers.h_results + querysum;

                    cudaMemcpyAsync(results,
                        params[batchid].results,
                        sizeof(AlignResultCompact) * params[batchid].n_queries,
                        D2H,
                        mybuffers.streams[batchid]); CUERR;

                    subjectindex++;
                    querysum += params[batchid].n_queries;
                }
            }            

            subjectindex = 0;
            querysum = 0;

			//wait for d2h transfer to complete and fetch results
            for(auto& b : batch){
                if(b.active){
                    batchid = subjectindex;
                    AlignResultCompact* results = mybuffers.h_results + querysum;

                    cudaStreamSynchronize(mybuffers.streams[batchid]); CUERR;

                    tpa = std::chrono::system_clock::now();

                    int count = 0;
                    for(auto& alignment : b.fwdAlignments){
                        alignment = results[count++];
                    }
                    for(auto& alignment : b.revcomplAlignments){
                        alignment = results[count++];
                    }

                    tpb = std::chrono::system_clock::now();
                    mybuffers.postprocessingtime += tpb - tpa;

                    subjectindex++;
                    querysum += params[batchid].n_queries;
                }
            }

        }else{ // use cpu for alignment

    #endif // __NVCC__

            tpa = std::chrono::system_clock::now();

            for(auto& b : batch){
                if(b.active){
                    const char* const subject = (const char*)b.fwdSequence->begin();
                    const int subjectLength = b.fwdSequence->length();

                    for(size_t i = 0; i < b.fwdSequences.size(); i++){
                        const char* query =  (const char*)b.fwdSequences[i]->begin();
                        const int queryLength = b.fwdSequences[i]->length();
                        b.fwdAlignments[i] = alignment::cpu_shifted_hamming_distance(subject, query, subjectLength, queryLength);
                    }

                    for(size_t i = 0; i < b.revcomplSequences.size(); i++){
                        const char* query =  (const char*)b.revcomplSequences[i]->begin();
                        const int queryLength = b.revcomplSequences[i]->length();
                        b.revcomplAlignments[i] = alignment::cpu_shifted_hamming_distance(subject, query, subjectLength, queryLength);
                    }
                }
            }

            tpb = std::chrono::system_clock::now();

            mybuffers.alignmenttime += tpb - tpa;
    #ifdef __NVCC__
        }
    #endif // __NVCC__
    }





    std::tuple<int,std::chrono::duration<double>,std::chrono::duration<double>>
	performCorrection(CorrectionBuffers& buffers, BatchElem& batchElem,
                double maxErrorRate,
                bool useQScores,
				bool correctQueries,
				int estimatedCoverage,
				double errorrate,
				double m,
				int kmerlength,
				bool useGpu){

            assert(batchElem.active && "batchElem.active");

			std::chrono::time_point<std::chrono::system_clock> tpc, tpd;

			tpc = std::chrono::system_clock::now();

			//determine number of columns in pileup image
			int startindex = 0;
			int endindex = batchElem.fwdSequenceString.length();

            for(size_t i = 0; i < batchElem.n_unique_candidates; i++){
				const int shift = batchElem.bestAlignments[i].shift;
				startindex = shift < startindex ? shift : startindex;
				const int queryEndsAt = batchElem.bestSequences[i]->length() + shift;
				endindex = queryEndsAt > endindex ? queryEndsAt : endindex;
            }

			const int columnsToCheck = endindex - startindex;
			const int subjectColumnsBegin_incl = std::max(-startindex,0);
			const int subjectColumnsEnd_excl = subjectColumnsBegin_incl + batchElem.fwdSequenceString.length();

			tpd = std::chrono::system_clock::now();
			buffers.preprocessingtime += tpd - tpc;
			tpc = std::chrono::system_clock::now();

			buffers.resize(columnsToCheck);

			tpd = std::chrono::system_clock::now();
			buffers.resizetime += tpd - tpc;
			tpc = std::chrono::system_clock::now();

			buffers.reset();

			tpd = std::chrono::system_clock::now();
			buffers.preprocessingtime += tpd - tpc;
			tpc = std::chrono::system_clock::now();

			auto majorityvotetime = correction::cpu_add_weights(&buffers, batchElem,
																			startindex, endindex,
																			columnsToCheck, subjectColumnsBegin_incl, subjectColumnsEnd_excl,
																			maxErrorRate,
																			useQScores);	
				
			correction::cpu_find_consensus(&buffers, batchElem,
							columnsToCheck,subjectColumnsBegin_incl);
				
			auto status_and_time = correction::cpu_correct(&buffers, batchElem,
												startindex, endindex,
												columnsToCheck, subjectColumnsBegin_incl, subjectColumnsEnd_excl,
												maxErrorRate,
												correctQueries,
												estimatedCoverage,
												errorrate,
												m,
												kmerlength);

			tpd = std::chrono::system_clock::now();
			buffers.correctiontime += tpd - tpc;

			return std::tie(std::get<0>(status_and_time), majorityvotetime, std::get<1>(status_and_time));
	}



















}// end namespace hammingtools
