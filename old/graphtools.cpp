#include "../inc/graphtools.hpp"
#include "../inc/alignment.hpp"
#include "../inc/sga.hpp"
#include "../inc/graph.hpp"
#include "../inc/batchelem.hpp"

#include <cassert>
#include <chrono>
#include <cstring>
#include <vector>

namespace care{
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

			cudaFreeHost(h_subjectsdata); CUERR;
			cudaMallocHost(&h_subjectsdata, sequencepitch * newmax); CUERR;

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
			cudaFree(data.d_subjectlengths); CUERR;
			cudaFree(data.d_querylengths); CUERR;

			cudaFreeHost(data.h_results); CUERR;
			cudaFreeHost(data.h_ops); CUERR;
			cudaFreeHost(data.h_subjectsdata); CUERR;
			cudaFreeHost(data.h_queriesdata); CUERR;
			cudaFreeHost(data.h_subjectlengths); CUERR;
			cudaFreeHost(data.h_querylengths); CUERR;

			for(int i = 0; i < 8; i++)
				cudaStreamDestroy(data.streams[i]); CUERR;
		#endif
	}

    void getMultipleAlignments(AlignerDataArrays& mybuffers, std::vector<BatchElem>& batch, bool useGpu){

		std::chrono::time_point<std::chrono::system_clock> tpa;
		std::chrono::time_point<std::chrono::system_clock> tpb;

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

			mybuffers.n_subjects = numberOfRealSubjects;
			mybuffers.n_queries = totalNumberOfAlignments;

			tpb = std::chrono::system_clock::now();

			mybuffers.resizetime += tpb - tpa;

			tpa = std::chrono::system_clock::now();

			int subjectindex = 0;
			int querysum = 0;
			int batchid = 0;
			std::vector<alignment::sgaparams> params(batch.size());

            for(auto& b : batch){
                if(b.active){
					tpa = std::chrono::system_clock::now();
					batchid = subjectindex;

					params[batchid].max_sequence_length = mybuffers.max_sequence_length;
					params[batchid].max_ops_per_alignment = mybuffers.max_ops_per_alignment;
					params[batchid].sequencepitch = mybuffers.sequencepitch;
					params[batchid].subjectlength = b.fwdSequence->length();
					params[batchid].n_queries = b.fwdSequences.size() + b.revcomplSequences.size();
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

                    assert(b.fwdSequence->length() <= mybuffers.max_sequence_length);

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
					alignment::call_cuda_semi_global_alignment_kernel_async(params[batchid], mybuffers.streams[batchid]);

                    querysum += count;
                    subjectindex++;

                    b.fwdAlignOps.resize(b.fwdSequences.size());
                    b.revcomplAlignOps.resize(b.revcomplSequences.size());
				}
			}


            subjectindex = 0;
            querysum = 0;
			//initialize transfer d2h
            for(auto& b : batch){
                if(b.active){
                    batchid = subjectindex;
                    AlignResultCompact* results = mybuffers.h_results + querysum;
                    AlignOp* ops = mybuffers.h_ops + querysum * mybuffers.max_ops_per_alignment;

                    cudaMemcpyAsync(results,
                        params[batchid].results,
                        sizeof(AlignResultCompact) * params[batchid].n_queries,
                        D2H,
                        mybuffers.streams[batchid]); CUERR;

					cudaMemcpyAsync(ops,
						params[batchid].ops,
						sizeof(AlignOp) * params[batchid].n_queries * mybuffers.max_ops_per_alignment,
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
                    AlignOp* ops = mybuffers.h_ops + querysum * mybuffers.max_ops_per_alignment;

                    cudaStreamSynchronize(mybuffers.streams[batchid]); CUERR;

                    tpa = std::chrono::system_clock::now();

                    int count = 0;
                    int localcount = 0;
                    for(auto& alignment : b.fwdAlignments){

                        alignment = results[count];
                        b.fwdAlignOps[localcount].resize(alignment.nOps);
                        std::reverse_copy(ops + count * mybuffers.max_ops_per_alignment,
                                  ops + count * mybuffers.max_ops_per_alignment + alignment.nOps,
                                  b.fwdAlignOps[localcount].begin());
                        count++;
                        localcount++;
                    }
                    localcount = 0;
                    for(auto& alignment : b.revcomplAlignments){

                        alignment = results[count];
                        b.revcomplAlignOps[localcount].resize(alignment.nOps);
                        std::copy(ops + count * mybuffers.max_ops_per_alignment,
                                  ops + count * mybuffers.max_ops_per_alignment + alignment.nOps,
                                  b.revcomplAlignOps[localcount].begin());
                        count++;
                        localcount++;
                    }


                    tpb = std::chrono::system_clock::now();
                    mybuffers.postprocessingtime += tpb - tpa;

                    subjectindex++;
                    querysum += params[batchid].n_queries;
                }
            }

		}else{ // use cpu for alignment

	#endif

            tpa = std::chrono::system_clock::now();

            for(auto& b : batch){
                if(b.active){
                    const char* const subject = (const char*)b.fwdSequence->begin();
                    const int subjectLength = b.fwdSequence->length();

                    for(size_t i = 0; i < b.fwdSequences.size(); i++){
                        const char* query =  (const char*)b.fwdSequences[i]->begin();
                        const int queryLength = b.fwdSequences[i]->length();
                        auto al = alignment::cpu_semi_global_alignment(&mybuffers, subject, query, subjectLength, queryLength);
                        b.fwdAlignments[i] = al.arc;
                        b.fwdAlignOps[i] = std::move(al.operations);
                    }

                    for(size_t i = 0; i < b.revcomplSequences.size(); i++){
                        const char* query =  (const char*)b.revcomplSequences[i]->begin();
                        const int queryLength = b.revcomplSequences[i]->length();
                        auto al = alignment::cpu_semi_global_alignment(&mybuffers, subject, query, subjectLength, queryLength);
                        b.revcomplAlignments[i] = al.arc;
                        b.revcomplAlignOps[i] = std::move(al.operations);
                    }
                }
            }

            tpb = std::chrono::system_clock::now();

            mybuffers.alignmenttime += tpb - tpa;

	#ifdef __NVCC__
		}
	#endif
	}


} //namespace end
}
