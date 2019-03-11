#include <gpu/kernels.hpp>
#include <msa.hpp>
#include <gpu/bestalignment.hpp>
#include <hpc_helpers.cuh>

#include <cassert>

#ifdef __NVCC__
#include <cub/cub.cuh>
#endif

namespace care{
namespace gpu{

#ifdef __NVCC__

    KernelLaunchHandle make_kernel_launch_handle(int deviceId){
        KernelLaunchHandle handle;
        handle.deviceId = deviceId;
        cudaGetDeviceProperties(&handle.deviceProperties, deviceId); CUERR;
        return handle;
    }





    __global__
    void msa_find_consensus_kernel(
                            char* __restrict__ d_consensus,
                            float* __restrict__ d_support,
                            int* __restrict__ d_coverage,
                            float* __restrict__ d_origWeights,
                            int* __restrict__ d_origCoverages,
                            int* __restrict__ d_countsA,
                            int* __restrict__ d_countsC,
                            int* __restrict__ d_countsG,
                            int* __restrict__ d_countsT,
                            float* __restrict__ d_weightsA,
                            float* __restrict__ d_weightsC,
                            float* __restrict__ d_weightsG,
                            float* __restrict__ d_weightsT,
                            const char* __restrict__ d_multiple_sequence_alignments,
                            const float* __restrict__ d_multiple_sequence_alignment_weights,
                            const MSAColumnProperties* __restrict__ d_msa_column_properties,
                            const int* __restrict__ d_candidates_per_subject_prefixsum,
                            const int* __restrict__ d_indices_per_subject,
                            const int* __restrict__ d_indices_per_subject_prefixsum,
                            int n_subjects,
                            int n_queries,
                            const int* __restrict__ d_num_indices,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            int msa_max_column_count,
                            int blocks_per_msa){

        constexpr char A_enc = 0x00;
        constexpr char C_enc = 0x01;
        constexpr char G_enc = 0x02;
        constexpr char T_enc = 0x03;

        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

        const int localBlockId = blockIdx.x % blocks_per_msa;
		//const int n_indices = *d_num_indices;

        //process multiple sequence alignment of each subject
        //for each column in msa, find consensus and support
        for(unsigned subjectIndex = blockIdx.x / blocks_per_msa; subjectIndex < n_subjects; subjectIndex += gridDim.x / blocks_per_msa){
            const int subjectColumnsBegin_incl = d_msa_column_properties[subjectIndex].subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = d_msa_column_properties[subjectIndex].subjectColumnsEnd_excl;
			const int columnsToCheck = d_msa_column_properties[subjectIndex].columnsToCheck;

            //number of rows in multiple sequence alignment for subject[subjectIndex]
            const int msa_rows = 1 + d_indices_per_subject[subjectIndex];

            const unsigned offset1 = msa_pitch * (subjectIndex + d_indices_per_subject_prefixsum[subjectIndex]);
            const unsigned offset2 = msa_weights_pitch_floats * (subjectIndex + d_indices_per_subject_prefixsum[subjectIndex]);

            const char* const my_multiple_sequence_alignment = d_multiple_sequence_alignments + offset1;
            const float* const my_multiple_sequence_alignment_weight = d_multiple_sequence_alignment_weights + offset2;

            char* const my_consensus = d_consensus + subjectIndex * msa_pitch;
            float* const my_support = d_support + subjectIndex * msa_weights_pitch_floats;
            int* const my_coverage = d_coverage + subjectIndex * msa_weights_pitch_floats;

            float* const my_orig_weights = d_origWeights + subjectIndex * msa_weights_pitch_floats;
            int* const my_orig_coverage = d_origCoverages + subjectIndex * msa_weights_pitch_floats;

            int* const my_countsA = d_countsA + subjectIndex * msa_weights_pitch_floats;
            int* const my_countsC = d_countsC + subjectIndex * msa_weights_pitch_floats;
            int* const my_countsG = d_countsG + subjectIndex * msa_weights_pitch_floats;
            int* const my_countsT = d_countsT + subjectIndex * msa_weights_pitch_floats;
            float* const my_weightsA = d_weightsA + subjectIndex * msa_weights_pitch_floats;
            float* const my_weightsC = d_weightsC + subjectIndex * msa_weights_pitch_floats;
            float* const my_weightsG = d_weightsG + subjectIndex * msa_weights_pitch_floats;
            float* const my_weightsT = d_weightsT + subjectIndex * msa_weights_pitch_floats;

            for(int column = localBlockId * blockDim.x + threadIdx.x; column < columnsToCheck; column += blocks_per_msa * blockDim.x){

                float Aw = 0.0f;
                float Cw = 0.0f;
                float Gw = 0.0f;
                float Tw = 0.0f;

                int As = 0;
                int Cs = 0;
                int Gs = 0;
                int Ts = 0;

                int columnCoverage = 0;

                for(int row = 0; row < msa_rows; ++row){
                    const char base = my_multiple_sequence_alignment[row * msa_pitch + column];
                    const float weight = my_multiple_sequence_alignment_weight[row * msa_weights_pitch_floats + column];

					//if(!(base == 'A' || base == 'C' || base == 'G' || base == 'T')){
					//	assert(base == 'A' || base == 'C' || base == 'G' || base == 'T');
					//}
#if 0
                    Aw += (base == 'A' ? weight : 0);
                    Cw += (base == 'C' ? weight : 0);
                    Gw += (base == 'G' ? weight : 0);
                    Tw += (base == 'T' ? weight : 0);

                    As += (base == 'A');
                    Cs += (base == 'C');
                    Gs += (base == 'G');
                    Ts += (base == 'T');
#else
                    Aw += (base == A_enc ? weight : 0);
                    Cw += (base == C_enc ? weight : 0);
                    Gw += (base == G_enc ? weight : 0);
                    Tw += (base == T_enc ? weight : 0);

                    As += (base == A_enc);
                    Cs += (base == C_enc);
                    Gs += (base == G_enc);
                    Ts += (base == T_enc);
#endif
                    //columnCoverage += (base == 'A' || base == 'C' || base == 'G' || base == 'T');
                    columnCoverage += (base == A_enc || base == C_enc || base == G_enc || base == T_enc);//!(base & 0xFC);
                }

                my_countsA[column] = As;
                my_countsC[column] = Cs;
                my_countsG[column] = Gs;
                my_countsT[column] = Ts;
                my_weightsA[column] = Aw;
                my_weightsC[column] = Cw;
                my_weightsG[column] = Gw;
                my_weightsT[column] = Tw;


				if(columnCoverage <= 0){
					//assert(columnCoverage > 0);
				}

                const float columnWeight = Aw + Cw + Gw + Tw;
                float consWeight = Aw;
#if 0
                char cons = 'A';

                cons = Cw > consWeight ? 'C' : cons;
                consWeight = Cw > consWeight ? Cw : consWeight;

                cons = Gw > consWeight ? 'G' : cons;
                consWeight = Gw > consWeight ? Gw : consWeight;

                cons = Tw > consWeight ? 'T' : cons;
                consWeight = Tw > consWeight ? Tw : consWeight;
#else
                char cons = 'A';

                cons = Cw > consWeight ? 'C' : cons;
                consWeight = Cw > consWeight ? Cw : consWeight;

                cons = Gw > consWeight ? 'G' : cons;
                consWeight = Gw > consWeight ? Gw : consWeight;

                cons = Tw > consWeight ? 'T' : cons;
                consWeight = Tw > consWeight ? Tw : consWeight;
#endif
                /*char consByCount = 'A';
                int consByCountCount = As;
                consByCount = Cs > consByCountCount ? 'C' : cons;
                consByCountCount = Cs > consByCountCount ? Cs : consWeight;

                consByCount = Gs > consByCountCount ? 'G' : cons;
                consByCountCount = Gs > consByCountCount ? Gs : consWeight;

                consByCount = Ts > consByCountCount ? 'T' : cons;
                consByCountCount = Ts > consByCountCount ? Ts : consWeight;

                if(consByCount != cons){
                    printf("cons differ\n");
                }*/

                my_consensus[column] = cons;
                my_support[column] = consWeight / columnWeight;
				//printf("subject %d, column %d, support %f\n", subjectIndex, column, my_support[column]);
                my_coverage[column] = columnCoverage;

                if(subjectColumnsBegin_incl <= column && column < subjectColumnsEnd_excl){
                    const char subjectbase = my_multiple_sequence_alignment[column];
                    if(subjectbase == A_enc){
                        my_orig_weights[column] = Aw;
                        my_orig_coverage[column] = As;
                    }else if(subjectbase == C_enc){
                        my_orig_weights[column] = Cw;
                        my_orig_coverage[column] = Cs;
                    }else if(subjectbase == G_enc){
                        my_orig_weights[column] = Gw;
                        my_orig_coverage[column] = Gs;
                    }else if(subjectbase == T_enc){
                        my_orig_weights[column] = Tw;
                        my_orig_coverage[column] = Ts;
                    }
                }
            }
        }
    }

    void call_msa_find_consensus_kernel_async(
                            char* d_consensus,
                            float* d_support,
                            int* d_coverage,
                            float* d_origWeights,
                            int* d_origCoverages,
                            int* d_countsA,
                            int* d_countsC,
                            int* d_countsG,
                            int* d_countsT,
                            float* d_weightsA,
                            float* d_weightsC,
                            float* d_weightsG,
                            float* d_weightsT,
                            const char* d_multiple_sequence_alignments,
                            const float* d_multiple_sequence_alignment_weights,
                            const MSAColumnProperties* d_msa_column_properties,
                            const int* d_candidates_per_subject_prefixsum,
                            const int* d_indices_per_subject,
                            const int* d_indices_per_subject_prefixsum,
                            int n_subjects,
                            int n_queries,
                            const int* d_num_indices,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            int msa_max_column_count,
                            cudaStream_t stream,
                            KernelLaunchHandle& handle){


        const int blocksize = 128;
        const std::size_t smem = 0;

        int max_blocks_per_device = 1;

        KernelLaunchConfig kernelLaunchConfig;
        kernelLaunchConfig.threads_per_block = blocksize;
        kernelLaunchConfig.smem = smem;

        auto iter = handle.kernelPropertiesMap.find(KernelId::MSAFindConsensus);
        if(iter == handle.kernelPropertiesMap.end()){

            std::map<KernelLaunchConfig, KernelProperties> mymap;

            #define getProp(blocksize) { \
                KernelLaunchConfig kernelLaunchConfig; \
                kernelLaunchConfig.threads_per_block = (blocksize); \
                kernelLaunchConfig.smem = 0; \
                KernelProperties kernelProperties; \
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                                                                msa_find_consensus_kernel, \
                                                                kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
                mymap[kernelLaunchConfig] = kernelProperties; \
            }

            getProp(32);
            getProp(64);
            getProp(96);
            getProp(128);
            getProp(160);
            getProp(192);
            getProp(224);
            getProp(256);

            const auto& kernelProperties = mymap[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

            handle.kernelPropertiesMap[KernelId::MSAFindConsensus] = std::move(mymap);

            #undef getProp
        }else{
            std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
            const KernelProperties& kernelProperties = map[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
        }


        const int blocks_per_msa = 2; //SDIV(msa_max_column_count, blocksize);

        dim3 block(blocksize, 1, 1);
        dim3 grid(std::min(max_blocks_per_device, n_subjects * blocks_per_msa), 1, 1);

        msa_find_consensus_kernel<<<grid, block, 0, stream>>>(d_consensus,
                                                            d_support,
                                                            d_coverage,
                                                            d_origWeights,
                                                            d_origCoverages,
                                                            d_countsA,
                                                            d_countsC,
                                                            d_countsG,
                                                            d_countsT,
                                                            d_weightsA,
                                                            d_weightsC,
                                                            d_weightsG,
                                                            d_weightsT,
                                                            d_multiple_sequence_alignments,
                                                            d_multiple_sequence_alignment_weights,
                                                            d_msa_column_properties,
                                                            d_candidates_per_subject_prefixsum,
                                                            d_indices_per_subject,
                                                            d_indices_per_subject_prefixsum,
                                                            n_subjects,
                                                            n_queries,
                                                            d_num_indices,
                                                            msa_pitch,
                                                            msa_weights_pitch,
                                                            msa_max_column_count,
                                                            blocks_per_msa); CUERR;

    }


    template<int BLOCKSIZE>
    __global__
    void msa_correct_subject_kernel(
                            const char* __restrict__ d_consensus,
                            const float* __restrict__ d_support,
                            const int* __restrict__ d_coverage,
                            const int* __restrict__ d_origCoverages,
                            const char* __restrict__ d_multiple_sequence_alignments,
                            const MSAColumnProperties* __restrict__ d_msa_column_properties,
                            const int* __restrict__ d_indices_per_subject_prefixsum,
                            bool* __restrict__ d_is_high_quality_subject,
                            char* __restrict__ d_corrected_subjects,
							bool* __restrict__ d_subject_is_corrected,
                            int n_subjects,
                            int n_queries,
                            const int* __restrict__ d_num_indices,
                            size_t sequence_pitch,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            float estimatedErrorrate,
                            float avg_support_threshold,
                            float min_support_threshold,
                            float min_coverage_threshold,
                            int k_region){

        using BlockReduceBool = cub::BlockReduce<bool, BLOCKSIZE>;
        using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;
        using BlockReduceFloat = cub::BlockReduce<float, BLOCKSIZE>;

        __shared__ union {
            typename BlockReduceBool::TempStorage boolreduce;
            typename BlockReduceInt::TempStorage intreduce;
            typename BlockReduceFloat::TempStorage floatreduce;
        } temp_storage;

        __shared__ bool broadcastbuffer;

        auto isGoodAvgSupport = [&](float avgsupport){
            return avgsupport >= avg_support_threshold;
        };
        auto isGoodMinSupport = [&](float minsupport){
            return minsupport >= min_support_threshold;
        };
        auto isGoodMinCoverage = [&](float mincoverage){
            return mincoverage >= min_coverage_threshold;
        };

        constexpr char A_enc = 0x00;
        constexpr char C_enc = 0x01;
        constexpr char G_enc = 0x02;
        constexpr char T_enc = 0x03;

        auto to_nuc = [](char c){
            switch(c){
            case A_enc: return 'A';
            case C_enc: return 'C';
            case G_enc: return 'G';
            case T_enc: return 'T';
            default: return 'F';
            }
        };

        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);
		//const int n_indices = *d_num_indices;

        for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
            const float* const my_support = d_support + msa_weights_pitch_floats * subjectIndex;
            const int* const my_coverage = d_coverage + msa_weights_pitch_floats * subjectIndex;
            const int* const my_orig_coverage = d_origCoverages + msa_weights_pitch_floats * subjectIndex;
            const char* const my_consensus = d_consensus + msa_pitch  * subjectIndex;
            char* const my_corrected_subject = d_corrected_subjects + subjectIndex * sequence_pitch;

            const MSAColumnProperties properties = d_msa_column_properties[subjectIndex];
            const int subjectColumnsBegin_incl = properties.subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = properties.subjectColumnsEnd_excl;

            float avg_support = 0;
            float min_support = 1.0f;
            //int max_coverage = 0;
            int min_coverage = std::numeric_limits<int>::max();

            for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                assert(i < properties.columnsToCheck);

                avg_support += my_support[i];
                min_support = min(my_support[i], min_support);
                //max_coverage = max(my_coverage[i], max_coverage);
                min_coverage = min(my_coverage[i], min_coverage);
            }

            avg_support = BlockReduceFloat(temp_storage.floatreduce).Sum(avg_support);
			__syncthreads();

            min_support = BlockReduceFloat(temp_storage.floatreduce).Reduce(min_support, cub::Min());
			__syncthreads();

            //max_coverage = BlockReduceInt(temp_storage.intreduce).Reduce(max_coverage, cub::Max());

            min_coverage = BlockReduceInt(temp_storage.intreduce).Reduce(min_coverage, cub::Min());
			__syncthreads();

            avg_support /= (subjectColumnsEnd_excl - subjectColumnsBegin_incl);

            bool isHQ = isGoodAvgSupport(avg_support) && isGoodMinSupport(min_support) && isGoodMinCoverage(min_coverage);

            if(threadIdx.x == 0){
                broadcastbuffer = isHQ;
                d_is_high_quality_subject[subjectIndex] = isHQ;
            }
            __syncthreads();

            isHQ = broadcastbuffer;

            if(isHQ){
                for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                    my_corrected_subject[i - subjectColumnsBegin_incl] = my_consensus[i];
                }
                if(threadIdx.x == 0){
                    d_subject_is_corrected[subjectIndex] = true;
                }
            }else{
                const unsigned offset1 = msa_pitch * (subjectIndex + d_indices_per_subject_prefixsum[subjectIndex]);
                const char* const my_multiple_sequence_alignment = d_multiple_sequence_alignments + offset1;

                //copy orignal sequence, which is in first row of msa, to corrected sequences
                for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                    my_corrected_subject[i - subjectColumnsBegin_incl] = to_nuc(my_multiple_sequence_alignment[i]);
                }

                const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;

                bool foundAColumn = false;
                for(int i = threadIdx.x; i < subjectLength; i += BLOCKSIZE){
                    const int globalIndex = subjectColumnsBegin_incl + i;

                    if(my_corrected_subject[i] != my_consensus[globalIndex]
                                && my_support[globalIndex] > 0.5f
                                && my_orig_coverage[globalIndex] <= min_coverage_threshold){
                        float avgsupportkregion = 0;
                        int c = 0;
                        bool kregioncoverageisgood = true;

                        for(int j = i - k_region/2; j <= i + k_region/2 && kregioncoverageisgood; j++){
                            if(j != i && j >= 0 && j < subjectLength){
                                avgsupportkregion += my_support[subjectColumnsBegin_incl + j];
                                kregioncoverageisgood &= (my_coverage[subjectColumnsBegin_incl + j] >= min_coverage_threshold);
                                //kregioncoverageisgood &= (my_coverage[subjectColumnsBegin_incl + j] >= 1);
                                c++;
                            }
                        }

                        avgsupportkregion /= c;

						//if(i == 33 || i == 34){
						//	printf("%d %f\n", i, avgsupportkregion);
						//}
                        if(kregioncoverageisgood && avgsupportkregion >= 1.0f-estimatedErrorrate){
                            my_corrected_subject[i] = my_consensus[globalIndex];
                            foundAColumn = true;
                        }
                    }
                }
                //perform block wide or-reduction on foundAColumn
                foundAColumn = BlockReduceBool(temp_storage.boolreduce).Reduce(foundAColumn, [](bool a, bool b){return a || b;});
				__syncthreads();

                if(threadIdx.x == 0){
                    d_subject_is_corrected[subjectIndex] = foundAColumn;
                }
            }
        }
    }

    void call_msa_correct_subject_kernel_async(
                            const char* d_consensus,
                            const float* d_support,
                            const int* d_coverage,
                            const int* d_origCoverages,
                            const char* d_multiple_sequence_alignments,
                            const MSAColumnProperties* d_msa_column_properties,
                            const int* d_indices_per_subject_prefixsum,
                            bool* d_is_high_quality_subject,
                            char* d_corrected_subjects,
							bool* d_subject_is_corrected,
                            int n_subjects,
                            int n_queries,
                            const int* d_num_indices,
                            size_t sequence_pitch,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            float estimatedErrorrate,
                            float avg_support_threshold,
                            float min_support_threshold,
                            float min_coverage_threshold,
                            int k_region,
                            int maximum_sequence_length,
                            cudaStream_t stream,
                            KernelLaunchHandle& handle){

        const int max_block_size = 256;
        const int blocksize = std::min(max_block_size, SDIV(maximum_sequence_length, 32) * 32);
        const std::size_t smem = 0;

        int max_blocks_per_device = 1;

        KernelLaunchConfig kernelLaunchConfig;
        kernelLaunchConfig.threads_per_block = blocksize;
        kernelLaunchConfig.smem = smem;

        auto iter = handle.kernelPropertiesMap.find(KernelId::MSACorrectSubject);
        if(iter == handle.kernelPropertiesMap.end()){

            std::map<KernelLaunchConfig, KernelProperties> mymap;

            #define getProp(blocksize) { \
                KernelLaunchConfig kernelLaunchConfig; \
                kernelLaunchConfig.threads_per_block = (blocksize); \
                kernelLaunchConfig.smem = 0; \
                KernelProperties kernelProperties; \
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                                                                msa_correct_subject_kernel<(blocksize)>, \
                                                                kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
                mymap[kernelLaunchConfig] = kernelProperties; \
            }

            getProp(32);
            getProp(64);
            getProp(96);
            getProp(128);
            getProp(160);
            getProp(192);
            getProp(224);
            getProp(256);

            const auto& kernelProperties = mymap[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

            handle.kernelPropertiesMap[KernelId::MSACorrectSubject] = std::move(mymap);

            #undef getProp
        }else{
            std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
            const KernelProperties& kernelProperties = map[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
        }



        dim3 block(blocksize, 1, 1);
        dim3 grid(std::min(n_subjects, max_blocks_per_device));

        #define mycall(blocksize) msa_correct_subject_kernel<(blocksize)> \
                                <<<grid, block, 0, stream>>>( \
                                    d_consensus, \
                                    d_support, \
                                    d_coverage, \
                                    d_origCoverages, \
                                    d_multiple_sequence_alignments, \
                                    d_msa_column_properties, \
                                    d_indices_per_subject_prefixsum, \
                                    d_is_high_quality_subject, \
                                    d_corrected_subjects, \
                                    d_subject_is_corrected, \
                                    n_subjects, \
                                    n_queries, \
                                    d_num_indices, \
                                    sequence_pitch, \
                                    msa_pitch, \
                                    msa_weights_pitch, \
                                    estimatedErrorrate, \
                                    avg_support_threshold, \
                                    min_support_threshold, \
                                    min_coverage_threshold, \
                                    k_region); CUERR;

        assert(blocksize > 0 && blocksize <= max_block_size);

        switch(blocksize){
            case 32: mycall(32); break;
            case 64: mycall(64); break;
            case 96: mycall(96); break;
            case 128: mycall(128); break;
            case 160: mycall(160); break;
            case 192: mycall(192); break;
            case 224: mycall(224); break;
            case 256: mycall(256); break;
            default: mycall(256); break;
        }
		#undef mycall
    }







    













#if 0
    __global__
    void msa_minimize_kernel(
                            char* __restrict__ d_multiple_sequence_alignments,
                            float* __restrict__ d_multiple_sequence_alignment_weights,
                            const char* __restrict__ d_consensus,
                            const float* __restrict__ d_support,
                            const int* __restrict__ d_coverage,
                            const float* __restrict__ d_origWeights,
                            const int* __restrict__ d_origCoverages,
                            const int* __restrict__ countsA,
                            const int* __restrict__ countsC,
                            const int* __restrict__ countsG,
                            const int* __restrict__ countsT,
                            const float* __restrict__ weightsA,
                            const float* __restrict__ weightsC,
                            const float* __restrict__ weightsG,
                            const float* __restrict__ weightsT,
                            const MSAColumnProperties* __restrict__ d_msa_column_properties,
                            const int* __restrict__ d_indices_per_subject,
                            const int* __restrict__ d_indices_per_subject_prefixsum,
                            int n_subjects,
                            int n_queries,
                            const int* __restrict__ d_num_indices,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            int msa_max_column_count,
                            int blocks_per_msa){

        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

        const int localBlockId = blockIdx.x % blocks_per_msa;
        //const int n_indices = *d_num_indices;

        //process multiple sequence alignment of each subject
        //for each column in msa, find consensus and support
        for(unsigned subjectIndex = blockIdx.x / blocks_per_msa; subjectIndex < n_subjects; subjectIndex += gridDim.x / blocks_per_msa){
            const int subjectColumnsBegin_incl = d_msa_column_properties[subjectIndex].subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = d_msa_column_properties[subjectIndex].subjectColumnsEnd_excl;
            const int columnsToCheck = d_msa_column_properties[subjectIndex].columnsToCheck;

            //number of rows in multiple sequence alignment for subject[subjectIndex]
            const int msa_rows = 1 + d_indices_per_subject[subjectIndex];

            const unsigned offset1 = msa_pitch * (subjectIndex + d_indices_per_subject_prefixsum[subjectIndex]);
            const unsigned offset2 = msa_weights_pitch_floats * (subjectIndex + d_indices_per_subject_prefixsum[subjectIndex]);

            const char* const my_multiple_sequence_alignment = d_multiple_sequence_alignments + offset1;
            const float* const my_multiple_sequence_alignment_weight = d_multiple_sequence_alignment_weights + offset2;

            const char* const my_consensus = d_consensus + subjectIndex * msa_pitch;
            const float* const my_support = d_support + subjectIndex * msa_weights_pitch_floats;
            const int* const my_coverage = d_coverage + subjectIndex * msa_weights_pitch_floats;

            const float* const my_orig_weights = d_origWeights + subjectIndex * msa_weights_pitch_floats;
            const int* const my_orig_coverage = d_origCoverages + subjectIndex * msa_weights_pitch_floats;

            int col = 0;
            bool foundColumn = false;
            char foundBase = 'F';
            int foundBaseIndex = 0;
            int consindex = 0;

            for(int columnindex = subjectColumnsBegin_incl + blockIdx.x * blockDim.x + threadIdx.x; columnindex < subjectColumnsEnd_excl && !foundColumn; columnindex += blockDim.x){

                int counts[4];
                int weights[4];
                counts[0] = countsA[columnindex];
                counts[1] = countsC[columnindex];
                counts[2] = countsG[columnindex];
                counts[3] = countsT[columnindex];
                weights[0] = weightsA[columnindex];
                weights[1] = weightsC[columnindex];
                weights[2] = weightsG[columnindex];
                weights[3] = weightsT[columnindex];

                int columnCoverage = 0;

                for(int row = 0; row < msa_rows; ++row){
                    const char base = my_multiple_sequence_alignment[row * msa_pitch + column];
                    const float weight = my_multiple_sequence_alignment_weight[row * msa_weights_pitch_floats + column];

                    //if(!(base == 'A' || base == 'C' || base == 'G' || base == 'T')){
                    //	assert(base == 'A' || base == 'C' || base == 'G' || base == 'T');
                    //}

                    Aw += (base == 'A' ? weight : 0);
                    Cw += (base == 'C' ? weight : 0);
                    Gw += (base == 'G' ? weight : 0);
                    Tw += (base == 'T' ? weight : 0);

                    As += (base == 'A');
                    Cs += (base == 'C');
                    Gs += (base == 'G');
                    Ts += (base == 'T');

                    columnCoverage += (base == 'A' || base == 'C' || base == 'G' || base == 'T');
                }

                if(columnCoverage <= 0){
                    //assert(columnCoverage > 0);
                }

                const float columnWeight = Aw + Cw + Gw + Tw;
                float consWeight = Aw;
                char cons = 'A';

                cons = Cw > consWeight ? 'C' : cons;
                consWeight = Cw > consWeight ? Cw : consWeight;

                cons = Gw > consWeight ? 'G' : cons;
                consWeight = Gw > consWeight ? Gw : consWeight;

                cons = Tw > consWeight ? 'T' : cons;
                consWeight = Tw > consWeight ? Tw : consWeight;

                char consByCount = 'A';
                int consByCountCount = As;
                consByCount = Cs > consByCountCount ? 'C' : cons;
                consByCountCount = Cs > consByCountCount ? Cs : consWeight;

                consByCount = Gs > consByCountCount ? 'G' : cons;
                consByCountCount = Gs > consByCountCount ? Gs : consWeight;

                consByCount = Ts > consByCountCount ? 'T' : cons;
                consByCountCount = Ts > consByCountCount ? Ts : consWeight;

                if(consByCount != cons){
                    printf("cons differ\n");
                }

                my_consensus[column] = cons;
                my_support[column] = consWeight / columnWeight;
                //printf("subject %d, column %d, support %f\n", subjectIndex, column, my_support[column]);
                my_coverage[column] = columnCoverage;

                if(subjectColumnsBegin_incl <= column && column < subjectColumnsEnd_excl){
                    const char subjectbase = my_multiple_sequence_alignment[column];
                    if(subjectbase == 'A'){
                        my_orig_weights[column] = Aw;
                        my_orig_coverage[column] = As;
                    }else if(subjectbase == 'C'){
                        my_orig_weights[column] = Cw;
                        my_orig_coverage[column] = Cs;
                    }else if(subjectbase == 'G'){
                        my_orig_weights[column] = Gw;
                        my_orig_coverage[column] = Gs;
                    }else if(subjectbase == 'T'){
                        my_orig_weights[column] = Tw;
                        my_orig_coverage[column] = Ts;
                    }
                }
            }
        }
    }
#endif

#endif

}
}
