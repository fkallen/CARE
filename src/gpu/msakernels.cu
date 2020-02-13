//#define NDEBUG

#include <gpu/kernels.hpp>
#include <gpu/devicefunctionsforkernels.cuh>

//#include <gpu/bestalignment.hpp>
#include <bestalignment.hpp>
#include <gpu/utility_kernels.cuh>
#include <gpu/cubcachingallocator.cuh>

#include <msa.hpp>
#include <sequence.hpp>




#include <hpc_helpers.cuh>
#include <config.hpp>

#include <cassert>


#include <cub/cub.cuh>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include <thrust/binary_search.h>


namespace care{
namespace gpu{



    template<int BLOCKSIZE>
    __global__
    void msaInitKernel(
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                const int* __restrict__ indices,
                const int* __restrict__ indices_per_subject,
                const int* __restrict__ candidatesPerSubjectPrefixSum,
                int n_subjects,
                const bool* __restrict__ canExecute){

        if(*canExecute){

            using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;            

            __shared__ union {
                typename BlockReduceInt::TempStorage reduce;
            } temp_storage;

            for(int subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x) {
                MSAColumnProperties* const properties_ptr = d_msapointers.msaColumnProperties + subjectIndex;

                // We only want to consider the candidates with good alignments. the indices of those were determined in a previous step
                const int num_indices_for_this_subject = indices_per_subject[subjectIndex];

                if(num_indices_for_this_subject > 0){
                    const int globalOffset = candidatesPerSubjectPrefixSum[subjectIndex];

                    const int* const myIndicesPtr = indices + globalOffset;
                    const int* const myShiftsPtr = d_alignmentresultpointers.shifts + globalOffset;
                    const BestAlignment_t* const myAlignmentFlagsPtr = d_alignmentresultpointers.bestAlignmentFlags + globalOffset;
                    const int* const myCandidateLengthsPtr = d_sequencePointers.candidateSequencesLength + globalOffset;

                    const int subjectLength = d_sequencePointers.subjectSequencesLength[subjectIndex];
                    int startindex = 0;
                    int endindex = subjectLength;

                    for(int k = threadIdx.x; k < num_indices_for_this_subject; k += BLOCKSIZE) {
                        const int localCandidateIndex = myIndicesPtr[k];

                        const int shift = myShiftsPtr[localCandidateIndex];
                        const BestAlignment_t flag = myAlignmentFlagsPtr[localCandidateIndex];
                        const int queryLength = myCandidateLengthsPtr[localCandidateIndex];

                        assert(flag != BestAlignment_t::None);

                        const int queryEndsAt = queryLength + shift;
                        startindex = min(startindex, shift);
                        endindex = max(endindex, queryEndsAt);
                    }

                    startindex = BlockReduceInt(temp_storage.reduce).Reduce(startindex, cub::Min());
                    __syncthreads();

                    endindex = BlockReduceInt(temp_storage.reduce).Reduce(endindex, cub::Max());
                    __syncthreads();

                    if(threadIdx.x == 0) {
                        MSAColumnProperties my_columnproperties;

                        my_columnproperties.subjectColumnsBegin_incl = max(-startindex, 0);
                        my_columnproperties.subjectColumnsEnd_excl = my_columnproperties.subjectColumnsBegin_incl + subjectLength;
                        my_columnproperties.firstColumn_incl = 0;
                        my_columnproperties.lastColumn_excl = endindex - startindex;

                        *properties_ptr = my_columnproperties;
                    }
                }/*else{
                    //empty MSA
                    if(threadIdx.x == 0) {
                        MSAColumnProperties my_columnproperties;

                        my_columnproperties.subjectColumnsBegin_incl = 0;
                        my_columnproperties.subjectColumnsEnd_excl = 0;
                        my_columnproperties.firstColumn_incl = 0;
                        my_columnproperties.lastColumn_excl = 0;

                        *properties_ptr = my_columnproperties;
                    }
                }*/
            }
        }
    }


    __global__
    void msa_update_properties_kernel(
                MSAPointers d_msapointers,
                const int* __restrict__ d_indices_per_subject,
                size_t msa_weights_pitch,
                int n_subjects,
                const bool* __restrict__ canExecute){

        if(*canExecute){

            const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

            for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x) {
                MSAColumnProperties* const properties_ptr = d_msapointers.msaColumnProperties + subjectIndex;
                const int firstColumn_incl = properties_ptr->firstColumn_incl;
                const int lastColumn_excl = properties_ptr->lastColumn_excl;

                // We only want to consider the candidates with good alignments. the indices of those were determined in a previous step
                const int num_indices_for_this_subject = d_indices_per_subject[subjectIndex];

                if(num_indices_for_this_subject > 0){
                    const int* const my_coverage = d_msapointers.coverage + subjectIndex * msa_weights_pitch_floats;

                    for(int column = threadIdx.x; firstColumn_incl <= column && column < lastColumn_excl-1; column += blockDim.x){
                        assert(my_coverage[column] >= 0);

                        if(my_coverage[column] == 0 && my_coverage[column+1] > 0){
                            properties_ptr->firstColumn_incl = column+1;
                        }

                        if(my_coverage[column] > 0 && my_coverage[column+1] == 0){
                            properties_ptr->lastColumn_excl = column+1;
                        }
                    }

                }else{
                    //clear MSA
                    if(threadIdx.x == 0) {
                        MSAColumnProperties my_columnproperties;

                        my_columnproperties.subjectColumnsBegin_incl = 0;
                        my_columnproperties.subjectColumnsEnd_excl = 0;
                        my_columnproperties.firstColumn_incl = 0;
                        my_columnproperties.lastColumn_excl = 0;

                        *properties_ptr = my_columnproperties;
                    }
                }
            }
        }
    }

    //TODO fix indices
    __global__
    void msa_add_sequences_kernel_implicit_global(
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                ReadQualitiesPointers d_qualityPointers,
    			const int* __restrict__ d_candidates_per_subject_prefixsum,
    			const int* __restrict__ d_indices,
    			const int* __restrict__ d_indices_per_subject,
    			int n_subjects,
    			int n_queries,
    			const int* __restrict__ d_num_indices,
    			bool canUseQualityScores,
    			float desiredAlignmentMaxErrorRate,
                int encodedSequencePitchInInts,
    			size_t qualityPitchInBytes,
    			size_t msa_row_pitch,
                size_t msa_weights_row_pitch,
                const bool* __restrict__ canExecute,
                bool debug){

        if(*canExecute){

            auto get = [] (const char* data, int length, int index){
                return getEncodedNuc2Bit((const unsigned int*)data, length, index, [](auto i){return i;});
            };

            auto make_unpacked_reverse_complement_inplace = [] (std::uint8_t* sequence, int sequencelength){
                return reverseComplementStringInplace((char*)sequence, sequencelength);
            };

            auto getSubjectPtr = [&] (int subjectIndex){
                const unsigned int* result = d_sequencePointers.subjectSequencesData + std::size_t(subjectIndex) * encodedSequencePitchInInts;
                return result;
            };

            auto getCandidatePtr = [&] (int candidateIndex){
                const unsigned int* result = d_sequencePointers.candidateSequencesData + std::size_t(candidateIndex) * encodedSequencePitchInInts;
                return result;
            };

            auto getSubjectQualityPtr = [&] (int subjectIndex){
                const char* result = d_qualityPointers.subjectQualities + std::size_t(subjectIndex) * qualityPitchInBytes;
                return result;
            };

            auto getCandidateQualityPtr = [&] (int candidateIndex){
                const char* result = d_qualityPointers.candidateQualities + std::size_t(candidateIndex) * qualityPitchInBytes;
                return result;
            };

            auto getSubjectLength = [&] (int subjectIndex){
                const int length = d_sequencePointers.subjectSequencesLength[subjectIndex];
                return length;
            };

            auto getCandidateLength = [&] __device__ (int candidateIndex){
                const int length = d_sequencePointers.candidateSequencesLength[candidateIndex];
                return length;
            };

            const size_t msa_weights_row_pitch_floats = msa_weights_row_pitch / sizeof(float);
            const int n_indices = *d_num_indices;

            //add subjects
            for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x) {
                if(d_indices_per_subject[subjectIndex] > 0){
                    const int subjectColumnsBegin_incl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
                    const int subjectColumnsEnd_excl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsEnd_excl;
                    const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
                    const unsigned int* const subject = getSubjectPtr(subjectIndex);
                    const char* const subjectQualityScore = getSubjectQualityPtr(subjectIndex);
                    const int shift = 0;

                    int* const my_coverage = d_msapointers.coverage + subjectIndex * msa_weights_row_pitch_floats;

                    //printf("subject: ");
                    for(int i = threadIdx.x; i < subjectLength; i+= blockDim.x){
                        const int globalIndex = subjectColumnsBegin_incl + shift + i;
                        const char base = get((const char*)subject, subjectLength, i);
                        //printf("%d ", int(base));
                        const float weight = canUseQualityScores ? getQualityWeight(subjectQualityScore[i]) : 1.0f;
                        const int ptrOffset = subjectIndex * 4 * msa_weights_row_pitch_floats + int(base) * msa_weights_row_pitch_floats;
                        atomicAdd(d_msapointers.counts + ptrOffset + globalIndex, 1);
                        atomicAdd(d_msapointers.weights + ptrOffset + globalIndex, weight);
                        atomicAdd(my_coverage + globalIndex, 1);
                    }
                }
            }
            //printf("\n");

            //add candidates //FIXME
            for(unsigned index = blockIdx.x; index < n_indices; index += gridDim.x) {
                const int queryIndex = d_indices[index];

                const int shift = d_alignmentresultpointers.shifts[queryIndex];
                const BestAlignment_t flag = d_alignmentresultpointers.bestAlignmentFlags[queryIndex];

                //find subjectindex
                int subjectIndex = 0;
                for(; subjectIndex < n_subjects; subjectIndex++) {
                    if(queryIndex < d_candidates_per_subject_prefixsum[subjectIndex+1])
                        break;
                }

                if(d_indices_per_subject[subjectIndex] > 0){

                    const int subjectColumnsBegin_incl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
                    const int subjectColumnsEnd_excl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsEnd_excl;
                    const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
                    const int defaultcolumnoffset = subjectColumnsBegin_incl + shift;

                    int* const my_coverage = d_msapointers.coverage + subjectIndex * msa_weights_row_pitch_floats;

                    const unsigned int* const query = getCandidatePtr(queryIndex);
                    const int queryLength = getCandidateLength(queryIndex);
                    const char* const queryQualityScore = getCandidateQualityPtr(index);

                    const int query_alignment_overlap = d_alignmentresultpointers.overlaps[queryIndex];
                    const int query_alignment_nops = d_alignmentresultpointers.nOps[queryIndex];

                    const float overlapweight = calculateOverlapWeight(subjectLength, query_alignment_nops, query_alignment_overlap);

                    assert(overlapweight <= 1.0f);
                    assert(overlapweight >= 0.0f);

                    assert(flag != BestAlignment_t::None);                 // indices should only be pointing to valid alignments
                    //printf("candidate %d, shift %d default %d: ", index, shift, defaultcolumnoffset);
                    //copy query into msa
                    if(flag == BestAlignment_t::Forward) {
                        for(int i = threadIdx.x; i < queryLength; i+= blockDim.x){
                            const int globalIndex = defaultcolumnoffset + i;
                            const char base = get((const char*)query, queryLength, i);
                            //printf("%d ", int(base));
                            const float weight = canUseQualityScores ? getQualityWeight(queryQualityScore[i]) * overlapweight : overlapweight;
                            const int ptrOffset = subjectIndex * 4 * msa_weights_row_pitch_floats + int(base) * msa_weights_row_pitch_floats;
                            atomicAdd(d_msapointers.counts + ptrOffset + globalIndex, 1);
                            atomicAdd(d_msapointers.weights + ptrOffset + globalIndex, weight);
                            atomicAdd(my_coverage + globalIndex, 1);
                        }
                    }else{
                        auto make_reverse_complement_byte = [](std::uint8_t in) -> std::uint8_t{
                            constexpr std::uint8_t mask = 0x03;
                            return (~in & mask);
                        };

                        for(int i = threadIdx.x; i < queryLength; i+= blockDim.x){
                            const int reverseIndex = queryLength - 1 - i;
                            const int globalIndex = defaultcolumnoffset + i;
                            const char base = get((const char*)query, queryLength, reverseIndex);
                            const char revCompl = make_reverse_complement_byte(base);
                            //printf("%d ", int(revCompl));
                            const float weight = canUseQualityScores ? getQualityWeight(queryQualityScore[reverseIndex]) * overlapweight : overlapweight;
                            const int ptrOffset = subjectIndex * 4 * msa_weights_row_pitch_floats + int(revCompl) * msa_weights_row_pitch_floats;
                            atomicAdd(d_msapointers.counts + ptrOffset + globalIndex, 1);
                            atomicAdd(d_msapointers.weights + ptrOffset + globalIndex, weight);
                            atomicAdd(my_coverage + globalIndex, 1);
                        }
                    }
                }

            }
        }
    }

    __global__
    void msaAddSequencesSmemWithSmallIfIntUnrolledQualitiesUnrolledKernel(
                char* __restrict__ consensus,
                float* __restrict__ support,
                int* __restrict__ coverage,
                float* __restrict__ origWeights,
                int* __restrict__ origCoverages,
                MSAColumnProperties* __restrict__ msaColumnProperties,
                int* __restrict__ counts,
                float* __restrict__ weights,
                const int* __restrict__ overlaps,
                const int* __restrict__ shifts,
                const int* __restrict__ nOps,
                const BestAlignment_t* __restrict__ bestAlignmentFlags,
                const unsigned int* __restrict__ subjectSequencesData,
                const unsigned int* __restrict__ candidateSequencesData,
                const int* __restrict__ subjectSequencesLength,
                const int* __restrict__ candidateSequencesLength,
                const char* __restrict__ subjectQualities,
                const char* __restrict__ candidateQualities,
                const int* __restrict__ d_candidates_per_subject_prefixsum,
                const int* __restrict__ d_indices,
                const int* __restrict__ d_indices_per_subject,
                const int* __restrict__ blocks_per_subject_prefixsum,
                int n_subjects,
                int n_queries,
                const int* __restrict__ d_num_indices,
                bool canUseQualityScores,
                size_t encodedSequencePitchInInts,
                size_t qualityPitchInBytes,
                size_t lengthOfMSARow,
                const bool* __restrict__ canExecute){

        constexpr bool candidatesAreTransposed = true;

        if(*canExecute){

            // sizeof(float) * 4 * msa_weights_row_pitch_floats // weights
            //+ sizeof(int) * 4 * msa_weights_row_pitch_floats // counts
            extern __shared__ float sharedmem[];

            auto get = [] (const unsigned int* data, int length, int index, auto trafo){
                return getEncodedNuc2Bit((const unsigned int*)data, length, index, trafo);
            };
            
            auto getEncodedNucFromInt2Bit = [](unsigned int data, int pos){
                return ((data >> (30 - 2*pos)) & 0x00000003);
            };

            const size_t msa_weights_row_pitch_floats = lengthOfMSARow;
            const int smemsizefloats = 4 * msa_weights_row_pitch_floats + 4 * msa_weights_row_pitch_floats;

            float* const shared_weights = sharedmem;
            int* const shared_counts = (int*)(shared_weights + 4 * msa_weights_row_pitch_floats);

            //const int requiredTiles = n_subjects;//blocks_per_subject_prefixsum[n_subjects];
            const int requiredTiles = blocks_per_subject_prefixsum[n_subjects];

            for(int logicalBlockId = blockIdx.x; logicalBlockId < requiredTiles; logicalBlockId += gridDim.x){
                //clear shared memory
                for(int i = threadIdx.x; i < smemsizefloats; i += blockDim.x){
                    sharedmem[i] = 0;
                }
                __syncthreads();

                const int subjectIndex = thrust::distance(blocks_per_subject_prefixsum,
                                                        thrust::lower_bound(
                                                            thrust::seq,
                                                            blocks_per_subject_prefixsum,
                                                            blocks_per_subject_prefixsum + n_subjects + 1,
                                                            logicalBlockId + 1))-1;

                if(d_indices_per_subject[subjectIndex] > 0){

                    const int blockForThisSubject = logicalBlockId - blocks_per_subject_prefixsum[subjectIndex];

                    const int globalOffset = d_candidates_per_subject_prefixsum[subjectIndex];
                    const int* const indices_for_this_subject = d_indices + globalOffset;
                    const int id = blockForThisSubject * blockDim.x + threadIdx.x;
                    const int maxid_excl = d_indices_per_subject[subjectIndex];

                    const int subjectColumnsBegin_incl = msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
                    const int subjectColumnsEnd_excl = msaColumnProperties[subjectIndex].subjectColumnsEnd_excl;
                    const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
                    const int columnsToCheck = msaColumnProperties[subjectIndex].lastColumn_excl;

                    int* const my_coverage = coverage + subjectIndex * msa_weights_row_pitch_floats;

                    //if(size_t(columnsToCheck) > msa_weights_row_pitch_floats){
                    //    printf("columnsToCheck %d, msa_weights_row_pitch_floats %lu\n", columnsToCheck, msa_weights_row_pitch_floats);
                        assert(columnsToCheck <= msa_weights_row_pitch_floats);
                    //}


                    //ensure that the subject is only inserted once, by the first block
                    if(blockForThisSubject == 0){
                        const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
                        const unsigned int* const subject = subjectSequencesData + subjectIndex * encodedSequencePitchInInts;
                        const char* const subjectQualityScore = subjectQualities + subjectIndex * qualityPitchInBytes;

                        for(int i = threadIdx.x; i < subjectLength; i+= blockDim.x){
                            const int shift = 0;
                            const int globalIndex = subjectColumnsBegin_incl + shift + i;
                            const char base = get(subject, subjectLength, i, [](auto i){return i;});

                            const float weight = canUseQualityScores ? getQualityWeight(subjectQualityScore[i]) : 1.0f;
                            const int ptrOffset = int(base) * msa_weights_row_pitch_floats;
                            atomicAdd(shared_counts + ptrOffset + globalIndex, 1);
                            atomicAdd(shared_weights + ptrOffset + globalIndex, weight);
                            atomicAdd(my_coverage + globalIndex, 1);
                        }
                    }

                    if(id < maxid_excl){
                        const int queryIndex = indices_for_this_subject[id] + globalOffset;
                        const int shift = shifts[queryIndex];
                        const BestAlignment_t flag = bestAlignmentFlags[queryIndex];
                        const int defaultcolumnoffset = subjectColumnsBegin_incl + shift;

                        const unsigned int* const query = candidateSequencesData + (candidatesAreTransposed ? queryIndex : queryIndex * encodedSequencePitchInInts);
                        const int queryLength = candidateSequencesLength[queryIndex];
                        const char* const queryQualityScore = candidateQualities + queryIndex * qualityPitchInBytes;

                        const int query_alignment_overlap = overlaps[queryIndex];
                        const int query_alignment_nops = nOps[queryIndex];

                        const float overlapweight = calculateOverlapWeight(subjectLength, query_alignment_nops, query_alignment_overlap);
                        assert(overlapweight <= 1.0f);
                        assert(overlapweight >= 0.0f);

                        assert(flag != BestAlignment_t::None); // indices should only be pointing to valid alignments

                        const bool isForward = (flag == BestAlignment_t::Forward);

                        constexpr int nucleotidesPerInt2Bit = 16;

                        const int fullInts = queryLength / nucleotidesPerInt2Bit;
                        for(int intIndex = 0; intIndex < fullInts; intIndex++){
                            const unsigned int currentDataInt = ((unsigned int*)query)[intIndex * (candidatesAreTransposed ? n_queries : 1)];

                            for(int k = 0; k < 4; k++){
                                alignas(4) char currentFourQualities[4];

                                assert(size_t(&currentFourQualities[0]) % 4 == 0);

                                if(canUseQualityScores){
                                    *((int*)&currentFourQualities[0]) = ((const int*)queryQualityScore)[intIndex * 4 + k];
                                }

                                //#pragma unroll
                                for(int l = 0; l < 4; l++){
                                    const int posInInt = k * 4 + l;

                                    unsigned int encodedBaseAsInt = getEncodedNucFromInt2Bit(currentDataInt, posInInt);
                                    if(!isForward){
                                        //reverse complement
                                        encodedBaseAsInt = (~encodedBaseAsInt & 0x00000003);
                                    }
                                    const float weight = canUseQualityScores ? getQualityWeight(currentFourQualities[l]) * overlapweight : overlapweight;

                                    assert(weight != 0);
                                    const int ptrOffset = encodedBaseAsInt * msa_weights_row_pitch_floats;
                                    const int globalIndex = defaultcolumnoffset + (isForward ? (intIndex * 16 + posInInt) : queryLength - 1 - (intIndex * 16 + posInInt));
                                    atomicAdd(shared_counts + ptrOffset + globalIndex, 1);
                                    atomicAdd(shared_weights + ptrOffset + globalIndex, weight);
                                    atomicAdd(my_coverage + globalIndex, 1);

                                }
                            }
                        }

                        //add remaining positions
                        if(queryLength % nucleotidesPerInt2Bit != 0){
                            const unsigned int currentDataInt = ((unsigned int*)query)[fullInts * (candidatesAreTransposed ? n_queries : 1)];
                            const int maxPos = queryLength - fullInts * 16;
                            for(int posInInt = 0; posInInt < maxPos; posInInt++){
                                unsigned int encodedBaseAsInt = getEncodedNucFromInt2Bit(currentDataInt, posInInt);
                                if(!isForward){
                                    //reverse complement
                                    encodedBaseAsInt = (~encodedBaseAsInt & 0x00000003);
                                }
                                const float weight = canUseQualityScores ? getQualityWeight(queryQualityScore[fullInts * 16 + posInInt]) * overlapweight : overlapweight;

                                assert(weight != 0);
                                const int ptrOffset = encodedBaseAsInt * msa_weights_row_pitch_floats;
                                const int globalIndex = defaultcolumnoffset + (isForward ? (fullInts * 16 + posInInt) : queryLength - 1 - (fullInts * 16 + posInInt));
                                atomicAdd(shared_counts + ptrOffset + globalIndex, 1);
                                atomicAdd(shared_weights + ptrOffset + globalIndex, weight);
                                atomicAdd(my_coverage + globalIndex, 1);
                            } 
                        }

                        //printf("\n");
                    }

                    __syncthreads();

                    for(int index = threadIdx.x; index < columnsToCheck; index += blockDim.x){
                        for(int k = 0; k < 4; k++){
                            const int* const srcCounts = shared_counts + k * msa_weights_row_pitch_floats + index;
                            int* const destCounts = counts + 4 * msa_weights_row_pitch_floats * subjectIndex + k * msa_weights_row_pitch_floats + index;
                            const float* const srcWeights = shared_weights + k * msa_weights_row_pitch_floats + index;
                            float* const destWeights = weights + 4 * msa_weights_row_pitch_floats * subjectIndex + k * msa_weights_row_pitch_floats + index;
                            atomicAdd(destCounts ,*srcCounts);
                            atomicAdd(destWeights, *srcWeights);
                        }
                    }

                    __syncthreads();
                }
            }
        }
    }



    __global__
    void check_built_msa_kernel(MSAPointers d_msapointers,
                                const int* __restrict__ d_indices_per_subject,
                                int nSubjects,
                                size_t msa_weights_row_pitch,
                                const bool* __restrict__ canExecute){

        if(*canExecute){

            const size_t msa_weights_row_pitch_floats = msa_weights_row_pitch / sizeof(float);

            for(int subjectIndex = blockIdx.x; subjectIndex < nSubjects; subjectIndex += gridDim.x){
                if(d_indices_per_subject[subjectIndex] > 0){
                    const int firstColumn_incl = d_msapointers.msaColumnProperties[subjectIndex].firstColumn_incl;
                    const int lastColumn_excl = d_msapointers.msaColumnProperties[subjectIndex].lastColumn_excl;

                    for(int column = firstColumn_incl + threadIdx.x; column < lastColumn_excl; column += blockDim.x){
                        const int* const counts = d_msapointers.counts + 4 * msa_weights_row_pitch_floats * subjectIndex + column;
                        const float* const weights = d_msapointers.weights + 4 * msa_weights_row_pitch_floats * subjectIndex + column;

                        for(int k = 0; k < 4; k++){
                            const int count = counts[k * msa_weights_row_pitch_floats];
                            const float weight = weights[k * msa_weights_row_pitch_floats];
                            if(count > 0 && weight <= 0.0f){
                                printf("msa check failed! subjectIndex %d, column %d, base %d, count %d, weight %f\n",
                                    subjectIndex, column, k, count, weight);
                                assert(false);
                            }

                            if(count <= 0 && weight > 0.0f){
                                printf("msa check failed! subjectIndex %d, column %d, base %d, count %d, weight %f\n",
                                    subjectIndex, column, k, count, weight);
                                assert(false);
                            }
                        }
                    }                        
                }
            }
        }
    }

    template<int BLOCKSIZE, int blocks_per_msa>
    __global__
    void msa_find_consensus_implicit_kernel(
                            MSAPointers d_msapointers,
                            ReadSequencesPointers d_sequencePointers,
                            const int* __restrict__ d_indices_per_subject,
                            int n_subjects,
                            int encodedSequencePitchInInts,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            const bool* __restrict__ canExecute){

        if(*canExecute){

            constexpr char A_enc = 0x00;
            constexpr char C_enc = 0x01;
            constexpr char G_enc = 0x02;
            constexpr char T_enc = 0x03;

            // using BlockReduceFloat = cub::BlockReduce<float, BLOCKSIZE>;

            // __shared__ union {
            //     typename BlockReduceFloat::TempStorage floatreduce;
            // } temp_storage;

            // __shared__ float avgCountPerWeight[4];

            auto get = [] (const char* data, int length, int index){
                return getEncodedNuc2Bit((const unsigned int*)data, length, index, [](auto i){return i;});
            };

            auto getSubjectPtr = [&] (int subjectIndex){
                const unsigned int* result = d_sequencePointers.subjectSequencesData + std::size_t(subjectIndex) * encodedSequencePitchInInts;
                return result;
            };

            const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

            const int localBlockId = blockIdx.x % blocks_per_msa;
            //const int n_indices = *d_num_indices;

            //process multiple sequence alignment of each subject
            //for each column in msa, find consensus and support
            for(int subjectIndex = blockIdx.x / blocks_per_msa; subjectIndex < n_subjects; subjectIndex += gridDim.x / blocks_per_msa){
                if(d_indices_per_subject[subjectIndex] > 0){
                    const int subjectColumnsBegin_incl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
                    const int subjectColumnsEnd_excl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsEnd_excl;
                    const int firstColumn_incl = d_msapointers.msaColumnProperties[subjectIndex].firstColumn_incl;
                    const int lastColumn_excl = d_msapointers.msaColumnProperties[subjectIndex].lastColumn_excl;

                    assert(lastColumn_excl <= msa_weights_pitch_floats);

                    const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
                    const unsigned int* const subject = getSubjectPtr(subjectIndex);

                    char* const my_consensus = d_msapointers.consensus + subjectIndex * msa_pitch;
                    float* const my_support = d_msapointers.support + subjectIndex * msa_weights_pitch_floats;

                    float* const my_orig_weights = d_msapointers.origWeights + subjectIndex * msa_weights_pitch_floats;
                    int* const my_orig_coverage = d_msapointers.origCoverages + subjectIndex * msa_weights_pitch_floats;

                    const int* const myCountsA = d_msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 0 * msa_weights_pitch_floats;
                    const int* const myCountsC = d_msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 1 * msa_weights_pitch_floats;
                    const int* const myCountsG = d_msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 2 * msa_weights_pitch_floats;
                    const int* const myCountsT = d_msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 3 * msa_weights_pitch_floats;

                    const float* const my_weightsA = d_msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 0 * msa_weights_pitch_floats;
                    const float* const my_weightsC = d_msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 1 * msa_weights_pitch_floats;
                    const float* const my_weightsG = d_msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 2 * msa_weights_pitch_floats;
                    const float* const my_weightsT = d_msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 3 * msa_weights_pitch_floats;

                    //calculate average count per weight
                    // float myaverageCountPerWeightA = 0.0f;
                    // float myaverageCountPerWeightG = 0.0f;
                    // float myaverageCountPerWeightC = 0.0f;
                    // float myaverageCountPerWeightT = 0.0f;

                    // for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                    //     assert(i < lastColumn_excl);

                    //     const int ca = myCountsA[i];
                    //     const int cc = myCountsC[i];
                    //     const int cg = myCountsG[i];
                    //     const int ct = myCountsT[i];
                    //     const float wa = my_weightsA[i];
                    //     const float wc = my_weightsC[i];
                    //     const float wg = my_weightsG[i];
                    //     const float wt = my_weightsT[i];

                    //     myaverageCountPerWeightA += ca / wa;
                    //     myaverageCountPerWeightC += cc / wc;
                    //     myaverageCountPerWeightG += cg / wg;
                    //     myaverageCountPerWeightT += ct / wt;
                    // }

                    // myaverageCountPerWeightA = BlockReduceFloat(temp_storage.floatreduce).Sum(myaverageCountPerWeightA);
                    // __syncthreads();
                    // myaverageCountPerWeightC = BlockReduceFloat(temp_storage.floatreduce).Sum(myaverageCountPerWeightC);
                    // __syncthreads();
                    // myaverageCountPerWeightG = BlockReduceFloat(temp_storage.floatreduce).Sum(myaverageCountPerWeightG);
                    // __syncthreads();
                    // myaverageCountPerWeightT = BlockReduceFloat(temp_storage.floatreduce).Sum(myaverageCountPerWeightT);

                    // if(threadIdx.x == 0){
                    //     avgCountPerWeight[0] = myaverageCountPerWeightA / (subjectColumnsEnd_excl - subjectColumnsBegin_incl);
                    //     avgCountPerWeight[1] = myaverageCountPerWeightC / (subjectColumnsEnd_excl - subjectColumnsBegin_incl);
                    //     avgCountPerWeight[2] = myaverageCountPerWeightG / (subjectColumnsEnd_excl - subjectColumnsBegin_incl);
                    //     avgCountPerWeight[3] = myaverageCountPerWeightT / (subjectColumnsEnd_excl - subjectColumnsBegin_incl);
                    // }
                    //__syncthreads();

                    for(int column = localBlockId * blockDim.x + threadIdx.x; firstColumn_incl <= column && column < lastColumn_excl; column += blocks_per_msa * BLOCKSIZE){
                        const int ca = myCountsA[column];
                        const int cc = myCountsC[column];
                        const int cg = myCountsG[column];
                        const int ct = myCountsT[column];
                        const float wa = my_weightsA[column];
                        const float wc = my_weightsC[column];
                        const float wg = my_weightsG[column];
                        const float wt = my_weightsT[column];

                        char cons = 'F';
                        float consWeight = 0.0f;
                        //float consWeightPerCount = 0.0f;
                        //float weightPerCountSum = 0.0f;
                        //if(ca != 0){
                        if(wa > consWeight){
                            cons = 'A';
                            consWeight = wa;
                            //consWeightPerCount = wa / ca;
                            //weightPerCountSum += wa / ca;
                        }
                        //if(cc != 0 && wc / cc > consWeightPerCount){
                        if(wc > consWeight){
                            cons = 'C';
                            consWeight = wc;
                            //consWeightPerCount = wc / cc;
                            //weightPerCountSum += wc / cc;
                        }
                        //if(cg != 0 && wg / cg > consWeightPerCount){
                        if(wg > consWeight){
                            cons = 'G';
                            consWeight = wg;
                            //consWeightPerCount = wg / cg;
                            //weightPerCountSum += wg / cg;
                        }
                        //if(ct != 0 && wt / ct > consWeightPerCount){
                        if(wt > consWeight){
                            cons = 'T';
                            consWeight = wt;
                            //consWeightPerCount = wt / ct;
                            //weightPerCountSum += wt / ct;
                        }
                        my_consensus[column] = cons;
                        const float columnWeight = wa + wc + wg + wt;
                        if(columnWeight == 0){
                            printf("s %d c %d\n", subjectIndex, column);
                            assert(columnWeight != 0);
                        }
                        //assert(weightPerCountSum != 0);
                        my_support[column] = consWeight / columnWeight;
                        //my_support[column] = consWeightPerCount / weightPerCountSum;


                        if(subjectColumnsBegin_incl <= column && column < subjectColumnsEnd_excl){

                            const int localIndex = column - subjectColumnsBegin_incl;
                            const char subjectbase = get((const char*)subject, subjectLength, localIndex);

                            if(subjectbase == A_enc){
                                my_orig_weights[column] = wa;
                                my_orig_coverage[column] = myCountsA[column];
                                //printf("%c", 'A');
                                //printf("%d %d %d %c\n", column, localIndex, my_orig_coverage[column], 'A');
                            }else if(subjectbase == C_enc){
                                my_orig_weights[column] = wc;
                                my_orig_coverage[column] = myCountsC[column];
                                //printf("%c", 'C');
                                //printf("%d %d %d %c\n", column, localIndex, my_orig_coverage[column], 'C');
                            }else if(subjectbase == G_enc){
                                my_orig_weights[column] = wg;
                                my_orig_coverage[column] = myCountsG[column];
                                //printf("%c", 'G');
                                //printf("%d %d %d %c\n", column, localIndex, my_orig_coverage[column], 'G');
                            }else if(subjectbase == T_enc){
                                my_orig_weights[column] = wt;
                                my_orig_coverage[column] = myCountsT[column];
                                //printf("%c", 'T');
                                //printf("%d %d %d %c\n", column, localIndex, my_orig_coverage[column], 'T');
                            }
                        }
                    }
                    //printf("\n");
                }
            }
        }
    }


/*
        This kernel inspects a msa and identifies candidates which could originate
        from a different genome region than the subject.

        the output element d_shouldBeKept[i] indicates whether
        the candidate referred to by d_indices[i] should remain in the msa
    */


    template<int BLOCKSIZE>
    __global__
    void msa_findCandidatesOfDifferentRegion_kernel(
            int* __restrict__ d_newIndices,
            int* __restrict__ d_newNumIndicesPerSubject,
            int* __restrict__ d_newNumIndices,

            MSAPointers d_msapointers,
            AlignmentResultPointers d_alignmentresultpointers,
            ReadSequencesPointers d_sequencePointers,
            bool* __restrict__ d_shouldBeKept,
            const int* __restrict__ d_candidates_per_subject_prefixsum,
            int n_subjects,
            int n_candidates,
            int encodedSequencePitchInInts,
            size_t msa_pitch,
            size_t msa_weights_pitch,
            const int* __restrict__ d_indices,
            const int* __restrict__ d_indices_per_subject,
            float desiredAlignmentMaxErrorRate,
            int dataset_coverage,
            const bool* __restrict__ canExecute,
            const unsigned int* d_readids,
            bool debug = false){

        if(*canExecute){

            // if(threadIdx.x + blockIdx.x * blockDim.x == 0){
            //     printf("msa_findCandidatesOfDifferentRegion_kernel\n");
            // }

            auto getNumBytes = [] (int sequencelength){
                return sizeof(unsigned int) * getEncodedNumInts2Bit(sequencelength);
            };

            auto get = [] (const char* data, int length, int index){
                return getEncodedNuc2Bit((const unsigned int*)data, length, index, [](auto i){return i;});
            };

            auto getSubjectPtr = [&] (int subjectIndex){
                const unsigned int* result = d_sequencePointers.subjectSequencesData + std::size_t(subjectIndex) * encodedSequencePitchInInts;
                return result;
            };

            auto getCandidatePtr = [&] (int candidateIndex){
                const unsigned int* result = d_sequencePointers.candidateSequencesData + std::size_t(candidateIndex) * encodedSequencePitchInInts;
                return result;
            };

            auto getSubjectLength = [&] (int subjectIndex){
                const int length = d_sequencePointers.subjectSequencesLength[subjectIndex];
                return length;
            };

            auto getCandidateLength = [&] (int candidateIndex){
                const int length = d_sequencePointers.candidateSequencesLength[candidateIndex];
                return length;
            };

            auto is_significant_count = [](int count, int coverage){
                /*if(ceil(estimatedErrorrate * coverage)*2 <= count ){
                    return true;
                }
                return false;*/
                if(int(coverage * 0.3f) <= count)
                    return true;
                return false;

            };

            const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);
            //const char index_to_base[4]{'A','C','G','T'};

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

            using BlockReduceBool = cub::BlockReduce<bool, BLOCKSIZE>;
            using BlockReduceInt2 = cub::BlockReduce<int2, BLOCKSIZE>;

            __shared__ union{
                typename BlockReduceBool::TempStorage boolreduce;
                typename BlockReduceInt2::TempStorage int2reduce;
            } temp_storage;

            __shared__ bool broadcastbufferbool;
            __shared__ int broadcastbufferint4[4];
            __shared__ int totalIndices;
            __shared__ int counts[1];

            extern __shared__ unsigned int sharedmemory[];

            if(threadIdx.x == 0){
                totalIndices = 0;
            }
            __syncthreads();

            for(int subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){

                const int globalOffset = d_candidates_per_subject_prefixsum[subjectIndex];

                const int* myIndices = d_indices + globalOffset;
                const int myNumIndices = d_indices_per_subject[subjectIndex];

                int* const myNewIndicesPtr = d_newIndices + globalOffset;
                int* const myNewNumIndicesPerSubjectPtr = d_newNumIndicesPerSubject + subjectIndex;

                if(debug && threadIdx.x == 0){
                    //printf("myNumIndices %d\n", myNumIndices);
                }

                if(myNumIndices > 0){

                    const unsigned int* const subjectptr = getSubjectPtr(subjectIndex);
                    const int subjectLength = getSubjectLength(subjectIndex);

                    const char* myConsensus = d_msapointers.consensus + subjectIndex * msa_pitch;

                    const int subjectColumnsBegin_incl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
                    const int subjectColumnsEnd_excl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsEnd_excl;

                    //check if subject and consensus differ at at least one position

                    bool hasMismatchToConsensus = false;

                    for(int pos = threadIdx.x; pos < subjectLength && !hasMismatchToConsensus; pos += blockDim.x){
                        const int column = subjectColumnsBegin_incl + pos;
                        const char consbase = myConsensus[column];
                        const char subjectbase = to_nuc(get((const char*)subjectptr, subjectLength, pos));

                        hasMismatchToConsensus |= (consbase != subjectbase);
                    }

                    hasMismatchToConsensus = BlockReduceBool(temp_storage.boolreduce).Reduce(hasMismatchToConsensus, [](auto l, auto r){return l || r;});

                    if(threadIdx.x == 0){
                        broadcastbufferbool = hasMismatchToConsensus;
                    }
                    __syncthreads();

                    hasMismatchToConsensus = broadcastbufferbool;

                    //if subject and consensus differ at at least one position, check columns in msa

                    if(hasMismatchToConsensus){
                        int col = std::numeric_limits<int>::max();
                        bool foundColumn = false;
                        char foundBase = 'F';
                        int foundBaseIndex = std::numeric_limits<int>::max();
                        int consindex = std::numeric_limits<int>::max();

                        const int* const myCountsA = d_msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 0 * msa_weights_pitch_floats;
                        const int* const myCountsC = d_msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 1 * msa_weights_pitch_floats;
                        const int* const myCountsG = d_msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 2 * msa_weights_pitch_floats;
                        const int* const myCountsT = d_msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 3 * msa_weights_pitch_floats;

                        const float* const myWeightsA = d_msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 0 * msa_weights_pitch_floats;
                        const float* const myWeightsC = d_msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 1 * msa_weights_pitch_floats;
                        const float* const myWeightsG = d_msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 2 * msa_weights_pitch_floats;
                        const float* const myWeightsT = d_msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 3 * msa_weights_pitch_floats;

                        const float* const mySupport = d_msapointers.support + subjectIndex * msa_weights_pitch_floats;

                        for(int columnindex = subjectColumnsBegin_incl + threadIdx.x; columnindex < subjectColumnsEnd_excl && !foundColumn; columnindex += blockDim.x){
                            int counts[4];
                            counts[0] = myCountsA[columnindex];
                            counts[1] = myCountsC[columnindex];
                            counts[2] = myCountsG[columnindex];
                            counts[3] = myCountsT[columnindex];

                            float weights[4];
                            weights[0] = myWeightsA[columnindex];
                            weights[1] = myWeightsC[columnindex];
                            weights[2] = myWeightsG[columnindex];
                            weights[3] = myWeightsT[columnindex];

                            const float support = mySupport[columnindex];

                            const char consbase = myConsensus[columnindex];
                            consindex = -1;

                            switch(consbase){
                                case 'A': consindex = 0;break;
                                case 'C': consindex = 1;break;
                                case 'G': consindex = 2;break;
                                case 'T': consindex = 3;break;
                            }

                            // char consensusByCount = 'A';
                            // int maxCount = counts[0];
                            // if(counts[1] > maxCount){
                            //     consensusByCount = 'C';
                            //     maxCount = counts[1];
                            // }
                            // if(counts[2] > maxCount){
                            //     consensusByCount = 'G';
                            //     maxCount = counts[2];
                            // }
                            // if(counts[3] > maxCount){
                            //     consensusByCount = 'T';
                            //     maxCount = counts[3];
                            // }
                            //
                            // if(consbase != consensusByCount){
                            //     printf("bycounts %c %.6f %.6f %.6f %.6f,\nbyweight %c %.6f %.6f %.6f %.6f\n\n",
                            //             consensusByCount, float(counts[0]), float(counts[1]), float(counts[2]), float(counts[3]),
                            //             consbase, weights[0], weights[1], weights[2], weights[3]);
                            // }

                            //find out if there is a non-consensus base with significant coverage
                            int significantBaseIndex = -1;

                            #pragma unroll
                            for(int i = 0; i < 4; i++){
                                if(i != consindex){
                                    //const bool significant = is_significant_count(counts[i], dataset_coverage);
                                    //const int columnCoverage = counts[0] + counts[1] +counts[2] + counts[3];

                                    const bool significant = is_significant_count(counts[i], dataset_coverage);

                                    //const bool significant = weights[i] / support >= 0.5f;

                                    significantBaseIndex = significant ? i : significantBaseIndex;
                                }
                            }

                            if(significantBaseIndex != -1){
                                foundColumn = true;
                                col = columnindex;
                                foundBaseIndex = significantBaseIndex;

                                // if(debug){
                                //     printf("found col %d, baseIndex %d\n", col, foundBaseIndex);
                                // }
                            }
                        }

                        int2 packed{col, foundBaseIndex};
                        //find packed value with smallest col
                        packed = BlockReduceInt2(temp_storage.int2reduce).Reduce(packed, [](auto l, auto r){
                            if(l.x < r.x){
                                return l;
                            }else{
                                return r;
                            }
                        });

                        if(threadIdx.x == 0){
                            if(packed.x != std::numeric_limits<int>::max()){
                                broadcastbufferint4[0] = 1;
                                broadcastbufferint4[1] = packed.x;
                                broadcastbufferint4[2] = to_nuc(packed.y);
                                broadcastbufferint4[3] = packed.y;
                            }else{
                                broadcastbufferint4[0] = 0;
                            }
                        }

                        __syncthreads();

                        foundColumn = (1 == broadcastbufferint4[0]);
                        col = broadcastbufferint4[1];
                        foundBase = broadcastbufferint4[2];
                        foundBaseIndex = broadcastbufferint4[3];

                        // if(debug && threadIdx.x == 0 /*&& d_readids[subjectIndex] == 207*/){
                        //     printf("reduced: found a column: %d, found col %d, found base %c, baseIndex %d\n", foundColumn, col, foundBase, foundBaseIndex);
                        // }

                        if(foundColumn){

                            //compare found base to original base
                            const char originalbase = to_nuc(get((const char*)subjectptr, subjectLength, col - subjectColumnsBegin_incl));

                            /*int counts[4];

                            counts[0] = myCountsA[col];
                            counts[1] = myCountsC[col];
                            counts[2] = myCountsG[col];
                            counts[3] = myCountsT[col];*/

                            auto discard_rows = [&](bool keepMatching){

                                for(int k = threadIdx.x; k < myNumIndices; k += blockDim.x){
                                    const int localCandidateIndex = myIndices[k];
                                    const int globalCandidateIndex = globalOffset + localCandidateIndex;
                                    const unsigned int* const candidateptr = getCandidatePtr(globalCandidateIndex);
                                    const int candidateLength = getCandidateLength(globalCandidateIndex);
                                    const int shift = d_alignmentresultpointers.shifts[globalCandidateIndex];
                                    const BestAlignment_t alignmentFlag = d_alignmentresultpointers.bestAlignmentFlags[globalCandidateIndex];

                                    //check if row is affected by column col
                                    const int row_begin_incl = subjectColumnsBegin_incl + shift;
                                    const int row_end_excl = row_begin_incl + candidateLength;
                                    const bool notAffected = (col < row_begin_incl || row_end_excl <= col);
                                    char base = 'F';
                                    if(!notAffected){
                                        if(alignmentFlag == BestAlignment_t::Forward){
                                            base = to_nuc(get((const char*)candidateptr, candidateLength, (col - row_begin_incl)));
                                        }else{
                                            assert(alignmentFlag == BestAlignment_t::ReverseComplement); //all candidates of MSA must not have alignmentflag None
                                            const char forwardbaseEncoded = get((const char*)candidateptr, candidateLength, row_end_excl-1 - col);
                                            base = to_nuc((~forwardbaseEncoded & 0x03));
                                        }
                                    }

                                    if(notAffected || (!(keepMatching ^ (base == foundBase)))){
                                        d_shouldBeKept[globalOffset + k] = true; //same region
                                    }else{
                                        d_shouldBeKept[globalOffset + k] = false; //different region
                                    }
                                }
    #if 1
                                //check that no candidate which should be removed has very good alignment.
                                //if there is such a candidate, none of the candidates will be removed.
                                bool veryGoodAlignment = false;
                                for(int k = threadIdx.x; k < myNumIndices && !veryGoodAlignment; k += blockDim.x){
                                    if(!d_shouldBeKept[globalOffset + k]){
                                        const int localCandidateIndex = myIndices[k];
                                        const int globalCandidateIndex = globalOffset + localCandidateIndex;
                                        const int nOps = d_alignmentresultpointers.nOps[globalCandidateIndex];
                                        const int overlapsize = d_alignmentresultpointers.overlaps[globalCandidateIndex];
                                        const float overlapweight = calculateOverlapWeight(subjectLength, nOps, overlapsize);
                                        assert(overlapweight <= 1.0f);
                                        assert(overlapweight >= 0.0f);

                                        if(overlapweight >= 0.90f){
                                            veryGoodAlignment = true;
                                        }
                                    }
                                }

                                veryGoodAlignment = BlockReduceBool(temp_storage.boolreduce).Reduce(veryGoodAlignment, [](auto l, auto r){return l || r;});

                                if(threadIdx.x == 0){
                                    broadcastbufferbool = veryGoodAlignment;
                                }
                                __syncthreads();

                                veryGoodAlignment = broadcastbufferbool;

                                if(veryGoodAlignment){
                                    for(int k = threadIdx.x; k < myNumIndices; k += blockDim.x){
                                        d_shouldBeKept[globalOffset + k] = true;
                                    }
                                }
    #endif

                                //select indices of candidates to keep and write them to new indices
                                if(threadIdx.x == 0){
                                    counts[0] = 0;
                                }
                                __syncthreads();

                                const int limit = SDIV(myNumIndices, BLOCKSIZE) * BLOCKSIZE;
                                for(int k = threadIdx.x; k < limit; k += BLOCKSIZE){
                                    bool keep = false;
                                    if(k < myNumIndices){
                                        keep = d_shouldBeKept[globalOffset + k];
                                    }                               
                        
                                    if(keep){
                                        cg::coalesced_group g = cg::coalesced_threads();
                                        int outputPos;
                                        if (g.thread_rank() == 0) {
                                            outputPos = atomicAdd(&counts[0], g.size());
                                            atomicAdd(&totalIndices, g.size());
                                        }
                                        outputPos = g.thread_rank() + g.shfl(outputPos, 0);
                                        myNewIndicesPtr[outputPos] = myIndices[k];
                                    }                        
                                }

                                __syncthreads();

                                if(threadIdx.x == 0){
                                    *myNewNumIndicesPerSubjectPtr = counts[0];
                                }

                                __syncthreads();


                            };



                            if(originalbase == foundBase){
                                //discard all candidates whose base in column col differs from foundBase
                                discard_rows(true);
                            }else{
                                //discard all candidates whose base in column col matches foundBase
                                discard_rows(false);
                            }

                        }else{
                            //did not find a significant columns

                            //remove no candidate
                            for(int k = threadIdx.x; k < myNumIndices; k += blockDim.x){
                                d_shouldBeKept[globalOffset + k] = true;
                                myNewIndicesPtr[k] = myIndices[k];
                            }
                            if(threadIdx.x == 0){
                                *myNewNumIndicesPerSubjectPtr = myNumIndices;
                                totalIndices += myNumIndices;
                            }
                        }

                    }else{
                        //no mismatch between consensus and subject

                        //remove no candidate
                        for(int k = threadIdx.x; k < myNumIndices; k += blockDim.x){
                            d_shouldBeKept[globalOffset + k] = true;
                            myNewIndicesPtr[k] = myIndices[k];
                        }
                        if(threadIdx.x == 0){
                            *myNewNumIndicesPerSubjectPtr = myNumIndices;
                            totalIndices += myNumIndices;
                        }
                    }
                }else{
                    ; //nothing to do if there are no candidates in msa
                }
            }

            __syncthreads();

            if(threadIdx.x == 0){
                atomicAdd(d_newNumIndices, totalIndices);
            }
        }
    }










    //####################   KERNEL DISPATCH   ####################
    
    

    void call_msa_init_kernel_async_exp(
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
    			const int* d_indices,
    			const int* d_indices_per_subject,
    			const int* d_candidates_per_subject_prefixsum,
    			int n_subjects,
                int n_queries,
                const bool* d_canExecute,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle){


    	constexpr int blocksize = 128;
    	const std::size_t smem = 0;

    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(KernelId::MSAInitExp);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    	    #define getProp(blocksize) { \
                KernelLaunchConfig kernelLaunchConfig; \
                kernelLaunchConfig.threads_per_block = (blocksize); \
                kernelLaunchConfig.smem = 0; \
                KernelProperties kernelProperties; \
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                    msaInitKernel<(blocksize)>, \
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

    		handle.kernelPropertiesMap[KernelId::MSAInitExp] = std::move(mymap);

    	    #undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
    	}

    	dim3 block(blocksize, 1, 1);
        dim3 grid(std::min(max_blocks_per_device, n_subjects), 1, 1);

		#define mycall(blocksize) msaInitKernel<(blocksize)> \
                <<<grid, block, 0, stream>>>(d_msapointers, \
                                               d_alignmentresultpointers, \
                                               d_sequencePointers, \
                                               d_indices, \
                                               d_indices_per_subject, \
                                               d_candidates_per_subject_prefixsum, \
                                               n_subjects, \
                                               d_canExecute); CUERR;

    	switch(blocksize) {
    	case 1: mycall(1); break;
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

    void call_msa_update_properties_kernel_async(
                    MSAPointers d_msapointers,
                    const int* d_indices_per_subject,
                    int n_subjects,
                    size_t msa_weights_pitch,
                    const bool* d_canExecute,
                    cudaStream_t stream,
                    KernelLaunchHandle& handle){

    	const int blocksize = 128;
    	const std::size_t smem = 0;

    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(KernelId::MSAUpdateProperties);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    		KernelLaunchConfig kernelLaunchConfig;
    		kernelLaunchConfig.threads_per_block = (blocksize);
    		kernelLaunchConfig.smem = 0;
    		KernelProperties kernelProperties;
    		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM,
            					msa_update_properties_kernel,
            					kernelLaunchConfig.threads_per_block,
                                kernelLaunchConfig.smem); CUERR;
    		mymap[kernelLaunchConfig] = kernelProperties;

    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

    		handle.kernelPropertiesMap[KernelId::MSAUpdateProperties] = std::move(mymap);

    	    #undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
    	}

    	dim3 block(blocksize, 1, 1);
    	dim3 grid(std::min(max_blocks_per_device, n_subjects), 1, 1);

        msa_update_properties_kernel<<<grid, block, 0, stream>>>(d_msapointers,
                                                                d_indices_per_subject,
                                                                msa_weights_pitch,
                                                                n_subjects,
                                                                d_canExecute); CUERR;




    }


    void call_msa_add_sequences_kernel_implicit_global_async(
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                ReadQualitiesPointers d_qualityPointers,
    			const int* d_candidates_per_subject_prefixsum,
    			const int* d_indices,
    			const int* d_indices_per_subject,
    			int n_subjects,
    			int n_queries,
    			const int* d_num_indices,
                float expectedAffectedIndicesFraction,
    			bool canUseQualityScores,
    			float desiredAlignmentMaxErrorRate,
                int encodedSequencePitchInInts,
    			size_t qualityPitchInBytes,
    			size_t msa_row_pitch,
                size_t msa_weights_row_pitch,
                const bool* d_canExecute,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle,
                bool debug){

        // set counts, weights, and coverages to zero for subjects with valid indices
        generic_kernel<<<n_subjects, 128, 0, stream>>>([=] __device__ (){
            if(*d_canExecute){
                for(int subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
                    if(d_indices_per_subject[subjectIndex] > 0){
                        const size_t msa_weights_pitch_floats = msa_weights_row_pitch / sizeof(float);

                        int* const mycounts = d_msapointers.counts + msa_weights_pitch_floats * 4 * subjectIndex;
                        float* const myweights = d_msapointers.weights + msa_weights_pitch_floats * 4 * subjectIndex;
                        int* const mycoverages = d_msapointers.coverage + msa_weights_pitch_floats * subjectIndex;

                        for(int column = threadIdx.x; column < msa_weights_pitch_floats * 4; column += blockDim.x){
                            mycounts[column] = 0;
                            myweights[column] = 0;
                        }

                        for(int column = threadIdx.x; column < msa_weights_pitch_floats; column += blockDim.x){
                            mycoverages[column] = 0;
                        }
                    }
                }
            }
        }); CUERR;

    	const int blocksize = 128;
    	const std::size_t smem = 0;

    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(KernelId::MSAAddSequencesImplicitGlobal);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    	    #define getProp(blocksize) { \
    		KernelLaunchConfig kernelLaunchConfig; \
    		kernelLaunchConfig.threads_per_block = (blocksize); \
    		kernelLaunchConfig.smem = 0; \
    		KernelProperties kernelProperties; \
    		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
    					msa_add_sequences_kernel_implicit_global, \
    					kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
    		mymap[kernelLaunchConfig] = kernelProperties; \
    }

            getProp(1);
    		getProp(32);
    		getProp(64);
    		getProp(96);
    		getProp(128);
    		getProp(160);
    		getProp(192);
    		getProp(224);
    		getProp(256);

    		const auto& kernelProperties = mymap[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM * 2;

    		handle.kernelPropertiesMap[KernelId::MSAAddSequencesImplicitGlobal] = std::move(mymap);

    	    #undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM * 2;
    		//std::cout << max_blocks_per_device << " = " << handle.deviceProperties.multiProcessorCount << " * " << kernelProperties.max_blocks_per_SM << std::endl;
    	}

    	dim3 block(blocksize, 1, 1);
        dim3 grid(std::min(n_queries, max_blocks_per_device), 1, 1);
        //dim3 grid(std::min(n_queries, max_blocks_per_device), 1, 1);

    	msa_add_sequences_kernel_implicit_global<<<grid, block, smem, stream>>>(
                                        d_msapointers,
                                        d_alignmentresultpointers,
                                        d_sequencePointers,
                                        d_qualityPointers,
                                        d_candidates_per_subject_prefixsum,
                                        d_indices,
                                        d_indices_per_subject,
                                        n_subjects,
                                        n_queries,
                                        d_num_indices,
                                        canUseQualityScores,
                                        desiredAlignmentMaxErrorRate,
                                        encodedSequencePitchInInts,
                                        qualityPitchInBytes,
                                        msa_row_pitch,
                                        msa_weights_row_pitch,
                                        d_canExecute,
                                        debug); CUERR;

        check_built_msa_kernel<<<n_subjects, 128, 0, stream>>>(d_msapointers,
                                                    d_indices_per_subject,
                                                    n_subjects,
                                                    msa_weights_row_pitch,
                                                    d_canExecute); CUERR;
    }
    
    
    void call_msa_add_sequences_kernel_implicit_shared_async(
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                ReadQualitiesPointers d_qualityPointers,
    			const int* d_candidates_per_subject_prefixsum,
    			const int* d_indices,
    			const int* d_indices_per_subject,
    			int n_subjects,
    			int n_queries,
    			const int* d_num_indices,
                float expectedAffectedIndicesFraction,
    			bool canUseQualityScores,
    			float desiredAlignmentMaxErrorRate,
    			int maximum_sequence_length,
                int encodedSequencePitchInInts,
    			size_t qualityPitchInBytes,
    			size_t msa_row_pitch,
                size_t msa_weights_row_pitch,
                const bool* d_canExecute,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle,
                bool debug){

        //constexpr bool transposeCandidates = true;

        // set counts, weights, and coverages to zero for subjects with valid indices
        generic_kernel<<<n_subjects, 128, 0, stream>>>([=] __device__ (){
            if(*d_canExecute){
                for(int subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
                    if(d_indices_per_subject[subjectIndex] > 0){
                        const size_t msa_weights_pitch_floats = msa_weights_row_pitch / sizeof(float);

                        int* const mycounts = d_msapointers.counts + msa_weights_pitch_floats * 4 * subjectIndex;
                        float* const myweights = d_msapointers.weights + msa_weights_pitch_floats * 4 * subjectIndex;
                        int* const mycoverages = d_msapointers.coverage + msa_weights_pitch_floats * subjectIndex;

                        for(int column = threadIdx.x; column < msa_weights_pitch_floats * 4; column += blockDim.x){
                            mycounts[column] = 0;
                            myweights[column] = 0;
                        }

                        for(int column = threadIdx.x; column < msa_weights_pitch_floats; column += blockDim.x){
                            mycoverages[column] = 0;
                        }
                    }
                }
            }
        }); CUERR;



//std::cerr << "n_subjects: " << n_subjects << ", n_queries: " << n_queries << ", *h_num_indices: " << *h_num_indices << '\n';
    	const int blocksize = 128;
        const std::size_t msa_weights_row_pitch_floats = msa_weights_row_pitch / sizeof(float);

    	//const std::size_t smem = sizeof(char) * maximum_sequence_length + sizeof(float) * maximum_sequence_length;
        const std::size_t smem = sizeof(float) * 4 * msa_weights_row_pitch_floats // weights
                                + sizeof(int) * 4 * msa_weights_row_pitch_floats; // counts

        //std::cerr << "msa_weights_row_pitch = " << msa_weights_row_pitch << " msa_weights_row_pitch_floats = " 
        //<< msa_weights_row_pitch_floats << "smem = " << smem << "\n";


    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(KernelId::MSAAddSequencesImplicitShared);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

            #define getProp(blocksize) { \
                KernelLaunchConfig kernelLaunchConfig; \
                kernelLaunchConfig.threads_per_block = (blocksize); \
                kernelLaunchConfig.smem = sizeof(float) * 4 * msa_weights_row_pitch_floats \
                                        + sizeof(int) * 4 * msa_weights_row_pitch_floats; \
                KernelProperties kernelProperties; \
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                    msaAddSequencesSmemWithSmallIfIntUnrolledQualitiesUnrolledKernel, \
                            kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
                mymap[kernelLaunchConfig] = kernelProperties; \
            }

            getProp(1);
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

    		handle.kernelPropertiesMap[KernelId::MSAAddSequencesImplicitShared] = std::move(mymap);

    	    #undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
            //std::cout << n_subjects << " " << n_queries << "\n";
    		//std::cout << max_blocks_per_device << " = " << handle.deviceProperties.multiProcessorCount << " * " << kernelProperties.max_blocks_per_SM << std::endl;
    	}


        int* d_blocksPerSubjectPrefixSum;
        cubCachingAllocator.DeviceAllocate((void**)&d_blocksPerSubjectPrefixSum, sizeof(int) * (n_subjects+1), stream);  CUERR;

        // calculate blocks per subject prefixsum
        auto getBlocksPerSubject = [=] __device__ (int indices_for_subject){
            return SDIV(indices_for_subject, blocksize);
        };
        cub::TransformInputIterator<int,decltype(getBlocksPerSubject), const int*>
            d_blocksPerSubject(d_indices_per_subject,
                          getBlocksPerSubject);

        void* tempstorage = nullptr;
        size_t tempstoragesize = 0;

        cub::DeviceScan::InclusiveSum(nullptr,
                    tempstoragesize,
                    d_blocksPerSubject,
                    d_blocksPerSubjectPrefixSum+1,
                    n_subjects,
                    stream); CUERR;

        cubCachingAllocator.DeviceAllocate((void**)&tempstorage, tempstoragesize, stream);  CUERR;

        cub::DeviceScan::InclusiveSum(tempstorage,
                    tempstoragesize,
                    d_blocksPerSubject,
                    d_blocksPerSubjectPrefixSum+1,
                    n_subjects,
                    stream); CUERR;

        cubCachingAllocator.DeviceFree(tempstorage);  CUERR;

        call_set_kernel_async(d_blocksPerSubjectPrefixSum,
                                0,
                                0,
                                stream);


        dim3 block(blocksize, 1, 1);
        
       // std::cerr << *h_num_indices << " " << expectedAffectedIndicesFraction << "\n";

        const int blocks = SDIV(n_queries, blocksize);
        //const int blocks = SDIV(n_queries, blocksize);
        dim3 grid(std::min(blocks, max_blocks_per_device), 1, 1);

        //msaAddSequencesSmemWithSmallIfIntUnrolledQualitiesUnrolledKernel<transposeCandidates><<<grid, block, smem, stream>>>(
        msaAddSequencesSmemWithSmallIfIntUnrolledQualitiesUnrolledKernel<<<grid, block, smem, stream>>>(
            d_msapointers.consensus,
            d_msapointers.support,
            d_msapointers.coverage,
            d_msapointers.origWeights,
            d_msapointers.origCoverages,
            d_msapointers.msaColumnProperties,
            d_msapointers.counts,
            d_msapointers.weights,
            d_alignmentresultpointers.overlaps,
            d_alignmentresultpointers.shifts,
            d_alignmentresultpointers.nOps,
            d_alignmentresultpointers.bestAlignmentFlags,
            d_sequencePointers.subjectSequencesData,
            d_sequencePointers.transposedCandidateSequencesData,
            d_sequencePointers.subjectSequencesLength,
            d_sequencePointers.candidateSequencesLength,
            d_qualityPointers.subjectQualities,
            d_qualityPointers.candidateQualities,
            d_candidates_per_subject_prefixsum,
            d_indices,
            d_indices_per_subject,
            d_blocksPerSubjectPrefixSum,
            n_subjects,
            n_queries,
            d_num_indices,
            canUseQualityScores,
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            msa_weights_row_pitch_floats,
            d_canExecute); CUERR;

        cubCachingAllocator.DeviceFree(d_blocksPerSubjectPrefixSum); CUERR;

        check_built_msa_kernel<<<n_subjects, 128, 0, stream>>>(d_msapointers,
                                                    d_indices_per_subject,
                                                    n_subjects,
                                                    msa_weights_row_pitch,
                                                    d_canExecute); CUERR;
    }    










    void call_msa_add_sequences_kernel_implicit_async(
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                ReadQualitiesPointers d_qualityPointers,
    			const int* d_candidates_per_subject_prefixsum,
    			const int* d_indices,
    			const int* d_indices_per_subject,
    			int n_subjects,
    			int n_queries,
    			const int* d_num_indices,
                float expectedAffectedIndicesFraction,
    			bool canUseQualityScores,
    			float desiredAlignmentMaxErrorRate,
    			int maximum_sequence_length,
                int encodedSequencePitchInInts,
    			size_t qualityPitchInBytes,
    			size_t msa_row_pitch,
                size_t msa_weights_row_pitch,
                const bool* d_canExecute,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle,
                bool debug){

        //std::cout << n_subjects << " " << *h_num_indices << " " << n_queries << std::endl;

    #if 0
        call_msa_add_sequences_kernel_implicit_global_async(d_msapointers,
                                                            d_alignmentresultpointers,
                                                            d_sequencePointers,
                                                            d_qualityPointers,
                                                            d_candidates_per_subject_prefixsum,
                                                            d_indices,
                                                            d_indices_per_subject,
                                                            n_subjects,
                                                            n_queries,
                                                            d_num_indices,
                                                            expectedAffectedIndicesFraction,
                                                            canUseQualityScores,
                                                            desiredAlignmentMaxErrorRate,
                                                            encodedSequencePitchInInts,
                                                            qualityPitchInBytes,
                                                            msa_row_pitch,
                                                            msa_weights_row_pitch,
                                                            d_canExecute,
                                                            stream,
                                                            handle,
                                                            debug); CUERR;
    #else


        call_msa_add_sequences_kernel_implicit_shared_async(d_msapointers,
                                                            d_alignmentresultpointers,
                                                            d_sequencePointers,
                                                            d_qualityPointers,
                                                            d_candidates_per_subject_prefixsum,
                                                            d_indices,
                                                            d_indices_per_subject,
                                                            n_subjects,
                                                            n_queries,
                                                            d_num_indices,
                                                            expectedAffectedIndicesFraction,
                                                            canUseQualityScores,
                                                            desiredAlignmentMaxErrorRate,
                                                            maximum_sequence_length,
                                                            encodedSequencePitchInInts,
                                                            qualityPitchInBytes,
                                                            msa_row_pitch,
                                                            msa_weights_row_pitch,
                                                            d_canExecute,
                                                            stream,
                                                            handle,
                                                            debug); CUERR;
    #endif
    }


    void call_msa_find_consensus_implicit_kernel_async(
                            MSAPointers d_msapointers,
                            ReadSequencesPointers d_sequencePointers,
                            const int* d_indices_per_subject,
                            int n_subjects,
                            int encodedSequencePitchInInts,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            const bool* d_canExecute,
                            cudaStream_t stream,
                            KernelLaunchHandle& handle){


        generic_kernel<<<n_subjects, 128, 0, stream>>>([=] __device__ (){
            if(*d_canExecute){
                for(int subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
                    if(d_indices_per_subject[subjectIndex] > 0){
                        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

                        float* const mysupport = d_msapointers.support + msa_weights_pitch_floats * subjectIndex;
                        float* const myorigweights = d_msapointers.origWeights + msa_weights_pitch_floats * subjectIndex;
                        int* const myorigcoverages = d_msapointers.origCoverages + msa_weights_pitch_floats * subjectIndex;
                        char* const myconsensus = d_msapointers.consensus + msa_pitch * subjectIndex;

                        for(int column = threadIdx.x; column < msa_weights_pitch_floats; column += blockDim.x){
                            mysupport[column] = 0;
                            myorigweights[column] = 0;
                            myorigcoverages[column] = 0;
                        }

                        for(int column = threadIdx.x; column < msa_pitch; column += blockDim.x){
                            myconsensus[column] = 0;
                        }
                    }
                }
            }
        }); CUERR;

        constexpr int blocksize = 128;
        constexpr int blocks_per_msa = 2;
        const std::size_t smem = 0;

        int max_blocks_per_device = 1;

        KernelLaunchConfig kernelLaunchConfig;
        kernelLaunchConfig.threads_per_block = blocksize;
        kernelLaunchConfig.smem = smem;

        auto iter = handle.kernelPropertiesMap.find(KernelId::MSAFindConsensusImplicit);
        if(iter == handle.kernelPropertiesMap.end()){

            std::map<KernelLaunchConfig, KernelProperties> mymap;

            #define getProp(blocksize) { \
                KernelLaunchConfig kernelLaunchConfig; \
                kernelLaunchConfig.threads_per_block = (blocksize); \
                kernelLaunchConfig.smem = 0; \
                KernelProperties kernelProperties; \
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                                                                msa_find_consensus_implicit_kernel<blocksize, blocks_per_msa>, \
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

            handle.kernelPropertiesMap[KernelId::MSAFindConsensusImplicit] = std::move(mymap);

            #undef getProp
        }else{
            std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
            const KernelProperties& kernelProperties = map[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
        }

        #define launch(blocksize) \
            dim3 block((blocksize), 1, 1); \
            dim3 grid(std::min(max_blocks_per_device, n_subjects), 1, 1); \
            msa_find_consensus_implicit_kernel<(blocksize), blocks_per_msa><<<grid, block, 0, stream>>>( \
                                                                d_msapointers, \
                                                                d_sequencePointers, \
                                                                d_indices_per_subject, \
                                                                n_subjects, \
                                                                encodedSequencePitchInInts, \
                                                                msa_pitch, \
                                                                msa_weights_pitch, \
                                                                d_canExecute); CUERR;

        launch(128);

        #undef launch

    }




    void call_msa_findCandidatesOfDifferentRegion_kernel_async(
                int* d_newIndices,
                int* d_newIndicesPerSubject,
                int* d_newNumIndices,
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                bool* d_shouldBeKept,
                const int* d_candidates_per_subject_prefixsum,
                int n_subjects,
                int n_candidates,
                int encodedSequencePitchInInts,
                size_t msa_pitch,
                size_t msa_weights_pitch,
                const int* d_indices,
                const int* d_indices_per_subject,
                float desiredAlignmentMaxErrorRate,
                int dataset_coverage,
                const bool* d_canExecute,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle,
                const unsigned int* d_readids,
                bool debug){

        cudaMemsetAsync(d_newNumIndices, 0, sizeof(int), stream); CUERR;
        cudaMemsetAsync(d_newIndicesPerSubject, 0, sizeof(int) * n_subjects, stream); CUERR;


    	constexpr int max_block_size = 256;
    	const int blocksize = 256;
    	const std::size_t smem = 0;

    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(KernelId::MSAFindCandidatesOfDifferentRegion);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    	    #define getProp(blocksize) { \
            		KernelLaunchConfig kernelLaunchConfig; \
            		kernelLaunchConfig.threads_per_block = (blocksize); \
            		kernelLaunchConfig.smem = 0; \
            		KernelProperties kernelProperties; \
            		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
            					msa_findCandidatesOfDifferentRegion_kernel<(blocksize)>, \
            					kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
            		mymap[kernelLaunchConfig] = kernelProperties; \
            }

            getProp(1);
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

    		handle.kernelPropertiesMap[KernelId::MSAFindCandidatesOfDifferentRegion] = std::move(mymap);

    	    #undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
    	}

    	dim3 block(blocksize, 1, 1);
    	dim3 grid(std::min(max_blocks_per_device, n_subjects));

    		#define mycall(blocksize) msa_findCandidatesOfDifferentRegion_kernel<(blocksize)> \
                <<<grid, block, 0, stream>>>( \
                    d_newIndices, \
                    d_newIndicesPerSubject, \
                    d_newNumIndices, \
                    d_msapointers, \
                    d_alignmentresultpointers, \
                    d_sequencePointers, \
                    d_shouldBeKept, \
                    d_candidates_per_subject_prefixsum, \
                    n_subjects, \
                    n_candidates, \
                    encodedSequencePitchInInts, \
                    msa_pitch, \
                    msa_weights_pitch, \
                    d_indices, \
                    d_indices_per_subject, \
                    desiredAlignmentMaxErrorRate, \
                    dataset_coverage, \
                    d_canExecute, \
                    d_readids, \
                    debug); CUERR;

    	assert(blocksize > 0 && blocksize <= max_block_size);

    	switch(blocksize) {
        case 1: mycall(1); break;
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



}
}