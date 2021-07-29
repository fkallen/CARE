#ifndef CARE_READEXTENDER_GPU_KERNELS_CUH
#define CARE_READEXTENDER_GPU_KERNELS_CUH

#include <cassert>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

namespace care{
namespace gpu{



namespace readextendergpukernels{

    //Utility


    template<class Iter1, class Iter2, class Iter3>
    __global__
    void vectorAddKernel(Iter1 input1, Iter2 input2, Iter3 output, int N){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        for(int i = tid; i < N; i += stride){
            *(output + i) = *(input1 + i) + *(input2 + i);
        }
    }


    template<class InputIter, class ConstantIter, class OutputIter>
    __global__
    void vectorAddConstantKernel(InputIter input1, ConstantIter constantiter, OutputIter output, int N){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        for(int i = tid; i < N; i += stride){
            *(output + i) = *(input1 + i) + *constantiter;
        }
    }

    template<class Sum, class T, class U>
    struct ConstantSumIterator{
        using value_type = Sum;

        T input1;
        U input2;

        __host__ __device__
        ConstantSumIterator(T i1, U i2) : input1(i1), input2(i2){}

        __host__ __device__
        value_type operator[](std::size_t i) const{
            return operator*();
        }

        __host__ __device__
        value_type operator*() const{
            return (*input1) + (*input2);
        }

        __host__ __device__
        value_type operator->() const{
            return (*input1) + (*input2);
        }        
    };

    template<class Sum, class T, class U>
    ConstantSumIterator<Sum, T, U> makeConstantSumIterator(T i1, U i2) {
        return ConstantSumIterator<Sum, T, U>{i1, i2};
    }


    template<class Iter1, class T>
    __global__
    void iotaKernel(Iter1 outputbegin, Iter1 outputend, T init){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;
        const int N = outputend - outputbegin;

        for(int i = tid; i < N; i += stride){
            *(outputbegin + i) = init + static_cast<T>(i);
        }
    }

    template<int groupsize, class Iter, class SizeIter, class OffsetIter>
    __global__
    void segmentedIotaKernel(
        Iter outputbegin, 
        int numSegments, 
        SizeIter segmentsizes, 
        OffsetIter segmentBeginOffsets
    ){
        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());

        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;
        
        const int numGroups = stride / groupsize;
        const int groupId = tid / groupsize;

        for(int s = groupId; s < numSegments; s += numGroups){
            const int num = segmentsizes[s];
            const int offset = segmentBeginOffsets[s];
            
            for(int i = group.thread_rank(); i < num; i += group.size()){
                *(outputbegin + offset + i) = i;
            }
        }
    }

    template<class Iter1, class T>
    __global__
    void fillKernel(Iter1 begin, int N, T value){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        for(int i = tid; i < N; i += stride){
            *(begin + i) = value;
        }
    }

    template<int blocksize, class InputIter, class OutputIter>
    __global__
    void minmaxSingleBlockKernel(InputIter begin, int N, OutputIter minmax){
        using value_type_in = typename std::iterator_traits<InputIter>::value_type;
        using value_type_out = typename std::iterator_traits<OutputIter>::value_type;
        static_assert(std::is_same_v<value_type_in, value_type_out>);

        using value_type = value_type_in;

        using BlockReduce = cub::BlockReduce<value_type, blocksize>;
        __shared__ typename BlockReduce::TempStorage temp1;
        __shared__ typename BlockReduce::TempStorage temp2;

        if(blockIdx.x == 0){

            const int tid = threadIdx.x;
            const int stride = blockDim.x;

            value_type myMin = std::numeric_limits<value_type>::max();
            value_type myMax = 0;

            for(int i = tid; i < N; i += stride){
                const value_type val = *(begin + i);
                myMin = min(myMin, val);
                myMax = max(myMax, val);
            }

            myMin = BlockReduce(temp1).Reduce(myMin, cub::Min{});
            myMax = BlockReduce(temp2).Reduce(myMax, cub::Max{});

            if(tid == 0){
                *(minmax + 0) = myMin;
                *(minmax + 1) = myMax;
            }
        }
    }


    //extender kernels


    template<int blocksize, int groupsize, class MapIter>
    __global__
    void taskGatherKernel1(
        MapIter d_mapBegin,
        MapIter d_mapEnd,
        int gathersize,
        std::size_t decodedSequencePitchInBytes,
        std::size_t qualityPitchInBytes,
        std::size_t encodedSequencePitchInInts,
        int* __restrict__ selection_soainputAnchorLengths,
        const int* __restrict__ soainputAnchorLengths,
        char* __restrict__ selection_soainputAnchorQualities,
        const char* __restrict__ soainputAnchorQualities,
        char* __restrict__ selection_soainputmateQualityScoresReversed,
        const char* __restrict__ soainputmateQualityScoresReversed,
        char* __restrict__ selection_soainputAnchorsDecoded,
        const char* __restrict__ soainputAnchorsDecoded,
        char* __restrict__ selection_soainputdecodedMateRevC,
        const char* __restrict__ soainputdecodedMateRevC,
        unsigned int* __restrict__ selection_inputEncodedMate,
        const unsigned int* __restrict__ inputEncodedMate,
        unsigned int* __restrict__ selection_inputAnchorsEncoded,
        const unsigned int* __restrict__ inputAnchorsEncoded
    ){
        const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
        const int gstride = blockDim.x * gridDim.x;

        for(int i = gtid; i < gathersize; i += gstride){
            const std::size_t srcindex = *(d_mapBegin + i);
            const std::size_t destindex = i;
            selection_soainputAnchorLengths[destindex] = soainputAnchorLengths[srcindex];
        }

        for(int i = blockIdx.x; i < gathersize; i += gridDim.x){
            const std::size_t srcindex = *(d_mapBegin + i);
            const std::size_t destindex = i;

            for(int k = threadIdx.x; k < decodedSequencePitchInBytes; k += blockDim.x){
                selection_soainputdecodedMateRevC[destindex * decodedSequencePitchInBytes + k]
                    = soainputdecodedMateRevC[srcindex * decodedSequencePitchInBytes + k];

                selection_soainputAnchorsDecoded[destindex * decodedSequencePitchInBytes + k]
                    = soainputAnchorsDecoded[srcindex * decodedSequencePitchInBytes + k];
            }

            for(int k = threadIdx.x; k < qualityPitchInBytes; k += blockDim.x){
                selection_soainputmateQualityScoresReversed[destindex * qualityPitchInBytes + k]
                    = soainputmateQualityScoresReversed[srcindex * qualityPitchInBytes + k];

                selection_soainputAnchorQualities[destindex * qualityPitchInBytes + k]
                    = soainputAnchorQualities[srcindex * qualityPitchInBytes + k];
            }

            for(int k = threadIdx.x; k < encodedSequencePitchInInts; k += blockDim.x){
                selection_inputEncodedMate[destindex * encodedSequencePitchInInts + k]
                    = inputEncodedMate[srcindex * encodedSequencePitchInInts + k];

                selection_inputAnchorsEncoded[destindex * encodedSequencePitchInInts + k]
                    = inputAnchorsEncoded[srcindex * encodedSequencePitchInInts + k];
            }
        }
    }

    template<int blocksize, int groupsize, class MapIter>
    __global__
    void taskGatherKernel2(
        MapIter d_mapBegin,
        MapIter d_mapEnd,
        int gathersize,
        std::size_t decodedSequencePitchInBytes,
        std::size_t qualityPitchInBytes,
        const int* __restrict__ selection_soaNumIterationResultsPerTaskPrefixSum,
        const int* __restrict__ soaNumIterationResultsPerTaskPrefixSum,
        const int* __restrict__ soaNumIterationResultsPerTask,
        int* __restrict__ selection_soatotalDecodedAnchorsLengths,
        const int* __restrict__ soatotalDecodedAnchorsLengths,
        int* __restrict__ selection_soatotalAnchorBeginInExtendedRead,
        const int* __restrict__ soatotalAnchorBeginInExtendedRead,
        char* __restrict__ selection_soatotalDecodedAnchorsFlat,
        const char* __restrict__ soatotalDecodedAnchorsFlat,
        char* __restrict__ selection_soatotalAnchorQualityScoresFlat,
        const char* __restrict__ soatotalAnchorQualityScoresFlat,
        const int* __restrict__ selection_d_numUsedReadIdsPerTaskPrefixSum,
        const int* __restrict__ d_numUsedReadIdsPerTaskPrefixSum,
        const int* __restrict__ d_numUsedReadIdsPerTask,
        read_number* __restrict__ selection_d_usedReadIds,
        const read_number* __restrict__ d_usedReadIds,
        const int* __restrict__  selection_d_numFullyUsedReadIdsPerTaskPrefixSum,
        const int* __restrict__  d_numFullyUsedReadIdsPerTaskPrefixSum,
        const int* __restrict__ d_numFullyUsedReadIdsPerTask,
        read_number* __restrict__ selection_d_fullyUsedReadIds,
        const read_number* __restrict__ d_fullyUsedReadIds
    ){
        for(int i = blockIdx.x; i < gathersize; i += gridDim.x){
            const std::size_t srcindex = *(d_mapBegin + i);
            const std::size_t destindex = i;

            //iteration results
            {
                int destoffset = selection_soaNumIterationResultsPerTaskPrefixSum[destindex];
                int srcoffset = soaNumIterationResultsPerTaskPrefixSum[srcindex];
                int num = soaNumIterationResultsPerTask[srcindex];

                for(int k = threadIdx.x; k < num; k += blockDim.x){
                    selection_soatotalDecodedAnchorsLengths[destoffset + k] 
                        = soatotalDecodedAnchorsLengths[srcoffset + k];

                    selection_soatotalAnchorBeginInExtendedRead[destoffset + k] 
                        = soatotalAnchorBeginInExtendedRead[srcoffset + k];
                }

                std::size_t pitchnum1 = decodedSequencePitchInBytes * num;

                for(int k = threadIdx.x; k < pitchnum1; k += blockDim.x){
                    selection_soatotalDecodedAnchorsFlat[destoffset * decodedSequencePitchInBytes + k] 
                        = soatotalDecodedAnchorsFlat[srcoffset * decodedSequencePitchInBytes+ k];
                }

                std::size_t pitchnum2 = qualityPitchInBytes * num;

                for(int k = threadIdx.x; k < pitchnum2; k += blockDim.x){
                    selection_soatotalAnchorQualityScoresFlat[destoffset * qualityPitchInBytes + k] 
                        = soatotalAnchorQualityScoresFlat[srcoffset * qualityPitchInBytes + k];
                }
            }

            //used ids
            {
                int destoffset = selection_d_numUsedReadIdsPerTaskPrefixSum[destindex];
                int srcoffset = d_numUsedReadIdsPerTaskPrefixSum[srcindex];
                int num = d_numUsedReadIdsPerTask[srcindex];

                for(int k = threadIdx.x; k < num; k += blockDim.x){
                    selection_d_usedReadIds[destoffset + k] 
                        = d_usedReadIds[srcoffset + k];
                }
            }

            //fully used ids
            {
                int destoffset = selection_d_numFullyUsedReadIdsPerTaskPrefixSum[destindex];
                int srcoffset = d_numFullyUsedReadIdsPerTaskPrefixSum[srcindex];
                int num = d_numFullyUsedReadIdsPerTask[srcindex];

                for(int k = threadIdx.x; k < num; k += blockDim.x){
                    selection_d_fullyUsedReadIds[destoffset + k] 
                        = d_fullyUsedReadIds[srcoffset + k];
                }
            }
        }
    }

    template<int blocksize>
    __global__
    void taskFixAppendedPrefixSumsKernel(
        int* __restrict__ soaNumIterationResultsPerTaskPrefixSum,
        int* __restrict__ d_numUsedReadIdsPerTaskPrefixSum,
        int* __restrict__ d_numFullyUsedReadIdsPerTaskPrefixSum,
        const int* __restrict__ soaNumIterationResultsPerTask,
        const int* __restrict__ d_numUsedReadIdsPerTask,
        const int* __restrict__ d_numFullyUsedReadIdsPerTask,
        int size,
        int rhssize
    ){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        if(size == 0){
            if(tid == 0){
                soaNumIterationResultsPerTaskPrefixSum[rhssize] 
                    = soaNumIterationResultsPerTaskPrefixSum[rhssize-1] + soaNumIterationResultsPerTask[rhssize-1];
                d_numUsedReadIdsPerTaskPrefixSum[rhssize] 
                    = d_numUsedReadIdsPerTaskPrefixSum[rhssize-1] + d_numUsedReadIdsPerTask[rhssize-1];
                d_numFullyUsedReadIdsPerTaskPrefixSum[rhssize] 
                    = d_numFullyUsedReadIdsPerTaskPrefixSum[rhssize-1] + d_numFullyUsedReadIdsPerTask[rhssize-1];
            }
        }else{
            for(int i = tid; i < rhssize+1; i += stride){
                soaNumIterationResultsPerTaskPrefixSum[size + i] 
                    += soaNumIterationResultsPerTaskPrefixSum[size-1] + soaNumIterationResultsPerTask[size - 1];
                d_numUsedReadIdsPerTaskPrefixSum[size + i] 
                    += d_numUsedReadIdsPerTaskPrefixSum[size-1] + d_numUsedReadIdsPerTask[size - 1];
                d_numFullyUsedReadIdsPerTaskPrefixSum[size + i] 
                    += d_numFullyUsedReadIdsPerTaskPrefixSum[size-1] + d_numFullyUsedReadIdsPerTask[size - 1];
            }
        }
    }

    template<int blocksize, int groupsize>
    __global__
    void taskAddIterationResultsKernel(
        int numTasks,
        std::size_t decodedSequencePitchInBytes,
        std::size_t qualityPitchInBytes,
        std::size_t addSequencesPitchInBytes,
        std::size_t addQualityPitchInBytes,
        const int* __restrict__ newNumEntriesPerTaskPrefixSum,
        char* __restrict__ newsoatotalDecodedAnchorsFlat,
        char* __restrict__ newsoatotalAnchorQualityScoresFlat,
        int* __restrict__ newsoatotalDecodedAnchorsLengths,
        int* __restrict__ newsoatotalAnchorBeginInExtendedRead,
        const int* __restrict__ soaNumIterationResultsPerTask,
        const int* __restrict__ soaNumIterationResultsPerTaskPrefixSum,
        const int* __restrict__ soatotalDecodedAnchorsLengths,
        const int* __restrict__ soatotalAnchorBeginInExtendedRead,
        const char* __restrict__ soatotalDecodedAnchorsFlat,
        const char* __restrict__ soatotalAnchorQualityScoresFlat,
        const int* __restrict__ addNumEntriesPerTask,
        const int* __restrict__ addNumEntriesPerTaskPrefixSum,
        const int* __restrict__ addTotalDecodedAnchorsLengths,
        const int* __restrict__ addTotalAnchorBeginInExtendedRead,
        const char* __restrict__ addTotalDecodedAnchorsFlat,
        const char* __restrict__ addTotalAnchorQualityScoresFlat
    ){
        for(int i = blockIdx.x; i < numTasks; i += gridDim.x){
            //copy current data to new buffer
            const int currentnum = soaNumIterationResultsPerTask[i];
            const int currentoffset = soaNumIterationResultsPerTaskPrefixSum[i];

            const int newoffset = newNumEntriesPerTaskPrefixSum[i];

            for(int k = threadIdx.x; k < currentnum; k += blockDim.x){
                newsoatotalDecodedAnchorsLengths[newoffset + k] = soatotalDecodedAnchorsLengths[currentoffset + k];
                newsoatotalAnchorBeginInExtendedRead[newoffset + k] = soatotalAnchorBeginInExtendedRead[currentoffset + k];
            }

            for(int k = threadIdx.x; k < decodedSequencePitchInBytes * currentnum; k += blockDim.x){
                newsoatotalDecodedAnchorsFlat[decodedSequencePitchInBytes * newoffset + k]
                    = soatotalDecodedAnchorsFlat[decodedSequencePitchInBytes * currentoffset + k];
            }

            for(int k = threadIdx.x; k < qualityPitchInBytes * currentnum; k += blockDim.x){
                newsoatotalAnchorQualityScoresFlat[qualityPitchInBytes * newoffset + k]
                    = soatotalAnchorQualityScoresFlat[qualityPitchInBytes * currentoffset + k];
            }

            //copy add data to new buffer
            const int addnum = addNumEntriesPerTask[i];
            if(addnum > 0){
                const int addoffset = addNumEntriesPerTaskPrefixSum[i];

                for(int k = threadIdx.x; k < addnum; k += blockDim.x){
                    newsoatotalDecodedAnchorsLengths[(newoffset + currentnum) + k] = addTotalDecodedAnchorsLengths[addoffset + k];
                    newsoatotalAnchorBeginInExtendedRead[(newoffset + currentnum) + k] = addTotalAnchorBeginInExtendedRead[addoffset + k];
                }

                for(int k = 0; k < addnum; k++){
                    for(int l = threadIdx.x; l < addSequencesPitchInBytes; l += blockDim.x){
                        newsoatotalDecodedAnchorsFlat[decodedSequencePitchInBytes * (newoffset + currentnum + k) + l]
                            = addTotalDecodedAnchorsFlat[addSequencesPitchInBytes * (addoffset + k) + l];
                    }  

                    for(int l = threadIdx.x; l < addQualityPitchInBytes; l += blockDim.x){
                        newsoatotalAnchorQualityScoresFlat[qualityPitchInBytes * (newoffset + currentnum + k) + l]
                            = addTotalAnchorQualityScoresFlat[addQualityPitchInBytes * (addoffset + k) + l];
                    }                       
                }
            }
        }
    }

    template<int blocksize>
    __global__
    void taskComputeActiveFlagsKernel(
        int numTasks,
        int insertSize,
        int insertSizeStddev,
        bool* __restrict__ d_flags,
        const int* __restrict__ iteration,
        const int* __restrict__ soaNumIterationResultsPerTask,
        const int* __restrict__ soaNumIterationResultsPerTaskPrefixSum,
        const int* __restrict__ soatotalAnchorBeginInExtendedRead,
        const int* __restrict__ soainputmateLengths,
        const extension::AbortReason* __restrict__ abortReason,
        const bool* __restrict__ mateHasBeenFound
    ){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        for(int i = tid; i < numTasks; i+= stride){
            const int num = soaNumIterationResultsPerTask[i];
            const int offset = soaNumIterationResultsPerTaskPrefixSum[i];
            const int accumExtensionLength = num > 0 ? soatotalAnchorBeginInExtendedRead[offset + num - 1] : 0;

            d_flags[i] = (iteration[i] < insertSize 
                && accumExtensionLength < insertSize - (soainputmateLengths[i]) + insertSizeStddev
                && (abortReason[i] == extension::AbortReason::None) 
                && !mateHasBeenFound[i]
            );
        }
    }

    template<int blocksize>
    __global__
    void taskUpdateScalarIterationResultsKernel(
        int numTasks,
        float* __restrict__ task_goodscore,
        extension::AbortReason* __restrict__ task_abortReason,
        bool* __restrict__ task_mateHasBeenFound,
        const float* __restrict__ d_goodscores,
        const extension::AbortReason* __restrict__ d_abortReasons,
        const bool* __restrict__ d_mateHasBeenFound
    ){
        const int tid = threadIdx.x + blockIdx.x * blocksize;
        const int stride = blocksize * gridDim.x;

        for(int i = tid; i < numTasks; i += stride){
            task_goodscore[i] += d_goodscores[i];
            task_abortReason[i] = d_abortReasons[i];
            task_mateHasBeenFound[i] = d_mateHasBeenFound[i];
        }
    }
        
    template<int blocksize>
    __global__
    void taskIncrementIterationKernel(
        int numTasks,
        const extension::ExtensionDirection* __restrict__ task_direction,
        const bool* __restrict__ task_pairedEnd,
        const bool* __restrict__ task_mateHasBeenFound,
        const int* __restrict__ task_pairId,
        const int* __restrict__ task_id,
        extension::AbortReason* __restrict__ task_abortReason,
        int* __restrict__ task_iteration
    ){
        const int tid = threadIdx.x + blockIdx.x * blocksize;
        const int stride = blocksize * gridDim.x;

        constexpr bool disableOtherStrand = false;

        for(int i = tid; i < numTasks; i += stride){
            task_iteration[i]++;
            
            const int whichtype = task_id[i] % 4;

            if(whichtype == 0){
                assert(task_direction[i] == extension::ExtensionDirection::LR);
                assert(task_pairedEnd[i] == true);

                if(task_mateHasBeenFound[i]){
                    for(int k = 1; k <= 4; k++){
                        if(i+k < numTasks){
                            if(task_pairId[i + k] == task_pairId[i]){
                                if(task_id[i + k] == task_id[i] + 1){
                                    //disable LR partner task
                                    task_abortReason[i + k] = extension::AbortReason::PairedAnchorFinished;
                                }else if(task_id[i+k] == task_id[i] + 2){
                                    //disable RL search task
                                    if(disableOtherStrand){
                                        task_abortReason[i + k] = extension::AbortReason::OtherStrandFoundMate;
                                    }
                                }
                            }else{
                                break;
                            }
                        }else{
                            break;
                        }
                    }
                }else if(task_abortReason[i] != extension::AbortReason::None){
                    for(int k = 1; k <= 4; k++){
                        if(i+k < numTasks){
                            if(task_pairId[i + k] == task_pairId[i]){
                                if(task_id[i + k] == task_id[i] + 1){
                                    //disable LR partner task  
                                    task_abortReason[i + k] = extension::AbortReason::PairedAnchorFinished;
                                    break;
                                }
                            }else{
                                break;
                            }
                        }else{
                            break;
                        }
                    }
                }
            }else if(whichtype == 2){
                assert(task_direction[i] == extension::ExtensionDirection::RL);
                assert(task_pairedEnd[i] == true);

                if(task_mateHasBeenFound[i]){
                    if(i+1 < numTasks){
                        if(task_pairId[i + 1] == task_pairId[i]){
                            if(task_id[i + 1] == task_id[i] + 1){
                                //disable RL partner task
                                task_abortReason[i + 1] = extension::AbortReason::PairedAnchorFinished;
                            }
                        }

                        for(int k = 1; k <= 2; k++){
                            if(i - k >= 0){
                                if(task_pairId[i - k] == task_pairId[i]){
                                    if(task_id[i - k] == task_id[i] - 2){
                                        //disable LR search task
                                        if(disableOtherStrand){
                                            task_abortReason[i - k] = extension::AbortReason::OtherStrandFoundMate;
                                        }
                                    }
                                }else{
                                    break;
                                }
                            }else{
                                break;
                            }
                        }
                    }
                    
                }else if(task_abortReason[i] != extension::AbortReason::None){
                    if(i+1 < numTasks){

                        if(task_pairId[i + 1] == task_pairId[i]){
                            if(task_id[i + 1] == task_id[i] + 1){
                                //disable RL partner task
                                task_abortReason[i + 1] = extension::AbortReason::PairedAnchorFinished;
                            }
                        }

                    }
                }
            }
        }
    }





        template<int blocksize>
    struct CheckAmbiguousColumns{

        using BlockReduce = cub::BlockReduce<int, blocksize>;
        //using BlockScan = cub::BlockScan<int, blocksize>;
        using BlockSort = cub::BlockRadixSort<std::uint64_t, blocksize, 1>;
        using BlockDiscontinuity = cub::BlockDiscontinuity<std::uint64_t, blocksize>;
        using MyBlockSelect = BlockSelect<std::uint64_t, blocksize>;

        const int* countsA;
        const int* countsC;
        const int* countsG;
        const int* countsT;
        const int* coverages;

        void* tempstorage;

        __host__ __device__
        CheckAmbiguousColumns(const int* cA, const int* cC, const int* cG, const int* cT, const int* cov) 
            : countsA(cA), countsC(cC), countsG(cG), countsT(cT), coverages(cov){}

        //thread 0 returns number of ambiguous columns in given range. Block-wide algorithm
        __device__
        int getAmbiguousColumnCount(int begin, int end, typename BlockReduce::TempStorage& temp) const{ 

            int myCount = 0;

            for(int col = threadIdx.x; col < end; col += blocksize){
                if(col >= begin){

                    int numNucs = 0;

                    auto checkNuc = [&](const auto& counts, const char nuc){
                        const float ratio = float(counts[col]) / float(coverages[col]);
                        if(counts[col] >= 2 && fgeq(ratio, 0.4f) && fleq(ratio, 0.6f)){
                            numNucs++;                                
                        }
                    };

                    checkNuc(countsA, 'A');
                    checkNuc(countsC, 'C');
                    checkNuc(countsG, 'G');
                    checkNuc(countsT, 'T');

                    if(numNucs > 0){
                        myCount++;
                    }
                }
            }

            myCount = BlockReduce(temp).Sum(myCount);

            return myCount;
        }

        struct SplitInfo{
            char nuc;
            int column;
            //float ratio;

            SplitInfo() = default;

            __host__ __device__
            SplitInfo(char c, int n) : nuc(c), column(n){}

            __host__ __device__
            SplitInfo(char c, int n, float f) : nuc(c), column(n)/*, ratio(f)*/{}
        };

        struct SplitInfos{       
            static constexpr int maxSplitInfos = 64;

            int numSplitInfos;
            SplitInfo splitInfos[maxSplitInfos];
        };

        struct TempStorage{
            int broadcastint;
            union{
                typename BlockReduce::TempStorage blockreducetemp;
                //typename BlockScan::TempStorage blockscantemp;
                typename BlockSort::TempStorage blocksorttemp;
                typename BlockDiscontinuity::TempStorage blockdiscontinuity;
                typename MyBlockSelect::TempStorage blockselect;
            } cub;

            static constexpr int maxNumEncodedRows = 128;
            int numEncodedRows;
            std::uint64_t encodedRows[maxNumEncodedRows];
            int flags[maxNumEncodedRows];
        };



        __device__
        void getSplitInfos(int begin, int end, float relativeCountLowerBound, float relativeCountUpperBound, SplitInfos& result) const{
            using PSC = MultipleSequenceAlignment::PossibleSplitColumn;

            if(threadIdx.x == 0){
                result.numSplitInfos = 0;
            }
            __syncthreads();

            //find columns where counts/coverage is in range [relativeCountLowerBound, relativeCountUpperBound]
            //if there are exactly 2 different bases for which this is true in a column, SplitInfos of both cases are saved in result.splitInfos
            for(int col = threadIdx.x; col < end; col += blocksize){
                if(col >= begin){

                    SplitInfo myResults[2];
                    int myNumResults = 0;

                    auto checkNuc = [&](const auto& counts, const char nuc){
                        const float ratio = float(counts[col]) / float(coverages[col]);
                        if(counts[col] >= 2 && fgeq(ratio, relativeCountLowerBound) && fleq(ratio, relativeCountUpperBound)){

                            #pragma unroll
                            for(int k = 0; k < 2; k++){
                                if(myNumResults == k){
                                    myResults[k] = {nuc, col, ratio};
                                    myNumResults++;
                                    break;
                                }
                            }
                              
                        }
                    };

                    checkNuc(countsA, 'A');
                    checkNuc(countsC, 'C');
                    checkNuc(countsG, 'G');
                    checkNuc(countsT, 'T');

                    if(myNumResults == 2){
                        if(result.numSplitInfos < SplitInfos::maxSplitInfos){
                            int firstPos = atomicAdd(&result.numSplitInfos, 2);
                            if(firstPos <= SplitInfos::maxSplitInfos - 2){
                                result.splitInfos[firstPos + 0] = myResults[0];
                                result.splitInfos[firstPos + 1] = myResults[1];
                            }
                        }
                    }
                }
            }

            __syncthreads();
            if(threadIdx.x == 0){
                if(result.numSplitInfos > SplitInfos::maxSplitInfos){
                    result.numSplitInfos = SplitInfos::maxSplitInfos;
                }
            }
            __syncthreads();
        }

        __device__
        int getNumberOfSplits(
            const SplitInfos& splitInfos, 
            const gpu::MSAColumnProperties& msaColumnProperties,
            int numCandidates, 
            const int* candidateShifts,
            const BestAlignment_t* bestAlignmentFlags,
            const int* candidateLengths,
            const unsigned int* encodedCandidates, 
            int encodedSequencePitchInInts, 
            TempStorage& temp
        ){
            assert(TempStorage::maxNumEncodedRows == blocksize);

            // 64-bit flags. 2 bit per column. 
            // 00 -> nuc does not match any of both splitInfos
            // 10 -> nuc  matches first splitInfos
            // 11 -> nuc  matches second splitInfos
            constexpr int numPossibleColumnsPerFlag = 32; 
            //

            if(splitInfos.numSplitInfos == 0) return 1;
            if(splitInfos.numSplitInfos == 2) return 2;

            if(threadIdx.x == 0){
                temp.numEncodedRows = 0;
            }
            __syncthreads();

            const int subjectColumnsBegin_incl = msaColumnProperties.subjectColumnsBegin_incl;
            const int numColumnsToCheck = std::min(numPossibleColumnsPerFlag, splitInfos.numSplitInfos / 2);
            const int maxCandidatesToCheck = std::min(blocksize, numCandidates);

            #if 0
            if(threadIdx.x == 0){                
                if(splitInfos.numSplitInfos > 0){
                    printf("numSplitInfos %d\n", splitInfos.numSplitInfos);
                    for(int i = 0; i < splitInfos.numSplitInfos; i++){
                        printf("(%c,%d, %f) ", 
                            splitInfos.splitInfos[i].nuc, 
                            splitInfos.splitInfos[i].column,
                            splitInfos.splitInfos[i].ratio);
                    }
                    printf("\n");

                    for(int c = 0; c < maxCandidatesToCheck; c++){
                        const unsigned int* const myCandidate = encodedCandidates + c * encodedSequencePitchInInts;
                        const int candidateShift = candidateShifts[c];
                        const int candidateLength = candidateLengths[c];
                        const BestAlignment_t alignmentFlag = bestAlignmentFlags[c];

                        for(int i = 0; i < candidateShift; i++){
                            printf("0");
                        }
                        for(int i = 0; i < candidateLength; i++){
                            char nuc = 'F';
                            if(alignmentFlag == BestAlignment_t::Forward){
                                const int positionInCandidate = i;
                                std::uint8_t encodedCandidateNuc = SequenceHelpers::getEncodedNuc2Bit(myCandidate, candidateLength, positionInCandidate);
                                nuc = SequenceHelpers::decodeBase(encodedCandidateNuc);
                            }else{
                                const int positionInCandidate = candidateLength - 1 - i;
                                std::uint8_t encodedCandidateNuc = SequenceHelpers::getEncodedNuc2Bit(myCandidate, candidateLength, positionInCandidate);
                                nuc = SequenceHelpers::complementBaseDecoded(SequenceHelpers::decodeBase(encodedCandidateNuc));
                            }
                            printf("%c", nuc);
                        }
                        printf("\n");
                    }
                }
            }

            __syncthreads(); //DEBUG
            #endif

            for(int c = threadIdx.x; c < maxCandidatesToCheck; c += blocksize){
                if(temp.numEncodedRows < TempStorage::maxNumEncodedRows){
                    std::uint64_t flags = 0;

                    const unsigned int* const myCandidate = encodedCandidates + c * encodedSequencePitchInInts;
                    const int candidateShift = candidateShifts[c];
                    const int candidateLength = candidateLengths[c];
                    const BestAlignment_t alignmentFlag = bestAlignmentFlags[c];

                    for(int k = 0; k < numColumnsToCheck; k++){
                        flags <<= 2;

                        const SplitInfo psc0 = splitInfos.splitInfos[2*k+0];
                        const SplitInfo psc1 = splitInfos.splitInfos[2*k+1];
                        assert(psc0.column == psc1.column);

                        const int candidateColumnsBegin_incl = candidateShift + subjectColumnsBegin_incl;
                        const int candidateColumnsEnd_excl = candidateLength + candidateColumnsBegin_incl;
                        
                        //column range check for row
                        if(candidateColumnsBegin_incl <= psc0.column && psc0.column < candidateColumnsEnd_excl){                        

                            char nuc = 'F';
                            if(alignmentFlag == BestAlignment_t::Forward){
                                const int positionInCandidate = psc0.column - candidateColumnsBegin_incl;
                                std::uint8_t encodedCandidateNuc = SequenceHelpers::getEncodedNuc2Bit(myCandidate, candidateLength, positionInCandidate);
                                nuc = SequenceHelpers::decodeBase(encodedCandidateNuc);
                            }else{
                                const int positionInCandidate = candidateLength - 1 - (psc0.column - candidateColumnsBegin_incl);
                                std::uint8_t encodedCandidateNuc = SequenceHelpers::getEncodedNuc2Bit(myCandidate, candidateLength, positionInCandidate);
                                nuc = SequenceHelpers::complementBaseDecoded(SequenceHelpers::decodeBase(encodedCandidateNuc));
                            }

                            //printf("cand %d col %d %c\n", c, psc0.column, nuc);

                            if(nuc == psc0.nuc){
                                flags = flags | 0b10;
                            }else if(nuc == psc1.nuc){
                                flags = flags | 0b11;
                            }else{
                                flags = flags | 0b00;
                            } 

                        }else{
                            flags = flags | 0b00;
                        } 

                    }

                    const int tempPos = atomicAdd(&temp.numEncodedRows, 1);
                    if(tempPos < TempStorage::maxNumEncodedRows){
                        temp.encodedRows[tempPos] = flags;
                    }
                }
            }
        
            __syncthreads();
            if(threadIdx.x == 0){
                if(temp.numEncodedRows > TempStorage::maxNumEncodedRows){
                    temp.numEncodedRows = TempStorage::maxNumEncodedRows;
                }
            }
            __syncthreads();

            {
                //sort the computed encoded rows, and make them unique

                std::uint64_t encodedRow[1];

                if(threadIdx.x < temp.numEncodedRows){
                    encodedRow[0] = temp.encodedRows[threadIdx.x];
                }else{
                    encodedRow[0] = std::numeric_limits<std::uint64_t>::max();
                }

                BlockSort(temp.cub.blocksorttemp).Sort(encodedRow);
                __syncthreads();

                if(threadIdx.x < temp.numEncodedRows){
                    temp.encodedRows[threadIdx.x] = encodedRow[0];
                    temp.flags[threadIdx.x] = 0;
                }

                __syncthreads();

                int headflag[1];

                if(threadIdx.x < temp.numEncodedRows){
                    encodedRow[0] = temp.encodedRows[threadIdx.x];
                }else{
                    encodedRow[0] = temp.encodedRows[temp.numEncodedRows-1];
                }

                BlockDiscontinuity(temp.cub.blockdiscontinuity).FlagHeads(headflag, encodedRow, cub::Inequality());

                __syncthreads();

                int numselected = MyBlockSelect(temp.cub.blockselect).Flagged(encodedRow, headflag, &temp.encodedRows[0], temp.numEncodedRows);

                __syncthreads();
                if(threadIdx.x == 0){
                    temp.numEncodedRows = numselected;
                }
                __syncthreads();
            }

            // if(threadIdx.x == 0){                
            //     if(splitInfos.numSplitInfos > 0){
            //         printf("numEncodedRows %d\n", temp.numEncodedRows);
            //         for(int i = 0; i < temp.numEncodedRows; i++){
            //             printf("%lu ", temp.encodedRows[i]);
            //         }
            //         printf("\n");
            //     }
            // }

            // __syncthreads(); //DEBUG

            for(int i = 0; i < temp.numEncodedRows - 1; i++){
                const std::uint64_t encodedRow = temp.encodedRows[i];

                std::uint64_t mask = 0;
                for(int s = 0; s < numColumnsToCheck; s++){
                    if(encodedRow >> (2*s+1) & 1){
                        mask = mask | (0x03 << (2*s));
                    }
                }
                
                if(i < threadIdx.x && threadIdx.x < temp.numEncodedRows){
                    //check if encodedRow is equal to another flag masked with mask. if yes, it can be removed
                    if((temp.encodedRows[threadIdx.x] & mask) == encodedRow){
                        // if(encodedRow == 172){
                        //     printf("i = %d, thread=%d temp.encodedRows[threadIdx.x] = %lu, mask = %lu", i, threadIdx.x, temp.encodedRows[threadIdx.x], mask)
                        // }
                        atomicAdd(&temp.flags[i], 1);
                    }
                }
            }

            __syncthreads();

            //count number of remaining row flags
            int count = 0;
            if(threadIdx.x < temp.numEncodedRows){
                count = temp.flags[threadIdx.x] == 0 ? 1 : 0;
            }
            count = BlockReduce(temp.cub.blockreducetemp).Sum(count);
            if(threadIdx.x == 0){
                temp.broadcastint = count;
                // if(splitInfos.numSplitInfos > 0){
                //     printf("count = %d\n", count);
                // }
            }
            __syncthreads();

            count = temp.broadcastint;

            return count;
        }

    };





    template<int blocksize>
    __global__
    void flagFullyUsedCandidatesKernel(
        int numTasks,
        const int* __restrict__ d_numCandidatesPerAnchor,
        const int* __restrict__ d_numCandidatesPerAnchorPrefixSum,
        const int* __restrict__ d_candidateSequencesLengths,
        const int* __restrict__ d_alignment_shifts,
        const int* __restrict__ d_anchorSequencesLength,
        const int* __restrict__ d_oldaccumExtensionsLengths,
        const int* __restrict__ d_newaccumExtensionsLengths,
        const extension::AbortReason* __restrict__ d_abortReasons,
        const bool* __restrict__ d_outputMateHasBeenFound,
        bool* __restrict__ d_isFullyUsedCandidate
    ){
        // d_isFullyUsedCandidate must be initialized with 0

        for(int task = blockIdx.x; task < numTasks; task += gridDim.x){
            const int numCandidates = d_numCandidatesPerAnchor[task];
            const auto abortReason = d_abortReasons[task];

            if(numCandidates > 0 && abortReason == extension::AbortReason::None){
                const int anchorLength = d_anchorSequencesLength[task];
                const int offset = d_numCandidatesPerAnchorPrefixSum[task];
                const int oldAccumExtensionsLength = d_oldaccumExtensionsLengths[task];
                const int newAccumExtensionsLength = d_newaccumExtensionsLengths[task];
                const int lengthOfExtension = newAccumExtensionsLength - oldAccumExtensionsLength;

                for(int c = threadIdx.x; c < numCandidates; c += blockDim.x){
                    const int candidateLength = d_candidateSequencesLengths[c];
                    const int shift = d_alignment_shifts[c];

                    if(candidateLength + shift <= anchorLength + lengthOfExtension){
                        d_isFullyUsedCandidate[offset + c] = true;
                    }
                }
            }
        }
    }


    template<int blocksize, int groupsize>
    __global__
    void updateWorkingSetFromTasksKernel(
        int numTasks,
        std::size_t qualityPitchInBytes,
        std::size_t decodedSequencePitchInBytes,
        const int* __restrict__ soaNumIterationResultsPerTask,
        const int* __restrict__ soaNumIterationResultsPerTaskPrefixSum,
        int* __restrict__ d_accumExtensionsLengths,
        int* __restrict__ d_anchorSequencesLength,
        char* __restrict__ d_anchorQualityScores,
        char* __restrict__ d_subjectSequencesDataDecoded,
        const int* __restrict__ soatotalAnchorBeginInExtendedRead,
        const int* __restrict__ soatotalDecodedAnchorsLengths,
        const int* __restrict__ soainputAnchorLengths,
        const char* __restrict__ soatotalAnchorQualityScoresFlat,
        const char* __restrict__ soainputAnchorQualities,
        const char* __restrict__ soatotalDecodedAnchorsFlat,
        const char* __restrict__ soainputAnchorsDecoded
    ){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int groupId = tid / groupsize;
        const int numGroups = stride / groupsize;

        for(int i = tid; i < numTasks; i += stride){
            const int num = soaNumIterationResultsPerTask[i];
            const int offset = soaNumIterationResultsPerTaskPrefixSum[i];
            if(num == 0){
                d_accumExtensionsLengths[i] = 0;
                d_anchorSequencesLength[i] = soainputAnchorLengths[i];
            }else{
                d_accumExtensionsLengths[i] = soatotalAnchorBeginInExtendedRead[offset + num - 1];
                d_anchorSequencesLength[i] = soatotalDecodedAnchorsLengths[offset + num - 1];
            }
        }

        for(int i = groupId; i < numTasks; i += numGroups){
            const int num = soaNumIterationResultsPerTask[i];
            const int offset = soaNumIterationResultsPerTaskPrefixSum[i];
            if(num == 0){
                for(int k = group.thread_rank(); k < qualityPitchInBytes; k += group.size()){
                    d_anchorQualityScores[qualityPitchInBytes * i + k]
                        = soainputAnchorQualities[qualityPitchInBytes * i + k];
                }
                for(int k = group.thread_rank(); k < decodedSequencePitchInBytes; k += group.size()){
                    d_subjectSequencesDataDecoded[decodedSequencePitchInBytes * i + k]
                        = soainputAnchorsDecoded[decodedSequencePitchInBytes * i + k];
                }
            }else{
                for(int k = group.thread_rank(); k < qualityPitchInBytes; k += group.size()){
                    d_anchorQualityScores[qualityPitchInBytes * i + k]
                        = soatotalAnchorQualityScoresFlat[qualityPitchInBytes * (offset + num - 1) + k];
                }
                for(int k = group.thread_rank(); k < decodedSequencePitchInBytes; k += group.size()){
                    d_subjectSequencesDataDecoded[decodedSequencePitchInBytes * i + k]
                        = soatotalDecodedAnchorsFlat[decodedSequencePitchInBytes * (offset + num - 1) + k];
                }
            }
        }
    }

    template<int blocksize>
    __global__
    void computeNumberOfSoaIterationResultsPerTaskKernel(
        int numTasks,
        int* __restrict__ d_addNumEntriesPerTask,
        int* __restrict__ d_addNumEntriesPerTaskPrefixSum,
        const extension::AbortReason* __restrict__ d_abortReasons,
        const bool* __restrict__ d_mateHasBeenFound,
        const int* __restrict__ d_sizeOfGapToMate
    ){
        const int tid = threadIdx.x + blockIdx.x * blocksize;
        const int stride = blocksize * gridDim.x;

        if(tid == 0){
            d_addNumEntriesPerTaskPrefixSum[0] = 0;
        }

        for(int i = tid; i < numTasks; i += stride){
            int num = 0;

            if(d_abortReasons[i] == extension::AbortReason::None){
                num = 1;

                if(d_mateHasBeenFound[i]){
                    if(d_sizeOfGapToMate[i] != 0){
                        num = 2;
                    }
                }
            }

            d_addNumEntriesPerTask[i] = num;
        }
    }

    template<int blocksize>
    __global__
    void makeSoAIterationResultsKernel(
        int numTasks,
        std::size_t outputAnchorPitchInBytes,
        std::size_t outputAnchorQualityPitchInBytes,
        const int* __restrict__ d_addNumEntriesPerTask,
        const int* __restrict__ d_addNumEntriesPerTaskPrefixSum,
        char* __restrict__ d_addTotalDecodedAnchorsFlat,
        char* __restrict__ d_addTotalAnchorQualityScoresFlat,
        int* __restrict__ d_addAnchorLengths,
        int* __restrict__ d_addAnchorBeginsInExtendedRead,
        std::size_t task_decodedSequencePitchInBytes,
        std::size_t task_qualityPitchInBytes,
        const extension::AbortReason* __restrict__ task_abortReason,
        const bool* __restrict__ task_mateHasBeenFound,
        const char* __restrict__ task_materevc,
        const char* __restrict__ task_materevcqual,
        const int* __restrict__ task_matelength,
        const int* __restrict__ d_sizeOfGapToMate,
        const int* __restrict__ d_outputAnchorLengths,
        const char* __restrict__ d_outputAnchors,
        const char* __restrict__ d_outputAnchorQualities,
        const int* __restrict__ d_accumExtensionsLengths
    ){
        for(int i = blockIdx.x; i < numTasks; i += gridDim.x){
            if(task_abortReason[i] == extension::AbortReason::None){                
                const int offset = d_addNumEntriesPerTaskPrefixSum[i];

                if(!task_mateHasBeenFound[i]){
                    const int length = d_outputAnchorLengths[i];
                    d_addAnchorLengths[offset] = length;

                    //copy result
                    for(int k = threadIdx.x; k < length; k += blockDim.x){
                        d_addTotalDecodedAnchorsFlat[offset * outputAnchorPitchInBytes + k] = d_outputAnchors[i * outputAnchorPitchInBytes + k];
                        d_addTotalAnchorQualityScoresFlat[offset * outputAnchorQualityPitchInBytes + k] = d_outputAnchorQualities[i * outputAnchorQualityPitchInBytes + k];
                    }
                    d_addAnchorBeginsInExtendedRead[offset] = d_accumExtensionsLengths[i];

                }else{
                    const int sizeofGap = d_sizeOfGapToMate[i];
                    if(sizeofGap == 0){                                
                        //copy mate revc

                        const int length = task_matelength[i];
                        d_addAnchorLengths[offset] = length;

                        for(int k = threadIdx.x; k < length; k += blockDim.x){
                            d_addTotalDecodedAnchorsFlat[offset * outputAnchorPitchInBytes + k] = task_materevc[i * task_decodedSequencePitchInBytes + k];
                            d_addTotalAnchorQualityScoresFlat[offset * outputAnchorQualityPitchInBytes + k] = task_materevcqual[i * task_qualityPitchInBytes + k];
                        }
                        d_addAnchorBeginsInExtendedRead[offset] = d_accumExtensionsLengths[i];          
                    }else{
                        //copy result
                        const int length = d_outputAnchorLengths[i];
                        d_addAnchorLengths[offset] = length;
                        
                        for(int k = threadIdx.x; k < length; k += blockDim.x){
                            d_addTotalDecodedAnchorsFlat[offset * outputAnchorPitchInBytes + k] = d_outputAnchors[i * outputAnchorPitchInBytes + k];
                            d_addTotalAnchorQualityScoresFlat[offset * outputAnchorQualityPitchInBytes + k] = d_outputAnchorQualities[i * outputAnchorQualityPitchInBytes + k];
                        }
                        d_addAnchorBeginsInExtendedRead[offset] = d_accumExtensionsLengths[i];
                        

                        //copy mate revc
                        const int length2 = task_matelength[i];
                        d_addAnchorLengths[(offset+1)] = length2;

                        for(int k = threadIdx.x; k < length; k += blockDim.x){
                            d_addTotalDecodedAnchorsFlat[(offset+1) * outputAnchorPitchInBytes + k] = task_materevc[i * task_decodedSequencePitchInBytes + k];
                            d_addTotalAnchorQualityScoresFlat[(offset+1) * outputAnchorQualityPitchInBytes + k] = task_materevcqual[i * task_qualityPitchInBytes + k];
                        }
                        d_addAnchorBeginsInExtendedRead[(offset+1)] = d_accumExtensionsLengths[i] + length;
                    }
                }
            }
        }
    };

    //replace positions which are covered by anchor and mate with the original data
    template<int blocksize, int groupsize>
    __global__
    void applyOriginalReadsToExtendedReads(
        std::size_t resultMSAColumnPitchInElements,
        int numFinishedTasks,
        char* __restrict__ d_decodedConsensus,
        char* __restrict__ d_consensusQuality,
        const int* __restrict__ d_resultLengths,
        const unsigned int* __restrict__ d_inputAnchorsEncoded,
        const int* __restrict__ d_inputAnchorLengths,
        const char* __restrict__ d_inputAnchorQualities,
        const bool* __restrict__ d_mateHasBeenFound,
        std::size_t  encodedSequencePitchInInts,
        std::size_t  qualityPitchInBytes
    ){
        const int numPairs = numFinishedTasks / 4;

        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int groupIdInBlock = threadIdx.x / groupsize;

        for(int pair = blockIdx.x; pair < numPairs; pair += gridDim.x){
            const int resultLength = d_resultLengths[4 * pair + groupIdInBlock];
            const int anchorLength = d_inputAnchorLengths[4 * pair + groupIdInBlock];
            const unsigned int* const inputAnchor = &d_inputAnchorsEncoded[(4 * pair + groupIdInBlock) * encodedSequencePitchInInts];
            char* const resultSequence = &d_decodedConsensus[(4 * pair + groupIdInBlock) * resultMSAColumnPitchInElements];
            const char* const inputQuality = &d_inputAnchorQualities[(4 * pair + groupIdInBlock) * qualityPitchInBytes];
            char* const resultQuality = &d_consensusQuality[(4 * pair + groupIdInBlock) * resultMSAColumnPitchInElements];

            SequenceHelpers::decodeSequence2Bit<int4>(group, inputAnchor, anchorLength, resultSequence);

            //copy anchor quality
            {
                const int numIters = anchorLength / sizeof(int);
                for(int i = group.thread_rank(); i < numIters; i += group.size()){
                    ((int*)resultQuality)[i] = ((const int*)inputQuality)[i];
                }
                const int remaining = anchorLength - sizeof(int) * numIters;
                if(remaining > 0){
                    for(int i = group.thread_rank(); i < remaining; i += group.size()){
                        resultQuality[sizeof(int) * numIters + i] = inputQuality[sizeof(int) * numIters + i];
                    }
                }
            }

            if(d_mateHasBeenFound[4 * pair + groupIdInBlock]){
                const int mateLength = d_inputAnchorLengths[4 * pair + groupIdInBlock + 1];
                const unsigned int* const anchorMate = &d_inputAnchorsEncoded[(4 * pair + groupIdInBlock + 1) * encodedSequencePitchInInts];
                const char* const anchorMateQuality = &d_inputAnchorQualities[(4 * pair + groupIdInBlock + 1) * qualityPitchInBytes];
                SequenceHelpers::decodeSequence2Bit<char>(group, anchorMate, mateLength, resultSequence + resultLength - mateLength);

                for(int i = group.thread_rank(); i < mateLength; i += group.size()){
                    resultQuality[resultLength - mateLength + i] = anchorMateQuality[i];
                }
            }
        }
    }

    template<int blocksize>
    __global__
    void flagFirstTasksOfConsecutivePairedTasks(
        int numTasks,
        bool* __restrict__ d_flags,
        const int* __restrict__ ids
    ){
        //d_flags must be zero'd

        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        for(int i = tid; i < numTasks - 1; i += stride){
            const bool areConsecutiveTasks = ids[i] + 1 == ids[i+1];
            const bool arePairedTasks = (ids[i] % 2) + 1 == (ids[i+1] % 2);

            if(areConsecutiveTasks && arePairedTasks){
                d_flags[i] = true;
            }
        }
    }

    template<int blocksize, int smemSizeBytes>
    __global__
    void flagPairedCandidatesKernel(
        const int* __restrict__ d_numChecks,
        const int* __restrict__ d_firstTasksOfPairsToCheck,
        const int* __restrict__ d_numCandidatesPerAnchor,
        const int* __restrict__ d_numCandidatesPerAnchorPrefixSum,
        const read_number* __restrict__ d_candidateReadIds,
        const int* __restrict__ d_numUsedReadIdsPerAnchor,
        const int* __restrict__ d_numUsedReadIdsPerAnchorPrefixSum,
        const read_number* __restrict__ d_usedReadIds,
        bool* __restrict__ d_isPairedCandidate
    ){

        constexpr int numSharedElements = SDIV(smemSizeBytes, sizeof(int));

        __shared__ read_number sharedElements[numSharedElements];

        //search elements of array1 in array2. if found, set output element to true
        //array1 and array2 must be sorted
        auto process = [&](
            const read_number* array1,
            int numElements1,
            const read_number* array2,
            int numElements2,
            bool* output
        ){
            const int numIterations = SDIV(numElements2, numSharedElements);

            for(int iteration = 0; iteration < numIterations; iteration++){

                const int begin = iteration * numSharedElements;
                const int end = min((iteration+1) * numSharedElements, numElements2);
                const int num = end - begin;

                for(int i = threadIdx.x; i < num; i += blockDim.x){
                    sharedElements[i] = array2[begin + i];
                }

                __syncthreads();

                //TODO in iteration > 0, we may skip elements at the beginning of first range

                for(int i = threadIdx.x; i < numElements1; i += blockDim.x){
                    if(!output[i]){
                        const read_number readId = array1[i];
                        const read_number readIdToFind = readId % 2 == 0 ? readId + 1 : readId - 1;

                        const bool found = thrust::binary_search(thrust::seq, sharedElements, sharedElements + num, readIdToFind);
                        if(found){
                            output[i] = true;
                        }
                    }
                }

                __syncthreads();
            }
        };

        const int numChecks = *d_numChecks;

        for(int a = blockIdx.x; a < numChecks; a += gridDim.x){
            const int firstTask = d_firstTasksOfPairsToCheck[a];
            //const int secondTask = firstTask + 1;

            //check for pairs in current candidates
            const int rangeBegin = d_numCandidatesPerAnchorPrefixSum[firstTask];                        
            const int rangeMid = d_numCandidatesPerAnchorPrefixSum[firstTask + 1];
            const int rangeEnd = rangeMid + d_numCandidatesPerAnchor[firstTask + 1];

            process(
                d_candidateReadIds + rangeBegin,
                rangeMid - rangeBegin,
                d_candidateReadIds + rangeMid,
                rangeEnd - rangeMid,
                d_isPairedCandidate + rangeBegin
            );

            process(
                d_candidateReadIds + rangeMid,
                rangeEnd - rangeMid,
                d_candidateReadIds + rangeBegin,
                rangeMid - rangeBegin,
                d_isPairedCandidate + rangeMid
            );

            //check for pairs in candidates of previous extension iterations

            const int usedRangeBegin = d_numUsedReadIdsPerAnchorPrefixSum[firstTask];                        
            const int usedRangeMid = d_numUsedReadIdsPerAnchorPrefixSum[firstTask + 1];
            const int usedRangeEnd = usedRangeMid + d_numUsedReadIdsPerAnchor[firstTask + 1];

            process(
                d_candidateReadIds + rangeBegin,
                rangeMid - rangeBegin,
                d_usedReadIds + usedRangeMid,
                usedRangeEnd - usedRangeMid,
                d_isPairedCandidate + rangeBegin
            );

            process(
                d_candidateReadIds + rangeMid,
                rangeEnd - rangeMid,
                d_usedReadIds + usedRangeBegin,
                usedRangeMid - usedRangeBegin,
                d_isPairedCandidate + rangeMid
            );
        }
    }

    template<int blocksize>
    __global__
    void flagGoodAlignmentsKernel(
        const BestAlignment_t* __restrict__ d_alignment_best_alignment_flags,
        const int* __restrict__ d_alignment_shifts,
        const int* __restrict__ d_alignment_overlaps,
        const int* __restrict__ d_anchorSequencesLength,
        const int* __restrict__ d_numCandidatesPerAnchor,
        const int* __restrict__ d_numCandidatesPerAnchorPrefixSum,
        const bool* __restrict__ d_isPairedCandidate,
        bool* __restrict__ d_keepflags,
        float min_overlap_ratio,
        int numAnchors,
        const int* __restrict__ currentNumCandidatesPtr,
        int initialNumCandidates
    ){
        using BlockReduceFloat = cub::BlockReduce<float, 128>;
        using BlockReduceInt = cub::BlockReduce<int, 128>;

        __shared__ union {
            typename BlockReduceFloat::TempStorage floatreduce;
            typename BlockReduceInt::TempStorage intreduce;
        } cubtemp;

        __shared__ int intbroadcast;
        __shared__ float floatbroadcast;

        for(int a = blockIdx.x; a < numAnchors; a += gridDim.x){
            const int num = d_numCandidatesPerAnchor[a];
            const int offset = d_numCandidatesPerAnchorPrefixSum[a];
            const float anchorLength = d_anchorSequencesLength[a];

            int threadReducedGoodAlignmentExists = 0;
            float threadReducedRelativeOverlapThreshold = 0.0f;

            for(int c = threadIdx.x; c < num; c += blockDim.x){
                d_keepflags[offset + c] = true;
            }

            //loop over candidates to compute relative overlap threshold

            for(int c = threadIdx.x; c < num; c += blockDim.x){
                const auto alignmentflag = d_alignment_best_alignment_flags[offset + c];
                const int shift = d_alignment_shifts[offset + c];

                if(alignmentflag != BestAlignment_t::None && shift >= 0){
                    if(!d_isPairedCandidate[offset+c]){
                        const float overlap = d_alignment_overlaps[offset + c];                            
                        const float relativeOverlap = overlap / anchorLength;
                        
                        if(relativeOverlap < 1.0f && fgeq(relativeOverlap, min_overlap_ratio)){
                            threadReducedGoodAlignmentExists = 1;
                            const float tmp = floorf(relativeOverlap * 10.0f) / 10.0f;
                            threadReducedRelativeOverlapThreshold = fmaxf(threadReducedRelativeOverlapThreshold, tmp);
                        }
                    }
                }else{
                    //remove alignment with negative shift or bad alignments
                    d_keepflags[offset + c] = false;
                }                       
            }

            int blockreducedGoodAlignmentExists = BlockReduceInt(cubtemp.intreduce)
                .Sum(threadReducedGoodAlignmentExists);
            if(threadIdx.x == 0){
                intbroadcast = blockreducedGoodAlignmentExists;
                //printf("task %d good: %d\n", a, blockreducedGoodAlignmentExists);
            }
            __syncthreads();

            blockreducedGoodAlignmentExists = intbroadcast;

            if(blockreducedGoodAlignmentExists > 0){
                float blockreducedRelativeOverlapThreshold = BlockReduceFloat(cubtemp.floatreduce)
                    .Reduce(threadReducedRelativeOverlapThreshold, cub::Max());
                if(threadIdx.x == 0){
                    floatbroadcast = blockreducedRelativeOverlapThreshold;
                    //printf("task %d thresh: %f\n", a, blockreducedRelativeOverlapThreshold);
                }
                __syncthreads();

                blockreducedRelativeOverlapThreshold = floatbroadcast;

                // loop over candidates and remove those with an alignment overlap threshold smaller than the computed threshold
                for(int c = threadIdx.x; c < num; c += blockDim.x){
                    if(!d_isPairedCandidate[offset+c]){
                        if(d_keepflags[offset + c]){
                            const float overlap = d_alignment_overlaps[offset + c];                            
                            const float relativeOverlap = overlap / anchorLength;                 

                            if(!fgeq(relativeOverlap, blockreducedRelativeOverlapThreshold)){
                                d_keepflags[offset + c] = false;
                            }
                        }
                    }
                }
            }else{
                //NOOP.
                //if no good alignment exists, no candidate is removed. we will try to work with the not-so-good alignments
                // if(threadIdx.x == 0){
                //     printf("no good alignment,nc %d\n", num);
                // }
            }

            __syncthreads();
        }
    
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;
        for(int i = *currentNumCandidatesPtr + tid; i < initialNumCandidates; i += stride){
            d_keepflags[i] = false;
        }
    }

    template<int blocksize, int groupsize>
    __global__
    void convertLocalIndicesInSegmentsToGlobalFlags(
        bool* __restrict__ d_flags,
        const int* __restrict__ indices,
        const int* __restrict__ segmentSizes,
        const int* __restrict__ segmentOffsets,
        int numSegments
    ){
        /*
            Input:
            indices: 0,1,2,0,0,0,0,3,5,0
            segmentSizes: 6,4,1
            segmentOffsets: 0,6,10,11

            Output:
            d_flags: 1,1,1,0,0,0,1,0,0,1,0,1

            d_flags must be initialized with 0
        */
        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());

        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;
        
        const int numGroups = stride / groupsize;
        const int groupId = tid / groupsize;


        for(int s = groupId; s < numSegments; s += numGroups){        
            const int num = segmentSizes[s];
            const int offset = segmentOffsets[s];

            for(int i = group.thread_rank(); i < num; i += group.size()){
                const int globalIndex = indices[offset + i] + offset;
                d_flags[globalIndex] = true;
            }
        }
    }


    template<int blocksize>
    __global__
    void computeTaskSplitGatherIndicesDefaultKernel(
        int numTasks,
        int* __restrict__ d_positions4,
        int* __restrict__ d_positionsNot4,
        int* __restrict__ d_numPositions4_out,
        int* __restrict__ d_numPositionsNot4_out,
        const int* __restrict__ d_run_endoffsets,
        const int* __restrict__ d_num_runs,
        const int* __restrict__ d_sortedindices,
        const int* __restrict__ task_ids,
        const int* __restrict__ d_outputoffsetsPos4,
        const int* __restrict__ d_outputoffsetsNotPos4
    ){
        __shared__ int count4;
        __shared__ int countNot4;

        if(threadIdx.x == 0){
            count4 = 0;
            countNot4 = 0;
        }
        __syncthreads();

        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        const int numRuns = *d_num_runs;

        auto group = cg::tiled_partition<4>(cg::this_thread_block());
        const int numGroups = stride / 4;
        const int groupId = tid / 4;

        for(int t = groupId; t < numRuns; t += numGroups){
            const int runBegin = (t == 0 ? 0 : d_run_endoffsets[t-1]);
            const int runEnd = d_run_endoffsets[t];

            const int size = runEnd - runBegin;
            if(size < 4){
                if(group.thread_rank() == 0){
                    atomicAdd(&countNot4, size);
                }

                if(group.thread_rank() < size){
                    d_positionsNot4[d_outputoffsetsNotPos4[t] + group.thread_rank()]
                        = d_sortedindices[runBegin + group.thread_rank()];
                }
            }else{
                if(size != 4){
                    if(group.thread_rank() == 0){
                        printf("error size %d\n", size);
                    }
                    group.sync(); //DEBUG
                }
                assert(size == 4);

                if(group.thread_rank() == 0){
                    atomicAdd(&count4, 4);
                }

                //sort 4 elements of same pairId by id. id is either 0,1,2, or 3
                const int position = d_sortedindices[runBegin + group.thread_rank()];
                const int id = task_ids[position];
                assert(0 <= id && id < 4);

                for(int x = 0; x < 4; x++){
                    if(id == x){
                        //d_positions4[groupoutputbegin + x] = position;
                        d_positions4[d_outputoffsetsPos4[t] + x] = position;
                    }
                }
            }
        }
    
        __syncthreads();
        if(threadIdx.x == 0){
            atomicAdd(d_numPositions4_out + 0, count4);
            atomicAdd(d_numPositionsNot4_out + 0, countNot4);
        }
    }

    
    //requires external shared memory of size (sizeof(int) * numTasks * 2) bytes;
    template<int blocksize, int elementsPerThread>
    __global__
    void computeTaskSplitGatherIndicesSmallInputGetStaticSmemSizeKernel(
        std::size_t* output
    ){
        using BlockLoad = cub::BlockLoad<int, blocksize, elementsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
        using BlockRadixSort = cub::BlockRadixSort<int, blocksize, elementsPerThread, int>;
        using BlockDiscontinuity = cub::BlockDiscontinuity<int, blocksize>;
        using BlockStore = cub::BlockStore<int, blocksize, elementsPerThread, cub::BLOCK_STORE_WARP_TRANSPOSE>;
        using BlockScan = cub::BlockScan<int, blocksize>;
        using BlockExchange = cub::BlockExchange<int, blocksize, elementsPerThread>;

        using TempType = union{
            typename BlockLoad::TempStorage load;
            typename BlockRadixSort::TempStorage sort;
            typename BlockDiscontinuity::TempStorage discontinuity;
            typename BlockStore::TempStorage store;
            typename BlockScan::TempStorage scan;
            typename BlockExchange::TempStorage exchange;
        };

        *output = sizeof(TempType);
    }

    template<int blocksize, int elementsPerThread>
    __global__
    void computeTaskSplitGatherIndicesSmallInputKernel(
        int numTasks,
        int* __restrict__ d_positions4,
        int* __restrict__ d_positionsNot4,
        int* __restrict__ d_numPositions4_out,
        int* __restrict__ d_numPositionsNot4_out,
        const int* __restrict__ task_pairIds,
        const int* __restrict__ task_ids,
        const int* __restrict__ d_minmax_pairId
    ){

        constexpr int maxInputSize = blocksize * elementsPerThread;

        assert(blockDim.x == blocksize);
        assert(gridDim.x == 1);
        assert(numTasks <= maxInputSize);

        using BlockLoad = cub::BlockLoad<int, blocksize, elementsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
        using BlockRadixSort = cub::BlockRadixSort<int, blocksize, elementsPerThread, int>;
        using BlockDiscontinuity = cub::BlockDiscontinuity<int, blocksize>;
        using BlockStore = cub::BlockStore<int, blocksize, elementsPerThread, cub::BLOCK_STORE_WARP_TRANSPOSE>;
        using BlockScan = cub::BlockScan<int, blocksize>;
        using BlockExchange = cub::BlockExchange<int, blocksize, elementsPerThread>;

        __shared__ union{
            typename BlockLoad::TempStorage load;
            typename BlockRadixSort::TempStorage sort;
            typename BlockDiscontinuity::TempStorage discontinuity;
            typename BlockStore::TempStorage store;
            typename BlockScan::TempStorage scan;
            typename BlockExchange::TempStorage exchange;
        } temp;

        extern __shared__ int extsmemTaskSplit[]; // (sizeof(int) * numTasks * 2) bytes
        int* const sharedCounts = &extsmemTaskSplit[0];
        int* const sharedIndices = sharedCounts + numTasks;

        int numRuns = 0;
        int prefixsum[elementsPerThread];

        {

            int myPairIds[elementsPerThread];
            int myIndices[elementsPerThread];
            int headFlags[elementsPerThread];
            int maxScan[elementsPerThread];
        
            #pragma unroll
            for(int i = 0; i < elementsPerThread; i++){
                myIndices[i] = elementsPerThread * threadIdx.x + i;
            }

            BlockLoad(temp.load).Load(task_pairIds, myPairIds, numTasks, std::numeric_limits<int>::max());
            __syncthreads();

            BlockRadixSort(temp.sort).Sort(myPairIds, myIndices);
            __syncthreads();

            BlockStore(temp.store).Store(sharedIndices, myIndices, numTasks);
            __syncthreads();
        
            BlockDiscontinuity(temp.discontinuity).FlagHeads(headFlags, myPairIds, cub::Inequality());
            __syncthreads();                    
            
            BlockScan(temp.scan).ExclusiveSum(headFlags, prefixsum, numRuns);
            __syncthreads();
            
            #pragma unroll
            for(int i = 0; i < elementsPerThread; i++){
                if(headFlags[i] > 0){
                    maxScan[i] = prefixsum[i];
                }else{
                    maxScan[i] = 0;
                }
            }                

            BlockScan(temp.scan).InclusiveScan(maxScan, maxScan, cub::Max{});
            __syncthreads();

            //compute counts of unique pair ids
            for(int i = threadIdx.x; i < numRuns; i += blockDim.x){
                sharedCounts[i] = 0;
            }

            __syncthreads();

            #pragma unroll
            for(int i = 0; i < elementsPerThread; i++){
                if(threadIdx.x * elementsPerThread + i < numTasks){
                    atomicAdd(sharedCounts + maxScan[i], 1);
                }
            }

            __syncthreads();
        }

        //compute output offsets and perform split
        
        int myCounts[elementsPerThread];
        int outputoffsets4[elementsPerThread];
        int outputoffsetsNot4[elementsPerThread];

        BlockLoad(temp.load).Load(sharedCounts, myCounts, numRuns, 0);
        __syncthreads();

        int myModifiedCounts[elementsPerThread];
        #pragma unroll
        for(int i = 0; i < elementsPerThread; i++){
            myModifiedCounts[i] = (myCounts[i] == 4) ? 4 : 0;
        }

        int numPos4 = 0;
        BlockScan(temp.scan).ExclusiveSum(myModifiedCounts, outputoffsets4, numPos4);
        __syncthreads();

        #pragma unroll
        for(int i = 0; i < elementsPerThread; i++){
            myModifiedCounts[i] = (myCounts[i] < 4) ? myCounts[i] : 0;
        }

        int numPosNot4 = 0;
        BlockScan(temp.scan).ExclusiveSum(myModifiedCounts, outputoffsetsNot4, numPosNot4);
        __syncthreads();

        BlockScan(temp.scan).ExclusiveSum(myCounts, prefixsum, numRuns);
        __syncthreads();

        BlockExchange(temp.exchange).BlockedToStriped(myCounts);
        __syncthreads();
        BlockExchange(temp.exchange).BlockedToStriped(outputoffsets4);
        __syncthreads();
        BlockExchange(temp.exchange).BlockedToStriped(outputoffsetsNot4);
        __syncthreads();
        BlockExchange(temp.exchange).BlockedToStriped(prefixsum);
        __syncthreads();

        //compact indices
        #pragma unroll
        for(int i = 0; i < elementsPerThread; i++){
            if(i * blocksize + threadIdx.x < numRuns){
                const int runBegin = prefixsum[i];
                const int runEnd = runBegin + myCounts[i];
                const int size = runEnd - runBegin;

                if(size < 4){
                    #pragma unroll
                    for(int k = 0; k < 4; k++){
                        if(k < size){
                            const int outputoffset = outputoffsetsNot4[i] + k;
                            d_positionsNot4[outputoffset] = sharedIndices[runBegin + k];
                        }
                    }
                }else{
                    int positions[4];
                    int ids[4];

                    #pragma unroll
                    for(int k = 0; k < 4; k++){
                        positions[k] = sharedIndices[runBegin + k];
                    }

                    #pragma unroll
                    for(int k = 0; k < 4; k++){
                        ids[k] = task_ids[positions[k]];
                        assert(0 <= ids[k] && ids[k] < 4);
                    }

                    // printf("thread %d. positions %d %d %d %d, runBegin %d\n", threadIdx.x, 
                    //     positions[0], positions[1], positions[2], positions[3],runBegin
                    // );

                    //sort 4 elements of same pairId by id and store them. id is either 0,1,2, or 3
                    #pragma unroll
                    for(int k = 0; k < 4; k++){
                        #pragma unroll
                        for(int l = 0; l < 4; l++){
                            if(ids[l] == k){
                                const int outputoffset = outputoffsets4[i] + k;
                                d_positions4[outputoffset] = positions[l];
                            }
                        }
                    }
                }

            }
        }

        __syncthreads();
        if(threadIdx.x == 0){
            atomicAdd(d_numPositions4_out, numPos4);
            atomicAdd(d_numPositionsNot4_out, numPosNot4);
        }

    }





    template<int blocksize>
    __global__
    void computeExtensionStepFromMsaKernel(
        int insertSize,
        int insertSizeStddev,
        const gpu::GPUMultiMSA multiMSA,
        const int* d_numCandidatesPerAnchor,
        const int* d_numCandidatesPerAnchorPrefixSum,
        const int* d_anchorSequencesLength,
        const int* d_accumExtensionsLengths,
        const int* d_inputMateLengths,
        extension::AbortReason* d_abortReasons,
        int* d_accumExtensionsLengthsOUT,
        char* d_outputAnchors,
        int outputAnchorPitchInBytes,
        char* d_outputAnchorQualities,
        int outputAnchorQualityPitchInBytes,
        int* d_outputAnchorLengths,
        const bool* d_isPairedTask,
        const unsigned int* d_inputanchormatedata,
        int encodedSequencePitchInInts,
        int decodedMatesRevCPitchInBytes,
        bool* d_outputMateHasBeenFound,
        int* d_sizeOfGapToMate,
        int minCoverageForExtension,
        int fixedStepsize
    ){

        using BlockReduce = cub::BlockReduce<int, blocksize>;
        using BlockReduceFloat = cub::BlockReduce<float, blocksize>;

        __shared__ union{
            typename BlockReduce::TempStorage reduce;
            typename BlockReduceFloat::TempStorage reduceFloat;
        } temp;

        constexpr int smemEncodedMateInts = 32;
        __shared__ unsigned int smemEncodedMate[smemEncodedMateInts];

        __shared__ int broadcastsmem_int;

        for(int t = blockIdx.x; t < multiMSA.numMSAs; t += gridDim.x){
            const int numCandidates = d_numCandidatesPerAnchor[t];

            if(numCandidates > 0){
                const gpu::GpuSingleMSA msa = multiMSA.getSingleMSA(t);

                const int anchorLength = d_anchorSequencesLength[t];
                int accumExtensionsLength = d_accumExtensionsLengths[t];
                const int mateLength = d_inputMateLengths[t];
                const bool isPaired = d_isPairedTask[t];

                const int consensusLength = msa.computeSize();

                auto consensusDecoded = msa.getDecodedConsensusIterator();
                auto consensusQuality = msa.getConsensusQualityIterator();

                extension::AbortReason* const abortReasonPtr = d_abortReasons + t;
                char* const outputAnchor = d_outputAnchors + t * outputAnchorPitchInBytes;
                char* const outputAnchorQuality = d_outputAnchorQualities + t * outputAnchorQualityPitchInBytes;
                int* const outputAnchorLengthPtr = d_outputAnchorLengths + t;
                bool* const mateHasBeenFoundPtr = d_outputMateHasBeenFound + t;

                int extendBy = std::min(
                    consensusLength - anchorLength, 
                    std::max(0, fixedStepsize)
                );
                //cannot extend over fragment 
                extendBy = std::min(extendBy, (insertSize + insertSizeStddev - mateLength) - accumExtensionsLength);

                //auto firstLowCoverageIter = std::find_if(coverage + anchorLength, coverage + consensusLength, [&](int cov){ return cov < minCoverageForExtension; });
                //coverage is monotonically decreasing. convert coverages to 1 if >= minCoverageForExtension, else 0. Find position of first 0
                int myPos = consensusLength;
                for(int i = anchorLength + threadIdx.x; i < consensusLength; i += blockDim.x){
                    int flag = msa.coverages[i] < minCoverageForExtension ? 0 : 1;
                    if(flag == 0 && i < myPos){
                        myPos = i;
                    }
                }

                myPos = BlockReduce(temp.reduce).Reduce(myPos, cub::Min{});

                if(threadIdx.x == 0){
                    broadcastsmem_int = myPos;
                }
                __syncthreads();
                myPos = broadcastsmem_int;
                __syncthreads();

                if(fixedStepsize <= 0){
                    extendBy = myPos - anchorLength;
                    extendBy = std::min(extendBy, (insertSize + insertSizeStddev - mateLength) - accumExtensionsLength);
                }

                auto makeAnchorForNextIteration = [&](){
                    if(extendBy == 0){
                        if(threadIdx.x == 0){
                            *abortReasonPtr = extension::AbortReason::MsaNotExtended;
                        }
                    }else{
                        if(threadIdx.x == 0){
                            d_accumExtensionsLengthsOUT[t] = accumExtensionsLength + extendBy;
                            *outputAnchorLengthPtr = anchorLength;
                        }           

                        for(int i = threadIdx.x; i < anchorLength; i += blockDim.x){
                            outputAnchor[i] = consensusDecoded[extendBy + i];
                            outputAnchorQuality[i] = consensusQuality[extendBy + i];
                        }
                    }
                };

                constexpr int requiredOverlapMate = 70; //TODO relative overlap 
                constexpr float maxRelativeMismatchesInOverlap = 0.06f;
                constexpr int maxAbsoluteMismatchesInOverlap = 10;

                const int maxNumMismatches = std::min(int(mateLength * maxRelativeMismatchesInOverlap), maxAbsoluteMismatchesInOverlap);

                

                if(isPaired && accumExtensionsLength + consensusLength - requiredOverlapMate + mateLength >= insertSize - insertSizeStddev){
                    //for each possibility to overlap the mate and consensus such that the merged sequence would end in the desired range [insertSize - insertSizeStddev, insertSize + insertSizeStddev]

                    const int firstStartpos = std::max(0, insertSize - insertSizeStddev - accumExtensionsLength - mateLength);
                    const int lastStartposExcl = std::min(
                        std::max(0, insertSize + insertSizeStddev - accumExtensionsLength - mateLength) + 1,
                        consensusLength - requiredOverlapMate
                    );

                    int bestOverlapMismatches = std::numeric_limits<int>::max();
                    int bestOverlapStartpos = -1;

                    const unsigned int* encodedMate = nullptr;
                    {
                        const unsigned int* const gmemEncodedMate = d_inputanchormatedata + t * encodedSequencePitchInInts;
                        const int requirednumints = SequenceHelpers::getEncodedNumInts2Bit(mateLength);
                        if(smemEncodedMateInts >= requirednumints){
                            for(int i = threadIdx.x; i < requirednumints; i += blockDim.x){
                                smemEncodedMate[i] = gmemEncodedMate[i];
                            }
                            encodedMate = &smemEncodedMate[0];
                            __syncthreads();
                        }else{
                            encodedMate = &gmemEncodedMate[0];
                        }
                    }

                    for(int startpos = firstStartpos; startpos < lastStartposExcl; startpos++){
                        //compute metrics of overlap

                        //Hamming distance. positions which do not overlap are not accounted for
                        int ham = 0;
                        for(int i = threadIdx.x; i < min(consensusLength - startpos, mateLength); i += blockDim.x){
                            std::uint8_t encbasemate = SequenceHelpers::getEncodedNuc2Bit(encodedMate, mateLength, mateLength - 1 - i);
                            std::uint8_t encbasematecomp = SequenceHelpers::complementBase2Bit(encbasemate);
                            char decbasematecomp = SequenceHelpers::decodeBase(encbasematecomp);

                            //TODO store consensusDecoded in smem ?
                            ham += (consensusDecoded[startpos + i] != decbasematecomp) ? 1 : 0;
                        }

                        ham = BlockReduce(temp.reduce).Sum(ham);

                        if(threadIdx.x == 0){
                            broadcastsmem_int = ham;
                        }
                        __syncthreads();
                        ham = broadcastsmem_int;
                        __syncthreads();

                        if(bestOverlapMismatches > ham){
                            bestOverlapMismatches = ham;
                            bestOverlapStartpos = startpos;
                        }

                        if(bestOverlapMismatches == 0){
                            break;
                        }
                    }

                    // if(threadIdx.x == 0){
                    //     printf("gpu: bestOverlapMismatches %d,bestOverlapStartpos %d\n", bestOverlapMismatches, bestOverlapStartpos);
                    // }

                    if(bestOverlapMismatches <= maxNumMismatches){
                        const int mateStartposInConsensus = bestOverlapStartpos;
                        const int missingPositionsBetweenAnchorEndAndMateBegin = std::max(0, mateStartposInConsensus - anchorLength);
                        // if(threadIdx.x == 0){
                        //     printf("missingPositionsBetweenAnchorEndAndMateBegin %d\n", missingPositionsBetweenAnchorEndAndMateBegin);
                        // }

                        if(missingPositionsBetweenAnchorEndAndMateBegin > 0){
                            //bridge the gap between current anchor and mate

                            for(int i = threadIdx.x; i < missingPositionsBetweenAnchorEndAndMateBegin; i += blockDim.x){
                                outputAnchor[i] = consensusDecoded[anchorLength + i];
                                outputAnchorQuality[i] = consensusQuality[anchorLength + i];
                            }

                            if(threadIdx.x == 0){
                                d_accumExtensionsLengthsOUT[t] = accumExtensionsLength + anchorLength;
                                *outputAnchorLengthPtr = missingPositionsBetweenAnchorEndAndMateBegin;
                                *mateHasBeenFoundPtr = true;
                                d_sizeOfGapToMate[t] = missingPositionsBetweenAnchorEndAndMateBegin;
                            }
                        }else{

                            if(threadIdx.x == 0){
                                d_accumExtensionsLengthsOUT[t] = accumExtensionsLength + mateStartposInConsensus;
                                *outputAnchorLengthPtr = 0;
                                *mateHasBeenFoundPtr = true;
                                d_sizeOfGapToMate[t] = 0;
                            }
                        }

                        
                    }else{
                        makeAnchorForNextIteration();
                    }
                }else{
                    makeAnchorForNextIteration();
                }

            }else{ //numCandidates == 0
                if(threadIdx.x == 0){
                    d_outputMateHasBeenFound[t] = false;
                    d_abortReasons[t] = extension::AbortReason::NoPairedCandidatesAfterAlignment;
                }
            }
        }
    }

    template<int blocksize>
    __global__
    void computeExtensionStepQualityKernel(
        float* d_goodscores,
        const gpu::GPUMultiMSA multiMSA,
        const extension::AbortReason* d_abortReasons,
        const bool* d_mateHasBeenFound,
        const int* accumExtensionLengthsBefore,
        const int* accumExtensionLengthsAfter,
        const int* anchorLengths,
        const int* d_numCandidatesPerAnchor,
        const int* d_numCandidatesPerAnchorPrefixSum,
        const int* d_candidateSequencesLengths,
        const int* d_alignment_shifts,
        const BestAlignment_t* d_alignment_best_alignment_flags,
        const unsigned int* d_candidateSequencesData,
        const gpu::MSAColumnProperties* d_msa_column_properties,
        int encodedSequencePitchInInts
    ){
        using BlockReduce = cub::BlockReduce<int, blocksize>;
        using BlockReduceFloat = cub::BlockReduce<float, blocksize>;
        using AmbiguousColumnsChecker = CheckAmbiguousColumns<blocksize>;

        __shared__ union{
            typename BlockReduce::TempStorage reduce;
            typename BlockReduceFloat::TempStorage reduceFloat;
            typename AmbiguousColumnsChecker::TempStorage columnschecker;
        } temp;

        __shared__ typename AmbiguousColumnsChecker::SplitInfos smemSplitInfos;

        for(int t = blockIdx.x; t < multiMSA.numMSAs; t += gridDim.x){
            if(d_abortReasons[t] == extension::AbortReason::None && !d_mateHasBeenFound[t]){

                const gpu::GpuSingleMSA msa = multiMSA.getSingleMSA(t);

                const int extendedBy = accumExtensionLengthsAfter[t] - accumExtensionLengthsBefore[t];
                const int anchorLength = anchorLengths[t];

                //const float* const mySupport = msa.support;
                const int* const myCounts = msa.counts;
                const int* const myCoverage = msa.coverages;

                const int* const myCandidateLengths = d_candidateSequencesLengths + d_numCandidatesPerAnchorPrefixSum[t];
                const int* const myCandidateShifts = d_alignment_shifts + d_numCandidatesPerAnchorPrefixSum[t];
                const BestAlignment_t* const myCandidateBestAlignmentFlags = d_alignment_best_alignment_flags + d_numCandidatesPerAnchorPrefixSum[t];
                const unsigned int* const myCandidateSequencesData = d_candidateSequencesData + encodedSequencePitchInInts * d_numCandidatesPerAnchorPrefixSum[t];

                // float supportSum = 0.0f;

                // for(int i = threadIdx.x; i < extendedBy; i += blockDim.x){
                //     supportSum += 1.0f - mySupport[anchorLength + i];
                // }

                // float reducedSupport = BlockReduceFloat(temp.reduceFloat).Sum(supportSum);

                // //printf("reducedSupport %f\n", reducedSupport);
                // if(threadIdx.x == 0){
                //     d_goodscores[t] = reducedSupport;
                // }

                AmbiguousColumnsChecker checker(
                    myCounts + 0 * msa.columnPitchInElements,
                    myCounts + 1 * msa.columnPitchInElements,
                    myCounts + 2 * msa.columnPitchInElements,
                    myCounts + 3 * msa.columnPitchInElements,
                    myCoverage
                );

                //int count = checker.getAmbiguousColumnCount(anchorLength, anchorLength + extendedBy, temp.reduce);

                //auto a = clock();

                checker.getSplitInfos(anchorLength, anchorLength + extendedBy, 0.4f, 0.6f, smemSplitInfos);

                //auto b = clock();

                int count = checker.getNumberOfSplits(
                    smemSplitInfos, 
                    d_msa_column_properties[t],
                    d_numCandidatesPerAnchor[t], 
                    myCandidateShifts,
                    myCandidateBestAlignmentFlags,
                    myCandidateLengths,
                    myCandidateSequencesData, 
                    encodedSequencePitchInInts, 
                    temp.columnschecker
                );

                //auto c = clock();

                if(threadIdx.x == 0){
                    d_goodscores[t] = count;
                    //printf("cand %d extendedBy %d, %lu %lu, infos %d, count %d\n", d_numCandidatesPerAnchor[t], extendedBy, b-a, c-b, smemSplitInfos.numSplitInfos, count);
                }
                
                __syncthreads();
            }
        }
    }

    template<int blocksize>
    __global__
    void makePairResultsFromFinishedTasksDryRunKernel(
        int numResults,
        int* __restrict__ outputLengthUpperBounds,
        const int* __restrict__ originalReadLengths,
        const int* __restrict__ dataExtendedReadLengths,
        const char* __restrict__ dataExtendedReadSequences,
        const char* __restrict__ dataExtendedReadQualities,
        const bool* __restrict__ dataMateHasBeenFound,
        const float* __restrict__ dataGoodScores,
        int inputPitch,
        int insertSize,
        int insertSizeStddev
    ){
        auto group = cg::this_thread_block();
        const int numGroupsInGrid = (blockDim.x * gridDim.x) / group.size();
        const int groupIdInGrid = (threadIdx.x + blockDim.x * blockIdx.x) / group.size();

        auto computeRead2Begin = [&](int i){
            if(dataMateHasBeenFound[i]){
                return dataExtendedReadLengths[i] - originalReadLengths[i+1];
            }else{
                return -1;
            }
        };

        auto mergelength = [&](int l, int r){
            assert(l+1 == r);
            assert(l % 2 == 0);

            const int lengthR = dataExtendedReadLengths[r];

            const int read2begin = computeRead2Begin(l);
   
            auto overlapstart = read2begin;

            const int resultsize = overlapstart + lengthR;
            
            return resultsize;
        };
    
        //process pair at position pairIdsToProcess[posInList] and store to result position posInList
        for(int posInList = groupIdInGrid; posInList < numResults; posInList += numGroupsInGrid){
            group.sync(); //reuse smem

            const int p = posInList;

            const int i0 = 4 * p + 0;
            const int i1 = 4 * p + 1;
            const int i2 = 4 * p + 2;
            const int i3 = 4 * p + 3;

            int* const myResultLengths = outputLengthUpperBounds + posInList;            

            auto LRmatefoundfunc = [&](){
                const int extendedReadLength3 = dataExtendedReadLengths[i3];
                const int originalLength3 = originalReadLengths[i3];

                int resultsize = mergelength(i0, i1);
                if(extendedReadLength3 > originalLength3){
                    resultsize += extendedReadLength3 - originalLength3;
                }                

                if(group.thread_rank() == 0){
                    *myResultLengths = resultsize;
                }
            };

            auto RLmatefoundfunc = [&](){
                const int extendedReadLength1 = dataExtendedReadLengths[i1];
                const int originalLength1 = originalReadLengths[i1];

                const int mergedLength = mergelength(i2, i3);
                int resultsize = mergedLength;
                if(extendedReadLength1 > originalLength1){
                    resultsize += extendedReadLength1 - originalLength1;
                }
                
                if(group.thread_rank() == 0){
                    *myResultLengths = resultsize;
                }
            };

            if(dataMateHasBeenFound[i0] && dataMateHasBeenFound[i2]){
                if(dataGoodScores[i0] < dataGoodScores[i2]){
                    LRmatefoundfunc();
                }else{
                    RLmatefoundfunc();
                }                
            }else 
            if(dataMateHasBeenFound[i0]){
                LRmatefoundfunc();                
            }else if(dataMateHasBeenFound[i2]){
                RLmatefoundfunc();                
            }else{
                constexpr int minimumOverlap = 40;

                int currentsize = 0;

                const int extendedReadLength0 = dataExtendedReadLengths[i0];
                const int extendedReadLength1 = dataExtendedReadLengths[i1];
                const int extendedReadLength2 = dataExtendedReadLengths[i2];
                const int extendedReadLength3 = dataExtendedReadLengths[i3];

                const int originalLength1 = originalReadLengths[i1];
                const int originalLength3 = originalReadLengths[i3];

                //insert extensions of reverse complement of d3 at beginning
                if(extendedReadLength3 > originalLength3){
                    currentsize = (extendedReadLength3 - originalLength3);
                }

                //try to find overlap of d0 and revc(d2)
                bool didMergeDifferentStrands = false;

                if(extendedReadLength0 + extendedReadLength2 >= insertSize - insertSizeStddev + minimumOverlap){
                    const int maxNumberOfPossibilities = 2*insertSizeStddev + 1;
                    const int resultLengthUpperBound = insertSize - insertSizeStddev + maxNumberOfPossibilities;
                    currentsize += resultLengthUpperBound;
                    didMergeDifferentStrands = true;
                }

                if(didMergeDifferentStrands){

                }else{
                    currentsize += extendedReadLength0;
                }

                if(didMergeDifferentStrands && extendedReadLength1 > originalLength1){

                    //insert extensions of d1 at end
                    currentsize += (extendedReadLength1 - originalLength1);
                }

                if(group.thread_rank() == 0){
                    *myResultLengths = currentsize;
                }
            }

        }
    }


    //requires external shared memory of 2*outputPitch per group in block
    template<int blocksize>
    __global__
    void makePairResultsFromFinishedTasksKernel(
        int numResults,
        bool* __restrict__ outputAnchorIsLR,
        char* __restrict__ outputSequences,
        char* __restrict__ outputQualities,
        int* __restrict__ outputLengths,
        int* __restrict__ outRead1Begins,
        int* __restrict__ outRead2Begins,
        bool* __restrict__ outMateHasBeenFound,
        bool* __restrict__ outMergedDifferentStrands,
        int outputPitch,
        const int* __restrict__ originalReadLengths,
        const int* __restrict__ dataExtendedReadLengths,
        const char* __restrict__ dataExtendedReadSequences,
        const char* __restrict__ dataExtendedReadQualities,
        const bool* __restrict__ dataMateHasBeenFound,
        const float* __restrict__ dataGoodScores,
        int inputPitch,
        int insertSize,
        int insertSizeStddev
    ){
        auto group = cg::this_thread_block();
        const int numGroupsInGrid = (blockDim.x * gridDim.x) / group.size();
        const int groupIdInGrid = (threadIdx.x + blockDim.x * blockIdx.x) / group.size();

        extern __shared__ int smemForResults[];
        char* const smemChars = (char*)&smemForResults[0];
        char* const smemSequence = smemChars;
        char* const smemSequence2 = smemSequence + outputPitch;
        char* const smemQualities = smemSequence2; //alias

        __shared__ typename gpu::MismatchRatioGlueDecider<blocksize>::TempStorage smemDecider;

        auto computeRead2Begin = [&](int i){
            if(dataMateHasBeenFound[i]){
                //extendedRead.length() - task.decodedMateRevC.length();
                return dataExtendedReadLengths[i] - originalReadLengths[i+1];
            }else{
                return -1;
            }
        };

        auto mergelength = [&](int l, int r){
            assert(l+1 == r);
            assert(l % 2 == 0);

            const int lengthR = dataExtendedReadLengths[r];

            const int read2begin = computeRead2Begin(l);
   
            auto overlapstart = read2begin;

            const int resultsize = overlapstart + lengthR;
            
            return resultsize;
        };

        //merge extensions of the same pair and same strand and append to result
        auto merge = [&](auto& group, int l, int r, char* sequenceOutput, char* qualityOutput){
            assert(l+1 == r);
            assert(l % 2 == 0);

            const int lengthL = dataExtendedReadLengths[l];
            const int lengthR = dataExtendedReadLengths[r];
            const int originalLengthR = originalReadLengths[r];
   
            auto overlapstart = computeRead2Begin(l);

            for(int i = group.thread_rank(); i < lengthL; i += group.size()){
                sequenceOutput[i] 
                    = dataExtendedReadSequences[l * inputPitch + i];
            }

            for(int i = group.thread_rank(); i < lengthR - originalLengthR; i += group.size()){
                sequenceOutput[lengthL + i] 
                    = dataExtendedReadSequences[r * inputPitch + originalLengthR + i];
            }

            for(int i = group.thread_rank(); i < lengthL; i += group.size()){
                qualityOutput[i] 
                    = dataExtendedReadQualities[l * inputPitch + i];
            }

            for(int i = group.thread_rank(); i < lengthR - originalLengthR; i += group.size()){
                qualityOutput[lengthL + i] 
                    = dataExtendedReadQualities[r * inputPitch + originalLengthR + i];
            }
        };
    
        //process pair at position pairIdsToProcess[posInList] and store to result position posInList
        for(int posInList = groupIdInGrid; posInList < numResults; posInList += numGroupsInGrid){
            group.sync(); //reuse smem

            const int p = posInList;

            const int i0 = 4 * p + 0;
            const int i1 = 4 * p + 1;
            const int i2 = 4 * p + 2;
            const int i3 = 4 * p + 3;

            char* const myResultSequence = outputSequences + posInList * outputPitch;
            char* const myResultQualities = outputQualities + posInList * outputPitch;
            int* const myResultRead1Begins = outRead1Begins + posInList;
            int* const myResultRead2Begins = outRead2Begins + posInList;
            int* const myResultLengths = outputLengths + posInList;
            bool* const myResultAnchorIsLR = outputAnchorIsLR + posInList;
            bool* const myResultMateHasBeenFound = outMateHasBeenFound + posInList;
            bool* const myResultMergedDifferentStrands = outMergedDifferentStrands + posInList;
            

            auto LRmatefoundfunc = [&](){
                const int extendedReadLength3 = dataExtendedReadLengths[i3];
                const int originalLength3 = originalReadLengths[i3];

                int resultsize = mergelength(i0, i1);
                if(extendedReadLength3 > originalLength3){
                    resultsize += extendedReadLength3 - originalLength3;
                }                

                int currentsize = 0;

                if(extendedReadLength3 > originalLength3){
                    //insert extensions of reverse complement of d3 at beginning

                    for(int k = group.thread_rank(); k < (extendedReadLength3 - originalLength3); k += group.size()){
                        myResultSequence[k] = SequenceHelpers::complementBaseDecoded(
                            dataExtendedReadSequences[i3 * inputPitch + originalLength3 + (extendedReadLength3 - originalLength3) - 1 - k]
                        );
                    }

                    for(int k = group.thread_rank(); k < (extendedReadLength3 - originalLength3); k += group.size()){
                        myResultQualities[k] = 
                            dataExtendedReadQualities[i3 * inputPitch + originalLength3 + (extendedReadLength3 - originalLength3) - 1 - k];
                    }

                    currentsize = (extendedReadLength3 - originalLength3);
                }

                merge(group, i0, i1, myResultSequence + currentsize, myResultQualities + currentsize);

                int read1begin = 0;
                int read2begin = computeRead2Begin(i0);

                if(extendedReadLength3 > originalLength3){                    
                    read1begin += (extendedReadLength3 - originalLength3);
                    read2begin += (extendedReadLength3 - originalLength3);
                }

                if(group.thread_rank() == 0){
                    *myResultRead1Begins = read1begin;
                    *myResultRead2Begins = read2begin;
                    *myResultLengths = resultsize;
                    *myResultAnchorIsLR = true;
                    *myResultMateHasBeenFound = true;
                    *myResultMergedDifferentStrands = false;
                }
            };

            auto RLmatefoundfunc = [&](){
                const int extendedReadLength1 = dataExtendedReadLengths[i1];
                const int originalLength1 = originalReadLengths[i1];

                const int mergedLength = mergelength(i2, i3);
                int resultsize = mergedLength;
                if(extendedReadLength1 > originalLength1){
                    resultsize += extendedReadLength1 - originalLength1;
                }

                merge(group, i2, i3, smemSequence, smemQualities);

                group.sync();

                for(int k = group.thread_rank(); k < mergedLength; k += group.size()){
                    myResultSequence[k] = SequenceHelpers::complementBaseDecoded(
                        smemSequence[mergedLength - 1 - k]
                    );
                }

                for(int k = group.thread_rank(); k < mergedLength; k += group.size()){
                    myResultQualities[k] = smemQualities[mergedLength - 1 - k];
                }

                group.sync();

                const int sizeOfRightExtension = mergedLength - (computeRead2Begin(i2) + originalReadLengths[i3]);

                int read1begin = 0;
                int newread2begin = mergedLength - (read1begin + originalReadLengths[i2]);
                int newread2length = originalReadLengths[i2];
                int newread1begin = sizeOfRightExtension;
                int newread1length = originalReadLengths[i3];

                assert(newread1begin >= 0);
                assert(newread2begin >= 0);
                assert(newread1begin + newread1length <= mergedLength);
                assert(newread2begin + newread2length <= mergedLength);

                if(extendedReadLength1 > originalLength1){
                    //insert extensions of d1 at end
                    for(int k = group.thread_rank(); k < (extendedReadLength1 - originalLength1); k += group.size()){
                        myResultSequence[mergedLength + k] = 
                            dataExtendedReadSequences[i1 * inputPitch + originalLength1 + k];                        
                    }

                    for(int k = group.thread_rank(); k < (extendedReadLength1 - originalLength1); k += group.size()){
                        myResultQualities[mergedLength + k] = 
                            dataExtendedReadQualities[i1 * inputPitch + originalLength1 + k];                        
                    }
                }

                read1begin = newread1begin;
                const int read2begin = newread2begin;
                
                if(group.thread_rank() == 0){
                    *myResultRead1Begins = read1begin;
                    *myResultRead2Begins = read2begin;
                    *myResultLengths = resultsize;
                    *myResultAnchorIsLR = false;
                    *myResultMateHasBeenFound = true;
                    *myResultMergedDifferentStrands = false;
                }
            };

            if(dataMateHasBeenFound[i0] && dataMateHasBeenFound[i2]){
                if(dataGoodScores[i0] < dataGoodScores[i2]){
                    LRmatefoundfunc();
                }else{
                    RLmatefoundfunc();
                }                
            }else 
            if(dataMateHasBeenFound[i0]){
                LRmatefoundfunc();                
            }else if(dataMateHasBeenFound[i2]){
                RLmatefoundfunc();                
            }else{
                
                constexpr int minimumOverlap = 40;
                constexpr float maxRelativeErrorInOverlap = 0.05;

                int read1begin = 0;
                int read2begin = computeRead2Begin(i0);
                int currentsize = 0;

                const int extendedReadLength0 = dataExtendedReadLengths[i0];
                const int extendedReadLength1 = dataExtendedReadLengths[i1];
                const int extendedReadLength2 = dataExtendedReadLengths[i2];
                const int extendedReadLength3 = dataExtendedReadLengths[i3];

                const int originalLength0 = originalReadLengths[i0];
                const int originalLength1 = originalReadLengths[i1];
                const int originalLength2 = originalReadLengths[i2];
                const int originalLength3 = originalReadLengths[i3];

                //insert extensions of reverse complement of d3 at beginning
                if(extendedReadLength3 > originalLength3){

                    for(int k = group.thread_rank(); k < (extendedReadLength3 - originalLength3); k += group.size()){
                        myResultSequence[k] = SequenceHelpers::complementBaseDecoded(
                            dataExtendedReadSequences[i3 * inputPitch + originalLength3 + (extendedReadLength3 - originalLength3) - 1 - k]
                        );
                    }

                    for(int k = group.thread_rank(); k < (extendedReadLength3 - originalLength3); k += group.size()){
                        myResultQualities[k] = 
                            dataExtendedReadQualities[i3 * inputPitch + originalLength3 + (extendedReadLength3 - originalLength3) - 1 - k];
                    }

                    currentsize = (extendedReadLength3 - originalLength3);
                    read1begin = (extendedReadLength3 - originalLength3);
                }

                //try to find overlap of d0 and revc(d2)                

                bool didMergeDifferentStrands = false;

                if(extendedReadLength0 + extendedReadLength2 >= insertSize - insertSizeStddev + minimumOverlap){
                    //copy sequences to smem

                    for(int k = group.thread_rank(); k < extendedReadLength0; k += group.size()){
                        smemSequence[k] = dataExtendedReadSequences[i0 * inputPitch + k];
                    }

                    for(int k = group.thread_rank(); k < extendedReadLength2; k += group.size()){
                        smemSequence2[k] = SequenceHelpers::complementBaseDecoded(
                            dataExtendedReadSequences[i2 * inputPitch + extendedReadLength2 - 1 - k]
                        );
                    }

                    group.sync();

                    const int maxNumberOfPossibilities = 2*insertSizeStddev + 1;

                    gpu::MismatchRatioGlueDecider<blocksize> decider(smemDecider, minimumOverlap, maxRelativeErrorInOverlap);
                    gpu::QualityWeightedGapGluer gluer(originalLength0, originalLength2);

                    for(int p = 0; p < maxNumberOfPossibilities; p++){                       

                        const int resultlength = insertSize - insertSizeStddev + p;

                        auto decision = decider(
                            MyStringView(smemSequence, extendedReadLength0), 
                            MyStringView(smemSequence2, extendedReadLength2), 
                            resultlength,
                            MyStringView(&dataExtendedReadQualities[i0 * inputPitch], extendedReadLength0),
                            MyStringView(&dataExtendedReadQualities[i2 * inputPitch], extendedReadLength2)
                        );

                        if(decision.valid){
                            gluer(group, decision, myResultSequence + currentsize, myResultQualities + currentsize);
                            currentsize += resultlength;
                            
                            didMergeDifferentStrands = true;
                            break;
                        }
                    }
                }

                if(didMergeDifferentStrands){
                    read2begin = currentsize - originalLength2;
                }else{
                    //initialize result with d0
                    for(int k = group.thread_rank(); k < extendedReadLength0; k += group.size()){
                        myResultSequence[currentsize + k] = dataExtendedReadSequences[i0 * inputPitch + k];
                    }

                    for(int k = group.thread_rank(); k < extendedReadLength0; k += group.size()){
                        myResultQualities[currentsize + k] = dataExtendedReadQualities[i0 * inputPitch + k];
                    }

                    currentsize += extendedReadLength0;
                }

                if(didMergeDifferentStrands && extendedReadLength1 > originalLength1){

                    //insert extensions of d1 at end

                    for(int k = group.thread_rank(); k < (extendedReadLength1 - originalLength1); k += group.size()){
                        myResultSequence[currentsize + k] = dataExtendedReadSequences[i1 * inputPitch + originalLength1 + k];
                    }

                    for(int k = group.thread_rank(); k < (extendedReadLength1 - originalLength1); k += group.size()){
                        myResultQualities[currentsize + k] = dataExtendedReadQualities[i1 * inputPitch + originalLength1 + k];
                    }

                    currentsize += (extendedReadLength1 - originalLength1);
                }

                if(group.thread_rank() == 0){
                    *myResultRead1Begins = read1begin;
                    *myResultRead2Begins = read2begin;
                    *myResultLengths = currentsize;
                    *myResultAnchorIsLR = true;
                    *myResultMateHasBeenFound = didMergeDifferentStrands;
                    *myResultMergedDifferentStrands = didMergeDifferentStrands;
                }
            }

        }
    }

    template<int blocksize, int itemsPerThread, bool inclusive, class T>
    __global__
    void prefixSumSingleBlockKernel(
        const T* input,
        T* output,
        int N
    ){
        struct BlockPrefixCallbackOp{
            // Running prefix
            int running_total;

            __device__
            BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
            // Callback operator to be entered by the first warp of threads in the block.
            // Thread-0 is responsible for returning a value for seeding the block-wide scan.
            __device__
            int operator()(int block_aggregate){
                int old_prefix = running_total;
                running_total += block_aggregate;
                return old_prefix;
            }
        };

        assert(blocksize == blockDim.x);

        using BlockScan = cub::BlockScan<T, blocksize>;
        using BlockLoad = cub::BlockLoad<T, blocksize, itemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
        using BlockStore = cub::BlockStore<T, blocksize, itemsPerThread, cub::BLOCK_STORE_WARP_TRANSPOSE>;

        __shared__ typename BlockScan::TempStorage blockscantemp;
        __shared__ union{
            typename BlockLoad::TempStorage load;
            typename BlockStore::TempStorage store;
        } temp;

        T items[itemsPerThread];

        BlockPrefixCallbackOp prefix_op(0);

        const int iterations = SDIV(N, blocksize);

        int remaining = N;

        const T* currentInput = input;
        T* currentOutput = output;

        for(int iteration = 0; iteration < iterations; iteration++){
            const int valid_items = min(itemsPerThread * blocksize, remaining);

            BlockLoad(temp.load).Load(currentInput, items, valid_items, 0);

            if(inclusive){
                BlockScan(blockscantemp).InclusiveSum(
                    items, items, prefix_op
                );
            }else{
                BlockScan(blockscantemp).ExclusiveSum(
                    items, items, prefix_op
                );
            }
            __syncthreads();

            BlockStore(temp.store).Store(currentOutput, items, valid_items);
            __syncthreads();

            remaining -= valid_items;
            currentInput += valid_items;
            currentOutput += valid_items;
        }
    }


    template<class T, class U>
    __global__ 
    void setFirstSegmentIdsKernel(
        const T* __restrict__ segmentSizes,
        int* __restrict__ segmentIds,
        const U* __restrict__ segmentOffsets,
        int N
    ){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        for(int i = tid; i < N; i += stride){
            if(segmentSizes[i] > 0){
                segmentIds[segmentOffsets[i]] = i;
            }
        }
    }



    //flag candidates to remove because they are equal to anchor id or equal to mate id
    __global__
    void flagCandidateIdsWhichAreEqualToAnchorOrMateKernel(
        const read_number* __restrict__ candidateReadIds,
        const read_number* __restrict__ anchorReadIds,
        const read_number* __restrict__ mateReadIds,
        const int* __restrict__ numCandidatesPerAnchorPrefixSum,
        const int* __restrict__ numCandidatesPerAnchor,
        bool* __restrict__ keepflags, // size numCandidates
        bool* __restrict__ mateRemovedFlags, //size numTasks
        int* __restrict__ numCandidatesPerAnchorOut,
        int numTasks,
        bool isPairedEnd
    ){

        using BlockReduceInt = cub::BlockReduce<int, 128>;

        __shared__ typename BlockReduceInt::TempStorage intreduce1;
        __shared__ typename BlockReduceInt::TempStorage intreduce2;

        for(int a = blockIdx.x; a < numTasks; a += gridDim.x){
            const int size = numCandidatesPerAnchor[a];
            const int offset = numCandidatesPerAnchorPrefixSum[a];
            const read_number anchorId = anchorReadIds[a];
            read_number mateId = 0;
            if(isPairedEnd){
                mateId = mateReadIds[a];
            }

            int mateIsRemoved = 0;
            int numRemoved = 0;

            // if(threadIdx.x == 0){
            //     printf("looking for anchor %u, mate %u\n", anchorId, mateId);
            // }
            __syncthreads();

            for(int i = threadIdx.x; i < size; i+= blockDim.x){
                bool keep = true;

                const read_number candidateId = candidateReadIds[offset + i];
                //printf("tid %d, comp %u at position %d\n", threadIdx.x, candidateId, offset + i);

                if(candidateId == anchorId){
                    keep = false;
                    numRemoved++;
                }

                if(isPairedEnd && candidateId == mateId){
                    if(keep){
                        keep = false;
                        numRemoved++;
                    }
                    mateIsRemoved++;
                    //printf("mate removed. i = %d\n", i);
                }

                keepflags[offset + i] = keep;
            }
            //printf("tid = %d, mateIsRemoved = %d\n", threadIdx.x, mateIsRemoved);
            int numRemovedBlock = BlockReduceInt(intreduce1).Sum(numRemoved);
            int numMateRemovedBlock = BlockReduceInt(intreduce2).Sum(mateIsRemoved);
            if(threadIdx.x == 0){
                numCandidatesPerAnchorOut[a] = size - numRemovedBlock;
                //printf("numMateRemovedBlock %d\n", numMateRemovedBlock);
                if(numMateRemovedBlock > 0){
                    mateRemovedFlags[a] = true;
                }else{
                    mateRemovedFlags[a] = false;
                }
            }
        }
    }
    

    template<int blocksize>
    __global__
    void reverseComplement2bitKernel(
        const int* __restrict__ lengths,
        const unsigned int* __restrict__ forward,
        unsigned int* __restrict__ reverse,
        int num,
        int encodedSequencePitchInInts
    ){

        for(int s = threadIdx.x + blockIdx.x * blockDim.x; s < num; s += blockDim.x * gridDim.x){
            const unsigned int* input = forward + encodedSequencePitchInInts * s;
            unsigned int* output = reverse + encodedSequencePitchInInts * s;
            const int length = lengths[s];

            SequenceHelpers::reverseComplementSequence2Bit(
                output,
                input,
                length,
                [](auto i){return i;},
                [](auto i){return i;}
            );
        }

        // constexpr int smemsizeints = blocksize * 16;
        // __shared__ unsigned int sharedsequences[smemsizeints]; //sequences will be stored transposed

        // const int sequencesPerSmem = std::min(blocksize, smemsizeints / encodedSequencePitchInInts);
        // assert(sequencesPerSmem > 0);

        // const int smemiterations = SDIV(num, sequencesPerSmem);

        // for(int smemiteration = blockIdx.x; smemiteration < smemiterations; smemiteration += gridDim.x){

        //     const int idBegin = smemiteration * sequencesPerSmem;
        //     const int idEnd = std::min((smemiteration+1) * sequencesPerSmem, num);

        //     __syncthreads();

        //     for(int s = idBegin + threadIdx.x; s < idEnd; s += blockDim.x){
        //         for(int intindex = 0; intindex < encodedSequencePitchInInts; intindex++){ //load intindex-th element of sequence s
        //             sharedsequences[intindex * sequencesPerSmem + s] = forward[encodedSequencePitchInInts * s + intindex];
        //         }
        //     }

        //     __syncthreads();

        //     for(int s = idBegin + threadIdx.x; s < idEnd; s += blockDim.x){
        //         SequenceHelpers::reverseComplementSequenceInplace2Bit(&sharedsequences[s], lengths[s], [&](auto i){return i * sequencesPerSmem;});
        //     }

        //     __syncthreads();

        //     for(int s = idBegin + threadIdx.x; s < idEnd; s += blockDim.x){
        //         for(int intindex = 0; intindex < encodedSequencePitchInInts; intindex++){ //load intindex-th element of sequence s
        //             reverse[encodedSequencePitchInInts * s + intindex] = sharedsequences[intindex * sequencesPerSmem + s];
        //         }
        //     }
        // }
    }

    template<int groupsize>
    __global__
    void encodeSequencesTo2BitKernel(
        unsigned int* __restrict__ encodedSequences,
        const char* __restrict__ decodedSequences,
        const int* __restrict__ sequenceLengths,
        int decodedSequencePitchInBytes,
        int encodedSequencePitchInInts,
        int numSequences
    ){
        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());

        const int numGroups = (blockDim.x * gridDim.x) / group.size();
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / group.size();

        for(int a = groupId; a < numSequences; a += numGroups){
            unsigned int* const out = encodedSequences + a * encodedSequencePitchInInts;
            const char* const in = decodedSequences + a * decodedSequencePitchInBytes;
            const int length = sequenceLengths[a];

            const int nInts = SequenceHelpers::getEncodedNumInts2Bit(length);
            constexpr int basesPerInt = SequenceHelpers::basesPerInt2Bit();

            for(int i = group.thread_rank(); i < nInts; i += group.size()){
                unsigned int data = 0;

                const int loopend = min((i+1) * basesPerInt, length);
                
                for(int nucIndex = i * basesPerInt; nucIndex < loopend; nucIndex++){
                    switch(in[nucIndex]) {
                    case 'A':
                        data = (data << 2) | SequenceHelpers::encodedbaseA();
                        break;
                    case 'C':
                        data = (data << 2) | SequenceHelpers::encodedbaseC();
                        break;
                    case 'G':
                        data = (data << 2) | SequenceHelpers::encodedbaseG();
                        break;
                    case 'T':
                        data = (data << 2) | SequenceHelpers::encodedbaseT();
                        break;
                    default:
                        data = (data << 2) | SequenceHelpers::encodedbaseA();
                        break;
                    }
                }

                if(i == nInts-1){
                    //pack bits of last integer into higher order bits
                    int leftoverbits = 2 * (nInts * basesPerInt - length);
                    if(leftoverbits > 0){
                        data <<= leftoverbits;
                    }
                }

                out[i] = data;
            }
        }
    }

    template<int groupsize>
    __global__
    void decodeSequencesFrom2BitKernel(
        char* __restrict__ decodedSequences,
        const unsigned int* __restrict__ encodedSequences,
        const int* __restrict__ sequenceLengths,
        int decodedSequencePitchInBytes,
        int encodedSequencePitchInInts,
        int numSequences
    ){
        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());

        const int numGroups = (blockDim.x * gridDim.x) / group.size();
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / group.size();

        for(int a = groupId; a < numSequences; a += numGroups){
            char* const out = decodedSequences + a * decodedSequencePitchInBytes;
            const unsigned int* const in = encodedSequences + a * encodedSequencePitchInInts;
            const int length = sequenceLengths[a];

            const int nInts = SequenceHelpers::getEncodedNumInts2Bit(length);
            constexpr int basesPerInt = SequenceHelpers::basesPerInt2Bit();

            for(int i = group.thread_rank(); i < nInts; i += group.size()){
                unsigned int data = in[i];

                if(i < nInts-1){
                    //not last iteration. int encodes 16 chars
                    __align__(16) char nucs[16];

                    #pragma unroll
                    for(int p = 0; p < 16; p++){
                        const std::uint8_t encodedBase = SequenceHelpers::getEncodedNucFromInt2Bit(data, p);
                        nucs[p] = SequenceHelpers::decodeBase(encodedBase);
                    }
                    ((int4*)out)[i] = *((const int4*)&nucs[0]);
                }else{
                    const int remaining = length - i * basesPerInt;

                    for(int p = 0; p < remaining; p++){
                        const std::uint8_t encodedBase = SequenceHelpers::getEncodedNucFromInt2Bit(data, p);
                        out[i * basesPerInt + p] = SequenceHelpers::decodeBase(encodedBase);
                    }
                }
            }
        }
    }


    template<int blocksize, int groupsize>
    __global__
    void filtermatekernel(
        const unsigned int* __restrict__ anchormatedata,
        const unsigned int* __restrict__ candidatefwddata,
        int encodedSequencePitchInInts,
        const int* __restrict__ numCandidatesPerAnchor,
        const int* __restrict__ numCandidatesPerAnchorPrefixSum,
        const bool* __restrict__ mateIdHasBeenRemoved,
        int numTasks,
        bool* __restrict__ output_keepflags,
        int initialNumCandidates,
        const int* currentNumCandidatesPtr
    ){

        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int groupindex = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int numgroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupindexinblock = threadIdx.x / groupsize;

        static_assert(blocksize % groupsize == 0);
        //constexpr int groupsperblock = blocksize / groupsize;

        extern __shared__ unsigned int smem[]; //sizeof(unsigned int) * groupsperblock * encodedSequencePitchInInts

        unsigned int* sharedMate = smem + groupindexinblock * encodedSequencePitchInInts;

        const int currentNumCandidates = *currentNumCandidatesPtr;

        for(int task = groupindex; task < numTasks; task += numgroups){
            const int numCandidates = numCandidatesPerAnchor[task];
            const int candidatesOffset = numCandidatesPerAnchorPrefixSum[task];

            if(mateIdHasBeenRemoved[task]){

                for(int p = group.thread_rank(); p < encodedSequencePitchInInts; p++){
                    sharedMate[p] = anchormatedata[encodedSequencePitchInInts * task + p];
                }
                group.sync();

                //compare mate to candidates. 1 thread per candidate
                for(int c = group.thread_rank(); c < numCandidates; c += group.size()){
                    bool doKeep = false;
                    const unsigned int* const candidateptr = candidatefwddata + encodedSequencePitchInInts * (candidatesOffset + c);

                    for(int p = 0; p < encodedSequencePitchInInts; p++){
                        const unsigned int aaa = sharedMate[p];
                        const unsigned int bbb = candidateptr[p];

                        if(aaa != bbb){
                            doKeep = true;
                            break;
                        }
                    }

                    output_keepflags[(candidatesOffset + c)] = doKeep;
                }

                group.sync();

            }
            // else{
            //     for(int c = group.thread_rank(); c < numCandidates; c += group.size()){
            //         output_keepflags[(candidatesOffset + c)] = true;
            //     }
            // }
        }

        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;
        for(int i = currentNumCandidates + tid; i < initialNumCandidates; i += stride){
            output_keepflags[i] = false;
        }
    }



    template<int groupsize>
    __global__
    void compactUsedIdsOfSelectedTasks(
        const int* __restrict__ indices,
        int numIndices,
        const read_number* __restrict__ d_usedReadIdsIn,
        read_number* __restrict__ d_usedReadIdsOut,
        //int* __restrict__ segmentIdsOut,
        const int* __restrict__ d_numUsedReadIdsPerAnchor,
        const int* __restrict__ inputSegmentOffsets,
        const int* __restrict__ outputSegmentOffsets
    ){
        const int warpid = (threadIdx.x + blockDim.x * blockIdx.x) / groupsize;
        const int numwarps = (blockDim.x * gridDim.x) / groupsize;
        const int lane = threadIdx.x % groupsize;

        for(int t = warpid; t < numIndices; t += numwarps){
            const int activeIndex = indices[t];
            const int num = d_numUsedReadIdsPerAnchor[t];
            const int inputOffset = inputSegmentOffsets[activeIndex];
            const int outputOffset = outputSegmentOffsets[t];

            for(int i = lane; i < num; i += groupsize){
                //copy read id
                d_usedReadIdsOut[outputOffset + i] = d_usedReadIdsIn[inputOffset + i];
                //set new segment id
                //segmentIdsOut[outputOffset + i] = t;
            }
        }
    }

    template<int blocksize, int groupsize>
    __global__
    void createGpuTaskData(
        int numReadPairs,
        const read_number* __restrict__ d_readpair_readIds,
        const int* __restrict__ d_readpair_readLengths,
        const unsigned int* __restrict__ d_readpair_sequences,
        const char* __restrict__ d_readpair_qualities,
        unsigned int* __restrict__ d_subjectSequencesData,
        int* __restrict__ d_anchorSequencesLength,
        char* __restrict__ d_anchorQualityScores,
        unsigned int* __restrict__ d_inputanchormatedata,
        int* __restrict__ d_inputMateLengths,
        bool* __restrict__ d_isPairedTask,
        read_number* __restrict__ d_anchorReadIds,
        read_number* __restrict__ d_mateReadIds,
        int* __restrict__ d_accumExtensionsLengths,
        int decodedSequencePitchInBytes,
        int qualityPitchInBytes,
        int encodedSequencePitchInInts
    ){
        constexpr int numGroupsInBlock = blocksize / groupsize;

        __shared__ unsigned int sharedEncodedSequence[numGroupsInBlock][32];

        assert(encodedSequencePitchInInts <= 32);

        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int groupIdInBlock = groupId % numGroupsInBlock;
        const int numGroups = (blockDim.x * gridDim.x) / groupsize;
        
        const int numTasks = numReadPairs * 4;

        //handle scalars
        #if 1
        for(int t = threadIdx.x + blockIdx.x * blockDim.x; t < numTasks; t += blockDim.x * gridDim.x){
            d_accumExtensionsLengths[t] = 0;

            const int pairId = t / 4;
            const int id = t % 4;

            if(id == 0){
                d_anchorReadIds[t] = d_readpair_readIds[2 * pairId + 0];
                d_mateReadIds[t] = d_readpair_readIds[2 * pairId + 1];
                d_anchorSequencesLength[t] = d_readpair_readLengths[2 * pairId + 0];
                d_inputMateLengths[t] = d_readpair_readLengths[2 * pairId + 1];
                d_isPairedTask[t] = true;
            }else if(id == 1){
                d_anchorReadIds[t] = d_readpair_readIds[2 * pairId + 1];
                d_mateReadIds[t] = std::numeric_limits<read_number>::max();
                d_anchorSequencesLength[t] = d_readpair_readLengths[2 * pairId + 1];
                d_inputMateLengths[t] = 0;
                d_isPairedTask[t] = false;
            }else if(id == 2){
                d_anchorReadIds[t] = d_readpair_readIds[2 * pairId + 1];
                d_mateReadIds[t] = d_readpair_readIds[2 * pairId + 0];
                d_anchorSequencesLength[t] = d_readpair_readLengths[2 * pairId + 1];
                d_inputMateLengths[t] = d_readpair_readLengths[2 * pairId + 0];
                d_isPairedTask[t] = true;
            }else{
                //id == 3
                d_anchorReadIds[t] = d_readpair_readIds[2 * pairId + 0];
                d_mateReadIds[t] = std::numeric_limits<read_number>::max();
                d_anchorSequencesLength[t] = d_readpair_readLengths[2 * pairId + 0];
                d_inputMateLengths[t] = 0;
                d_isPairedTask[t] = false;
            }
        }

        #endif

        #if 1
        //handle sequences
        for(int t = groupId; t < numTasks; t += numGroups){
            const int pairId = t / 4;
            const int id = t % 4;

            const unsigned int* const myReadpairSequences = d_readpair_sequences + 2 * pairId * encodedSequencePitchInInts;
            unsigned int* const myAnchorSequence = d_subjectSequencesData + t * encodedSequencePitchInInts;
            unsigned int* const myMateSequence = d_inputanchormatedata + t * encodedSequencePitchInInts;

            if(id == 0){
                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    myAnchorSequence[k] = myReadpairSequences[k];
                    myMateSequence[k] = myReadpairSequences[encodedSequencePitchInInts + k];
                }
            }else if(id == 1){
                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    sharedEncodedSequence[groupIdInBlock][k] = myReadpairSequences[encodedSequencePitchInInts + k];
                }
                group.sync();
                if(group.thread_rank() == 0){
                    SequenceHelpers::reverseComplementSequence2Bit(
                        myAnchorSequence,
                        &sharedEncodedSequence[groupIdInBlock][0],
                        d_readpair_readLengths[2 * pairId + 1],
                        [](auto i){return i;},
                        [](auto i){return i;}
                    );
                }
                group.sync();
            }else if(id == 2){
                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    myAnchorSequence[k] = myReadpairSequences[encodedSequencePitchInInts + k];
                    myMateSequence[k] = myReadpairSequences[k];
                }
            }else{
                //id == 3
                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    sharedEncodedSequence[groupIdInBlock][k] = myReadpairSequences[k];
                }
                group.sync();
                if(group.thread_rank() == 0){
                    SequenceHelpers::reverseComplementSequence2Bit(
                        myAnchorSequence,
                        &sharedEncodedSequence[groupIdInBlock][0],
                        d_readpair_readLengths[2 * pairId + 0],
                        [](auto i){return i;},
                        [](auto i){return i;}
                    );
                }
                group.sync();
            }
        }
        #endif

        #if 1

        //handle qualities
        for(int t = blockIdx.x; t < numTasks; t += gridDim.x){
            const int pairId = t / 4;
            const int id = t % 4;

            const int* const myReadPairLengths = d_readpair_readLengths + 2 * pairId;
            const char* const myReadpairQualities = d_readpair_qualities + 2 * pairId * qualityPitchInBytes;
            char* const myAnchorQualities = d_anchorQualityScores + t * qualityPitchInBytes;

            //const int numInts = qualityPitchInBytes / sizeof(int);
            int l0 = myReadPairLengths[0];
            int l1 = myReadPairLengths[1];

            if(id == 0){
                for(int k = threadIdx.x; k < l0; k += blockDim.x){
                    myAnchorQualities[k] = myReadpairQualities[k];
                }
            }else if(id == 1){
                for(int k = threadIdx.x; k < l1; k += blockDim.x){
                    myAnchorQualities[k] = myReadpairQualities[qualityPitchInBytes + l1 - 1 - k];
                }
            }else if(id == 2){
                for(int k = threadIdx.x; k < l1; k += blockDim.x){
                    myAnchorQualities[k] = myReadpairQualities[qualityPitchInBytes + k];
                }
            }else{
                //id == 3
                for(int k = threadIdx.x; k < l0; k += blockDim.x){
                    myAnchorQualities[k] = myReadpairQualities[l0 - 1 - k];
                }
            }
        }

        #endif
    }







    template<int blocksize, int groupsize>
    __global__
    void createGpuTaskData(
        int numReadPairs,
        const read_number* __restrict__ d_readpair_readIds,
        const int* __restrict__ d_readpair_readLengths,
        const unsigned int* __restrict__ d_readpair_sequences,
        const char* __restrict__ d_readpair_qualities,

        bool* __restrict__ pairedEnd, // assgdf
        bool* __restrict__ mateHasBeenFound,// assgdf
        int* __restrict__ ids,// assgdf
        int* __restrict__ pairIds,// assgdf
        int* __restrict__ iteration,// assgdf
        float* __restrict__ goodscore,// assgdf
        read_number* __restrict__ d_anchorReadIds,// assgdf
        read_number* __restrict__ d_mateReadIds,// assgdf
        extension::AbortReason* __restrict__ abortReason,// assgdf
        extension::ExtensionDirection* __restrict__ direction,// assgdf
        unsigned int* __restrict__ inputEncodedMate,// assgdf
        char* __restrict__ inputdecodedMateRevC,// assgdf
        char* __restrict__ inputmateQualityScoresReversed,// assgdf
        int* __restrict__ inputmateLengths,// assgdf
        unsigned int* __restrict__ inputAnchorsEncoded,// assgdf
        char* __restrict__ inputAnchorsDecoded,// assgdf
        char* __restrict__ inputAnchorQualities,// assgdf
        int* __restrict__ inputAnchorLengths,// assgdf
        int* __restrict__ soaNumIterationResultsPerTask,
        int* __restrict__ soaNumIterationResultsPerTaskPrefixSum,
        int decodedSequencePitchInBytes,
        int qualityPitchInBytes,
        int encodedSequencePitchInInts
    ){
        constexpr int numGroupsInBlock = blocksize / groupsize;

        __shared__ unsigned int sharedEncodedSequence[numGroupsInBlock][32];
        __shared__ unsigned int sharedEncodedSequence2[numGroupsInBlock][32];

        assert(encodedSequencePitchInInts <= 32);

        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int groupIdInBlock = groupId % numGroupsInBlock;
        const int numGroups = (blockDim.x * gridDim.x) / groupsize;
        
        const int numTasks = numReadPairs * 4;

        //handle scalars

        for(int t = threadIdx.x + blockIdx.x * blockDim.x; t < numTasks; t += blockDim.x * gridDim.x){

            const int inputPairId = t / 4;
            const int id = t % 4;

            mateHasBeenFound[t] = false;
            ids[t] = id;
            pairIds[t] = d_readpair_readIds[2 * inputPairId + 0] / 2;
            iteration[t] = 0;
            goodscore[t] = 0.0f;
            abortReason[t] = extension::AbortReason::None;
            soaNumIterationResultsPerTask[t] = 0;
            soaNumIterationResultsPerTaskPrefixSum[t] = 0;

            if(id == 0){
                d_anchorReadIds[t] = d_readpair_readIds[2 * inputPairId + 0];
                d_mateReadIds[t] = d_readpair_readIds[2 * inputPairId + 1];
                inputAnchorLengths[t] = d_readpair_readLengths[2 * inputPairId + 0];
                inputmateLengths[t] = d_readpair_readLengths[2 * inputPairId + 1];
                pairedEnd[t] = true;
                direction[t] = extension::ExtensionDirection::LR;
            }else if(id == 1){
                d_anchorReadIds[t] = d_readpair_readIds[2 * inputPairId + 1];
                d_mateReadIds[t] = std::numeric_limits<read_number>::max();
                inputAnchorLengths[t] = d_readpair_readLengths[2 * inputPairId + 1];
                inputmateLengths[t] = 0;
                pairedEnd[t] = false;
                direction[t] = extension::ExtensionDirection::LR;
            }else if(id == 2){
                d_anchorReadIds[t] = d_readpair_readIds[2 * inputPairId + 1];
                d_mateReadIds[t] = d_readpair_readIds[2 * inputPairId + 0];
                inputAnchorLengths[t] = d_readpair_readLengths[2 * inputPairId + 1];
                inputmateLengths[t] = d_readpair_readLengths[2 * inputPairId + 0];
                pairedEnd[t] = true;
                direction[t] = extension::ExtensionDirection::RL;
            }else{
                //id == 3
                d_anchorReadIds[t] = d_readpair_readIds[2 * inputPairId + 0];
                d_mateReadIds[t] = std::numeric_limits<read_number>::max();
                inputAnchorLengths[t] = d_readpair_readLengths[2 * inputPairId + 0];
                inputmateLengths[t] = 0;
                pairedEnd[t] = false;
                direction[t] = extension::ExtensionDirection::RL;
            }
        }

        //handle sequences
        for(int t = groupId; t < numTasks; t += numGroups){
            const int inputPairId = t / 4;
            const int id = t % 4;

            const unsigned int* const myReadpairSequences = d_readpair_sequences + 2 * inputPairId * encodedSequencePitchInInts;
            const int* const myReadPairLengths = d_readpair_readLengths + 2 * inputPairId;

            unsigned int* const myAnchorSequence = inputAnchorsEncoded + t * encodedSequencePitchInInts;
            unsigned int* const myMateSequence = inputEncodedMate + t * encodedSequencePitchInInts;
            char* const myDecodedMateRevC = inputdecodedMateRevC + t * decodedSequencePitchInBytes;
            char* const myDecodedAnchor = inputAnchorsDecoded + t * decodedSequencePitchInBytes;

            if(id == 0){
                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    sharedEncodedSequence[groupIdInBlock][k] = myReadpairSequences[k]; //anchor in shared
                    sharedEncodedSequence2[groupIdInBlock][k] = myReadpairSequences[encodedSequencePitchInInts + k]; //mate in shared2
                }
                group.sync();

                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    myAnchorSequence[k] = sharedEncodedSequence[groupIdInBlock][k];
                    myMateSequence[k] = sharedEncodedSequence2[groupIdInBlock][k];
                }

                SequenceHelpers::decodeSequence2Bit(group, &sharedEncodedSequence[groupIdInBlock][0], myReadPairLengths[0], myDecodedAnchor);
                group.sync();

                if(group.thread_rank() == 0){
                    //store reverse complement mate to smem1
                    SequenceHelpers::reverseComplementSequence2Bit(
                        &sharedEncodedSequence[groupIdInBlock][0],
                        &sharedEncodedSequence2[groupIdInBlock][0],
                        myReadPairLengths[1],
                        [](auto i){return i;},
                        [](auto i){return i;}
                    );
                }
                group.sync();
                SequenceHelpers::decodeSequence2Bit(group, &sharedEncodedSequence[groupIdInBlock][0], myReadPairLengths[1], myDecodedMateRevC);
                
            }else if(id == 1){
                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    sharedEncodedSequence[groupIdInBlock][k] = myReadpairSequences[encodedSequencePitchInInts + k];
                }
                group.sync();
                if(group.thread_rank() == 0){
                    SequenceHelpers::reverseComplementSequence2Bit(
                        &sharedEncodedSequence2[groupIdInBlock][0],
                        &sharedEncodedSequence[groupIdInBlock][0],
                        myReadPairLengths[1],
                        [](auto i){return i;},
                        [](auto i){return i;}
                    );
                }
                group.sync();

                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    myAnchorSequence[k] = sharedEncodedSequence2[groupIdInBlock][k];
                }
                SequenceHelpers::decodeSequence2Bit(group, &sharedEncodedSequence2[groupIdInBlock][0], myReadPairLengths[1], myDecodedAnchor);
            }else if(id == 2){
                // for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                //     myAnchorSequence[k] = myReadpairSequences[encodedSequencePitchInInts + k];
                //     myMateSequence[k] = myReadpairSequences[k];
                // }

                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    sharedEncodedSequence[groupIdInBlock][k] = myReadpairSequences[encodedSequencePitchInInts + k]; //anchor in shared
                    sharedEncodedSequence2[groupIdInBlock][k] = myReadpairSequences[k]; //mate in shared2
                }
                group.sync();

                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    myAnchorSequence[k] = sharedEncodedSequence[groupIdInBlock][k];
                    myMateSequence[k] = sharedEncodedSequence2[groupIdInBlock][k];
                }

                SequenceHelpers::decodeSequence2Bit(group, &sharedEncodedSequence[groupIdInBlock][0], myReadPairLengths[0], myDecodedAnchor);
                group.sync();

                if(group.thread_rank() == 0){
                    //store reverse complement mate to smem1
                    SequenceHelpers::reverseComplementSequence2Bit(
                        &sharedEncodedSequence[groupIdInBlock][0],
                        &sharedEncodedSequence2[groupIdInBlock][0],
                        myReadPairLengths[0],
                        [](auto i){return i;},
                        [](auto i){return i;}
                    );
                }
                group.sync();
                SequenceHelpers::decodeSequence2Bit(group, &sharedEncodedSequence[groupIdInBlock][0], myReadPairLengths[0], myDecodedMateRevC);
            }else{
                //id == 3
                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    sharedEncodedSequence[groupIdInBlock][k] = myReadpairSequences[k];
                }
                group.sync();
                if(group.thread_rank() == 0){
                    SequenceHelpers::reverseComplementSequence2Bit(
                        &sharedEncodedSequence2[groupIdInBlock][0],
                        &sharedEncodedSequence[groupIdInBlock][0],
                        myReadPairLengths[0],
                        [](auto i){return i;},
                        [](auto i){return i;}
                    );
                }
                group.sync();
                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    myAnchorSequence[k] = sharedEncodedSequence2[groupIdInBlock][k];
                }
                SequenceHelpers::decodeSequence2Bit(group, &sharedEncodedSequence2[groupIdInBlock][0], myReadPairLengths[0], myDecodedAnchor);
            }
        }

        //handle qualities
        for(int t = blockIdx.x; t < numTasks; t += gridDim.x){
            const int inputPairId = t / 4;
            const int id = t % 4;

            const int* const myReadPairLengths = d_readpair_readLengths + 2 * inputPairId;
            const char* const myReadpairQualities = d_readpair_qualities + 2 * inputPairId * qualityPitchInBytes;
            char* const myAnchorQualities = inputAnchorQualities + t * qualityPitchInBytes;
            char* const myMateQualityScoresReversed = inputmateQualityScoresReversed + t * qualityPitchInBytes;

            //const int numInts = qualityPitchInBytes / sizeof(int);
            int l0 = myReadPairLengths[0];
            int l1 = myReadPairLengths[1];

            if(id == 0){
                for(int k = threadIdx.x; k < l0; k += blockDim.x){
                    myAnchorQualities[k] = myReadpairQualities[k];
                }
                for(int k = threadIdx.x; k < l1; k += blockDim.x){
                    myMateQualityScoresReversed[k] = myReadpairQualities[qualityPitchInBytes + l1 - 1 - k];
                }
            }else if(id == 1){
                for(int k = threadIdx.x; k < l1; k += blockDim.x){
                    myAnchorQualities[k] = myReadpairQualities[qualityPitchInBytes + l1 - 1 - k];
                }
            }else if(id == 2){
                for(int k = threadIdx.x; k < l1; k += blockDim.x){
                    myAnchorQualities[k] = myReadpairQualities[qualityPitchInBytes + k];
                }
                for(int k = threadIdx.x; k < l0; k += blockDim.x){
                    myMateQualityScoresReversed[k] = myReadpairQualities[l0 - 1 - k];
                }
            }else{
                //id == 3
                for(int k = threadIdx.x; k < l0; k += blockDim.x){
                    myAnchorQualities[k] = myReadpairQualities[l0 - 1 - k];
                }
            }
        }

    }


    struct ComputeTaskSplitGatherIndicesSmallInput{

        std::size_t staticSharedMemory = 0;
        std::size_t maxDynamicSharedMemory = 0;

        ComputeTaskSplitGatherIndicesSmallInput(){
            std::size_t* d_output;
            cudaMalloc(&d_output, sizeof(std::size_t));

            readextendergpukernels::computeTaskSplitGatherIndicesSmallInputGetStaticSmemSizeKernel<256,16><<<1, 1, 0, cudaStreamPerThread>>>(d_output); CUERR;

            cudaMemcpyAsync(&staticSharedMemory, d_output, sizeof(std::size_t), D2H, cudaStreamPerThread); CUERR;
            cudaStreamSynchronize(cudaStreamPerThread); CUERR;

            cudaFree(d_output); CUERR;

            int device = 0;
            cudaGetDevice(&device); CUERR;

            int smemoptin = 0;
            cudaDeviceGetAttribute(&smemoptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device); CUERR;
            
            cudaFuncSetAttribute(
                readextendergpukernels::computeTaskSplitGatherIndicesSmallInputKernel<256,16>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 
                smemoptin - staticSharedMemory
            ); CUERR;

            maxDynamicSharedMemory = smemoptin - staticSharedMemory;
        }

        std::size_t getRequiredDynamicSharedMemory(int numTasks) const{
            return sizeof(int) * numTasks * 2;
        }

        bool computationPossible(int numTasks) const{
            return numTasks <= (256 * 16) && getRequiredDynamicSharedMemory(numTasks) <= maxDynamicSharedMemory;
        }

        void compute(
            int numTasks,
            int* d_positions4,
            int* d_positionsNot4,
            int* d_numPositions4_out,
            int* d_numPositionsNot4_out,
            const int* task_pairIds,
            const int* task_ids,
            const int* d_minmax_pairId,
            cudaStream_t stream
        ) const{

            std::size_t smem = getRequiredDynamicSharedMemory(numTasks);
            readextendergpukernels::computeTaskSplitGatherIndicesSmallInputKernel<256,16><<<1, 256, smem, stream>>>(
                numTasks,
                d_positions4,
                d_positionsNot4,
                d_numPositions4_out,
                d_numPositionsNot4_out,
                task_pairIds,
                task_ids,
                d_minmax_pairId
            ); CUERR;
        }
    };

}

    
}
}


#endif