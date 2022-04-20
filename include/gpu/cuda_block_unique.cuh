#ifndef CARE_CUDA_BLOCK_UNIQUE_CUH
#define CARE_CUDA_BLOCK_UNIQUE_CUH

#include <cub/cub.cuh>

namespace care{
namespace gpu{


    template<class T, int blocksize, int elemsPerThread>
    struct BlockUnique{
    private:

        using BlockDiscontinuity = cub::BlockDiscontinuity<T, blocksize>;
        using BlockScan = cub::BlockScan<int, blocksize>; 

    public:
        struct TempStorage{
            union{
                union{
                    typename BlockDiscontinuity::TempStorage discontinuity;
                    typename BlockScan::TempStorage scan;
                } cubtemp;

            } data;
        };


        __device__
        BlockUnique(TempStorage& temp) : temp_storage(temp){}

        //return size of output
        __device__
        int execute(T (&input)[elemsPerThread], int validInputSize, T* output){
            constexpr bool isFirstChunk = true;
            return execute_impl<isFirstChunk>(input, validInputSize, T{}, output);
        }

        //return size of output
        __device__
        int execute(T (&input)[elemsPerThread], int validInputSize, T lastElementOfPreviousChunk, T* output){
            constexpr bool isFirstChunk = false;
            return execute_impl<isFirstChunk>(input, validInputSize, lastElementOfPreviousChunk, output);
        }

    private:

        TempStorage& temp_storage;

        template<bool isFirstChunk>
        __device__
        int execute_impl(T (&input)[elemsPerThread], int validInputSize, T lastElementOfPreviousChunk, T* output){
            int prefixsum[elemsPerThread];
            int head_flags[elemsPerThread];

            if(isFirstChunk){
                BlockDiscontinuity(temp_storage.data.cubtemp.discontinuity).FlagHeads(
                    head_flags, 
                    input, 
                    cub::Inequality()
                );
                __syncthreads();
            }else{
                BlockDiscontinuity(temp_storage.data.cubtemp.discontinuity).FlagHeads(
                    head_flags, 
                    input, 
                    cub::Inequality(),
                    lastElementOfPreviousChunk
                );
                __syncthreads();
            }

            #pragma unroll
            for(int i = 0; i < elemsPerThread; i++){
                if(threadIdx.x * elemsPerThread + i >= validInputSize){
                    head_flags[i] = 0;
                }
            }

            int numSelected = 0;
            BlockScan(temp_storage.data.cubtemp.scan).ExclusiveSum(head_flags, prefixsum, numSelected);
            __syncthreads();

            #pragma unroll
            for(int i = 0; i < elemsPerThread; i++){
                if(threadIdx.x * elemsPerThread + i < validInputSize){
                    if(head_flags[i]){
                        output[prefixsum[i]] = input[i];
                    }
                }
            }
            return numSelected;
        }

    };

#if 0
    template<class T, int blocksize, int elemsPerThread>
    struct BlockUniqueWithLimit{
    private:
        struct SegmentIdCountPair{
            int segmentId;
            int count;
        };

        struct PairScanOp{
            __device__
            SegmentIdCountPair operator()(const SegmentIdCountPair& left, const SegmentIdCountPair& right) const{
                SegmentIdCountPair result;
                result.segmentId = right.segmentId;
                if(left.segmentId == right.segmentId){
                    result.count = right.count + left.count;
                }else{
                    result.count = right.count;                
                }
                return result;
            }
        };

        using BlockDiscontinuity = cub::BlockDiscontinuity<T, blocksize>;
        using BlockScan = cub::BlockScan<int, blocksize>;
        using BlockScanPair = cub::BlockScan<SegmentIdCountPair, blocksize>;   

    public:
        struct TempStorage{
            struct{
                union{
                    typename BlockDiscontinuity::TempStorage discontinuity;
                    typename BlockScan::TempStorage scan;
                    typename BlockScanPair::TempStorage scanpair;
                } cubtemp;

                int countOfLastSegment;
                int numSelected;
            } data;
        };

        TempStorage& temp_storage;

        __device__
        BlockUniqueWithLimit(TempStorage& temp) : temp_storage(temp){}

        /*
            For each segment of equal elements in input, if the segment size is >= minimumSegmentSize,
            copy one of its element to the output. output order is stable.
            input is in blocked arrangement. 
            output and numUnique must point to shared memory or global memory
            numUnique is written by thread threadIdx.x == 0
        */
        __device__
        int executeSingleChunk(T (&input)[elemsPerThread], int validInputSize, int minimumSegmentSize, T* output, int* numUnique){
            constexpr bool isFirstChunk = true;
            constexpr bool isLastChunk = true;
            return execute_impl<isFirstChunk, isLastChunk>(input, validInputSize, minimumSegmentSize, T{}, 0, T{}, output, numUnique);
        }

        __device__
        int executeFirstChunk(T (&input)[elemsPerThread], int validInputSize, int minimumSegmentSize, T firstElementOfNextChunk, T* output, int* numUnique){
            constexpr bool isFirstChunk = true;
            constexpr bool isLastChunk = false;
            return execute_impl<isFirstChunk, isLastChunk>(input, validInputSize, minimumSegmentSize, T{}, 0, firstElementOfNextChunk, output, numUnique);
        }

        __device__
        int executeLastChunk(T (&input)[elemsPerThread], int validInputSize, int minimumSegmentSize, int lastSegmentSizeOfPreviousChunk, T lastElementOfPreviousChunk, T* output, int* numUnique){
            constexpr bool isFirstChunk = false;
            constexpr bool isLastChunk = true;
            return execute_impl<isFirstChunk, isLastChunk>(input, validInputSize, minimumSegmentSize, lastElementOfPreviousChunk, lastSegmentSizeOfPreviousChunk, T{}, output, numUnique);
        }

        __device__
        int executeIntermediateChunk(T (&input)[elemsPerThread], int validInputSize, int minimumSegmentSize, T firstElementOfNextChunk, int lastSegmentSizeOfPreviousChunk, T lastElementOfPreviousChunk, T* output, int* numUnique){
            constexpr bool isFirstChunk = false;
            constexpr bool isLastChunk = false;
            return execute_impl<isFirstChunk, isLastChunk>(input, validInputSize, minimumSegmentSize, lastElementOfPreviousChunk, lastSegmentSizeOfPreviousChunk, lastElementOfPreviousChunk, output, numUnique);
        }



    private:
        template<bool isFirstChunk, bool isLastChunk>
        __device__
        int execute_impl(
            T (&input)[elemsPerThread], 
            int validInputSize, 
            int minimumSegmentSize, 
            T firstElementOfNextChunk, 
            int lastSegmentSizeOfPreviousChunk,
            T lastElementOfPreviousChunk, 
            T* output, 
            int* numUnique
        ){

            int prefixsum[elemsPerThread];
            int tail_flags[elemsPerThread];

            BlockDiscontinuity(temp_storage.data.cubtemp.discontinuity).FlagTails(
                tail_flags, 
                input, 
                cub::Inequality()
            );
            __syncthreads();

            #pragma unroll
            for(int i = 0; i < elemsPerThread; i++){
                if(threadIdx.x * elemsPerThread + i >= validInputSize){
                    tail_flags[i] = 0;
                }
            }
            
            BlockScan(temp_storage.data.cubtemp.scan).ExclusiveSum(tail_flags, prefixsum);
            __syncthreads();

            SegmentIdCountPair zipped[elemsPerThread];

            #pragma unroll
            for(int i = 0; i < elemsPerThread; i++){
                zipped[i].count = 1;
                zipped[i].segmentId = prefixsum[i];
            }

            if(!isFirstChunk){
                if(threadIdx.x == 0){
                    if(input[0] == lastElementOfPreviousChunk){
                        zipped[0].count = 1 + lastSegmentSizeOfPreviousChunk;
                    }
                }
            }

            SegmentIdCountPair zippedprefixsum[elemsPerThread];
            BlockScanPair(temp_storage.data.cubtemp.scanpair).InclusiveScan(
                zipped, 
                zippedprefixsum, 
                PairScanOp{}
            );
            __syncthreads();

            #pragma unroll
            for(int i = 0; i < elemsPerThread; i++){
                if(threadIdx.x * elemsPerThread + i < validInputSize){
                    if(tail_flags[i]){
                        if(zippedprefixsum[i].count < minimumSegmentSize){
                            tail_flags[i] = 0;
                        }
                    }
                }else{
                    tail_flags[i] = 0;
                }
            }

            int numSelected = 0;
            BlockScan(temp_storage.data.cubtemp.scan).ExclusiveSum(tail_flags, prefixsum, numSelected);
            __syncthreads();

            #pragma unroll
            for(int i = 0; i < elemsPerThread; i++){
                if(threadIdx.x * elemsPerThread + i == validInputSize - 1){
                    temp_storage.data.countOfLastSegment = zippedprefixsum[i].count;
                }
            }
            if(threadIdx.x == 0){
                temp_storage.data.numSelected = numSelected;
            }
            __syncthreads();

            if(isLastChunk){
                #pragma unroll
                for(int i = 0; i < elemsPerThread; i++){
                    if(threadIdx.x * elemsPerThread + i < validInputSize){
                        if(tail_flags[i]){
                            output[prefixsum[i]] = input[i];
                        }
                    }
                }
            }else{
                #pragma unroll
                for(int i = 0; i < elemsPerThread; i++){
                    if(threadIdx.x * elemsPerThread + i < validInputSize){
                        if(tail_flags[i]){
                            if(threadIdx.x * elemsPerThread + i == validInputSize - 1){
                                if(input[i] != firstElementOfNextChunk){
                                    output[prefixsum[i]] = input[i];
                                }else{
                                    temp_storage.data.numSelected -= 1;
                                }
                            }else{
                                output[prefixsum[i]] = input[i];
                            }
                        }
                    }
                }
            }
            __syncthreads();

            *numUnique = temp_storage.data.numSelected;           

            return temp_storage.data.countOfLastSegment;
        }

    };

#endif

} //namespace gpu
} //namespace care




#endif