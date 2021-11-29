#ifndef CARE_CUDA_BLOCK_UNIQUE_CUH
#define CARE_CUDA_BLOCK_UNIQUE_CUH

#include <cub/cub.cuh>

namespace care{
namespace gpu{


    template<class T, int blocksize, int elemsPerThread>
    struct BlockUnique{
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
            union{
                typename BlockDiscontinuity::TempStorage discontinuity;
                typename BlockScan::TempStorage scan;
                typename BlockScanPair::TempStorage scanpair;
            } cubtemp;
        };

        TempStorage& temp_storage;

        __device__
        BlockUnique(TempStorage& temp) : temp_storage(temp){}

        /*
            For each segment of equal elements in input, if the segment size is >= minimumSegmentSize,
            copy one of its element to the output. output order is stable.
            input is in blocked arrangement. 
            output and numUnique must point to shared memory or global memory
            numUnique is written by thread threadIdx.x == 0
        */
        __device__
        void execute(T (&input)[elemsPerThread], int validInputSize, int minimumSegmentSize, T* output, int* numUnique){
            execute_impl<true>(input, validInputSize, minimumSegmentSize, output, numUnique);
        }

        __device__
        void execute(T (&input)[elemsPerThread], int validInputSize, T* output, int* numUnique){
            execute_impl<false>(input, validInputSize, 1, output, numUnique);
        }

    private:
        template<bool checksegmentsize>
        __device__
        void execute_impl(T (&input)[elemsPerThread], int validInputSize, int minimumSegmentSize, T* output, int* numUnique){
            int prefixsum[elemsPerThread];
            int tail_flags[elemsPerThread];

            BlockDiscontinuity(temp_storage.cubtemp.discontinuity).FlagTails(
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

            int numSelected = 0;
            BlockScan(temp_storage.cubtemp.scan).ExclusiveSum(tail_flags, prefixsum, numSelected);
            __syncthreads();

            // if(threadIdx.x == 0){
            //     printf("validInputSize %d, numSelected %d\n", validInputSize, numSelected);
            // }

            if(checksegmentsize){

                SegmentIdCountPair zipped[elemsPerThread];

                #pragma unroll
                for(int i = 0; i < elemsPerThread; i++){
                    zipped[i].count = 1;
                    zipped[i].segmentId = prefixsum[i];
                }

                SegmentIdCountPair zippedprefixsum[elemsPerThread];
                BlockScanPair(temp_storage.cubtemp.scanpair).InclusiveScan(
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

                BlockScan(temp_storage.cubtemp.scan).ExclusiveSum(tail_flags, prefixsum, numSelected);
                __syncthreads();

            }

            #pragma unroll
            for(int i = 0; i < elemsPerThread; i++){
                if(threadIdx.x * elemsPerThread + i < validInputSize){
                    if(tail_flags[i]){
                        // if(input[i] == std::numeric_limits<T>::max()){
                        //     printf("maxint observed. threadIdx.x %d, i %d, elemsPerThread %d, validInputSize %d\n", threadIdx.x, i, elemsPerThread, validInputSize);
                        //     // printf("my tail flags:\n");
                        //     // for(int k = 0; k < elemsPerThread)
                        //     assert(input[i] != std::numeric_limits<T>::max());
                        // }
                        output[prefixsum[i]] = input[i];
                    }
                }
            }
            if(threadIdx.x == 0){
                *numUnique = numSelected;
            }
            __syncthreads();
        }

    };

} //namespace gpu
} //namespace care




#endif