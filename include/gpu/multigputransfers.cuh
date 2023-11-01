#ifndef MULTIGPU_TRANSFERS_CUH        
#define MULTIGPU_TRANSFERS_CUH

#include <gpu/cudaerrorcheck.cuh>
#include <vector>
        
    //send srcBuffers[s][d] on device srcDeviceIds[s]
    //to dstBuffers[s][d] on device dstDeviceIds[d] using stream streams[s]
    __inline__
    void multigpu_transfer(
        const std::vector<int>& srcDeviceIds,
        const std::vector<std::vector<const void*>>& srcBuffers,
        const std::vector<std::vector<size_t>>& transferSizesBytes,
        const std::vector<cudaStream_t>& streams,
        const std::vector<int>& dstDeviceIds,
        const std::vector<std::vector<void*>>& dstBuffers
    ){
        const int numSrcGpus = srcDeviceIds.size();
        const int numDstGpus = dstDeviceIds.size();
        if(srcDeviceIds == dstDeviceIds){
            //all-to-all
            const int numGpus = numSrcGpus;

            for(int distance = 0; distance < numGpus; distance++){
                for(int g = 0; g < numGpus; g++){
                    CUDACHECK(cudaSetDevice(srcDeviceIds[g]));
        
                    const int d = (g + distance) % numGpus;
                    //send from g to d

                    CUDACHECK(cudaMemcpyPeerAsync(
                        dstBuffers[g][d],
                        dstDeviceIds[d],
                        srcBuffers[g][d],
                        srcDeviceIds[g],
                        transferSizesBytes[g][d],
                        streams[g]
                    ));
                }
            }
        }else{
            for(int d = 0; d < numDstGpus; d++){
                for(int g = 0; g < numSrcGpus; g++){
                    CUDACHECK(cudaSetDevice(srcDeviceIds[g]));

                    CUDACHECK(cudaMemcpyPeerAsync(
                        dstBuffers[g][d],
                        dstDeviceIds[d],
                        srcBuffers[g][d],
                        srcDeviceIds[g],
                        transferSizesBytes[g][d],
                        streams[g]
                    ));
                }
            }
        }
    }

#endif