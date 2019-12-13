#include <gpu/kernels.hpp>
#include <gpu/utility_kernels.cuh>
#include <gpu/cubcachingallocator.cuh>
#include <hpc_helpers.cuh>
#include <config.hpp>
#include <sequence.hpp>

#include <cassert>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace care{
namespace gpu{

 
template<int groupsize>
__global__
void convert2BitTo2BitHiloKernelNN(
        const unsigned int* const __restrict__ inputdata,
        size_t inputpitchInInts, // max num ints per input sequence
        unsigned int*  const __restrict__ outputdata,
        size_t outputpitchInInts, // max num ints per output sequence
        const int* const __restrict__ sequenceLengths,
        int numSequences){

    auto inputStartIndex = [&](auto i){return i * inputpitchInInts;};
    auto outputStartIndex = [&](auto i){return i * outputpitchInInts;};
    auto inputTrafo = [&](auto i){return i;};
    auto outputTrafo = [&](auto i){return i;};

    auto extractEvenBits = [](unsigned int x){
        x = x & 0x55555555;
        x = (x | (x >> 1)) & 0x33333333;
        x = (x | (x >> 2)) & 0x0F0F0F0F;
        x = (x | (x >> 4)) & 0x00FF00FF;
        x = (x | (x >> 8)) & 0x0000FFFF;
        return x;
    };

    auto convert = [&](auto group,
                        unsigned int* out,
                        const unsigned int* in,
                        int length,
                        auto inindextrafo,
                        auto outindextrafo){

        const int inInts = getEncodedNumInts2Bit(length);
        const int outInts = getEncodedNumInts2BitHiLo(length);

        unsigned int* const outHi = out;
        unsigned int* const outLo = out + outindextrafo(outInts/2);

        for(int i = group.thread_rank(); i < outInts / 2; i += group.size()){
            const int outIndex = outindextrafo(i);
            const int inindex1 = inindextrafo(i*2);

            const unsigned int data1 = in[inindex1];
            const unsigned int even161 = extractEvenBits(data1);
            const unsigned int odd161 = extractEvenBits(data1 >> 1);

            unsigned int resultHi = odd161;
            unsigned int resultLo = even161;

            if((i < outInts / 2 - 1) || ((length-1) % 32) >= 16){
                const int inindex2 = inindextrafo(i*2 + 1);

                const unsigned int data2 = in[inindex2];
                const unsigned int even162 = extractEvenBits(data2);
                const unsigned int odd162 = extractEvenBits(data2 >> 1);

                resultHi = resultHi | (odd162 << 16);
                resultLo = resultLo | (even162 << 16) ;
            }

            outHi[outIndex] = resultHi;
            outLo[outIndex] = resultLo;
        }
    };

    auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());
    const int numGroups = (blockDim.x * gridDim.x) / groupsize;
    const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;

    for(int index = groupId; index < numSequences; index += numGroups){
        const int sequenceLength = sequenceLengths[index];
        const unsigned int* const in = inputdata + inputStartIndex(index);
        unsigned int* const out = outputdata + outputStartIndex(index);            

        convert(
            group,
            out,
            in,
            sequenceLength,
            inputTrafo,
            outputTrafo
        );
    } 
}

__global__
void convert2BitTo2BitHiloKernelNT(
        const unsigned int* const __restrict__ inputdata,
        size_t inputpitchInInts, // max num ints per input sequence
        unsigned int*  const __restrict__ outputdata,
        size_t outputpitchInInts, // max num ints per output sequence
        const int* const __restrict__ sequenceLengths,
        int numSequences){

    auto inputStartIndex = [&](auto i){return i * inputpitchInInts;};
    auto outputStartIndex = [&](auto i){return i;};
    auto inputTrafo = [&](auto i){return i;};
    auto outputTrafo = [&](auto i){return i * numSequences;};

    auto extractEvenBits = [](unsigned int x){
        x = x & 0x55555555;
        x = (x | (x >> 1)) & 0x33333333;
        x = (x | (x >> 2)) & 0x0F0F0F0F;
        x = (x | (x >> 4)) & 0x00FF00FF;
        x = (x | (x >> 8)) & 0x0000FFFF;
        return x;
    };

    auto convert = [&](auto group,
                        unsigned int* out,
                        const unsigned int* in,
                        int length,
                        auto inindextrafo,
                        auto outindextrafo){

        const int inInts = getEncodedNumInts2Bit(length);
        const int outInts = getEncodedNumInts2BitHiLo(length);

        unsigned int* const outHi = out;
        unsigned int* const outLo = out + outindextrafo(outInts/2);

        for(int i = group.thread_rank(); i < outInts / 2; i += group.size()){
            const int outIndex = outindextrafo(i);
            const int inindex1 = inindextrafo(i*2);

            const unsigned int data1 = in[inindex1];
            const unsigned int even161 = extractEvenBits(data1);
            const unsigned int odd161 = extractEvenBits(data1 >> 1);

            unsigned int resultHi = odd161;
            unsigned int resultLo = even161;

            if((i < outInts / 2 - 1) || ((length-1) % 32) >= 16){
                const int inindex2 = inindextrafo(i*2 + 1);

                const unsigned int data2 = in[inindex2];
                const unsigned int even162 = extractEvenBits(data2);
                const unsigned int odd162 = extractEvenBits(data2 >> 1);

                resultHi = resultHi | (odd162 << 16);
                resultLo = resultLo | (even162 << 16) ;
            }

            outHi[outIndex] = resultHi;
            outLo[outIndex] = resultLo;
        }
    };

    auto group = cg::tiled_partition<1>(cg::this_thread_block());
    const int numGroups = (blockDim.x * gridDim.x) / group.size();
    const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / group.size();

    for(int index = groupId; index < numSequences; index += numGroups){
        const int sequenceLength = sequenceLengths[index];
        const unsigned int* const in = inputdata + inputStartIndex(index);
        unsigned int* const out = outputdata + outputStartIndex(index);            

        convert(
            group,
            out,
            in,
            sequenceLength,
            inputTrafo,
            outputTrafo
        );
    } 
}



__global__
void convert2BitTo2BitHiloKernelTT(
        const unsigned int* const __restrict__ inputdata,
        size_t inputpitchInInts, // max num ints per input sequence
        unsigned int*  const __restrict__ outputdata,
        size_t outputpitchInInts, // max num ints per output sequence
        const int* const __restrict__ sequenceLengths,
        int numSequences){

    auto inputStartIndex = [&](auto i){return i;};
    auto outputStartIndex = [&](auto i){return i;};
    auto inputTrafo = [&](auto i){return i * numSequences;};
    auto outputTrafo = [&](auto i){return i * numSequences;};

    auto extractEvenBits = [](unsigned int x){
        x = x & 0x55555555;
        x = (x | (x >> 1)) & 0x33333333;
        x = (x | (x >> 2)) & 0x0F0F0F0F;
        x = (x | (x >> 4)) & 0x00FF00FF;
        x = (x | (x >> 8)) & 0x0000FFFF;
        return x;
    };

    auto convert = [&](auto group,
                        unsigned int* out,
                        const unsigned int* in,
                        int length,
                        auto inindextrafo,
                        auto outindextrafo){

        const int inInts = getEncodedNumInts2Bit(length);
        const int outInts = getEncodedNumInts2BitHiLo(length);

        unsigned int* const outHi = out;
        unsigned int* const outLo = out + outindextrafo(outInts/2);

        for(int i = group.thread_rank(); i < outInts / 2; i += group.size()){
            const int outIndex = outindextrafo(i);
            const int inindex1 = inindextrafo(i*2);

            const unsigned int data1 = in[inindex1];
            const unsigned int even161 = extractEvenBits(data1);
            const unsigned int odd161 = extractEvenBits(data1 >> 1);

            unsigned int resultHi = odd161;
            unsigned int resultLo = even161;

            if((i < outInts / 2 - 1) || ((length-1) % 32) >= 16){
                const int inindex2 = inindextrafo(i*2 + 1);

                const unsigned int data2 = in[inindex2];
                const unsigned int even162 = extractEvenBits(data2);
                const unsigned int odd162 = extractEvenBits(data2 >> 1);

                resultHi = resultHi | (odd162 << 16);
                resultLo = resultLo | (even162 << 16) ;
            }

            outHi[outIndex] = resultHi;
            outLo[outIndex] = resultLo;
        }
    };

    auto group = cg::tiled_partition<1>(cg::this_thread_block());
    const int numGroups = (blockDim.x * gridDim.x) / group.size();
    const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / group.size();

    for(int index = groupId; index < numSequences; index += numGroups){
        const int sequenceLength = sequenceLengths[index];
        const unsigned int* const in = inputdata + inputStartIndex(index);
        unsigned int* const out = outputdata + outputStartIndex(index);            

        convert(
            group,
            out,
            in,
            sequenceLength,
            inputTrafo,
            outputTrafo
        );
    } 
}





void callConversionKernel2BitTo2BitHiLoNN(
        const unsigned int* d_inputdata,
        size_t inputpitchInInts,
        unsigned int* d_outputdata,
        size_t outputpitchInInts,
        int* d_sequenceLengths,
        int numSequences,
        cudaStream_t stream,
        KernelLaunchHandle& handle){

    
    constexpr int groupsize = 8;        
    constexpr int blocksize = 128;
    constexpr size_t smem = 0;
    
    int max_blocks_per_device = 1;

    KernelLaunchConfig kernelLaunchConfig;
    kernelLaunchConfig.threads_per_block = blocksize;
    kernelLaunchConfig.smem = smem;

    auto iter = handle.kernelPropertiesMap.find(KernelId::Conversion2BitTo2BitHiLoNN);
    if(iter == handle.kernelPropertiesMap.end()) {

        std::map<KernelLaunchConfig, KernelProperties> mymap;

        #define getProp(blocksize) { \
                KernelLaunchConfig kernelLaunchConfig; \
                kernelLaunchConfig.threads_per_block = (blocksize); \
                kernelLaunchConfig.smem = 0; \
                KernelProperties kernelProperties; \
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                    convert2BitTo2BitHiloKernelNN<groupsize>, \
                            kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
                mymap[kernelLaunchConfig] = kernelProperties; \
        }
        //getProp(1);
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

        handle.kernelPropertiesMap[KernelId::Conversion2BitTo2BitHiLoNN] = std::move(mymap);

        #undef getProp
    }else{
        std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
        const KernelProperties& kernelProperties = map[kernelLaunchConfig];
        max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
    }

    dim3 block(blocksize,1,1);
    dim3 grid(std::min(max_blocks_per_device, SDIV(numSequences * groupsize, blocksize)), 1, 1);

    convert2BitTo2BitHiloKernelNN<groupsize><<<grid, block, 0, stream>>>(
        d_inputdata,
        inputpitchInInts,
        d_outputdata,
        outputpitchInInts,
        d_sequenceLengths,
        numSequences); CUERR;
}

void callConversionKernel2BitTo2BitHiLoNT(
        const unsigned int* d_inputdata,
        size_t inputpitchInInts,
        unsigned int* d_outputdata,
        size_t outputpitchInInts,
        int* d_sequenceLengths,
        int numSequences,
        cudaStream_t stream,
        KernelLaunchHandle& handle){

    int max_blocks_per_device = 1;

    constexpr int blocksize = 128;
    constexpr size_t smem = 0;

    KernelLaunchConfig kernelLaunchConfig;
    kernelLaunchConfig.threads_per_block = blocksize;
    kernelLaunchConfig.smem = smem;

    auto iter = handle.kernelPropertiesMap.find(KernelId::Conversion2BitTo2BitHiLoNT);
    if(iter == handle.kernelPropertiesMap.end()) {

        std::map<KernelLaunchConfig, KernelProperties> mymap;

        #define getProp(blocksize) { \
                KernelLaunchConfig kernelLaunchConfig; \
                kernelLaunchConfig.threads_per_block = (blocksize); \
                kernelLaunchConfig.smem = 0; \
                KernelProperties kernelProperties; \
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                    convert2BitTo2BitHiloKernelNT, \
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

        handle.kernelPropertiesMap[KernelId::Conversion2BitTo2BitHiLoNT] = std::move(mymap);

        #undef getProp
    }else{
        std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
        const KernelProperties& kernelProperties = map[kernelLaunchConfig];
        max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
    }

    dim3 block(blocksize,1,1);
    dim3 grid(std::min(max_blocks_per_device, SDIV(numSequences, blocksize)), 1, 1);

    convert2BitTo2BitHiloKernelNT<<<grid, block, 0, stream>>>(
        d_inputdata,
        inputpitchInInts,
        d_outputdata,
        outputpitchInInts,
        d_sequenceLengths,
        numSequences); CUERR;
}

void callConversionKernel2BitTo2BitHiLoTT(
        const unsigned int* d_inputdata,
        size_t inputpitchInInts,
        unsigned int* d_outputdata,
        size_t outputpitchInInts,
        int* d_sequenceLengths,
        int numSequences,
        cudaStream_t stream,
        KernelLaunchHandle& handle){

    int max_blocks_per_device = 1;

    constexpr int blocksize = 128;
    constexpr size_t smem = 0;

    KernelLaunchConfig kernelLaunchConfig;
    kernelLaunchConfig.threads_per_block = blocksize;
    kernelLaunchConfig.smem = smem;

    auto iter = handle.kernelPropertiesMap.find(KernelId::Conversion2BitTo2BitHiLoTT);
    if(iter == handle.kernelPropertiesMap.end()) {

        std::map<KernelLaunchConfig, KernelProperties> mymap;

        #define getProp(blocksize) { \
                KernelLaunchConfig kernelLaunchConfig; \
                kernelLaunchConfig.threads_per_block = (blocksize); \
                kernelLaunchConfig.smem = 0; \
                KernelProperties kernelProperties; \
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                    convert2BitTo2BitHiloKernelTT, \
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

        handle.kernelPropertiesMap[KernelId::Conversion2BitTo2BitHiLoTT] = std::move(mymap);

        #undef getProp
    }else{
        std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
        const KernelProperties& kernelProperties = map[kernelLaunchConfig];
        max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
    }

    dim3 block(blocksize,1,1);
    dim3 grid(std::min(max_blocks_per_device, SDIV(numSequences, blocksize)), 1, 1);

    convert2BitTo2BitHiloKernelTT<<<grid, block, 0, stream>>>(
        d_inputdata,
        inputpitchInInts,
        d_outputdata,
        outputpitchInInts,
        d_sequenceLengths,
        numSequences); CUERR;
}



}
}




