#include <gpu/kernels.hpp>
#include <gpu/cudaerrorcheck.cuh>
#include <hpc_helpers.cuh>
#include <config.hpp>
#include <sequencehelpers.hpp>

#include <cassert>
#include <cooperative_groups.h>


//#define DO_CHECK_CONVERSIONS

namespace cg = cooperative_groups;

namespace care{
namespace gpu{

template<class First2Bit, class First2BitHilo, class Trafo2Bit, class Trafo2BitHilo>
__global__
void checkSequenceConversionKernel(const unsigned int* const __restrict__ normalData,
        size_t normalpitchInInts, // max num ints per input sequence
        const unsigned int*  const __restrict__ hiloData,
        size_t hilopitchInInts, // max num ints per output sequence
        const int* const __restrict__ sequenceLengths,
        int numSequences,
        First2Bit first2Bit,
        First2BitHilo first2BitHilo,
        Trafo2Bit trafo2Bit,
        Trafo2BitHilo trafo2BitHilo){

    auto to_nuc = [](std::uint8_t enc){
        return SequenceHelpers::decodeBase(enc);
    };

    //use one block per sequence
    for(int index = blockIdx.x; index < numSequences; index += gridDim.x){
        const int sequenceLength = sequenceLengths[index];
        const unsigned int* const normalSeq = normalData + first2Bit(index);
        const unsigned int* const hiloSeq = hiloData + first2BitHilo(index);    
        
        for(int p = threadIdx.x; p < sequenceLength; p += blockDim.x){
            std::uint8_t encnormal = SequenceHelpers::getEncodedNuc2Bit(normalSeq, sequenceLength, p, trafo2Bit);
            char basenormal = to_nuc(encnormal);
            std::uint8_t enchilo = SequenceHelpers::getEncodedNuc2BitHiLo(hiloSeq, sequenceLength, p, trafo2BitHilo);
            char basehilo = to_nuc(enchilo);
            if(basenormal != basehilo){
                printf("error seq %d position %d, normal %c hilo %c\n", index, p, basenormal, basehilo);
            }
            assert(basenormal == basehilo);
        }
    }
}   

void callCheckSequenceConversionKernelNN(const unsigned int* normalData,
        size_t normalpitchInInts,
        const unsigned int* hiloData,
        size_t hilopitchInInts,
        const int* sequenceLengths,
        int numSequences,
        cudaStream_t stream){

    auto first2Bit = [=] __device__ (auto i){return i * normalpitchInInts;};
    auto first2BitHilo = [=] __device__ (auto i){return i * hilopitchInInts;};
    auto trafo2Bit = [=] __device__ (auto i){return i;};
    auto trafo2BitHilo = [=] __device__ (auto i){return i;};

    const int blocksize = 128;
    const int gridsize = std::min(numSequences, 65535);

    checkSequenceConversionKernel<<<gridsize,blocksize, 0, stream>>>(
        normalData,
        normalpitchInInts,
        hiloData,
        hilopitchInInts,
        sequenceLengths,
        numSequences,
        first2Bit,
        first2BitHilo,
        trafo2Bit,
        trafo2BitHilo
    ); CUDACHECKASYNC;
}

void callCheckSequenceConversionKernelNT(const unsigned int* normalData,
        size_t normalpitchInInts,
        const unsigned int* hiloData,
        size_t hilopitchInInts,
        const int* sequenceLengths,
        int numSequences,
        cudaStream_t stream){

    auto first2Bit = [=] __device__ (auto i){return i * normalpitchInInts;};
    auto first2BitHilo = [=] __device__ (auto i){return i;};
    auto trafo2Bit = [=] __device__ (auto i){return i;};
    auto trafo2BitHilo = [=] __device__ (auto i){return i * numSequences;};

    const int blocksize = 128;
    const int gridsize = std::min(numSequences, 65535);

    checkSequenceConversionKernel<<<gridsize,blocksize, 0, stream>>>(
        normalData,
        normalpitchInInts,
        hiloData,
        hilopitchInInts,
        sequenceLengths,
        numSequences,
        first2Bit,
        first2BitHilo,
        trafo2Bit,
        trafo2BitHilo
    ); CUDACHECKASYNC;
}

void callCheckSequenceConversionKernelTT(const unsigned int* normalData,
        size_t normalpitchInInts,
        const unsigned int* hiloData,
        size_t hilopitchInInts,
        const int* sequenceLengths,
        int numSequences,
        cudaStream_t stream){

    auto first2Bit = [=] __device__ (auto i){return i;};
    auto first2BitHilo = [=] __device__ (auto i){return i;};
    auto trafo2Bit = [=] __device__ (auto i){return i * numSequences;};
    auto trafo2BitHilo = [=] __device__ (auto i){return i * numSequences;};

    const int blocksize = 128;
    const int gridsize = std::min(numSequences, 65535);

    checkSequenceConversionKernel<<<gridsize, blocksize, 0, stream>>>(
        normalData,
        normalpitchInInts,
        hiloData,
        hilopitchInInts,
        sequenceLengths,
        numSequences,
        first2Bit,
        first2BitHilo,
        trafo2Bit,
        trafo2BitHilo
    ); CUDACHECKASYNC;
}

 
template<int groupsize>
__global__
void convert2BitTo2BitHiloKernelNN(
    const unsigned int* const __restrict__ inputdata,
    size_t inputpitchInInts, // max num ints per input sequence
    unsigned int*  const __restrict__ outputdata,
    size_t outputpitchInInts, // max num ints per output sequence
    const int* const __restrict__ sequenceLengths,
    int numSequences
){

    auto inputStartIndex = [&](auto i){return i * inputpitchInInts;};
    auto outputStartIndex = [&](auto i){return i * outputpitchInInts;};
    auto inputTrafo = [&](auto i){return i;};
    auto outputTrafo = [&](auto i){return i;};

    auto convert = [&](auto group,
                        unsigned int* out,
                        const unsigned int* in,
                        int length,
                        auto inindextrafo,
                        auto outindextrafo){

        const int inInts = SequenceHelpers::getEncodedNumInts2Bit(length);
        const int outInts = SequenceHelpers::getEncodedNumInts2BitHiLo(length);

        unsigned int* const outHi = out;
        unsigned int* const outLo = out + outindextrafo(outInts/2);

        for(int i = group.thread_rank(); i < outInts / 2; i += group.size()){
            const int outIndex = outindextrafo(i);
            const int inindex1 = inindextrafo(i*2);

            const unsigned int data1 = in[inindex1];
            const unsigned int even161 = SequenceHelpers::extractEvenBits(data1);
            const unsigned int odd161 = SequenceHelpers::extractEvenBits(data1 >> 1);

            unsigned int resultHi = odd161 << 16;
            unsigned int resultLo = even161 << 16;

            if((i < outInts / 2 - 1) || ((length-1) % 32) >= 16){
                const int inindex2 = inindextrafo(i*2 + 1);

                const unsigned int data2 = in[inindex2];
                const unsigned int even162 = SequenceHelpers::extractEvenBits(data2);
                const unsigned int odd162 = SequenceHelpers::extractEvenBits(data2 >> 1);

                resultHi = resultHi | odd162;
                resultLo = resultLo | even162;
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
    int numSequences
){

    auto inputStartIndex = [&](auto i){return i * inputpitchInInts;};
    auto outputStartIndex = [&](auto i){return i;};
    auto inputTrafo = [&](auto i){return i;};
    auto outputTrafo = [&](auto i){return i * numSequences;};

    auto convert = [&](auto group,
                        unsigned int* out,
                        const unsigned int* in,
                        int length,
                        auto inindextrafo,
                        auto outindextrafo){

        const int inInts = SequenceHelpers::getEncodedNumInts2Bit(length);
        const int outInts = SequenceHelpers::getEncodedNumInts2BitHiLo(length);

        unsigned int* const outHi = out;
        unsigned int* const outLo = out + outindextrafo(outInts/2);

        for(int i = group.thread_rank(); i < outInts / 2; i += group.size()){
            const int outIndex = outindextrafo(i);
            const int inindex1 = inindextrafo(i*2);

            const unsigned int data1 = in[inindex1];
            const unsigned int even161 = SequenceHelpers::extractEvenBits(data1);
            const unsigned int odd161 = SequenceHelpers::extractEvenBits(data1 >> 1);

            unsigned int resultHi = odd161 << 16;
            unsigned int resultLo = even161 << 16;

            if((i < outInts / 2 - 1) || ((length-1) % 32) >= 16){
                const int inindex2 = inindextrafo(i*2 + 1);

                const unsigned int data2 = in[inindex2];
                const unsigned int even162 = SequenceHelpers::extractEvenBits(data2);
                const unsigned int odd162 = SequenceHelpers::extractEvenBits(data2 >> 1);

                resultHi = resultHi | odd162;
                resultLo = resultLo | even162;
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
    int numSequences
){

    auto inputStartIndex = [&](auto i){return i;};
    auto outputStartIndex = [&](auto i){return i;};
    auto inputTrafo = [&](auto i){return i * numSequences;};
    auto outputTrafo = [&](auto i){return i * numSequences;};

    auto convert = [&](auto group,
                        unsigned int* out,
                        const unsigned int* in,
                        int length,
                        auto inindextrafo,
                        auto outindextrafo){

        const int inInts = SequenceHelpers::getEncodedNumInts2Bit(length);
        const int outInts = SequenceHelpers::getEncodedNumInts2BitHiLo(length);

        unsigned int* const outHi = out;
        unsigned int* const outLo = out + outindextrafo(outInts/2);

        for(int i = group.thread_rank(); i < outInts / 2; i += group.size()){
            const int outIndex = outindextrafo(i);
            const int inindex1 = inindextrafo(i*2);

            const unsigned int data1 = in[inindex1];
            const unsigned int even161 = SequenceHelpers::extractEvenBits(data1);
            const unsigned int odd161 = SequenceHelpers::extractEvenBits(data1 >> 1);

            unsigned int resultHi = odd161 << 16;
            unsigned int resultLo = even161 << 16;

            if((i < outInts / 2 - 1) || ((length-1) % 32) >= 16){
                const int inindex2 = inindextrafo(i*2 + 1);

                const unsigned int data2 = in[inindex2];
                const unsigned int even162 = SequenceHelpers::extractEvenBits(data2);
                const unsigned int odd162 = SequenceHelpers::extractEvenBits(data2 >> 1);

                resultHi = resultHi | odd162;
                resultLo = resultLo | even162;
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
    const int* d_sequenceLengths,
    int numSequences,
    cudaStream_t stream
){

    
    constexpr int groupsize = 8;        
    constexpr int blocksize = 128;
    constexpr size_t smem = 0;
    
    int deviceId = 0;
    int numSMs = 0;
    int maxBlocksPerSM = 0;
    CUDACHECK(cudaGetDevice(&deviceId));
    CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        convert2BitTo2BitHiloKernelNN<groupsize>,
        blocksize, 
        smem
    ));

    const int maxBlocks = maxBlocksPerSM * numSMs;

    dim3 block(blocksize,1,1);
    dim3 grid(std::min(maxBlocks, SDIV(numSequences * groupsize, blocksize)), 1, 1);

    convert2BitTo2BitHiloKernelNN<groupsize><<<grid, block, 0, stream>>>(
        d_inputdata,
        inputpitchInInts,
        d_outputdata,
        outputpitchInInts,
        d_sequenceLengths,
        numSequences
    ); CUDACHECKASYNC;

#ifdef DO_CHECK_CONVERSIONS        

    callCheckSequenceConversionKernelNN(d_inputdata,
        inputpitchInInts,
        d_outputdata,
        outputpitchInInts,
        d_sequenceLengths,
        numSequences,
        stream);

#endif

}

void callConversionKernel2BitTo2BitHiLoNT(
    const unsigned int* d_inputdata,
    size_t inputpitchInInts,
    unsigned int* d_outputdata,
    size_t outputpitchInInts,
    const int* d_sequenceLengths,
    int numSequences,
    cudaStream_t stream
){

    constexpr int blocksize = 128;
    constexpr size_t smem = 0;

    int deviceId = 0;
    int numSMs = 0;
    int maxBlocksPerSM = 0;
    CUDACHECK(cudaGetDevice(&deviceId));
    CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        convert2BitTo2BitHiloKernelNT,
        blocksize, 
        smem
    ));

    const int maxBlocks = maxBlocksPerSM * numSMs;

    dim3 block(blocksize,1,1);
    dim3 grid(std::min(maxBlocks, SDIV(numSequences, blocksize)), 1, 1);

    convert2BitTo2BitHiloKernelNT<<<grid, block, 0, stream>>>(
        d_inputdata,
        inputpitchInInts,
        d_outputdata,
        outputpitchInInts,
        d_sequenceLengths,
        numSequences
    ); CUDACHECKASYNC;

#if 0    

    callCheckSequenceConversionKernelNT(d_inputdata,
        inputpitchInInts,
        d_outputdata,
        outputpitchInInts,
        d_sequenceLengths,
        numSequences,
        stream);

#endif

}

void callConversionKernel2BitTo2BitHiLoTT(
    const unsigned int* d_inputdata,
    size_t inputpitchInInts,
    unsigned int* d_outputdata,
    size_t outputpitchInInts,
    const int* d_sequenceLengths,
    int numSequences,
    cudaStream_t stream
){

    constexpr int blocksize = 128;
    constexpr size_t smem = 0;

    int deviceId = 0;
    int numSMs = 0;
    int maxBlocksPerSM = 0;
    CUDACHECK(cudaGetDevice(&deviceId));
    CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        convert2BitTo2BitHiloKernelTT,
        blocksize, 
        smem
    ));

    const int maxBlocks = maxBlocksPerSM * numSMs;

    dim3 block(blocksize,1,1);
    dim3 grid(std::min(maxBlocks, SDIV(numSequences, blocksize)), 1, 1);

    convert2BitTo2BitHiloKernelTT<<<grid, block, 0, stream>>>(
        d_inputdata,
        inputpitchInInts,
        d_outputdata,
        outputpitchInInts,
        d_sequenceLengths,
        numSequences
    ); CUDACHECKASYNC;

#if 0            

    callCheckSequenceConversionKernelTT(d_inputdata,
        inputpitchInInts,
        d_outputdata,
        outputpitchInInts,
        d_sequenceLengths,
        numSequences,
        stream);

#endif 
        
}



template<int groupsize>
__global__
void encodeSequencesTo2BitKernel(
    unsigned int* __restrict__ encodedSequences,
    const char* __restrict__ decodedSequences,
    const int* __restrict__ sequenceLengths,
    size_t decodedSequencePitchInBytes,
    size_t encodedSequencePitchInInts,
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

            auto encodeNuc = [&](char nuc){
                switch(nuc) {
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
            };

            if(i < nInts - 1){
                //not last iteration. int encodes 16 chars
                __align__(16) char nucs[16];
                ((int4*)nucs)[0] = *((const int4*)&in[i * 16]);

                #pragma unroll
                for(int p = 0; p < 16; p++){
                    encodeNuc(nucs[p]);
                }
            }else{        
                for(int nucIndex = i * basesPerInt; nucIndex < length; nucIndex++){
                    encodeNuc(in[nucIndex]);
                }

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

void callEncodeSequencesTo2BitKernel(
    unsigned int* d_encodedSequences,
    const char* d_decodedSequences,
    const int* d_sequenceLengths,
    size_t decodedSequencePitchInBytes,
    size_t encodedSequencePitchInInts,
    int numSequences,
    int groupsize,
    cudaStream_t stream
){
    #define callkernel(s){ \
        dim3 block = 256; \
        dim3 grid = SDIV(numSequences, block.x / s); \
                                                    \
        encodeSequencesTo2BitKernel<s><<<grid, block, 0, stream>>>( \
            d_encodedSequences, \
            d_decodedSequences, \
            d_sequenceLengths, \
            decodedSequencePitchInBytes, \
            encodedSequencePitchInInts, \
            numSequences \
        ); CUDACHECKASYNC; \
    }

    switch(groupsize){
        case 1: callkernel(1); break;
        case 2: callkernel(2); break;
        case 4: callkernel(4); break;
        case 8: callkernel(8); break;
        case 16: callkernel(16); break;
        case 32: callkernel(32); break;
        default: assert(false);
    }

    #undef callkernel
}

template<int groupsize>
__global__
void decodeSequencesFrom2BitKernel(
    char* __restrict__ decodedSequences,
    const unsigned int* __restrict__ encodedSequences,
    const int* __restrict__ sequenceLengths,
    size_t decodedSequencePitchInBytes,
    size_t encodedSequencePitchInInts,
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

void callDecodeSequencesFrom2BitKernel(
    char* d_decodedSequences,
    const unsigned int* d_encodedSequences,
    const int* d_sequenceLengths,
    size_t decodedSequencePitchInBytes,
    size_t encodedSequencePitchInInts,
    int numSequences,
    int groupsize,
    cudaStream_t stream
){
    #define callkernel(s){ \
        dim3 block = 256; \
        dim3 grid = SDIV(numSequences, block.x / s); \
                                                   \
        decodeSequencesFrom2BitKernel<s><<<grid, block, 0, stream>>>( \
            d_decodedSequences, \
            d_encodedSequences, \
            d_sequenceLengths, \
            decodedSequencePitchInBytes, \
            encodedSequencePitchInInts, \
            numSequences \
        ); CUDACHECKASYNC; \
    }

    switch(groupsize){
        case 1: callkernel(1); break;
        case 2: callkernel(2); break;
        case 4: callkernel(4); break;
        case 8: callkernel(8); break;
        case 16: callkernel(16); break;
        case 32: callkernel(32); break;
        default: assert(false);
    }

    #undef callkernel
}



}
}




