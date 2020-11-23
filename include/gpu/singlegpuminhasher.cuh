#ifndef CARE_SINGLE_GPU_MINHASHER_CUH
#define CARE_SINGLE_GPU_MINHASHER_CUH

#include <config.hpp>

#include <warpcore.cuh>

#include <gpu/distributedreadstorage.hpp>
#include <gpu/cuda_unique.cuh>
#include <cpuhashtable.hpp>
#include <gpu/gpuhashtable.cuh>

#include <options.hpp>
#include <util.hpp>
#include <hpc_helpers.cuh>
#include <filehelpers.hpp>

#include <sequencehelpers.hpp>
#include <memorymanagement.hpp>
#include <threadpool.hpp>

#include <cub/cub.cub>

#include <vector>
#include <memory>
#include <limits>
#include <string>
#include <fstream>
#include <algorithm>

namespace care{
namespace gpu{

    namespace sgpuminhasherkernels{

        __global__
        void minhashSignaturesKernel(
            std::uint64_t* __restrict__ signatures,
            std::size_t signaturesRowPitchElements,
            const unsigned int* __restrict__ sequences2Bit,
            std::size_t sequenceRowPitchElements,
            int numSequences,
            const int* __restrict__ sequenceLengths,
            int k,
            int numHashFuncs,
            int firstHashFunc
        ){
                    
            //constexpr int blocksize = 128;
            constexpr int maximum_kmer_length = max_k<std::uint64_t>::value;

            const int tid = threadIdx.x + blockIdx.x * blockDim.x;

            if(tid < numSequences * numHashFuncs){
                const int mySequenceIndex = tid / numHashFuncs;
                const int myNumHashFunc = tid % numHashFuncs;
                const int hashFuncId = myNumHashFunc + firstHashFunc;

                const unsigned int* const mySequence = sequences2Bit + mySequenceIndex * sequenceRowPitchElements;
                const int myLength = sequenceLengths[mySequenceIndex];

                std::uint64_t* const mySignature = signatures + mySequenceIndex * signaturesRowPitchElements;

                std::uint64_t minHashValue = std::numeric_limits<std::uint64_t>::max();

                auto handlekmer = [&](auto fwd, auto rc){
                    using hasher = hashers::MurmurHash<std::uint64_t>;

                    const auto smallest = min(fwd, rc);
                    const auto hashvalue = hasher::hash(smallest + hashFuncId);
                    minHashValue = min(minHashValue, hashvalue);
                };

                if(myLength >= k){
                    //const int numKmers = myLength - k + 1;
                    const std::uint64_t kmer_mask = std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - k) * 2);
                    const int rcshiftamount = (maximum_kmer_length - k) * 2;

                    //Compute the first kmer
                    std::uint64_t kmer_encoded = mySequence[0];
                    if(k <= 16){
                        kmer_encoded >>= (16 - k) * 2;
                    }else{
                        kmer_encoded = (kmer_encoded << 32) | mySequence[1];
                        kmer_encoded >>= (32 - k) * 2;
                    }

                    kmer_encoded >>= 2; //k-1 bases, allows easier loop

                    std::uint64_t rc_kmer_encoded = SequenceHelpers::reverseComplementInt2Bit(kmer_encoded);

                    auto addBase = [&](std::uint64_t encBase){
                        kmer_encoded <<= 2;
                        rc_kmer_encoded >>= 2;

                        const std::uint64_t revcBase = (~encBase) & 3;
                        kmer_encoded |= encBase;
                        rc_kmer_encoded |= revcBase << 62;
                    };

                    constexpr int basesPerInt = SequenceHelpers::basesPerInt2Bit();

                    const int itersend1 = min(SDIV(k-1, basesPerInt) * basesPerInt, myLength);

                    //process sequence positions one by one
                    // until the next encoded sequence data element is reached
                    for(int nextSequencePos = k - 1; nextSequencePos < itersend1; nextSequencePos++){
                        const int nextIntIndex = nextSequencePos / basesPerInt;
                        const int nextPositionInInt = nextSequencePos % basesPerInt;

                        const std::uint64_t nextBase = mySequence[nextIntIndex] >> (30 - 2 * nextPositionInInt);

                        addBase(nextBase);

                        handlekmer(
                            kmer_encoded & kmer_mask, 
                            rc_kmer_encoded >> rcshiftamount
                        );
                    }

                    const int fullIntIters = (myLength - itersend1) / basesPerInt;

                    //process all fully occupied encoded sequence data elements
                    // improves memory access
                    for(int iter = 0; iter < fullIntIters; iter++){
                        const int intIndex = (itersend1 + iter * basesPerInt) / basesPerInt;
                        const unsigned int data = mySequence[intIndex];

                        #pragma unroll
                        for(int posInInt = 0; posInInt < basesPerInt; posInInt++){
                            const std::uint64_t nextBase = data >> (30 - 2 * posInInt);

                            addBase(nextBase);

                            handlekmer(
                                kmer_encoded & kmer_mask, 
                                rc_kmer_encoded >> rcshiftamount
                            );
                        }
                    }

                    //process remaining positions one by one
                    for(int nextSequencePos = fullIntIters * basesPerInt + itersend1; nextSequencePos < myLength; nextSequencePos++){
                        const int nextIntIndex = nextSequencePos / basesPerInt;
                        const int nextPositionInInt = nextSequencePos % basesPerInt;

                        const std::uint64_t nextBase = mySequence[nextIntIndex] >> (30 - 2 * nextPositionInInt);

                        addBase(nextBase);

                        handlekmer(
                            kmer_encoded & kmer_mask, 
                            rc_kmer_encoded >> rcshiftamount
                        );
                    }

                    mySignature[myNumHashFunc] = minHashValue;

                }else{
                    mySignature[myNumHashFunc] = std::numeric_limits<std::uint64_t>::max();
                }
            }
        }
    
        template<class T, class IsValidFunc>
        __global__
        void fixTableKeysKernel(T* __restrict__ keys, int numKeys, IsValidFunc isValid){
            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
            const int stride = blockDim.x * gridDim.x;

            for(int i = tid; i < numKeys; i += stride){
                T key = keys[i];
                int changed = 0;
                while(!isValid(key)){
                    key++;
                    changed = 1;
                }
                if(changed == 1){
                    keys[i] = key;
                }
            }
        }
    
    }


    class SingleGpuMinhasher{
    public:
        using Key = kmer_type;
        using Value = read_number;

        using GpuTable = GpuHashtable<Key, Value>;

        using DevicerSwitcher = cub::SwitchDevice;


        SingleGpuMinhasher(int maxNumKeys_, int maxKeysPerValue, int k)
            : maxNumKeys({)maxNumKeys_), kmerSize(k), resultsPerMapThreshold(maxKeysPerValue)
        {
            cudaGetDevice(&deviceId); CUERR;
        }

        int addHashfunctions(int numExtraFunctions){
            DevicerSwitcher ds(deviceId);

            int added = 0;
            int cur = gpuHashTables.size();

            for(int i = 0; i < numExtraFunctions; i++){
                gpuHashTables.emplace_back(
                    maxNumKeys / getLoad(),
                    warpcore::defaults::seed<kmer_type>(),
                    resultsPerMapThreshold + 1
                );

                auto status = gpuHashTables.back().pop_status();
                cudaDeviceSynchronize();
                assert(!status.has_any()); 
                //TODO errorhandling

                added++;
            }

            return added;
        }

        void insert(
            const unsigned int* d_sequenceData2Bit,
            int numSequences,
            const int* d_sequenceLengths,
            std::size_t encodedSequencePitchInInts,
            const read_number* d_readIds,
            int firstHashfunction,
            int numHashfunctions,
            cudaStream_t stream
        ){
            assert(firstHashfunction + numHashfunctions <= int(gpuHashTables.size()));

            DevicerSwitcher ds(deviceId);

            const std::size_t signaturesRowPitchElements = numHashfunctions;

            helpers::SimpleAllocationDevice<std::uint64_t> d_sig(numHashfunctions * numSequences);
            helpers::SimpleAllocationDevice<std::uint64_t> d_sig_trans(numHashfunctions * numSequences);

            std::uint64_t* d_signatures = d_sig.data();
            std::uint64_t* d_signatures_transposed = d_sig_trans.data();


            dim3 block(128,1,1);
            dim3 grid(SDIV(numHashfunctions * numSequences, block.x),1,1);

            minhashSignaturesKernel<<<grid, block, 0, stream>>>(
                d_signatures,
                signaturesRowPitchElements,
                d_sequenceData2Bit,
                encodedSequencePitchInInts,
                numSequences,
                d_sequenceLengths,
                getKmerSize(),
                numHashfunctions,
                firstHashfunction
            ); CUERR;

            helpers::call_transpose_kernel(
                d_signatures_transposed, 
                d_signatures, 
                numSequences, 
                signaturesRowPitchElements, 
                signaturesRowPitchElements,
                stream
            );

            fixTableKeysKernel<<<SDIV(numSequences * numHashfunctions, 128), 128, 0, stream>>>(
                d_signatures_transposed, 
                numSequences * numHashfunctions, 
                [] __device__ (const Key key){
                    return GpuTable::isValidKey(key);
                }                
            ); CUERR;

            for(int i = 0; i < numHashfunctions; i++){
                gpuHashTables[firstHashfunction + i]->insert(
                    d_signatures_transposed + i * numSequences,
                    d_readIds,
                    numSequences,
                    stream
                );

                // for(std::size_t k = 0; k < curBatchsize; k++){
                //     if(h_insertionStatus[k].has_any()){
                //         std::cerr << "Error table " << i << ", batch " << iter << ", position " << k << ": " << h_insertionStatus[k] << "\n";
                //     }
                // }
            }

            cudaStreamSynchronize(stream);

            for(int i = 0; i < numHashfunctions; i++){
                status_type status = gpuHashTables[firstHashfunction + i]->pop_status(stream);
                cudaStreamSynchronize(stream);

                if(h_insertionStatus[k].has_any()){
                    std::cerr << "Error table " << (firstHashfunction + i) << " after insertion: " << status << "\n";
                }
            }
        }

        void queryExcludingSelf(
            read_number* d_values,
            int* d_numValuesPerSequence,
            int* d_offsets, //numSequences + 1
            const unsigned int* d_sequenceData2Bit,
            int numSequences,
            const int* d_sequenceLengths,
            std::size_t encodedSequencePitchInInts,
            const read_number* d_readIds,
            cudaStream_t stream
        ){

            DevicerSwitcher ds(deviceId);

            const int numHashfunctions = gpuHashTables.size();
            const int firstHashfunction = 0;

            const std::size_t signaturesRowPitchElements = numHashfunctions;

            helpers::SimpleAllocationDevice<std::uint64_t, 0> d_sig(numHashfunctions * numSequences);
            helpers::SimpleAllocationDevice<std::uint64_t, 0> d_sig_trans(numHashfunctions * numSequences);

            helpers::SimpleAllocationDevice<int, 0> d_int_temp(numSequences * numHashfunctions);
            helpers::SimpleAllocationDevice<int, 0> d_cubsum(1 + numSequences);

            std::size_t cubtempbytes = 0;

            cub::DeviceScan::ExclusiveSum(
                nullptr,
                cubtempbytes,
                (int*)nullptr, 
                (int*)nullptr, 
                numSequences,
                stream
            );

            std::size_t cubtempbytes2 = 0;
            cub::DeviceScan::ExclusiveSum(
                nullptr,
                cubtempbytes2,
                (int*)nullptr, 
                (int*)nullptr, 
                numHashfunctions,
                stream
            );

            cubtempbytes = std::max(cubtempbytes, cubtempbytes2);

            cub::DeviceReduce::Sum(
                nullptr, 
                cubtempbytes2, 
                (int*)nullptr, 
                (int*)nullptr, 
                numSequences, 
                stream
            );

            cubtempbytes = std::max(cubtempbytes, cubtempbytes2);

            helpers::SimpleAllocationDevice<char, 0> d_cub_temp(cubtempbytes);

            std::uint64_t* d_signatures = d_sig.data();
            std::uint64_t* d_signatures_transposed = d_sig_trans.data();
            void* d_cubTemp = d_cub_temp.data(),

            dim3 block(128,1,1);
            dim3 grid(SDIV(numHashfunctions * numSequences, block.x),1,1);

            minhashSignaturesKernel<<<grid, block, 0, stream>>>(
                d_signatures,
                signaturesRowPitchElements,
                d_sequenceData2Bit,
                encodedSequencePitchInInts,
                numSequences,
                d_sequenceLengths,
                getKmerSize(),
                numHashfunctions,
                firstHashfunction
            ); CUERR;

            helpers::call_transpose_kernel(
                d_signatures_transposed, 
                d_signatures, 
                numSequences, 
                signaturesRowPitchElements, 
                signaturesRowPitchElements,
                stream
            );

            fixTableKeysKernel<<<SDIV(numSequences * numHashfunctions, 128), 128, 0, stream>>>(
                d_signatures_transposed, 
                numSequences * numHashfunctions, 
                [] __device__ (const Key key){
                    return GpuTable::isValidKey(key);
                }                
            ); CUERR;

            //determine number of values per hashfunction per sequence
            for(int i = 0; i < numHashfunctions; i++){
                gpuHashTables[i]->numValuesPerKeyCompact(
                    d_signatures_transposed + i * numSequences,
                    numSequences,
                    d_int_temp.data() + i * numSequences,
                    stream
                );
            }

            //cudaMemsetAsync(d_numValuesPerSequence, 0, sizeof(int) * numSequences, stream); CUERR;

            //accumulate number of values per sequence
            helpers::lambda_kernel<<<1024, 256, 0, stream>>>(
                [=, d_int_temp = d_int_temp.data()] __device__ (){
                    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    const int stride = blockDim.x * gridDim.x;

                    for(int i = tid; i < numSequences; i += stride){
                        d_numValuesPerSequence[i] = 0;
                    }
                    
                    for(int i = tid; i < numSequences; i += stride){
                        for(int k = 0; k < numTables; k++){
                            d_numValuesPerSequence[i] += d_int_temp[k * numSequences + i];
                        }
                    }
                }
            );

            //calculate total number of values
            cub::DeviceReduce::Sum(
                d_cubTemp, 
                cubtempbytes, 
                d_numValuesPerSequence, 
                d_cubsum + numSequences, 
                numSequences, 
                stream
            );

            //calculate total number of values per table
            for(int i = 0; i < numHashfunctions; i++){
                cub::DeviceReduce::Sum(
                    d_cubTemp, 
                    cubtempbytes, 
                    d_int_temp.data() + i * numSequences, 
                    d_cubsum + i, 
                    numSequences, 
                    stream
                );
            }

            //calculate global begin offset for each table
            cub::DeviceScan::ExclusiveSum(
                d_cubTemp, 
                cubtempbytes, 
                d_cubsum,
                d_cubsum,
                numHashfunctions,
                stream
            );

            std::vector<int> h_cubsum(numSequences + 1);

            cudaMemcpyAsync(h_cubsum.data, d_cubsum.data(), sizeof(int) * (numHashfunctions + 1), D2H, stream); CUERR;
            cudaStreamSynchronize(stream);

            const int totalNumValues = h_cubsum.back();

            helpers::SimpleAllocationDevice<Value> d_values_tmp(totalNumValues);

            //retrieve values

            for(int i = 0; i < numHashfunctions; i++){
                d_signatures_transposed + i * numSequences,
                    numSequences,
                    d_int_temp.data() + i * numSequences,
                    stream
                gpuHashTables[i]->retrieveCompact(
                   d_signatures_transposed + i * numSequences,
                    const Offset* d_beginOffsets,
                    numSequences,
                    d_values_tmp,
                    stream
                );
            }

            // cudaMemsetAsync(d_offsets, 0, sizeof(int), stream); CUERR;

            // cub::DeviceScan::InclusiveSum(
            //     nullptr,
            //     cubtempbytes,
            //     d_numValuesPerSequence,
            //     d_offsets + 1,
            //     numSequences,
            //     stream
            // );



            
            for(int i = 0; i < numHashfunctions; i++){

                gpuHashTables[i]->numValuesPerKeyCompact(
                    d_signatures_transposed + i * numSequences,
                    numSequences,
                    d_numValuesPerSequence,
                    stream
                );

                cub::DeviceScan::ExclusiveSum(
                    nullptr,
                    cubtempbytes,
                    d_numValuesPerSequence,
                    d_beginOffsets,
                    numSequences,
                    stream
                );

                gpuHashTables[i]->retrieveCompact(
                    d_signatures_transposed + i * numSequences,
                    d_beginOffsets,
                    numSequences,
                    d_values
                    cudaStream_t stream
                ) const 

                // for(std::size_t k = 0; k < curBatchsize; k++){
                //     if(h_insertionStatus[k].has_any()){
                //         std::cerr << "Error table " << i << ", batch " << iter << ", position " << k << ": " << h_insertionStatus[k] << "\n";
                //     }
                // }
            }

            cudaStreamSynchronize(stream);

            for(int i = 0; i < numHashfunctions; i++){
                status_type status = gpuHashTables[firstHashfunction + i]->pop_status(stream);
                cudaStreamSynchronize(stream);

                if(h_insertionStatus[k].has_any()){
                    std::cerr << "Error table " << (firstHashfunction + i) << " after insertion: " << status << "\n";
                }
            }
        }

        void compact(){
            DevicerSwitcher ds(deviceId);

            for(auto& table : gpuHashTables){
                table->compact();
            }
        }

        void finalize(){
            compact();
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage mem{};

            for(const auto& table : gpuHashTables){
                mem += table->getMemoryInfo();
            }

            return mem;
        }

        constexpr int getKmerSize() const noexcept{
            return kmerSize;
        }

        constexpr float getLoad() const noexcept{
            return 0.8f;
        }

        int deviceId{};
        int maxNumKeys{};
        int kmerSize{};
        int resultsPerMapThreshold{};
        std::vector<std::unique_ptr<GpuTable>> gpuHashTables{};
    };


}
}





#endif