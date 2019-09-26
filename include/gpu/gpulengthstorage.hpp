#ifndef CARE_GPU_LENGTH_STORAGE_HPP
#define CARE_GPU_LENGTH_STORAGE_HPP

#include <gpu/utility_kernels.cuh>
#include <hpc_helpers.cuh>
#include <lengthstorage.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

namespace care{

#ifdef __NVCC__    

struct GPULengthStore{
    using Data_t = LengthStore::Data_t;

    GPULengthStore() = default;

    GPULengthStore(LengthStore&& lStore, std::vector<int> deviceIds_) 
            : deviceIds(deviceIds_), 
            lengthStore(std::move(lStore)){

        deviceDataPointers.resize(deviceIds.size());
        int oldDevice = 0;
        cudaGetDevice(&oldDevice); CUERR;
        for(int i = 0; i < int(deviceIds.size()); i++){
            int deviceId = deviceIds[i];
            cudaSetDevice(deviceId);
            cudaMalloc(&deviceDataPointers[i], lengthStore.getRawSizeInBytes()); CUERR;
            cudaMemcpy(deviceDataPointers[i], lengthStore.getRaw(), lengthStore.getRawSizeInBytes(), cudaMemcpyHostToDevice); CUERR;
        }
        cudaSetDevice(oldDevice); CUERR;
    }

    ~GPULengthStore(){
        int oldDevice = 0;
        cudaGetDevice(&oldDevice); CUERR;
        for(int i = 0; i < int(deviceIds.size()); i++){
            cudaSetDevice(deviceIds[i]); CUERR;
            cudaFree(deviceDataPointers[i]); CUERR;
        }
        cudaSetDevice(oldDevice); CUERR;
    }

    GPULengthStore(const GPULengthStore&) = delete;
    GPULengthStore(GPULengthStore&&) = default;
    GPULengthStore& operator=(const GPULengthStore&) = delete;
    GPULengthStore& operator=(GPULengthStore&&) = default;

    int getMinLength() const{
        return lengthStore.getMinLength();
    }

    int getMaxLength() const{
        return lengthStore.getMaxLength();
    }

    std::int64_t getNumElements() const{
        return lengthStore.getNumElements();
    }

    void gatherLengthsOnHost(int* result, const std::int64_t* ids, std::int64_t numIds){
        auto loop = [&](auto begin, auto end){
            for(decltype(begin) i = begin; i < end; i++){
                result[i] =  lengthStore.getLength(ids[i]);
            }
        };

        loop(0, numIds);
    }

    void gatherLengthsOnDeviceAsync(int* d_result, 
                                    int resultDeviceId,
                                    const std::int64_t* d_ids, 
                                    int numIds,
                                    cudaStream_t stream){
        auto it = std::find(deviceIds.begin(), deviceIds.end(), resultDeviceId);
        assert(it != deviceIds.end());

        int gpuIndex = std::distance(deviceIds.begin(), it);

        int oldDevice = 0;
        cudaGetDevice(&oldDevice); CUERR;
        cudaSetDevice(resultDeviceId); CUERR;

        const Data_t* gpuData = deviceDataPointers[gpuIndex];
        int minLen = getMinLength();
        int maxLen = getMaxLength();
        //std::int64_t size = getSize();
        int databits = DataTBits;
        int numRawElements = lengthStore.getRawSizeInElements();
        int bitsPerLength = lengthStore.getRawBitsPerLength();

        if(minLen == maxLen){
            call_fill_kernel_async(d_result, numIds, minLen, stream); CUERR;
        }else{

            dim3 block(128,1,1);
            dim3 grid(SDIV(numIds, block.x));

            generic_kernel<<<grid, block, 0, stream>>>([=] __device__ (){
                auto getBits = [=](Data_t l, Data_t r, int begin, int endExcl){
                    assert(0 <= begin && begin < endExcl && endExcl <= 2 * databits);
                    
                    const int lbegin = min(databits, begin);
                    const int lendExcl = min(databits, endExcl);

                    const int rbegin = max(0, begin - databits);
                    const int rendExcl = max(0, endExcl - databits);

                    Data_t lmask = 0;
                    for(int i = lbegin; i < lendExcl; i++){
                        lmask = (lmask << 1) | 1;
                    }

                    Data_t rmask = 0;
                    for(int i = rbegin; i < rendExcl; i++){
                        rmask = (rmask << 1) | 1;
                    }

                    const Data_t lpiece = (l >> (databits - lendExcl)) & lmask;
                    const Data_t rpiece = (r >> (databits - rendExcl)) & rmask;
                    Data_t result = lpiece << (bitsPerLength - (lendExcl - lbegin)) | rpiece;
                    return result;
                };

                for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < numIds; i += blockDim.x * gridDim.x){
                    std::int64_t index = d_ids[i];
                    const std::uint64_t firstBit = bitsPerLength * index;
                    const std::uint64_t lastBitExcl = bitsPerLength * (index+1);
                    const std::uint64_t firstuintindex = firstBit / databits;
                    const int begin = firstBit - firstuintindex * databits;
                    const int endExcl = lastBitExcl - firstuintindex * databits;

                    const auto first = gpuData[firstuintindex];
                    //prevent oob access
                    const auto second = firstuintindex == numRawElements - 1 ? gpuData[firstuintindex] : gpuData[firstuintindex + 1];
                    const Data_t lengthBits = getBits(first, second, begin, endExcl);

                    d_result[i] = int(lengthBits) + minLen;
                }
                
            }); CUERR;
        }

        cudaSetDevice(oldDevice); CUERR;
    }

    int getLength(std::int64_t index) const{
        return lengthStore.getLength(index);
    }

private:
    int DataTBits = 8 * sizeof(Data_t);
    std::vector<Data_t*> deviceDataPointers;
    std::vector<int> deviceIds;
    LengthStore lengthStore;    
};

}


#endif

#endif