#ifndef MULTI_DEVICE_ALLOCATION_CUH
#define MULTI_DEVICE_ALLOCATION_CUH

#include <hpc_helpers.cuh>
#include <vector>
#include <cuda.h> //driver api

inline
CUresult multiDeviceFree(CUdeviceptr dptr, size_t size);

//create a contiguous allocation backed by memory of multiple devices.
// device[i] stores at least bytesPerGpu[i] bytes
inline
CUresult multiDeviceAlloc(
    CUdeviceptr *dptr, 
    size_t* totalAllocationSize, 
    const std::vector<CUdevice>& backingDeviceIds,
    std::vector<size_t> bytesPerGpu,
    const std::vector<CUdevice>& accessingDeviceIds
) {
    CUresult status = CUDA_SUCCESS;
    size_t granularity = 0;
    const int numBackingGpus = backingDeviceIds.size();
    const int numAccessingGpus = accessingDeviceIds.size();

    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(CUmemAllocationProp));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    for(auto id : backingDeviceIds){
        size_t min_granularity = 0;

        prop.location.id = id;
        status = cuMemGetAllocationGranularity(
            &granularity, 
            &prop,
            CU_MEM_ALLOC_GRANULARITY_MINIMUM
        );
        if(status != CUDA_SUCCESS){
            return status;
        }
        granularity = std::max(min_granularity, granularity);
    }
    for(auto id : accessingDeviceIds){
        size_t min_granularity = 0;

        prop.location.id = id;
        status = cuMemGetAllocationGranularity(
            &granularity, 
            &prop,
            CU_MEM_ALLOC_GRANULARITY_MINIMUM
        );
        if(status != CUDA_SUCCESS){
            return status;
        }
        granularity = std::max(min_granularity, granularity);
    }

    *totalAllocationSize = 0;
    for(auto& size : bytesPerGpu){
        size = SDIV(size, granularity) * granularity;
        *totalAllocationSize += size;
    }
    if(*totalAllocationSize == 0){
        return CUDA_ERROR_INVALID_VALUE;
    }

    //std::cout << *totalAllocationSize << "\n";
    status = cuMemAddressReserve(dptr, *totalAllocationSize, 0, 0, 0);
    if (status != CUDA_SUCCESS) {
        return status;
    }

    size_t mapOffset = 0;

    for(int g = 0; g < numBackingGpus; g++){
        if(bytesPerGpu[g] > 0){
            CUresult status2 = CUDA_SUCCESS;
            prop.location.id = backingDeviceIds[g];

            CUmemGenericAllocationHandle allocationHandle;
            status = cuMemCreate(&allocationHandle, bytesPerGpu[g], &prop, 0);
            if (status != CUDA_SUCCESS) {
                multiDeviceFree(*dptr, *totalAllocationSize);
                return status;
            }
            status = cuMemMap(
                *dptr + mapOffset, 
                bytesPerGpu[g], 
                0,
                allocationHandle, 
                0
            );
            status2 = cuMemRelease(allocationHandle);

            if (status == CUDA_SUCCESS) {
                status = status2;
            }
            if (status != CUDA_SUCCESS) {
                multiDeviceFree(*dptr, *totalAllocationSize);
                return status;
            }
            mapOffset += bytesPerGpu[g];
        }
    }

    std::vector<CUmemAccessDesc> accessDescriptors;
    accessDescriptors.resize(numAccessingGpus);
    for(int g = 0; g < numAccessingGpus; g++){
        accessDescriptors[g].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDescriptors[g].location.id = accessingDeviceIds[g];
        accessDescriptors[g].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    status = cuMemSetAccess(
        *dptr, 
        *totalAllocationSize, 
        accessDescriptors.data(),
        accessDescriptors.size()
    );
    if (status != CUDA_SUCCESS) {
        multiDeviceFree(*dptr, *totalAllocationSize);
        return status;
    }

    return status;
}

inline
CUresult multiDeviceFree(CUdeviceptr dptr, size_t size) {
    CUresult status = CUDA_SUCCESS;

    status = cuMemUnmap(dptr, size);
    if (status != CUDA_SUCCESS) {
        return status;
    }

    status = cuMemAddressFree(dptr, size);

    return status;
}

#endif