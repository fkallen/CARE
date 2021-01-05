#ifndef CARE_MULTIGPUREADSTORAGE_CUH
#define CARE_MULTIGPUREADSTORAGE_CUH

#include <readstorage.hpp>
#include <gpu/gpureadstorage.cuh>

#include <2darray.hpp>
#include <gpu/multigpuarray.cuh>
#include <sequencehelpers.hpp>
#include <lengthstorage.hpp>
#include <gpu/gpulengthstorage.hpp>
#include <sharedmutex.hpp>


#include <vector>
#include <cstdint>
#include <memory>
#include <map>

#include <cub/cub.cuh>

namespace care{
namespace gpu{


class MultiGpuReadStororage : public GpuReadStorage {
public:

    struct TempData{
        struct CallerData{

        };

        CallerData& getCallerData(int deviceId){
            cub::SwitchDevice sd(deviceId);
            return callerDataMap[deviceId];
        }

        std::map<int, CallerData> callerDataMap{};

        MemoryUsage getMemoryInfo() const override{
            MemoryUsage result{};

            return result;
        }
    };

    MultiGpuReadStororage(
        const cpu::ContiguousReadStorage& cpuReadStorage_, 
        std::vector<int> deviceIds_, 
        std::vector<std::size_t> memoryLimitsPerDevice_
    ) : cpuReadStorage{&cpuReadStorage_},
        deviceIds{std::move(deviceIds_)},
        memoryLimitsPerDevice{std::move(memoryLimitsPerDevice_)}
    {

        const int numGpus = deviceIds.size();
        const std::size_t numReads = cpuReadStorage->getNumberOfReads();

        gpuLengthStorage = std::move(GPULengthStore2(cpuReadStorage->getLengthStore(), deviceIds));

        //handle sequences
        readsPerSequencesPartition.resize(numGpus, 0);
        const int numColumnsSequences = SequenceHelpers::getEncodedNumInts2Bit(cpuReadStorage->getSequenceLengthUpperBound());
        std::size_t remaining = numReads;

        for(int d = 0; d < numGpus; d++){
            cub::SwitchDevice sd{deviceIds[d]};

            const std::size_t bytesPerSequence = Gpu2dArrayManaged<unsigned int>::computePitch(numColumnsSequences, sizeof(unsigned int));
            const std::size_t elementsForGpu = std::min(remaining, memoryLimitsPerDevice[d] / bytesPerSequence);

            Gpu2dArrayManaged<unsigned int> gpuarray(elementsForGpu, numColumnsSequences, sizeof(unsigned int));

            if(elementsForGpu > 0){
                cudaMemcpy2D(
                    gpuarray.getGpuData(),
                    gpuarray.getPitch(),
                    (const char*)(cpuReadStorage->getSequenceArray()) + (numReads - remaining) * cpuReadStorage->getSequencePitch(),
                    cpuReadStorage->getSequencePitch(),
                    numColumnsSequences,
                    elementsForGpu,
                    H2D
                ); CUERR;

                remaining -= elementsForGpu;

                memoryLimitsPerDevice[d] -= elementsForGpu * bytesPerSequence;             
            }

            sequencesGpuPartitions.emplace_back(std::move(gpuarray));

            readsPerSequencesPartition[d] = elementsForGpu;
        }

        readsPerSequencesPartition.emplace_back(remaining); // elements which to not fit on gpus
        readsPerSequencesPartitionPrefixSum.resize(readsPerSequencesPartitionPrefixSum.size()+1, 0);
        std::partial_sum(
            readsPerSequencesPartition.begin(), 
            readsPerSequencesPartition.end(),
            readsPerSequencesPartitionPrefixSum.begin() + 1
        );

        //handle qualities
        readsPerQualitiesPartition.resize(numGpus, 0);
        const int numColumnsQualities = cpuReadStorage->getSequenceLengthUpperBound();
        remaining = numReads;

        for(int d = 0; d < numGpus; d++){
            cub::SwitchDevice sd{deviceIds[d]};

            const std::size_t bytesPerSequence = Gpu2dArrayManaged<unsigned int>::computePitch(numColumnsSequences, sizeof(char));
            const std::size_t elementsForGpu = std::min(remaining, memoryLimitsPerDevice[d] / bytesPerSequence);

            Gpu2dArrayManaged<char> gpuarray(elementsForGpu, numColumnsQualities, sizeof(char));

            if(elementsForGpu > 0){
                cudaMemcpy2D(
                    gpuarray.getGpuData(),
                    gpuarray.getPitch(),
                    (const char*)(cpuReadStorage->getQualityArray()) + (numReads - remaining) * cpuReadStorage->getQualityPitch(),
                    cpuReadStorage->getQualityPitch(),
                    numColumnsQualities,
                    elementsForGpu,
                    H2D
                ); CUERR;

                remaining -= elementsForGpu;

                memoryLimitsPerDevice[d] -= elementsForGpu * bytesPerSequence;                
            }

            qualitiesGpuPartitions.emplace_back(std::move(gpuarray));

            readsPerQualitiesPartition[d] = elementsForGpu;
        }

        readsPerQualitiesPartition.emplace_back(remaining); // elements which to not fit on gpus
        readsPerQualitiesPartitionPrefixSum.resize(readsPerQualitiesPartitionPrefixSum.size()+1, 0);
        std::partial_sum(
            readsPerQualitiesPartition.begin(), 
            readsPerQualitiesPartition.end(),
            readsPerQualitiesPartitionPrefixSum.begin() + 1
        );
    }

public: //inherited GPUReadStorage interface

    Handle makeHandle() const override {
        auto data = std::make_unique<TempData>();

        std::unique_lock<SharedMutex> lock(sharedmutex);
        const int handleid = counter++;
        Handle h = constructHandle(handleid);

        tempdataVector.emplace_back(std::move(data));
        return h;
    }

    void areSequencesAmbiguous(
        Handle& handle,
        bool* d_result, 
        const read_number* d_readIds, 
        int numSequences, 
        cudaStream_t stream
    ) const override{
        //TODO correct implementation

        cudaMemsetAsync(d_result, 0, sizeof(bool) * numSequences, stream); CUERR;
    }

    void gatherSequences(
        Handle& handle,
        unsigned int* d_sequence_data,
        size_t outSequencePitchInInts,
        const read_number* h_readIds,
        const read_number* d_readIds,
        int numSequences,
        cudaStream_t stream
    ) const override{
        TempData* tempData = getTempDataFromHandle(handle);

        const int numGpus = deviceIds.size();

        for()

        //TODO
    }

    virtual void gatherQualities(
        Handle& handle,
        char* d_quality_data,
        size_t out_quality_pitch,
        const read_number* h_readIds,
        const read_number* d_readIds,
        int numSequences,
        cudaStream_t stream
    ) const override{

        //TODO
    }

    void gatherSequenceLengths(
        Handle& handle,
        int* d_lengths,
        const read_number* d_readIds,
        int numSequences,    
        cudaStream_t stream
    ) const override{

        gpuLengthStorage.gatherLengthsOnDeviceAsync(
            d_lengths, 
            d_readIds, 
            numSequences, 
            stream
        );

    }

    std::int64_t getNumberOfReadsWithN() const override{
        return cpuReadStorage->getNumberOfReadsWithN();
    }

    MemoryUsage getMemoryInfo() const override{
        MemoryUsage result;

        const int numGpus = deviceIds.size();

        for(int i = 0; i < numGpus; i++){
            result.device[sequencesGpuPartitions[i].getDeviceId()] += sequencesGpuPartitions[i].getPitch() * sequencesGpuPartitions[i].getNumColumns();

            if(canUseQualityScores()){
                result.device[qualitiesGpuPartitions[i].getDeviceId()] += qualitiesGpuPartitions[i].getPitch() * qualitiesGpuPartitions[i].getNumColumns();
            }
        }

        result += gpuLengthStorage.getMemoryInfo();

        return result;
    }

    MemoryUsage getMemoryInfo(const Handle& handle) const override{
        int deviceId = 0;
        cudaGetDevice(&deviceId); CUERR;

        TempData* tempData = getTempDataFromHandle(handle);
        auto& callerData = tempData->getCallerData(deviceId);
 
        return tempData->getMemoryInfo();
    }

    read_number getNumberOfReads() const override{
        return cpuReadStorage->getNumberOfReads();
    }

    bool canUseQualityScores() const override{
        return cpuReadStorage->canUseQualityScores();
    }

    int getSequenceLengthLowerBound() const override{
        return cpuReadStorage->getSequenceLengthLowerBound();
    }

    int getSequenceLengthUpperBound() const override{
        return cpuReadStorage->getSequenceLengthUpperBound();
    }

    void destroy() override{
        auto deallocVector = [](auto& vec){
            using T = typename std::remove_reference<decltype(vec)>::type;
            T tmp{};
            vec.swap(tmp);
        };
        const int numGpus = deviceIds.size();

        for(int i = 0; i < numGpus; i++){
            cub::SwitchDevice sd(deviceIds[i]);

            cudaDeviceSynchronize(); CUERR;

            sequencesGpuPartitions[i].destroy();

            if(canUseQualityScores()){
                qualitiesGpuPartitions[i].destroy();
            }
        }

        gpuLengthStorage.destroyGpuData();

        deallocVector(sequencesGpuPartitions);
        deallocVector(qualitiesGpuPartitions);
        deallocVector(readsPerSequencesPartition);
        deallocVector(readsPerSequencesPartitionPrefixSum);
        deallocVector(readsPerQualitiesPartition);
        deallocVector(readsPerQualitiesPartitionPrefixSum);
        deallocVector(tempdataVector);

    }

public: //private, but nvcc..

    template<class T>
    void gatherImplSingleGpu(
        Handle& handle,
        T* d_sequence_data,
        size_t outSequencePitchInBytes,
        const read_number* h_readIds,
        const read_number* d_readIds,
        int numSequences,
        cudaStream_t stream,
        const std::vector<Gpu2dArrayManaged<T>>& gpuArrays,
        const std::vector<std::size_t>& readsPerPartition,
        const std::vector<std::size_t>& readsPerPartitionPrefixSum
    ) const override{
        TempData* tempData = getTempDataFromHandle(handle);

        const int numGpus = deviceIds.size();

        for(int i = 0; i < numGpus; i++){
            if(readsPerPartition[i] > 0){
                cub::SwitchDevice sd{deviceIds[i]};

                break;
            }
        }
    }

private:
    TempData* getTempDataFromHandle(const Handle& handle) const{
        std::shared_lock<SharedMutex> lock(sharedmutex);

        return tempdataVector[handle.getId()].get();
    }

    bool hasHostSequences() const noexcept{
        return readsPerSequencesPartition.back() > 0;
    }

    bool hasHostQualities() const noexcept{
        return canUseQualityScores() && readsPerQualitiesPartition.back() > 0;
    }

    bool isHostElementSequence(std::size_t index) const noexcept{
        const int hostposition = readsPerSequencesPartitionPrefixSum.size() - 2;
        return index < getNumberOfReads() && readsPerSequencesPartitionPrefixSum[hostposition] < getNumberOfReads();
    }

    bool isHostElementQualityScore(std::size_t index) const noexcept{
        const int hostposition = readsPerQualitiesPartitionPrefixSum.size() - 2;
        return index < getNumberOfReads() && readsPerQualitiesPartitionPrefixSum[hostposition] < getNumberOfReads();
    }
    
    const cpu::ContiguousReadStorage* cpuReadStorage{};

    std::vector<Gpu2dArrayManaged<unsigned int>> sequencesGpuPartitions{};
    std::vector<Gpu2dArrayManaged<char>> qualitiesGpuPartitions{};
    std::vector<std::size_t> readsPerSequencesPartition{};
    std::vector<std::size_t> readsPerSequencesPartitionPrefixSum{};
    std::vector<std::size_t> readsPerQualitiesPartition{};
    std::vector<std::size_t> readsPerQualitiesPartitionPrefixSum{};

    GPULengthStore2<std::uint32_t> gpuLengthStorage{};

    std::vector<int> deviceIds{};
    std::vector<std::size_t> memoryLimitsPerDevice{};

    mutable int counter = 0;
    mutable SharedMutex sharedmutex{};
    mutable std::vector<std::unique_ptr<TempData>> tempdataVector{};
};
    
}
}






#endif