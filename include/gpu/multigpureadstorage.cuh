#ifndef CARE_MULTIGPUREADSTORAGE_CUH
#define CARE_MULTIGPUREADSTORAGE_CUH

#include <readstorage.hpp>
#include <gpu/gpureadstorage.cuh>

#include <2darray.hpp>
#include <gpu/multigpuarray.cuh>
#include <sequencehelpers.hpp>

#include <vector>
#include <cstdint>
#include <memory>

#include <cub/cub.cuh>

namespace care{
namespace gpu{


class MultiGpuReadStororage : public GpuReadStorage {
public:
    std::vector<Gpu2dArrayManaged<unsigned int>> sequencesGpuPartitions;
    std::vector<Gpu2dArrayManaged<char>> qualitiesGpuPartitions;
    std::vector<int> readsPerSequencesPartition;
    std::vector<int> readsPerSequencesPartitionPrefixSum;
    std::vector<int> readsPerQualitiesPartition;
    std::vector<int> readsPerQualitiesPartitionPrefixSum;

    MultiGpuReadStororage(
        const cpu::ContiguousReadStorage& cpuReadStorage_, 
        std::vector<int> deviceIds_, 
        std::vector<std::size_t> memoryLimitsPerDevice_
    ) : cpuReadStorage{&cpuReadStorage_},
        deviceIds{std::move(deviceIds_)},
        memoryLimitsPerDevice{std::move(memoryLimitsPerDevice_)}
    {
        //Gpu2dArrayManaged(size_t numRows, size_t numColumns, size_t alignmentInBytes)
        const int numGpus = deviceIds.size();


        const std::size_t numReads = cpuReadStorage->getNumberOfReads();

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

public: //inherited interface

    virtual Handle makeHandle() const = 0;

    virtual void areSequencesAmbiguous(
        Handle& handle,
        bool* d_result, 
        const read_number* d_readIds, 
        int numSequences, 
        cudaStream_t stream
    ) const = 0;

    virtual void gatherSequences(
        Handle& handle,
        unsigned int* d_sequence_data,
        size_t outSequencePitchInInts,
        const read_number* h_readIds,
        const read_number* d_readIds,
        int numSequences,
        cudaStream_t stream
    ) const = 0;

    virtual void gatherQualities(
        Handle& handle,
        char* d_quality_data,
        size_t out_quality_pitch,
        const read_number* h_readIds,
        const read_number* d_readIds,
        int numSequences,
        cudaStream_t stream
    ) const = 0;

    virtual void gatherSequenceLengths(
        Handle& handle,
        int* d_lengths,
        const read_number* d_readIds,
        int numSequences,    
        cudaStream_t stream
    ) const = 0;

    virtual std::int64_t getNumberOfReadsWithN() const = 0;

    virtual MemoryUsage getMemoryInfo() const = 0;

    virtual MemoryUsage getMemoryInfo(const Handle& handle) const = 0;

    virtual read_number getNumberOfReads() const = 0;

    virtual bool canUseQualityScores() const = 0;

    virtual int getSequenceLengthLowerBound() const = 0;

    virtual int getSequenceLengthUpperBound() const = 0;

    virtual void destroy() = 0;

private:
    const cpu::ContiguousReadStorage* cpuReadStorage{};

    std::vector<int> deviceIds{};
    std::vector<std::size_t> memoryLimitsPerDevice{};
};
    
}
}






#endif