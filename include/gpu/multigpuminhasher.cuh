#ifndef CARE_MULTI_GPU_MINHASHER_CUH
#define CARE_MULTI_GPU_MINHASHER_CUH

#include <config.hpp>


#include <gpu/distributedreadstorage.hpp>
#include <gpu/cuda_unique.cuh>
#include <gpu/singlegpuminhasher.cuh>
#include <gpu/kernels.hpp>


#include <options.hpp>
#include <util.hpp>
#include <hpc_helpers.cuh>
#include <filehelpers.hpp>

#include <sequencehelpers.hpp>
#include <memorymanagement.hpp>
#include <threadpool.hpp>

#include <cub/cub.cuh>

#include <vector>
#include <memory>
#include <limits>
#include <string>
#include <fstream>
#include <algorithm>
#include <numeric>

namespace care{
namespace gpu{


    class MultiGpuMinhasher{
    public:
        using Key = kmer_type;
        using Value = read_number;

        using GpuTable = GpuHashtable<Key, Value>;

        using DeviceSwitcher = cub::SwitchDevice;

        template<class T>
        using HostBuffer = helpers::SimpleAllocationPinnedHost<T, 5>;
        template<class T>
        using DeviceBuffer = helpers::SimpleAllocationDevice<T, 5>;


        MultiGpuMinhasher(int maxNumKeys_, int maxValuesPerKey, int k, std::vector<int> deviceIds_)
            : maxNumKeys(maxNumKeys_), kmerSize(k), resultsPerMapThreshold(maxValuesPerKey), deviceIds(deviceIds_)
        {
            
        }

        int constructFromReadStorage(
            const RuntimeOptions &runtimeOptions,
            std::uint64_t nReads,
            const DistributedReadStorage& gpuReadStorage,
            int upperBoundSequenceLength,
            int maxNumHashfunctions,
            int hashFunctionOffset = 0
        ){
            
            const int numDevices = deviceIds.size();
            sgpuMinhashers.clear();
            usableDeviceIds.clear();

            int currentHashFunctionOffset = hashFunctionOffset;

            vec_h_currentHashFunctionNumbers.resize(numDevices);

            int remainingNumHashfunctions = maxNumHashfunctions;

            for(int d = 0; d < numDevices; d++){
                DeviceSwitcher ds(deviceIds[d]);

                if(remainingNumHashfunctions > 0){

                    SingleGpuMinhasher mh(nReads, resultsPerMapThreshold, kmerSize);
                    const int createdTables =  mh.constructFromReadStorage(
                        runtimeOptions, 
                        nReads, 
                        gpuReadStorage, 
                        upperBoundSequenceLength, 
                        remainingNumHashfunctions, 
                        currentHashFunctionOffset
                    );

                    if(createdTables > 0){
                        vec_h_currentHashFunctionNumbers.push_back({});
                        sgpuMinhashers.emplace_back(std::move(mh));

                        auto& buffer = vec_h_currentHashFunctionNumbers[d];
                        buffer.resize(createdTables);
                        std::iota(buffer.begin(), buffer.end(), currentHashFunctionOffset);


                        currentHashFunctionOffset += createdTables;
                        remainingNumHashfunctions -= createdTables;

                        usableDeviceIds.emplace_back(deviceIds[d]);
                    }
                }                
            }

            const int numberOfAvailableHashFunctions = maxNumHashfunctions - remainingNumHashfunctions;

            return numberOfAvailableHashFunctions; 
        }

        struct QueryHandleStruct{
            std::vector<SingleGpuMinhasher::QueryHandle> sgpuHandles;
            std::vector<DeviceBuffer<Value>> remoteResultsPerGpu;

            std::vector<CudaStream> streams;
            std::vector<CudaEvent> events;

            std::vector<int> deviceIds;

            MemoryUsage getMemoryInfo() const{
                MemoryUsage mem{};

                const int numIds = deviceIds.size();

                for(const auto& h : sgpuHandles){
                    mem += h->getMemoryInfo();
                }

                for(int i = 0; i < numIds; i++){
                    mem.device[deviceIds[i]] = remoteResultsPerGpu[i].capacityInBytes();
                }

                return mem;
            }
        };

        using QueryHandle = std::shared_ptr<QueryHandleStruct>;

        QueryHandle makeQueryHandle(){
            auto ptr = std::make_shared<QueryHandleStruct>();

            const int numMinhashers = sgpuMinhashers.size();

            ptr->sgpuHandles.resize(numMinhashers);
            ptr->remoteResultsPerGpu.resize(numMinhashers);
            ptr->streams.resize(numMinhashers);
            ptr->events.resize(numMinhashers);
            ptr->deviceIds = usableDeviceIds;

            for(int i = 0; i < numMinhashers; i++){
                DeviceSwitcher ds(sgpuMinhashers[i].getDeviceId());
                ptr->sgpuHandles[i] = SingleGpuMinhasher::makeQueryHandle();
                ptr->streams[i] = std::move(CudaStream{});
                ptr->events[i] = std::move(CudaEvent{cudaEventDisableTiming});
            }

            return ptr;
        }

        void query(
            QueryHandle& handle,
            const unsigned int* d_encodedSequences,
            std::size_t encodedSequencePitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            int deviceId, 
            cudaStream_t stream,
            read_number* d_similarReadIds,
            int* d_similarReadsPerSequence,
            int* d_similarReadsPerSequencePrefixSum
        ) const{
            query_impl(
                handle,
                nullptr,
                d_encodedSequences,
                encodedSequencePitchInInts,
                d_sequenceLengths,
                numSequences,
                deviceId,
                stream, 
                d_similarReadIds,
                d_similarReadsPerSequence,
                d_similarReadsPerSequencePrefixSum
            );
        }

        void queryExcludingSelf(
            QueryHandle& handle,
            const read_number* d_readIds,
            const unsigned int* d_encodedSequences,
            std::size_t encodedSequencePitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            int deviceId, 
            cudaStream_t stream,
            read_number* d_similarReadIds,
            int* d_similarReadsPerSequence,
            int* d_similarReadsPerSequencePrefixSum
        ) const{
            query_impl(
                handle,
                d_readIds,
                d_encodedSequences,
                encodedSequencePitchInInts,
                d_sequenceLengths,
                numSequences,
                deviceId,
                stream, 
                d_similarReadIds,
                d_similarReadsPerSequence,
                d_similarReadsPerSequencePrefixSum
            );
        }


        void query_impl(
            QueryHandle& queryHandle,
            const read_number* d_readIds,
            const unsigned int* d_sequenceData2Bit,
            std::size_t encodedSequencePitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            int deviceId, 
            cudaStream_t stream,
            read_number* d_values,
            int* d_numValuesPerSequence,
            int* d_offsets //numSequences + 1
        ) const {

            for(int d = 0; d < int(usableDeviceIds.size()); i++){
                DeviceSwitcher ds(usableDeviceIds[d]);

                //allocate remote values
                //copy input data to remote gpus

                //query remote
            }

            //copy remote results into local results
        }

        void compact(){
            for(auto& minhasher : sgpuMinhashers){
                DeviceSwitcher ds(minhasher.getDeviceId());

                minhasher.compact();
            }            
        }

        void finalize(){
            compact();
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage mem{};

            for(const auto& minhasher : sgpuMinhashers){
                mem += minhasher.getMemoryInfo();
            }

            return mem;
        }

        constexpr int getKmerSize() const noexcept{
            return kmerSize;
        }

        std::uint64_t getKmerMask() const{
            constexpr int maximum_kmer_length = max_k<std::uint64_t>::value;

            return std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - getKmerSize()) * 2);
        }

        constexpr float getLoad() const noexcept{
            return 0.8f;
        }

        constexpr int getNumResultsPerMapThreshold() const noexcept{
            return resultsPerMapThreshold;
        }
        
        int getNumberOfMaps() const noexcept{
            int num = 0;
            for(const auto& minhasher : sgpuMinhashers){
                num += minhasher.getNumberOfMaps();
            }
            return num;
        }

        void destroy(){
            for(auto& minhasher : sgpuMinhashers){
                DeviceSwitcher sd(minhasher.getDeviceId());
                minhasher.destroy();
            }
        }

        int maxNumKeys{};
        int kmerSize{};
        int resultsPerMapThreshold{};
        std::vector<int> deviceIds;
        std::vector<int> usableDeviceIds;
        std::vector<HostBuffer<int>> vec_h_currentHashFunctionNumbers{};
        std::vector<SingleGpuMinhasher> sgpuMinhashers{};
    };


}
}




#endif