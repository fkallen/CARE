#include <readstorage.hpp>
#include <gpu/readstorage.hpp>
#include <gpu/utility_kernels.cuh>
#include <hpc_helpers.cuh>
#include <config.hpp>

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <string>
#include <iostream>
#include <vector>

namespace care {
namespace gpu {

#ifdef __NVCC__

    __global__
    void ContiguousReadStorage_sequence_test_kernel(char* result, const char* d_sequence_data, int maximum_allowed_sequence_bytes, std::uint64_t readId){
    	for(int i = threadIdx.x; i < maximum_allowed_sequence_bytes; i += blockDim.x) {
    		result[i] = d_sequence_data[readId * maximum_allowed_sequence_bytes + i];
    	}
    }

    template<class Length_t>
    __global__
    void ContiguousReadStorage_sequencelength_test_kernel(Length_t* result, const Length_t* d_sequence_lengths, std::uint64_t readId){
    	if(threadIdx.x == 0)
    		result[0] = d_sequence_lengths[readId];
    }

    __global__
    void ContiguousReadStorage_quality_test_kernel(char* result, const char* d_quality_data, int maximum_allowed_sequence_length, std::uint64_t readId){
    	for(int i = threadIdx.x; i < maximum_allowed_sequence_length; i += blockDim.x) {
    		result[i] = d_quality_data[readId * maximum_allowed_sequence_length + i];
    	}
    }


        ContiguousReadStorage::ContiguousReadStorage(const cpu::ContiguousReadStorage* readStorage, const std::vector<int>& deviceIds)
                        : cpuReadStorage(readStorage), deviceIds(deviceIds){
            const std::size_t requiredSequenceMem = cpuReadStorage->sequence_data_bytes + cpuReadStorage->sequence_lengths_bytes; //sequences and sequence lengths
            const std::size_t requiredQualityMem = cpuReadStorage->useQualityScores ? cpuReadStorage->quality_data_bytes : 0;

    		dataProperties = findDataProperties(requiredSequenceMem, requiredQualityMem);


    	}

    	ContiguousReadStorage::ContiguousReadStorage(ContiguousReadStorage&& other){
    		*this = std::move(other);
    	}

    	ContiguousReadStorage::ContiguousReadStorage& ContiguousReadStorage::operator=(ContiguousReadStorage&& other){
    		destroy();

    		dataProperties = other.dataProperties;
    		deviceIds = std::move(other.deviceIds);
            gpuData = std::move(other.gpuData);
    		hasMoved = other.hasMoved;

    		other.destroy();
    		other.hasMoved = true;

    		return *this;
    	}

    	bool ContiguousReadStorage::operator==(const ContiguousReadStorage& other) const{
            if(*cpuReadStorage != *(other.cpuReadStorage))
                return false;

    		if(deviceIds != other.deviceIds)
    			return false;
    		if(dataProperties != other.dataProperties)
    			return false;
    		if(hasMoved != other.hasMoved)
    			return false;

    		//don't compare gpu memory

    		return true;
    	}

    	bool ContiguousReadStorage::operator!=(const ContiguousReadStorage& other) const{
    		return !(*this == other);
    	}

    	std::size_t ContiguousReadStorage::size() const {
            return cpuReadStorage->size();
    	}

    	void ContiguousReadStorage::destroy(){

    		if(!hasMoved) {
                int oldId;

                cudaGetDevice(&oldId); CUERR;

                for(auto& p : gpuData) {
                    auto& data = p.second;
                    cudaSetDevice(data.id); CUERR;
                    cudaFree(data.d_sequence_data); CUERR;
                    cudaFree(data.d_sequence_lengths); CUERR;
                    cudaFree(data.d_quality_data); CUERR;
                }

                cudaSetDevice(oldId); CUERR;
    		}

    		dataProperties = DataProperties{};

    		//sequenceType = ContiguousReadStorage::Type::None;
    		//qualityType = ContiguousReadStorage::Type::None;
    	}

    	const char* ContiguousReadStorage::fetchQuality_ptr(read_number readNumber) const {
            return cpuReadStorage->fetchQuality_ptr(readNumber);
    	}

    	const char* ContiguousReadStorage::fetchSequenceData_ptr(read_number readNumber) const {
            return cpuReadStorage->fetchSequenceData_ptr(readNumber);
    	}

    	int ContiguousReadStorage::fetchSequenceLength(read_number readNumber) const {
            return cpuReadStorage->fetchSequenceLength(readNumber);
    	}

    	std::uint64_t ContiguousReadStorage::getNumberOfSequences() const {
            return cpuReadStorage->getNumberOfSequences();
    	}

    	void ContiguousReadStorage::initGPUData(){
    		int oldId;
    		cudaGetDevice(&oldId); CUERR;

    		for(auto deviceId : deviceIds) {
    			auto datait = gpuData.find(deviceId);
    			if(datait == gpuData.end()) {

    				cudaSetDevice(deviceId); CUERR;

    				GPUData data;
    				data.id = deviceId;

    				if(dataProperties.sequenceType == ContiguousReadStorage::Type::Full) {
    					cudaMalloc(&data.d_sequence_data, cpuReadStorage->sequence_data_bytes); CUERR;
                        cudaMalloc(&data.d_sequence_lengths, cpuReadStorage->sequence_lengths_bytes); CUERR;

                        cudaMemcpy(data.d_sequence_data, cpuReadStorage->h_sequence_data.get(), cpuReadStorage->sequence_data_bytes, H2D); CUERR;
                        cudaMemcpy(data.d_sequence_lengths, cpuReadStorage->h_sequence_lengths.get(), cpuReadStorage->sequence_lengths_bytes, H2D); CUERR;

    					data.sequenceType = ContiguousReadStorage::Type::Full;
    				}

    				if(dataProperties.qualityType == ContiguousReadStorage::Type::Full) {
                        cudaMalloc(&data.d_quality_data, cpuReadStorage->quality_data_bytes); CUERR;

                        cudaMemcpy(data.d_quality_data, cpuReadStorage->h_quality_data.get(), cpuReadStorage->quality_data_bytes, H2D); CUERR;

    					data.qualityType = ContiguousReadStorage::Type::Full;
    				}

    				cudaSetDevice(oldId); CUERR;

    				gpuData[deviceId] = data;

    	    #if 0
    				if(data.isValidSequenceData()) {
    					//verify sequence data
    					{
    						char* h_test, *d_test;
    						cudaMallocHost(&h_test, maximum_allowed_sequence_bytes); CUERR;
    						cudaMalloc(&d_test, maximum_allowed_sequence_bytes); CUERR;

    						std::mt19937 gen;
    						gen.seed(std::random_device()());
    						std::uniform_int_distribution<read_number> dist(0, getNumberOfSequences()-1); // distribution in range [1, 6]

    						for(read_number i = 0; i < getNumberOfSequences(); i++) {
    							read_number readId = i;//dist(gen);
    							ContiguousReadStorage_sequence_test_kernel<<<1,32>>>(d_test, data.d_sequence_data, maximum_allowed_sequence_bytes, readId); CUERR;
    							cudaMemcpy(h_test, d_test, maximum_allowed_sequence_bytes, D2H); CUERR;
    							cudaDeviceSynchronize(); CUERR;

    							//const Sequence_t* sequence = cpurs.fetchSequence_ptr(readId);
    							const char* sequence = fetchSequenceData_ptr(readId);
    							const int len = fetchSequenceLength(readId);

    							int result = std::memcmp(sequence, h_test, Sequence_t::getNumBytes(len));
    							if(result != 0) {
    								std::cout << readId << std::endl;
    								for(int k = 0; k < Sequence_t::getNumBytes(len); ++k)
    									std::cout << int(sequence[k]) << " " << int(h_test[k]) << std::endl;
    							}
    							assert(result == 0);
    						}

    						std::cout << "ContiguousReadStorage_sequence_test ok" << std::endl;

    						cudaFree(d_test); CUERR;
    						cudaFreeHost(h_test); CUERR;
    					}
    					{
    						Length_t* h_test, *d_test;
    						cudaMallocHost(&h_test, sizeof(Length_t)); CUERR;
    						cudaMalloc(&d_test, sizeof(Length_t)); CUERR;

    						std::mt19937 gen;
    						gen.seed(std::random_device()());
    						std::uniform_int_distribution<read_number> dist(0, getNumberOfSequences()-1); // distribution in range [1, 6]

    						for(read_number i = 0; i < getNumberOfSequences(); i++) {
    							read_number readId = i;//dist(gen);
    							ContiguousReadStorage_sequencelength_test_kernel<<<1,1>>>(d_test, data.d_sequence_lengths, readId); CUERR;
    							cudaMemcpy(h_test, d_test, sizeof(Length_t), D2H); CUERR;
    							cudaDeviceSynchronize(); CUERR;

    							const int length = fetchSequenceLength(readId);

    							bool equal = length == *h_test;
    							if(!equal) {
    								std::cout << readId << std::endl;
    								std::cout << length << " " << *h_test << std::endl;
    							}
    							assert(equal);
    						}

    						std::cout << "ContiguousReadStorage_sequencelength_test ok" << std::endl;

    						cudaFree(d_test); CUERR;
    						cudaFreeHost(h_test); CUERR;
    					}
    				}
    	    #endif

    	    #if 0
    				if(data.isValidQualityData()) {
    					//verify quality scores

    					char* h_test, *d_test;
    					cudaMallocHost(&h_test, maximum_allowed_sequence_length); CUERR;
    					cudaMalloc(&d_test, maximum_allowed_sequence_length); CUERR;

    					std::mt19937 gen;
    					gen.seed(std::random_device()());
    					std::uniform_int_distribution<read_number> dist(0, getNumberOfSequences()-1); // distribution in range [1, 6]

    					for(read_number i = 0; i < getNumberOfSequences(); i++) {
    						read_number readId = i;//dist(gen);
    						ContiguousReadStorage_quality_test_kernel<<<1,128>>>(d_test, data.d_quality_data, maximum_allowed_sequence_length, readId); CUERR;
    						cudaMemcpy(h_test, d_test, maximum_allowed_sequence_length, D2H); CUERR;
    						cudaDeviceSynchronize(); CUERR;

    						//const std::string* qualityptr = cpurs.fetchQuality_ptr(readId);
    						const char* quality = fetchQuality_ptr(readId);
    						const int len = fetchSequenceLength(readId);

    						int result = std::memcmp(quality, h_test, len);
    						if(result != 0) {
    							std::cout << readId << std::endl;
    							for(int k = 0; k < len; ++k)
    								std::cout << int(quality[k]) << " " << int(h_test[k]) << std::endl;
    						}
    						assert(result == 0);
    					}

    					std::cout << "ContiguousReadStorage_quality_test ok" << std::endl;

    					cudaFree(d_test); CUERR;
    					cudaFreeHost(h_test); CUERR;
    				}
    	    #endif
    			}
    		}

    		cudaSetDevice(oldId); CUERR;
    	}

    	ContiguousReadStorage::GPUData ContiguousReadStorage::getGPUData(int deviceId) const {
    		auto it = std::find(deviceIds.begin(), deviceIds.end(), deviceId);
    		if(it == deviceIds.end()) {
    			GPUData data;
    			data.id = deviceId;
    			return data;
    		}else{

    			auto datait = gpuData.find(deviceId);
    			if(datait != gpuData.end()) {
    				return datait->second;
    			}else{
    				std::cerr << "getGPUData(" << deviceId << ") not found";

    				GPUData data;
    				data.id = deviceId;

    				return data;
    			}
    		}
    	}

    	ContiguousReadStorage::DataProperties ContiguousReadStorage::findDataProperties(std::size_t requiredSequenceMem,
    				std::size_t requiredQualityMem) const {

    		constexpr float maxPercentOfTotalGPUMem = 0.8;

    		const std::size_t requiredTotalMem = requiredSequenceMem + requiredQualityMem;

    		int oldId;
    		cudaGetDevice(&oldId); CUERR;


    		const bool everyDeviceCanStoreSequences = std::all_of(deviceIds.begin(), deviceIds.end(), [&](int deviceId){
    					cudaSetDevice(deviceId); CUERR;

    					std::size_t freeMem;
    					std::size_t totalMem;
    					cudaMemGetInfo(&freeMem, &totalMem); CUERR;

    					bool isEnoughMemForSequences = (requiredSequenceMem < maxPercentOfTotalGPUMem * totalMem && requiredSequenceMem < freeMem);

    		            return isEnoughMemForSequences;
    				});

    		const bool everyDeviceCanStoreBothSequencesAndQualities = std::all_of(deviceIds.begin(), deviceIds.end(), [&](int deviceId){
    					cudaSetDevice(deviceId); CUERR;

    					std::size_t freeMem;
    					std::size_t totalMem;
    					cudaMemGetInfo(&freeMem, &totalMem); CUERR;

    					bool isEnoughMemForSequencesAndQualities = (requiredTotalMem < maxPercentOfTotalGPUMem * totalMem && requiredTotalMem < freeMem);

    					return isEnoughMemForSequencesAndQualities;
    				});

    		cudaSetDevice(oldId);

    		DataProperties result;

    		if(!everyDeviceCanStoreSequences) {
    			result.sequenceType = ContiguousReadStorage::Type::None;
    		}else{ // everyDeviceCanStoreSequences == true
    			result.sequenceType = ContiguousReadStorage::Type::Full;
    		}

    		if(cpuReadStorage->useQualityScores) {
    			if(!everyDeviceCanStoreBothSequencesAndQualities) {
    				result.qualityType = ContiguousReadStorage::Type::None;
    			}else{ // everyDeviceCanStoreBothSequencesAndQualities == true
    				result.qualityType = ContiguousReadStorage::Type::Full;
    			}
    		}

            //result.sequenceType = ContiguousReadStorage::Type::None;
            //result.qualityType = ContiguousReadStorage::Type::None;

    		return result;
    	}

    	std::string ContiguousReadStorage::nameOf(ContiguousReadStorage::Type type) const {
    		switch(type) {
    		case ContiguousReadStorage::Type::None: return "ContiguousReadStorage::Type::None";
    		case ContiguousReadStorage::Type::Full: return "ContiguousReadStorage::Type::Full";
    		case ContiguousReadStorage::Type::Managed: return "ContiguousReadStorage::Type::Managed";
    		default: return "Error. ContiguousReadStorage::nameOf default case";
    		}
    	}

    	std::string ContiguousReadStorage::getNameOfSequenceType() const {
    		return nameOf(dataProperties.sequenceType);
    	}

    	std::string ContiguousReadStorage::getNameOfQualityType() const {
    		return nameOf(dataProperties.qualityType);
    	}

    	int ContiguousReadStorage::getSequencePitch() const {
            return cpuReadStorage->maximum_allowed_sequence_bytes;
    	}

    	int ContiguousReadStorage::getQualityPitch() const {
            return cpuReadStorage->maximum_allowed_sequence_length;
    	}

    	void ContiguousReadStorage::copyGpuLengthsToGpuBufferAsync(Length_t* d_lengths, const read_number* d_readIds, int nReadIds, int deviceId, cudaStream_t stream) const{
            auto gpuData = getGPUData(deviceId);

            const int* const rs_sequence_lengths = gpuData.d_sequence_lengths;

            dim3 grid(SDIV(nReadIds, 128),1,1);
            dim3 block(128,1,1);

            generic_kernel<<<grid, block,0, stream>>>([=] __device__ (){
                for(int index = threadIdx.x + blockDim.x * blockIdx.x; index < nReadIds; index += blockDim.x * gridDim.x){
                    const read_number readId = d_readIds[index];
                    d_lengths[index] = rs_sequence_lengths[readId];
                }
            });
        }

        void ContiguousReadStorage::copyGpuSequenceDataToGpuBufferAsync(char* d_sequence_data, size_t out_sequence_pitch, const read_number* d_readIds, int nReadIds, int deviceId, cudaStream_t stream) const{
            assert(size_t(cpuReadStorage->maximum_allowed_sequence_bytes) <= out_sequence_pitch);

            auto gpuData = getGPUData(deviceId);

            const char* const rs_sequence_data = gpuData.d_sequence_data;
            const size_t rs_sequence_pitch = std::size_t(getSequencePitch());

            dim3 grid(SDIV(nReadIds, 128),1,1);
            dim3 block(128,1,1);

            generic_kernel<<<grid, block,0, stream>>>([=] __device__ (){
                const int intiters = out_sequence_pitch / sizeof(int);

                for(int index = threadIdx.x + blockDim.x * blockIdx.x; index < nReadIds; index += blockDim.x * gridDim.x){
                    const read_number readId = d_readIds[index];

                    for(int k = 0; k < intiters; k++){
                        ((int*)&d_sequence_data[index * out_sequence_pitch])[k] = ((int*)&rs_sequence_data[size_t(readId) * rs_sequence_pitch])[k];
                    }
                    for(int k = intiters * sizeof(int); k < out_sequence_pitch; k++){
                        d_sequence_data[index * out_sequence_pitch + k] = rs_sequence_data[size_t(readId) * rs_sequence_pitch + k];
                    }
                }
            });
        }

        void ContiguousReadStorage::copyGpuQualityDataToGpuBufferAsync(char* d_quality_data, size_t out_quality_pitch, const read_number* d_readIds, int nReadIds, int deviceId, cudaStream_t stream) const{
            //std::cout << size_t(cpuReadStorage->maximum_allowed_sequence_length) <<  " "  << out_quality_pitch << std::endl;
            //assert(size_t(cpuReadStorage->maximum_allowed_sequence_length) <= out_quality_pitch);

            auto gpuData = getGPUData(deviceId);

            const int* const rs_sequence_lengths = gpuData.d_sequence_lengths;
            const char* const rs_quality_data = gpuData.d_quality_data;
            const size_t rs_quality_pitch = std::size_t(getQualityPitch());

            dim3 grid(std::min(nReadIds, (1<<16)),1,1);
            dim3 block(64,1,1);

            generic_kernel<<<grid, block,0, stream>>>([=] __device__ (){

                for(int index = blockIdx.x; index < nReadIds; index += gridDim.x){
                    const read_number readId = d_readIds[index];
                    const int length = rs_sequence_lengths[index];
                    for(int k = threadIdx.x; k < length; k += blockDim.x){
                        d_quality_data[index * out_quality_pitch + k]
                                = rs_quality_data[size_t(readId) * rs_quality_pitch + k];
                    }
                }
            });
        }




#endif


}
}
