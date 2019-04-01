#ifndef CARE_GPU_CONTIG_READ_STORAGE_HPP
#define CARE_GPU_CONTIG_READ_STORAGE_HPP

#include "../hpc_helpers.cuh"
#include <readstorage.hpp>

#include <config.hpp>

#include <iostream>
#include <limits>
#include <random>
#include <cstring>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <map>
#include <mutex>

#include <omp.h>

namespace care {
namespace gpu {

#ifdef __NVCC__


struct ContiguousReadStorage {

    using Length_t = cpu::ContiguousReadStorage::Length_t;
    using Sequence_t = cpu::ContiguousReadStorage::Sequence_t;
    
	using SequenceStatistics = cpu::SequenceStatistics;

	static constexpr bool has_reverse_complement = false;
	static constexpr int serialization_id = 100000;

	enum class Type {
		None,
		Full,
		Managed
	};

	struct GPUData {
		char* d_sequence_data = nullptr;
		Length_t* d_sequence_lengths = nullptr;
		char* d_quality_data = nullptr;

		int id;

		ContiguousReadStorage::Type sequenceType = ContiguousReadStorage::Type::None;
		ContiguousReadStorage::Type qualityType = ContiguousReadStorage::Type::None;

		bool isValidSequenceData() const {
			return sequenceType != ContiguousReadStorage::Type::None;
		}

		bool isValidQualityData() const {
			return qualityType != ContiguousReadStorage::Type::None;
		}
	};

	struct DataProperties {
		ContiguousReadStorage::Type sequenceType = ContiguousReadStorage::Type::None;
		ContiguousReadStorage::Type qualityType = ContiguousReadStorage::Type::None;

		bool operator==(const DataProperties& rhs) const {
			return sequenceType == rhs.sequenceType
			       && qualityType == rhs.qualityType;
		}
		bool operator!=(const DataProperties& rhs) const {
			return !(*this == rhs);
		}
	};
    
    const cpu::ContiguousReadStorage* cpuReadStorage;



	//managed gpu memory
	char* d_sequence_data = nullptr;
	Length_t* d_sequence_lengths = nullptr;
	char* d_quality_data = nullptr;




	//ContiguousReadStorage::Type sequenceType = ContiguousReadStorage::Type::None;
	//ContiguousReadStorage::Type qualityType = ContiguousReadStorage::Type::None;

	DataProperties dataProperties;

	std::vector<int> deviceIds;
	std::map<int, GPUData> gpuData;
	bool hasMoved = false;

	std::mutex mutex;

    ContiguousReadStorage(const cpu::ContiguousReadStorage* readStorage, const std::vector<int>& deviceIds);

    ContiguousReadStorage(const ContiguousReadStorage& other) = delete;
    ContiguousReadStorage& operator=(const ContiguousReadStorage& other) = delete;

	ContiguousReadStorage(ContiguousReadStorage&& other);

	ContiguousReadStorage& operator=(ContiguousReadStorage&& other);

	bool operator==(const ContiguousReadStorage& other) const;

	bool operator!=(const ContiguousReadStorage& other) const;

	std::size_t size() const ;

	void destroy();

	const char* fetchQuality_ptr(read_number readNumber) const;

	const char* fetchSequenceData_ptr(read_number readNumber) const;

	int fetchSequenceLength(read_number readNumber) const;

	std::uint64_t getNumberOfSequences() const;

	void initGPUData();

	GPUData getGPUData(int deviceId) const;

	DataProperties findDataProperties(std::size_t requiredSequenceMem,
				std::size_t requiredQualityMem) const;


	std::string nameOf(ContiguousReadStorage::Type type) const;

	std::string getNameOfSequenceType() const;

	std::string getNameOfQualityType() const;

	int getSequencePitch() const;

	int getQualityPitch() const;

	SequenceStatistics getSequenceStatistics() const;

	SequenceStatistics getSequenceStatistics(int numThreads) const;
    
    void copyGpuLengthsToGpuBufferAsync(Length_t* d_lengths, const read_number* d_readIds, int nReadIds, int deviceId, cudaStream_t stream) const;
    
    void copyGpuSequenceDataToGpuBufferAsync(char* d_sequence_data, size_t out_sequence_pitch, const read_number* d_readIds, int nReadIds, int deviceId, cudaStream_t stream) const;
    
    void copyGpuQualityDataToGpuBufferAsync(char* d_quality_data, size_t out_quality_pitch, const read_number* d_readIds, int nReadIds, int deviceId, cudaStream_t stream) const;

};

#endif

}
}

#endif
