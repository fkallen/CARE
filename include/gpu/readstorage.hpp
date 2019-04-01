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

    using Length_t = int;
    using Sequence_t = care::Sequence2BitHiLo;
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


	char* h_sequence_data = nullptr;
	Length_t* h_sequence_lengths = nullptr;
	char* h_quality_data = nullptr;
	//managed gpu memory
	char* d_sequence_data = nullptr;
	Length_t* d_sequence_lengths = nullptr;
	char* d_quality_data = nullptr;

	int maximum_allowed_sequence_length = 0;
	int maximum_allowed_sequence_bytes = 0;
	bool useQualityScores = false;
	read_number num_sequences = 0;
	std::size_t sequence_data_bytes = 0;
	std::size_t sequence_lengths_bytes = 0;
	std::size_t quality_data_bytes = 0;

	//ContiguousReadStorage::Type sequenceType = ContiguousReadStorage::Type::None;
	//ContiguousReadStorage::Type qualityType = ContiguousReadStorage::Type::None;

	DataProperties dataProperties;

	std::vector<int> deviceIds;
	std::map<int, GPUData> gpuData;
	bool hasMoved = false;

	std::mutex mutex;

	ContiguousReadStorage(read_number nSequences);

	ContiguousReadStorage(read_number nSequences, bool b);

	ContiguousReadStorage(read_number nSequences, bool useQualityScores, int maximum_allowed_sequence_length, const std::vector<int>& deviceIds);

    ContiguousReadStorage(const ContiguousReadStorage& other) = delete;
    ContiguousReadStorage& operator=(const ContiguousReadStorage& other) = delete;

	ContiguousReadStorage(ContiguousReadStorage&& other);

	ContiguousReadStorage& operator=(ContiguousReadStorage&& other);

	bool operator==(const ContiguousReadStorage& other);

	bool operator!=(const ContiguousReadStorage& other);

	std::size_t size() const ;

	void resize(read_number nReads);

	void destroy();

private:
	void insertSequence(read_number readNumber, const std::string& sequence);
public:
	void insertRead(read_number readNumber, const std::string& sequence);

	void insertRead(read_number readNumber, const std::string& sequence, const std::string& quality);

	const char* fetchQuality_ptr(read_number readNumber) const;

	const char* fetchSequenceData_ptr(read_number readNumber) const;

	int fetchSequenceLength(read_number readNumber) const;

	std::uint64_t getNumberOfSequences() const;

	void initGPUData();

	GPUData getGPUData(int deviceId) const;

	DataProperties findDataProperties(std::size_t requiredSequenceMem,
				std::size_t requiredQualityMem) const;

	void saveToFile(const std::string& filename) const;

	void loadFromFile(const std::string& filename);

	std::string nameOf(ContiguousReadStorage::Type type) const;

	std::string getNameOfSequenceType() const;

	std::string getNameOfQualityType() const;

	int getSequencePitch() const;

	int getQualityPitch() const;

	SequenceStatistics getSequenceStatistics() const;

	SequenceStatistics getSequenceStatistics(int numThreads) const;

};

#endif

}
}

#endif
