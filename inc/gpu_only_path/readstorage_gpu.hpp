#ifndef CARE_GPU_READ_STORAGE_HPP
#define CARE_GPU_READ_STORAGE_HPP

#include "../hpc_helpers.cuh"

#include <iostream>
#include <limits>
#include <random>
#include <cstring>
#include <cstdint>

namespace care{

#ifdef __NVCC__
    __global__
    void GPUReadStorage_sequence_test_kernel(char* result, const char* d_sequence_data, int max_sequence_bytes, std::uint64_t readId){
        for(int i = threadIdx.x; i < max_sequence_bytes; i += blockDim.x){
            result[i] = d_sequence_data[readId * max_sequence_bytes + i];
        }
    }

    __global__
    void GPUReadStorage_quality_test_kernel(char* result, const char* d_quality_data, int max_sequence_length, std::uint64_t readId){
        for(int i = threadIdx.x; i < max_sequence_length; i += blockDim.x){
            result[i] = d_quality_data[readId * max_sequence_length + i];
        }
    }

#endif

enum class GPUReadStorageType : int{
	None = 0,
	Sequences = 1,
	SequencesAndQualities = 2,
};

struct GPUReadStorage{

    char* d_sequence_data = nullptr;
    char* d_quality_data = nullptr;
    int max_sequence_bytes = 0;
    int max_sequence_length = 0;
    int deviceId = -1;

	GPUReadStorageType type = GPUReadStorageType::None;

    static std::string nameOf(GPUReadStorageType type){
        switch(type){
            case GPUReadStorageType::None: return "GPUReadStorageType::None";
            case GPUReadStorageType::Sequences: return "GPUReadStorageType::Sequences";
            case GPUReadStorageType::SequencesAndQualities: return "GPUReadStorageType::SequencesAndQualities";
            default: return "Error. GPUReadStorageType::nameOf default case";
        }
    }

#ifdef __NVCC__
    template<class CPUReadStorage>
    static GPUReadStorageType getBestPossibleType(const CPUReadStorage& cpurs,
                                                    int max_sequence_bytes,
                                                    int max_sequence_length,
                                                    float maxPercentOfTotalMem,
                                                    int deviceId){
        int oldId;
        cudaGetDevice(&oldId); CUERR;
        cudaSetDevice(deviceId); CUERR;

        const std::uint64_t requiredSequenceMem = max_sequence_bytes * cpurs.sequences.size();
        const std::uint64_t requiredQualityMem = max_sequence_length * cpurs.sequences.size();

		//std::uint64_t requiredQualityMem =
        std::size_t freeMem;
        std::size_t totalMem;
        cudaMemGetInfo(&freeMem, &totalMem); CUERR;

        std::cout << "GPUReadStorage: required " << requiredSequenceMem << ", free " << freeMem << ", total " << totalMem << std::endl;

        GPUReadStorageType result = GPUReadStorageType::None;
        if(requiredSequenceMem < maxPercentOfTotalMem * totalMem && requiredSequenceMem < freeMem){
            result = GPUReadStorageType::Sequences;
        }

        if(requiredSequenceMem + requiredQualityMem < maxPercentOfTotalMem * totalMem && requiredSequenceMem + requiredQualityMem < freeMem){
            result = GPUReadStorageType::SequencesAndQualities;
        }

        cudaSetDevice(oldId); CUERR;

        return result;
    }

    template<class CPUReadStorage>
    static GPUReadStorage createFrom(const CPUReadStorage& cpurs,
                                    GPUReadStorageType type,
                                    int max_sequence_bytes,
                                    int max_sequence_length,
                                    int deviceId){
        using ReadId_t = typename CPUReadStorage::ReadId_t;
        using Sequence_t = typename CPUReadStorage::Sequence_t;

        static_assert(std::numeric_limits<ReadId_t>::max() <= std::numeric_limits<std::uint64_t>::max());



		assert(type != GPUReadStorageType::None);

        int oldId;
        cudaGetDevice(&oldId); CUERR;
        cudaSetDevice(deviceId); CUERR;

        const std::uint64_t nSequences = cpurs.sequences.size();

        GPUReadStorage gpurs;
        gpurs.deviceId = deviceId;
        gpurs.max_sequence_bytes = max_sequence_bytes;
        gpurs.max_sequence_length = max_sequence_length;
		gpurs.type = type;

		if(type == GPUReadStorageType::Sequences || type == GPUReadStorageType::SequencesAndQualities){
            constexpr std::uint64_t maxcopybatchsequences = std::size_t(10000000);

			//copy sequences to GPU

            const std::uint64_t requiredSequenceMem = max_sequence_bytes * nSequences;
            cudaMalloc(&gpurs.d_sequence_data, requiredSequenceMem); CUERR;

			const std::uint64_t copybatchsequences = std::min(nSequences, maxcopybatchsequences);

			std::uint64_t tmpstoragesize = copybatchsequences * max_sequence_bytes;
			char* h_tmp;
			//h_tmp = new char[tmpstoragesize];
			cudaMallocHost(&h_tmp, tmpstoragesize); CUERR;

			assert(h_tmp != nullptr);

			const int iters = SDIV(nSequences, copybatchsequences);

			for(int iter = 0; iter < iters; ++iter){
				std::memset(h_tmp, 0, tmpstoragesize);

				ReadId_t localcount = 0;

				for(ReadId_t readId = iter * copybatchsequences; readId < std::min((iter + 1) * copybatchsequences, nSequences); ++readId){
					const Sequence_t* sequence = cpurs.fetchSequence_ptr(readId);

					assert(sequence->getNumBytes() <= max_sequence_bytes );

					std::memcpy(h_tmp + localcount * max_sequence_bytes,
								sequence->begin(),
								sequence->getNumBytes());

					++localcount;
				}

				cudaMemcpy(gpurs.d_sequence_data + iter * copybatchsequences * max_sequence_bytes,
								h_tmp,
								localcount * max_sequence_bytes,
								H2D); CUERR;
			}

			cudaDeviceSynchronize(); CUERR;

			//delete [] h_tmp;
			cudaFreeHost(h_tmp); CUERR;

			//test code
#if 0
			{
				char* h_test, *d_test;
				cudaMallocHost(&h_test, max_sequence_bytes); CUERR;
				cudaMalloc(&d_test, max_sequence_bytes); CUERR;

				std::mt19937 gen;
				gen.seed(std::random_device()());
				std::uniform_int_distribution<ReadId_t> dist(0, nSequences-1); // distribution in range [1, 6]

				for(ReadId_t i = 0; i < nSequences; i++){
					ReadId_t readId = i;//dist(gen);
					GPUReadStorage_quality_test_kernel<<<1,32>>>(d_test, gpurs.d_sequence_data, gpurs.max_sequence_bytes, readId); CUERR;
					cudaMemcpy(h_test, d_test, gpurs.max_sequence_bytes, D2H); CUERR;
					cudaDeviceSynchronize(); CUERR;

					const Sequence_t* sequence = cpurs.fetchSequence_ptr(readId);

					int result = std::memcmp(sequence->begin(), h_test, sequence->getNumBytes());
					if(result != 0){
						std::cout << readId << std::endl;
						for(int k = 0; k < sequence->getNumBytes(); ++k)
							std::cout << int(sequence->begin()[k]) << " " << int(h_test[k]) << std::endl;
					}
					assert(result == 0);
				}

				std::cout << "GPUReadStorage_sequence_test ok" << std::endl;

				cudaFree(d_test); CUERR;
				cudaFreeHost(h_test); CUERR;
			}
#endif

		}

        if(type == GPUReadStorageType::SequencesAndQualities){
            constexpr std::uint64_t maxcopybatchsequences = std::size_t(1000000);
			//copy qualities to GPU

            const std::uint64_t requiredQualityMem = max_sequence_length * nSequences;
            cudaMalloc(&gpurs.d_quality_data, requiredQualityMem); CUERR;

			const std::uint64_t copybatchsequences = std::min(nSequences, maxcopybatchsequences);

			std::uint64_t tmpstoragesize = copybatchsequences * max_sequence_length;
			char* h_tmp;
			//h_tmp = new char[tmpstoragesize];
			cudaMallocHost(&h_tmp, tmpstoragesize); CUERR;

			assert(h_tmp != nullptr);

			const int iters = SDIV(nSequences, copybatchsequences);

			for(int iter = 0; iter < iters; ++iter){
				std::memset(h_tmp, 0, tmpstoragesize);

				ReadId_t localcount = 0;

				for(ReadId_t readId = iter * copybatchsequences; readId < std::min((iter + 1) * copybatchsequences, nSequences); ++readId){
					const std::string* qualityptr = cpurs.fetchQuality_ptr(readId);

					assert(int(qualityptr->size()) <= max_sequence_length);

					std::memcpy(h_tmp + localcount * max_sequence_length,
								qualityptr->data(),
								qualityptr->size());

					++localcount;
				}

				cudaMemcpy(gpurs.d_quality_data + iter * copybatchsequences * max_sequence_length,
								h_tmp,
								localcount * max_sequence_length,
								H2D); CUERR;
			}

			cudaDeviceSynchronize(); CUERR;


			//delete [] h_tmp;
			cudaFreeHost(h_tmp); CUERR;

			//test code
#if 0
			{
				char* h_test, *d_test;
				cudaMallocHost(&h_test, max_sequence_length); CUERR;
				cudaMalloc(&d_test, max_sequence_length); CUERR;

				std::mt19937 gen;
				gen.seed(std::random_device()());
				std::uniform_int_distribution<ReadId_t> dist(0, nSequences-1); // distribution in range [1, 6]

				for(ReadId_t i = 0; i < nSequences; i++){
					ReadId_t readId = i;//dist(gen);
					GPUReadStorage_quality_test_kernel<<<1,128>>>(d_test, gpurs.d_quality_data, gpurs.max_sequence_length, readId); CUERR;
					cudaMemcpy(h_test, d_test, gpurs.max_sequence_length, D2H); CUERR;
					cudaDeviceSynchronize(); CUERR;

					const std::string* qualityptr = cpurs.fetchQuality_ptr(readId);

					int result = std::memcmp(qualityptr->data(), h_test, max_sequence_length);
					if(result != 0){
						std::cout << readId << std::endl;
						for(int k = 0; k < int(qualityptr->size()); ++k)
							std::cout << int(qualityptr->begin()[k]) << " " << int(h_test[k]) << std::endl;
					}
					assert(result == 0);
				}

				std::cout << "GPUReadStorage_quality_test ok" << std::endl;

				cudaFree(d_test); CUERR;
				cudaFreeHost(h_test); CUERR;
			}
#endif

		}

        cudaSetDevice(oldId); CUERR;

        return gpurs;
    }

    static void destroy(GPUReadStorage& gpurs){
        int oldId;
        cudaGetDevice(&oldId); CUERR;
        cudaSetDevice(gpurs.deviceId); CUERR;

        cudaFree(gpurs.d_sequence_data); CUERR;
        cudaFree(gpurs.d_quality_data); CUERR;

        gpurs.d_sequence_data = nullptr;
        gpurs.d_quality_data = nullptr;
        gpurs.max_sequence_bytes = 0;
        gpurs.max_sequence_length = 0;
        gpurs.deviceId = -2;
        gpurs.type = GPUReadStorageType::None;

        cudaSetDevice(oldId); CUERR;
    }

#endif
};

}

#endif
