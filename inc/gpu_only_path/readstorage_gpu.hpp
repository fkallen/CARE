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

    template<class Length_t>
    __global__
    void GPUReadStorage_sequencelength_test_kernel(Length_t* result, const Length_t* d_sequence_lengths, std::uint64_t readId){
        if(threadIdx.x == 0)
            result[0] = d_sequence_lengths[readId];
    }

    __global__
    void GPUReadStorage_quality_test_kernel(char* result, const char* d_quality_data, int max_sequence_length, std::uint64_t readId){
        for(int i = threadIdx.x; i < max_sequence_length; i += blockDim.x){
            result[i] = d_quality_data[readId * max_sequence_length + i];
        }
    }

#endif

/*enum class GPUReadStorageType : int{
	None = 0,
	Sequences = 1,
	SequencesAndQualities = 2,
};*/

struct GPUReadStorage{

	enum class Type{
		None,
		Full,
		Managed
	};

    using Length_t = int;

    char* d_sequence_data = nullptr;
    char* d_quality_data = nullptr;
    Length_t* d_sequence_lengths = nullptr;
    int max_sequence_bytes = 0;
    int max_sequence_length = 0;
    int deviceId = -1;
	bool useQualityScores = false;

	//GPUReadStorageType type = GPUReadStorageType::None;

	GPUReadStorage::Type sequenceType = GPUReadStorage::Type::None;
	GPUReadStorage::Type qualityType = GPUReadStorage::Type::None;
    GPUReadStorage::Type sequencelengthType = GPUReadStorage::Type::None;

    std::string nameOf(GPUReadStorage::Type type) const {
        switch(type){
            case GPUReadStorage::Type::None: return "GPUReadStorage::Type::None";
            case GPUReadStorage::Type::Full: return "GPUReadStorage::Type::Full";
            case GPUReadStorage::Type::Managed: return "GPUReadStorage::Type::Managed";
            default: return "Error. GPUReadStorage::nameOf default case";
        }
    }

    std::string getNameOfSequenceType() const{
		return nameOf(sequenceType);
	}

	std::string getNameOfQualityType() const{
		return nameOf(qualityType);
	}
#if 0
    std::string getNameOfSequenceLengthType() const{
		return nameOf(sequencelengthType);
	}
#endif
	bool hasSequences() const{
		return sequenceType != GPUReadStorage::Type::None;
	}

	bool hasFullSequences() const{
		return sequenceType == GPUReadStorage::Type::Full;
	}

	bool hasManagedSequences() const{
		return sequenceType == GPUReadStorage::Type::Managed;
	}

	bool hasQualities() const{
		return useQualityScores && qualityType != GPUReadStorage::Type::None;
	}

	bool hasFullQualities() const{
		return useQualityScores && qualityType == GPUReadStorage::Type::Full;
	}

	bool hasManagedQualities() const{
		return useQualityScores && qualityType == GPUReadStorage::Type::Managed;
	}
#if 0
    bool hasSequenceLengths() const{
        return sequencelengthType != GPUReadStorage::Type::None;
    }

    bool hasFullSequenceLengths() const{
        return sequencelengthType == GPUReadStorage::Type::Full;
    }

    bool hasManagedSequenceLengths() const{
        return sequencelengthType == GPUReadStorage::Type::Managed;
    }
#endif

#ifdef __NVCC__


	template<class CPUReadStorage>
    static GPUReadStorage createFrom(const CPUReadStorage& cpurs,
                                    int max_sequence_bytes,
                                    int max_sequence_length,
									float maxPercentOfTotalMem,
									bool canUseManagedMemory,
                                    int deviceId){
        using ReadId_t = typename CPUReadStorage::ReadId_t;
        using Sequence_t = typename CPUReadStorage::Sequence_t;

        static_assert(std::numeric_limits<ReadId_t>::max() <= std::numeric_limits<std::uint64_t>::max());

        int oldId;
        cudaGetDevice(&oldId); CUERR;
        cudaSetDevice(deviceId); CUERR;

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, deviceId);

		bool isCapableOfUsingManagedMemory = prop.concurrentManagedAccess == 1;

		std::size_t freeMem;
        std::size_t totalMem;
        cudaMemGetInfo(&freeMem, &totalMem); CUERR;

        const std::uint64_t nSequences = cpurs.sequences.size();

        GPUReadStorage gpurs;
        gpurs.deviceId = deviceId;
        gpurs.max_sequence_bytes = max_sequence_bytes;
        gpurs.max_sequence_length = max_sequence_length;
		gpurs.useQualityScores = cpurs.useQualityScores;

        //return gpurs;
		canUseManagedMemory = false;

		const std::uint64_t requiredSequenceMem = max_sequence_bytes * nSequences + sizeof(Length_t) * nSequences; //sequences and sequence lengths
        //const std::uint64_t requiredSequenceLengthsMem = sizeof(Length_t) * nSequences;
		const std::uint64_t requiredQualityMem = cpurs.useQualityScores ? max_sequence_length * nSequences : 0;
		const std::uint64_t requiredTotalMem = requiredSequenceMem + requiredQualityMem;

		const bool isEnoughMemForSequences = (requiredSequenceMem < maxPercentOfTotalMem * totalMem && requiredSequenceMem < freeMem);
		const bool isEnoughMemForSequencesAndQualities = (requiredTotalMem < maxPercentOfTotalMem * totalMem && requiredTotalMem < freeMem);

		bool canTestSequences = false;

		//make sequences in gpu mem
		if(isEnoughMemForSequences){

			constexpr std::uint64_t maxcopybatchsequences = std::size_t(10000000);

			//copy sequences to GPU

            const std::uint64_t requiredSequenceMem = max_sequence_bytes * nSequences;
			const std::uint64_t requiredSequenceLengthsMem = sizeof(Length_t) * nSequences;
			
            cudaMalloc(&gpurs.d_sequence_data, requiredSequenceMem); CUERR;
			cudaMalloc(&gpurs.d_sequence_lengths, requiredSequenceLengthsMem); CUERR;

			const std::uint64_t copybatchsequences = std::min(nSequences, maxcopybatchsequences);

			std::uint64_t tmpstoragesize_seq = copybatchsequences * max_sequence_bytes;
			std::uint64_t tmpstoragesize_lengths = copybatchsequences * sizeof(Length_t);
			char* h_tmp_seq;
			char* h_tmp_lengths;
			cudaMallocHost(&h_tmp_seq, tmpstoragesize_seq); CUERR;
			cudaMallocHost(&h_tmp_lengths, tmpstoragesize_lengths); CUERR;

			assert(h_tmp_seq != nullptr);
			assert(h_tmp_lengths != nullptr);

			const int iters = SDIV(nSequences, copybatchsequences);

			for(int iter = 0; iter < iters; ++iter){
				std::memset(h_tmp_seq, 0, tmpstoragesize_seq);
				std::memset(h_tmp_lengths, 0, tmpstoragesize_lengths);

				ReadId_t localcount = 0;

				for(ReadId_t readId = iter * copybatchsequences; readId < std::min((iter + 1) * copybatchsequences, nSequences); ++readId){
					const Sequence_t* sequence = cpurs.fetchSequence_ptr(readId);

					assert(sequence->getNumBytes() <= max_sequence_bytes );
					assert(sequence->length() <= std::numeric_limits<Length_t>::max());
					
					Length_t len = sequence->length();

					std::memcpy(h_tmp_seq + localcount * max_sequence_bytes,
								sequence->begin(),
								sequence->getNumBytes());
					//h_tmp_lengths[localcount] = len;
					std::memcpy(h_tmp_lengths + localcount * sizeof(Length_t),
								&len,
								sizeof(Length_t));

					++localcount;
				}

				cudaMemcpy(gpurs.d_sequence_data + iter * copybatchsequences * max_sequence_bytes,
								h_tmp_seq,
								localcount * max_sequence_bytes,
								H2D); CUERR;
								
				cudaMemcpy(gpurs.d_sequence_lengths + iter * copybatchsequences,
								h_tmp_lengths,
								localcount * sizeof(Length_t),
								H2D); CUERR;
			}

			cudaDeviceSynchronize(); CUERR;

			cudaFreeHost(h_tmp_seq); CUERR;
			cudaFreeHost(h_tmp_lengths); CUERR;

			gpurs.sequenceType = GPUReadStorage::Type::Full;

			canTestSequences = true;
		}else{
			//use managed memory
			if(isCapableOfUsingManagedMemory && canUseManagedMemory){
				const std::uint64_t requiredSequenceMem = max_sequence_bytes * nSequences;
				const std::uint64_t requiredSequenceLengthsMem = sizeof(Length_t) * nSequences;
				cudaMallocManaged(&gpurs.d_sequence_data, requiredSequenceMem); CUERR;
				cudaMallocManaged(&gpurs.d_sequence_lengths, requiredSequenceLengthsMem); CUERR;

				cudaMemAdvise(gpurs.d_sequence_data,
								requiredSequenceMem,
								cudaMemAdviseSetReadMostly,
								0); CUERR; //last argument is ignored for cudaMemAdviseSetReadMostly
								
				cudaMemAdvise(gpurs.d_sequence_lengths,
								requiredSequenceLengthsMem,
								cudaMemAdviseSetReadMostly,
								0); CUERR; //last argument is ignored for cudaMemAdviseSetReadMostly

				for(ReadId_t readId = 0; readId < nSequences; ++readId){
					const Sequence_t* sequence = cpurs.fetchSequence_ptr(readId);

					assert(sequence->getNumBytes() <= max_sequence_bytes);
					assert(sequence->length() <= std::numeric_limits<Length_t>::max());
					
					Length_t len = sequence->length();

					std::memcpy(gpurs.d_sequence_data + readId * max_sequence_bytes,
								sequence->begin(),
								sequence->getNumBytes());
					gpurs.d_sequence_lengths[readId] = len;
				}

				gpurs.sequenceType = GPUReadStorage::Type::Managed;

				canTestSequences = true;
			}
		}

		if(canTestSequences){
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

#if 0
            {
                Length_t* h_test, *d_test;
                cudaMallocHost(&h_test, sizeof(Length_t)); CUERR;
                cudaMalloc(&d_test, sizeof(Length_t)); CUERR;

                std::mt19937 gen;
                gen.seed(std::random_device()());
                std::uniform_int_distribution<ReadId_t> dist(0, nSequences-1); // distribution in range [1, 6]

                for(ReadId_t i = 0; i < nSequences; i++){
                    ReadId_t readId = i;//dist(gen);
                    GPUReadStorage_sequencelength_test_kernel<<<1,1>>>(d_test, gpurs.d_sequence_lengths, readId); CUERR;
                    cudaMemcpy(h_test, d_test, sizeof(Length_t), D2H); CUERR;
                    cudaDeviceSynchronize(); CUERR;

                    const Sequence_t* sequence = cpurs.fetchSequence_ptr(readId);

                    bool equal = sequence->length() == *h_test;
                    if(!equal){
                        std::cout << readId << std::endl;
                        std::cout << sequence->length() << " " << *h_test << std::endl;
                    }
                    assert(equal);
                }

                std::cout << "GPUReadStorage_sequencelength_test ok" << std::endl;

                cudaFree(d_test); CUERR;
                cudaFreeHost(h_test); CUERR;
            }
#endif

		}
#if 0
        bool canTestSequenceLengths = false;

        if(isEnoughMemForSequenceLengths){

			constexpr std::uint64_t maxcopybatchsequences = std::size_t(10000000);

			//copy sequences to GPU

            const std::uint64_t requiredSequenceLengthsMem = sizeof(Length_t) * nSequences;
            cudaMalloc(&gpurs.d_sequence_lengths, requiredSequenceLengthsMem); CUERR;

			const std::uint64_t copybatchsequences = std::min(nSequences, maxcopybatchsequences);

			std::uint64_t tmpstoragesize = copybatchsequences * sizeof(Length_t);
			char* h_tmp;
			cudaMallocHost(&h_tmp, tmpstoragesize); CUERR;

			assert(h_tmp != nullptr);

			const int iters = SDIV(nSequences, copybatchsequences);

			for(int iter = 0; iter < iters; ++iter){
				std::memset(h_tmp, 0, tmpstoragesize);

				ReadId_t localcount = 0;

				for(ReadId_t readId = iter * copybatchsequences; readId < std::min((iter + 1) * copybatchsequences, nSequences); ++readId){
					const Sequence_t* sequence = cpurs.fetchSequence_ptr(readId);

					assert(sequence->length() <= std::numeric_limits<Length_t>::max());
                    Length_t len = sequence->length();
					std::memcpy(h_tmp + localcount * sizeof(Length_t),
								&len,
								sizeof(Length_t));

					++localcount;
				}

				cudaMemcpy(gpurs.d_sequence_lengths + iter * copybatchsequences,
								h_tmp,
								localcount * sizeof(Length_t),
								H2D); CUERR;
			}

			cudaDeviceSynchronize(); CUERR;

			cudaFreeHost(h_tmp); CUERR;

			gpurs.sequencelengthType = GPUReadStorage::Type::Full;

			canTestSequenceLengths = true;
		}else{
			//use managed memory
			if(isCapableOfUsingManagedMemory && canUseManagedMemory){
				const std::uint64_t requiredSequenceLengthsMem = sizeof(Length_t) * nSequences;
				cudaMallocManaged(&gpurs.d_sequence_lengths, requiredSequenceLengthsMem); CUERR;

				cudaMemAdvise(gpurs.d_sequence_lengths,
								requiredSequenceMem,
								cudaMemAdviseSetReadMostly,
								0); CUERR; //last argument is ignored for cudaMemAdviseSetReadMostly

				for(ReadId_t readId = 0; readId < nSequences; ++readId){
					const Sequence_t* sequence = cpurs.fetchSequence_ptr(readId);

					assert(sequence->length() <= std::numeric_limits<Length_t>::max());
                    Length_t len = sequence->length();
					std::memcpy(gpurs.d_sequence_lengths + readId,
								&len,
								sizeof(Length_t));
				}

				gpurs.sequencelengthType = GPUReadStorage::Type::Managed;

				canTestSequenceLengths = true;
			}
		}

        if(canTestSequenceLengths){
            //test code
#if 0
            {
                Length_t* h_test, *d_test;
                cudaMallocHost(&h_test, sizeof(Length_t)); CUERR;
                cudaMalloc(&d_test, sizeof(Length_t)); CUERR;

                std::mt19937 gen;
                gen.seed(std::random_device()());
                std::uniform_int_distribution<ReadId_t> dist(0, nSequences-1); // distribution in range [1, 6]

                for(ReadId_t i = 0; i < nSequences; i++){
                    ReadId_t readId = i;//dist(gen);
                    GPUReadStorage_sequencelength_test_kernel<<<1,1>>>(d_test, gpurs.d_sequence_lengths, readId); CUERR;
                    cudaMemcpy(h_test, d_test, sizeof(Length_t), D2H); CUERR;
                    cudaDeviceSynchronize(); CUERR;

                    const Sequence_t* sequence = cpurs.fetchSequence_ptr(readId);

                    bool equal = sequence->length() == *h_test;
                    if(!equal){
                        std::cout << readId << std::endl;
                        std::cout << sequence->length() << " " << *h_test << std::endl;
                    }
                    assert(equal);
                }

                std::cout << "GPUReadStorage_sequencelength_test ok" << std::endl;

                cudaFree(d_test); CUERR;
                cudaFreeHost(h_test); CUERR;
            }
#endif
        }
#endif

		bool canTestQualities = false;

		if(cpurs.useQualityScores){
			if(isEnoughMemForSequencesAndQualities){
				constexpr std::uint64_t maxcopybatchsequences = std::size_t(1000000);
				//copy qualities to GPU

				const std::uint64_t requiredQualityMem = max_sequence_length * nSequences;
				cudaMalloc(&gpurs.d_quality_data, requiredQualityMem); CUERR;

				const std::uint64_t copybatchsequences = std::min(nSequences, maxcopybatchsequences);

				std::uint64_t tmpstoragesize = copybatchsequences * max_sequence_length;
				char* h_tmp;
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

				cudaFreeHost(h_tmp); CUERR;

				gpurs.qualityType = GPUReadStorage::Type::Full;

				canTestQualities = true;
			}else{
				//use managed memory
				if(isCapableOfUsingManagedMemory && canUseManagedMemory){
					cudaMallocManaged(&gpurs.d_quality_data, requiredQualityMem); CUERR;

					cudaMemAdvise(gpurs.d_quality_data,
									requiredQualityMem,
									cudaMemAdviseSetReadMostly,
									0); CUERR; //last argument is ignored for cudaMemAdviseSetReadMostly

					for(ReadId_t readId = 0; readId < nSequences; ++readId){
						const std::string* qualityptr = cpurs.fetchQuality_ptr(readId);

						assert(int(qualityptr->size()) <= max_sequence_length);

						std::memcpy(gpurs.d_quality_data + readId * max_sequence_length,
									qualityptr->data(),
									qualityptr->size());
					}

					gpurs.qualityType = GPUReadStorage::Type::Managed;

					canTestQualities = true;
				}
			}

			if(canTestQualities){
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
		}

        cudaSetDevice(oldId); CUERR;
		
		//gpurs.sequenceType = GPUReadStorage::Type::None;
		//gpurs.qualityType = GPUReadStorage::Type::None;
		//gpurs.sequencelengthType = GPUReadStorage::Type::None;

        return gpurs;
    }

/*





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

        std::cout << "GPUReadStorage: requiredSequenceMem " << requiredSequenceMem << ", requiredQualityMem " << requiredQualityMem << ", free " << freeMem << ", total " << totalMem << std::endl;

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
*/
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
        //gpurs.type = GPUReadStorageType::None;
		gpurs.sequenceType = GPUReadStorage::Type::None;
		gpurs.qualityType = GPUReadStorage::Type::None;

        cudaSetDevice(oldId); CUERR;
    }

#endif
};



}

#endif
