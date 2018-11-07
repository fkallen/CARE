#ifndef CARE_GPU_CONTIG_READ_STORAGE_HPP
#define CARE_GPU_CONTIG_READ_STORAGE_HPP

#include "../hpc_helpers.cuh"
#include "../readstorage.hpp"

#include <iostream>
#include <limits>
#include <random>
#include <cstring>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <map>

namespace care{
namespace gpu{

#ifdef __NVCC__

    template<class sequence_t,
             class readId_t>
    struct ContiguousReadStorage{

        using Length_t = int;
        using Sequence_t = sequence_t;
        using ReadId_t = readId_t;
        using SequenceStatistics = cpu::SequenceStatistics;

        static constexpr bool has_reverse_complement = false;

        enum class Type{
            None,
            Full,
            Managed
        };

        struct GPUData{
            char* d_sequence_data = nullptr;
            Length_t* d_sequence_lengths = nullptr;
            char* d_quality_data = nullptr;

            int id;

            ContiguousReadStorage::Type sequenceType = ContiguousReadStorage::Type::None;
        	ContiguousReadStorage::Type qualityType = ContiguousReadStorage::Type::None;

            bool isValidSequenceData() const{
                return sequenceType != ContiguousReadStorage::Type::None;
            }

            bool isValidQualityData() const{
                return qualityType != ContiguousReadStorage::Type::None;
            }
        };



        char* h_sequence_data = nullptr;
        Length_t* h_sequence_lengths = nullptr;
        char* h_quality_data = nullptr;
        //managed gpu memory
        char* d_sequence_data = nullptr;
        Length_t* d_sequence_lengths = nullptr;
        char* d_quality_data = nullptr;

        int max_sequence_length = 0;
        int max_sequence_bytes = 0;
        bool useQualityScores = false;
        ReadId_t num_sequences = 0;
        std::size_t sequence_data_bytes = 0;
        std::size_t sequence_lengths_bytes = 0;
        std::size_t quality_data_bytes = 0;

        ContiguousReadStorage::Type sequenceType = ContiguousReadStorage::Type::None;
    	ContiguousReadStorage::Type qualityType = ContiguousReadStorage::Type::None;

        std::vector<int> deviceIds;
        std::map<int, GPUData> gpuData;
        bool hasMoved = false;

        std::string nameOf(ContiguousReadStorage::Type type) const {
            switch(type){
                case ContiguousReadStorage::Type::None: return "ContiguousReadStorage::Type::None";
                case ContiguousReadStorage::Type::Full: return "ContiguousReadStorage::Type::Full";
                case ContiguousReadStorage::Type::Managed: return "ContiguousReadStorage::Type::Managed";
                default: return "Error. ContiguousReadStorage::nameOf default case";
            }
        }

        ContiguousReadStorage(ReadId_t nSequences) : ContiguousReadStorage(nSequences, false){}

        ContiguousReadStorage(ReadId_t nSequences, bool b) : ContiguousReadStorage(nSequences, b, 0, {}){
        }

        ContiguousReadStorage(ReadId_t nSequences, bool b, int maximum_sequence_length, const std::vector<int>& deviceIds)
            : max_sequence_length(maximum_sequence_length),
                max_sequence_bytes(Sequence_t::getNumBytes(maximum_sequence_length)),
                useQualityScores(b),
                num_sequences(nSequences),
                deviceIds(deviceIds){

            constexpr bool allowUVM = true;
            constexpr float maxPercentOfTotalGPUMem = 0.8;

            sequence_data_bytes = sizeof(char) * std::size_t(num_sequences) * max_sequence_bytes;
            sequence_lengths_bytes = sizeof(Length_t) * std::size_t(num_sequences);
            if(useQualityScores){
                quality_data_bytes = sizeof(char) * std::size_t(num_sequences) * max_sequence_length;
            }

            int oldId;
            cudaGetDevice(&oldId); CUERR;

            bool everyDeviceSupportsUVM = deviceIds.size() > 0
                            && std::all_of(deviceIds.begin(), deviceIds.end(), [](int deviceId){
                                    cudaSetDevice(deviceId); CUERR;
                                    cudaDeviceProp prop;
                                    cudaGetDeviceProperties(&prop, deviceId);

                                    return prop.major >= 6; // check if compute capability >= 6.0
                                });

            const std::uint64_t requiredSequenceMem = sequence_data_bytes + sequence_lengths_bytes; //sequences and sequence lengths
    		const std::uint64_t requiredQualityMem = useQualityScores ? quality_data_bytes : 0;
    		const std::uint64_t requiredTotalMem = requiredSequenceMem + requiredQualityMem;

            bool everyDeviceCanStoreSequences = std::all_of(deviceIds.begin(), deviceIds.end(), [&](int deviceId){
                cudaSetDevice(deviceId); CUERR;

                std::size_t freeMem;
                std::size_t totalMem;
                cudaMemGetInfo(&freeMem, &totalMem); CUERR;

                bool isEnoughMemForSequences = (requiredSequenceMem < maxPercentOfTotalGPUMem * totalMem && requiredSequenceMem < freeMem);

                return isEnoughMemForSequences;
            });

            bool everyDeviceCanStoreBothSequencesAndQualities = std::all_of(deviceIds.begin(), deviceIds.end(), [&](int deviceId){
                cudaSetDevice(deviceId); CUERR;

                std::size_t freeMem;
                std::size_t totalMem;
                cudaMemGetInfo(&freeMem, &totalMem); CUERR;

                bool isEnoughMemForSequencesAndQualities = (requiredTotalMem < maxPercentOfTotalGPUMem * totalMem && requiredTotalMem < freeMem);

                return isEnoughMemForSequencesAndQualities;
            });

            cudaSetDevice(oldId);

            if(allowUVM && !everyDeviceCanStoreSequences){
                cudaMallocManaged(&h_sequence_data, sequence_data_bytes); CUERR;
                cudaMallocManaged(&h_sequence_lengths, sequence_lengths_bytes); CUERR;

                sequenceType = ContiguousReadStorage::Type::Managed;
            }else if(!allowUVM && !everyDeviceCanStoreSequences){
                cudaMallocHost(&h_sequence_data, sequence_data_bytes); CUERR;
                cudaMallocHost(&h_sequence_lengths, sequence_lengths_bytes); CUERR;

                sequenceType = ContiguousReadStorage::Type::None;
            }else{ // everyDeviceCanStoreSequences == true
                cudaMallocHost(&h_sequence_data, sequence_data_bytes); CUERR;
                cudaMallocHost(&h_sequence_lengths, sequence_lengths_bytes); CUERR;

                sequenceType = ContiguousReadStorage::Type::Full;
            }

            if(useQualityScores){
                if(allowUVM && !everyDeviceCanStoreBothSequencesAndQualities){
                    cudaMallocManaged(&h_quality_data, quality_data_bytes); CUERR;

                    qualityType = ContiguousReadStorage::Type::Managed;
                }else if(!allowUVM && !everyDeviceCanStoreBothSequencesAndQualities){
                    cudaMallocHost(&h_quality_data, quality_data_bytes); CUERR;

                    qualityType = ContiguousReadStorage::Type::None;
                }else{ // everyDeviceCanStoreBothSequencesAndQualities == true
                    cudaMallocHost(&h_quality_data, quality_data_bytes); CUERR;

                    qualityType = ContiguousReadStorage::Type::Full;
                }
            }

            std::fill(&h_sequence_data[0], &h_sequence_data[sequence_data_bytes], 0);
            std::fill(&h_sequence_lengths[0], &h_sequence_lengths[num_sequences], 0);
            std::fill(&h_quality_data[0], &h_quality_data[quality_data_bytes], 0);
        }

        ContiguousReadStorage(const ContiguousReadStorage& other) = delete;
        ContiguousReadStorage& operator=(const ContiguousReadStorage& other) = delete;

        ContiguousReadStorage(ContiguousReadStorage&& other){
            *this = std::move(other);
        }

        ContiguousReadStorage& operator=(ContiguousReadStorage&& other){
            destroy();

            h_sequence_data = other.h_sequence_data;
            h_sequence_lengths = other.h_sequence_lengths;
            h_quality_data = other.h_quality_data;
            d_sequence_data = other.d_sequence_data;
            d_sequence_lengths = other.d_sequence_lengths;
            d_quality_data = other.d_quality_data;
            max_sequence_length = other.max_sequence_length;
            max_sequence_bytes = other.max_sequence_bytes;
            useQualityScores = other.useQualityScores;
            num_sequences = other.num_sequences;
            sequence_data_bytes = other.sequence_data_bytes;
            sequence_lengths_bytes = other.sequence_lengths_bytes;
            quality_data_bytes = other.quality_data_bytes;
            sequenceType = other.sequenceType;
            qualityType = other.qualityType;
            deviceIds = std::move(other.deviceIds);
            gpuData = other.gpuData;
            hasMoved = other.hasMoved;

            other.hasMoved = true;

            return *this;
        }

        bool operator==(const ContiguousReadStorage& other){
            if(max_sequence_length != other.max_sequence_length)
                return false;
            if(max_sequence_bytes != other.max_sequence_bytes)
                return false;
            if(useQualityScores != other.useQualityScores)
                return false;
            if(num_sequences != other.num_sequences)
                return false;
            if(sequence_data_bytes != other.sequence_data_bytes)
                return false;
            if(sequence_lengths_bytes != other.sequence_lengths_bytes)
                return false;
            if(quality_data_bytes != other.quality_data_bytes)
                return false;
            if(useQualityScores != other.useQualityScores)
                return false;
            if(deviceIds != other.deviceIds)
                return false;
            if(sequenceType != other.sequenceType)
                return false;
            if(qualityType != other.qualityType)
                return false;
            if(hasMoved != other.hasMoved)
                return false;

            if(0 != std::memcmp(h_sequence_data, other.h_sequence_data, sequence_data_bytes))
                return false;
            if(0 != std::memcmp(h_sequence_lengths, other.h_sequence_lengths, sequence_lengths_bytes))
                return false;
            if(0 != std::memcmp(h_quality_data, other.h_quality_data, quality_data_bytes))
                return false;

            //don't compare gpu memory

            return true;
        }

        bool operator!=(const ContiguousReadStorage& other){
            return !(*this == other);
        }

        std::size_t size() const{

            std::size_t result = 0;
            result += sequence_data_bytes;
            result += sequence_lengths_bytes;

            if(useQualityScores){
                result += quality_data_bytes;
            }

            return result;
        }

        void resize(ReadId_t nReads){
            assert(getNumberOfSequences() >= nReads);

            num_sequences = nReads;
        }

        void destroy(){
            if(!hasMoved){
                if(sequenceType == ContiguousReadStorage::Type::Managed){
                    cudaFree(h_sequence_data); CUERR;
                    cudaFree(h_sequence_lengths); CUERR;
                }else{
                    cudaFreeHost(h_sequence_data); CUERR;
                    cudaFreeHost(h_sequence_lengths); CUERR;

                    if(sequenceType == ContiguousReadStorage::Type::Full){
                        int oldId;
                        cudaGetDevice(&oldId); CUERR;
                        for(auto& p : gpuData){
                            auto& data = p.second;
                            cudaSetDevice(data.id); CUERR;
                            cudaFree(data.d_sequence_data); CUERR;
                            cudaFree(data.d_sequence_lengths); CUERR;
                        }
                        cudaSetDevice(oldId); CUERR;
                    }
                }

                if(qualityType == ContiguousReadStorage::Type::Managed){
                    cudaFree(h_quality_data); CUERR;
                }else{
                    cudaFreeHost(h_quality_data); CUERR;

                    if(qualityType == ContiguousReadStorage::Type::Full){
                        int oldId;
                        cudaGetDevice(&oldId); CUERR;
                        for(auto& p : gpuData){
                            auto& data = p.second;
                            cudaSetDevice(data.id); CUERR;
                            cudaFree(data.d_quality_data); CUERR;
                        }
                        cudaSetDevice(oldId); CUERR;
                    }
                }
            }
        }

private:
        void insertSequence(ReadId_t readNumber, const std::string& sequence){
            Sequence_t seq(sequence);
            std::memcpy(&h_sequence_data[readNumber * max_sequence_bytes],
                        seq.begin(),
                        seq.getNumBytes());

            h_sequence_lengths[readNumber] = Length_t(sequence.length());
        }
public:
        void insertRead(ReadId_t readNumber, const std::string& sequence){
            assert(readNumber < getNumberOfSequences());
            assert(sequence.length() <= max_sequence_length);

            if(useQualityScores){
                insertRead(readNumber, sequence, std::string(sequence.length(), 'A'));
            }else{
                insertSequence(readNumber, sequence);
            }
        }

        void insertRead(ReadId_t readNumber, const std::string& sequence, const std::string& quality){
            assert(readNumber < getNumberOfSequences());
            assert(sequence.length() <= max_sequence_length);
            assert(quality.length() <= max_sequence_length);
            assert(sequence.length() == quality.length());

            insertSequence(readNumber, sequence);

            if(useQualityScores){
                std::memcpy(&h_quality_data[readNumber * max_sequence_length],
                            quality.c_str(),
                            sizeof(char) * quality.length());
            }
        }

        const char* fetchQuality2_ptr(ReadId_t readNumber) const{
            if(useQualityScores){
                return &h_quality_data[readNumber * max_sequence_length];
            }else{
                return nullptr;
            }
        }

        const std::string* fetchQuality_ptr(ReadId_t readNumber) const{
            return nullptr;
        }

       const char* fetchSequenceData_ptr(ReadId_t readNumber) const{
            return &h_sequence_data[readNumber * max_sequence_bytes];
       }

       int fetchSequenceLength(ReadId_t readNumber) const{
            return h_sequence_lengths[readNumber];
       }

        std::uint64_t getNumberOfSequences() const{
            return num_sequences;
        }

        GPUData getGPUData(int deviceId){

            auto it = std::find(deviceIds.begin(), deviceIds.end(), deviceId);
            if(it == deviceIds.end()){
                GPUData data;
                data.id = deviceId;
                return data;
            }else{
                auto datait = std::find(gpuData.begin(), gpuData.end(), deviceId);
                if(datait != gpuData.end()){
                    return *datait;
                }else{
                    int oldId;
                    cudaGetDevice(&oldId); CUERR;
                    cudaSetDevice(deviceId); CUERR;

                    GPUData data;
                    data.id = deviceId;

                    if(sequenceType == ContiguousReadStorage::Type::Managed){
                        data.d_sequence_data = h_sequence_data;
                        data.d_sequence_lengths = h_sequence_lengths;
                        data.sequenceType == ContiguousReadStorage::Type::Managed;
                    }else if(sequenceType == ContiguousReadStorage::Type::Full){
                        cudaMalloc(&data.d_sequence_data, sequence_data_bytes); CUERR;
                        cudaMalloc(&data.d_sequence_lengths, sequence_lengths_bytes); CUERR;

                        cudaMemcpy(data.d_sequence_data, h_sequence_data, sequence_data_bytes, H2D); CUERR;
                        cudaMemcpy(data.d_sequence_lengths, h_sequence_lengths, sequence_lengths_bytes, H2D); CUERR;

                        data.sequenceType == ContiguousReadStorage::Type::Full;
                    }

                    if(qualityType == ContiguousReadStorage::Type::Managed){
                        data.d_quality_data = h_quality_data;
                        data.qualityType == ContiguousReadStorage::Type::Managed;
                    }else if(sequenceType == ContiguousReadStorage::Type::Full){
                        cudaMalloc(&data.d_quality_data, quality_data_bytes); CUERR;

                        cudaMemcpy(data.d_quality_data, h_quality_data, quality_data_bytes, H2D); CUERR;

                        data.qualityType == ContiguousReadStorage::Type::Full;
                    }

                    cudaSetDevice(oldId); CUERR;

                    gpuData[deviceId] = data;

                    return data;
                }
            }

        }

        void saveToFile(const std::string& filename) const{
            std::cout << "gpu::ContiguousReadStorage::saveToFile is not implemented yet!" << std::endl;
            /*std::ofstream stream(filename, std::ios::binary);

            auto writesequence = [&](const Sequence_t& seq){
                const int length = seq.length();
                const int bytes = seq.getNumBytes();
                stream.write(reinterpret_cast<const char*>(&length), sizeof(int));
                stream.write(reinterpret_cast<const char*>(&bytes), sizeof(int));
                stream.write(reinterpret_cast<const char*>(seq.begin()), bytes);
            };

            auto writequality = [&](const std::string& qual){
                const std::size_t bytes = qual.length();
                stream.write(reinterpret_cast<const char*>(&bytes), sizeof(int));
                stream.write(reinterpret_cast<const char*>(qual.c_str()), bytes);
            };

            std::size_t numReads = sequences.size();
            stream.write(reinterpret_cast<const char*>(&numReads), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(&useQualityScores), sizeof(bool));

            for(std::size_t i = 0; i < numReads; i++){
                writesequence(sequences[i]);
                if(useQualityScores)
                    writequality(qualityscores[i]);
            }*/
        }

        void loadFromFile(const std::string& filename){
            throw std::runtime_error("gpu::ContiguousReadStorage::loadFromFile is not implemented yet!");
            /*std::ifstream stream(filename);
            if(!stream)
                throw std::runtime_error("cannot load binary sequences from file " + filename);

            auto readsequence = [&](){
                int length;
                int bytes;
                stream.read(reinterpret_cast<char*>(&length), sizeof(int));
                stream.read(reinterpret_cast<char*>(&bytes), sizeof(int));

                auto data = std::make_unique<std::uint8_t[]>(bytes);
                stream.read(reinterpret_cast<char*>(data.get()), bytes);

                return Sequence_t{data.get(), length};
            };

            auto readquality = [&](){
                static_assert(sizeof(char) == 1);

                std::size_t bytes;
                stream.read(reinterpret_cast<char*>(&bytes), sizeof(int));
                auto data = std::make_unique<char[]>(bytes);

                stream.read(reinterpret_cast<char*>(data.get()), bytes);

                return std::string{data.get(), bytes};
            };

            std::size_t numReads;
            stream.read(reinterpret_cast<char*>(&numReads), sizeof(std::size_t));
            stream.read(reinterpret_cast<char*>(&useQualityScores), sizeof(bool));

            if(numReads > getNumberOfSequences()){
                throw std::runtime_error("Readstorage was constructed for "
                                        + std::to_string(getNumberOfSequences())
                                        + " sequences, but binary file contains "
                                        + std::to_string(numReads) + " sequences!");
            }

            sequences.clear();
            sequences.reserve(numReads);
            if(useQualityScores){
                qualityscores.clear();
                qualityscores.reserve(numReads);
            }

            for(std::size_t i = 0; i < numReads; i++){
                sequences.emplace_back(readsequence());
                if(useQualityScores)
                    qualityscores.emplace_back(readquality());
            }*/
        }

        SequenceStatistics getSequenceStatistics() const{
            return getSequenceStatistics(1);
        }

        SequenceStatistics getSequenceStatistics(int numThreads) const{
            int maxSequenceLength = 0;
            int minSequenceLength = std::numeric_limits<int>::max();

            const int oldnumthreads = omp_get_thread_num();

            omp_set_num_threads(numThreads);

            #pragma omp parallel for reduction(max:maxSequenceLength) reduction(min:minSequenceLength)
            for(std::size_t readId = 0; readId < getNumberOfSequences(); readId++){
                const int len = fetchSequenceLength(readId);
                if(len > maxSequenceLength)
                    maxSequenceLength = len;
                if(len < minSequenceLength)
                    minSequenceLength = len;
            }

            omp_set_num_threads(oldnumthreads);

            SequenceStatistics stats;
            stats.minSequenceLength = minSequenceLength;
            stats.maxSequenceLength = maxSequenceLength;

            return stats;
        }

    };

#endif

}
}

#endif
