#ifndef READ_STORAGE_HPP
#define READ_STORAGE_HPP

#include <config.hpp>
#include <sequence.hpp>
#include <lengthstorage.hpp>
#include <memorymanagement.hpp>

#include <threadpool.hpp>
#include <util.hpp>

#include <algorithm>
#include <limits>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>
#include <omp.h>
#include <map>
#include <fstream>
#include <memory>
#include <cstring>
#include <mutex>
#include <atomic>


namespace care{

namespace cpu{

    struct ContiguousReadStorage{

        struct Statistics{
            int maximumSequenceLength = 0;
            int minimumSequenceLength = std::numeric_limits<int>::max();

            bool operator==(const Statistics& rhs) const noexcept {
                return maximumSequenceLength == rhs.maximumSequenceLength
                    && minimumSequenceLength == rhs.minimumSequenceLength;
            }
            bool operator!=(const Statistics& rhs) const noexcept{
                return !(operator==(rhs));
            }
        };

        struct GatherHandle{
            std::vector<int> permutation;
        };

        enum class GatherType {Direct, Sorted};

        static constexpr bool has_reverse_complement = false;
        static constexpr int serialization_id = 1;

        using Length_t = int;
        using LengthStore_t = LengthStore<std::uint32_t>;

        std::unique_ptr<unsigned int[]> h_sequence_data = nullptr;
        std::unique_ptr<char[]> h_quality_data = nullptr;
        int sequenceLengthLowerBound = 0;
        int sequenceLengthUpperBound = 0;
        int sequenceDataPitchInInts = 0;
        int sequenceQualitiesPitchInBytes = 0;
        bool useQualityScores = false;
        read_number maximumNumberOfSequences = 0;
        std::size_t sequence_data_bytes = 0;
        std::size_t quality_data_bytes = 0;
        std::vector<read_number> readIdsOfReadsWithUndeterminedBase; //sorted in ascending order
        std::mutex mutexUndeterminedBaseReads;
        Statistics statistics;
        std::atomic<read_number> numberOfInsertedReads{0};
        LengthStore_t lengthStorage;



        ContiguousReadStorage() : ContiguousReadStorage(0){}

        ContiguousReadStorage(read_number nSequences) : ContiguousReadStorage(nSequences, false){}

        ContiguousReadStorage(read_number nSequences, bool b) : ContiguousReadStorage(nSequences, b, 0, 0){
        }

        ContiguousReadStorage(read_number nSequences, bool b, int minimum_sequence_length, int maximum_sequence_length){
            init(nSequences, b, minimum_sequence_length, maximum_sequence_length);
        }

        void init(read_number nSequences, bool b, int minimum_sequence_length, int maximum_sequence_length){
            sequenceLengthLowerBound = minimum_sequence_length,
            sequenceLengthUpperBound = maximum_sequence_length,
            sequenceDataPitchInInts = getEncodedNumInts2Bit(maximum_sequence_length),
            sequenceQualitiesPitchInBytes = maximum_sequence_length,
            useQualityScores = b,
            maximumNumberOfSequences = nSequences;

            lengthStorage = std::move(LengthStore_t(sequenceLengthLowerBound, sequenceLengthUpperBound, nSequences));

            h_sequence_data.reset(new unsigned int[std::size_t(maximumNumberOfSequences) * sequenceDataPitchInInts]);
            sequence_data_bytes = sizeof(unsigned int) * std::size_t(maximumNumberOfSequences) * sequenceDataPitchInInts;

            if(useQualityScores){
                h_quality_data.reset(new char[std::size_t(maximumNumberOfSequences) * sequenceQualitiesPitchInBytes]);
                quality_data_bytes = sizeof(char) * std::size_t(maximumNumberOfSequences) * sequenceQualitiesPitchInBytes;
            }

            std::fill_n(&h_sequence_data[0], std::size_t(maximumNumberOfSequences) * sequenceDataPitchInInts, 0);
            std::fill(&h_quality_data[0], &h_quality_data[quality_data_bytes], 0);
        }

        ContiguousReadStorage(const ContiguousReadStorage& other) = delete;
        ContiguousReadStorage& operator=(const ContiguousReadStorage& other) = delete;

        ContiguousReadStorage(ContiguousReadStorage&& other)
            : h_sequence_data(std::move(other.h_sequence_data)),
              h_quality_data(std::move(other.h_quality_data)),
              sequenceLengthLowerBound(other.sequenceLengthLowerBound),
              sequenceLengthUpperBound(other.sequenceLengthUpperBound),
              sequenceDataPitchInInts(other.sequenceDataPitchInInts),
              sequenceQualitiesPitchInBytes(other.sequenceQualitiesPitchInBytes),
              useQualityScores(other.useQualityScores),
              maximumNumberOfSequences(other.maximumNumberOfSequences),
              sequence_data_bytes(other.sequence_data_bytes),
              quality_data_bytes(other.quality_data_bytes),
              readIdsOfReadsWithUndeterminedBase(std::move(other.readIdsOfReadsWithUndeterminedBase)),
              statistics(std::move(other.statistics)),
              numberOfInsertedReads(other.numberOfInsertedReads.load()),
              lengthStorage(std::move(other.lengthStorage)){

            other.numberOfInsertedReads = 0;
            other.statistics = Statistics{};

        }

        ContiguousReadStorage& operator=(ContiguousReadStorage&& other){
            h_sequence_data = std::move(other.h_sequence_data);
            h_quality_data = std::move(other.h_quality_data);
            sequenceLengthLowerBound = other.sequenceLengthLowerBound;
            sequenceLengthUpperBound = other.sequenceLengthUpperBound;
            sequenceDataPitchInInts = other.sequenceDataPitchInInts;
            sequenceQualitiesPitchInBytes = other.sequenceQualitiesPitchInBytes;
            useQualityScores = other.useQualityScores;
            maximumNumberOfSequences = other.maximumNumberOfSequences;
            sequence_data_bytes = other.sequence_data_bytes;
            quality_data_bytes = other.quality_data_bytes;
            readIdsOfReadsWithUndeterminedBase = std::move(other.readIdsOfReadsWithUndeterminedBase);
            statistics = std::move(other.statistics);
            numberOfInsertedReads = other.numberOfInsertedReads.load();
            lengthStorage = std::move(other.lengthStorage);

            other.numberOfInsertedReads = 0;
            other.statistics = Statistics{};

            return *this;
        }

        bool operator==(const ContiguousReadStorage& other) const{
            if(sequenceLengthLowerBound != other.sequenceLengthLowerBound)
                return false;
            if(sequenceLengthUpperBound != other.sequenceLengthUpperBound)
                return false;
            if(sequenceDataPitchInInts != other.sequenceDataPitchInInts)
                return false;
            if(sequenceQualitiesPitchInBytes != other.sequenceQualitiesPitchInBytes)
                return false;
            if(useQualityScores != other.useQualityScores)
                return false;
            if(maximumNumberOfSequences != other.maximumNumberOfSequences)
                return false;
            if(useQualityScores != other.useQualityScores)
                return false;
            if(sequence_data_bytes != other.sequence_data_bytes)
                return false;
            if(quality_data_bytes != other.quality_data_bytes)
                return false;
            if(readIdsOfReadsWithUndeterminedBase != other.readIdsOfReadsWithUndeterminedBase){
                return false;
            }
            if(statistics != other.statistics){
                return false;
            }

            if(lengthStorage != other.lengthStorage){
                return false;
            }

            if(0 != std::memcmp(h_sequence_data.get(), other.h_sequence_data.get(), sequence_data_bytes))
                return false;
            if(0 != std::memcmp(h_quality_data.get(), other.h_quality_data.get(), quality_data_bytes))
                return false;

            return true;
        }

        bool operator!=(const ContiguousReadStorage& other) const{
            return !(*this == other);
        }

        void construct(
            std::vector<std::string> inputfiles,
            bool useQualityScores,
            read_number expectedNumberOfReads,
            int expectedMinimumReadLength,
            int expectedMaximumReadLength,
            int threads,
            bool showProgress
        ){
            constexpr std::array<char, 4> bases = {'A', 'C', 'G', 'T'};

            auto checkRead = [&, this](read_number readIndex, Read& read, int& Ncount){
                const int readLength = int(read.sequence.size());

                if(readIndex >= expectedNumberOfReads){
                    throw std::runtime_error("Error! Expected " + std::to_string(expectedNumberOfReads)
                                            + " reads, but file contains at least "
                                            + std::to_string(readIndex+1) + " reads.");
                }

                if(readLength > expectedMaximumReadLength){
                    throw std::runtime_error("Error! Expected maximum read length = "
                                            + std::to_string(expectedMaximumReadLength)
                                            + ", but read " + std::to_string(readIndex)
                                            + "has length " + std::to_string(readLength));
                }

                auto isValidBase = [](char c){
                    constexpr std::array<char, 10> validBases{'A','C','G','T','a','c','g','t'};
                    return validBases.end() != std::find(validBases.begin(), validBases.end(), c);
                };

                const int undeterminedBasesInRead = std::count_if(read.sequence.begin(), read.sequence.end(), [&](char c){
                    return !isValidBase(c);
                });

                //nmap[undeterminedBasesInRead]++;

                if(undeterminedBasesInRead > 0){
                    setReadContainsN(readIndex, true);
                }

                for(auto& c : read.sequence){
                    if(c == 'a') c = 'A';
                    else if(c == 'c') c = 'C';
                    else if(c == 'g') c = 'G';
                    else if(c == 't') c = 'T';
                    else if(!isValidBase(c)){
                        c = bases[Ncount];
                        Ncount = (Ncount + 1) % 4;
                    }
                }
            };

            constexpr int maxbuffersize = 65536;
            constexpr int numBuffers = 3;

            std::array<std::vector<read_number>, numBuffers> indicesBuffers;
            std::array<std::vector<Read>, numBuffers> readsBuffers;
            std::array<int, numBuffers> bufferSizes;
            std::array<bool, numBuffers> canBeUsed;
            std::array<std::mutex, numBuffers> mutex;
            std::array<std::condition_variable, numBuffers> cv;

            //using Inserter_t = decltype(makeReadInserter());

            //std::array<Inserter_t, 3> inserterFuncs{makeReadInserter(), makeReadInserter(), makeReadInserter()};

            ThreadPool threadPool(numBuffers);
            ThreadPool pforPool(std::max(1, threads - numBuffers));

            for(int i = 0; i < numBuffers; i++){
                indicesBuffers[i].resize(maxbuffersize);
                readsBuffers[i].resize(maxbuffersize);
                canBeUsed[i] = true;
                bufferSizes[i] = 0;

            }

            int bufferindex = 0;
            read_number globalReadId = 0;

            auto showProgressFunc = [show = showProgress](auto totalCount, auto seconds){
                if(show){
                    std::cout << "Processed " << totalCount << " reads in file. Elapsed time: " 
                                    << seconds << " seconds." << std::endl;
                }
            };

            auto updateShowProgressInterval = [](auto duration){
                return duration * 2;
            };

            ProgressThread<read_number> progressThread(
                expectedNumberOfReads, 
                showProgressFunc, 
                updateShowProgressInterval
            );

            for(const auto& inputfile : inputfiles){
                std::cout << "Converting reads of file " << inputfile << ", storing them in memory\n";

                forEachReadInFile(inputfile,
                                [&](auto /*readnum*/, auto& read){

                        if(!canBeUsed[bufferindex]){
                            std::unique_lock<std::mutex> ul(mutex[bufferindex]);
                            if(!canBeUsed[bufferindex]){
                                //std::cerr << "waiting for other buffer\n";
                                //nvtx::push_range("wait for doublebuffer", 0);
                                cv[bufferindex].wait(ul, [&](){ return canBeUsed[bufferindex]; });
                                //nvtx::pop_range();
                            }
                        }

                        auto indicesBufferPtr = &indicesBuffers[bufferindex];
                        auto readsBufferPtr = &readsBuffers[bufferindex];
                        auto& buffersize = bufferSizes[bufferindex];

                        (*indicesBufferPtr)[buffersize] = globalReadId;
                        std::swap((*readsBufferPtr)[buffersize], read);

                        ++buffersize;

                        ++globalReadId;

                        progressThread.addProgress(1);

                        if(buffersize >= maxbuffersize){
                            canBeUsed[bufferindex] = false;

                            //std::cerr << "launch other thread\n";
                            //nvtx::push_range("enqeue", 0);
                            threadPool.enqueue([&, indicesBufferPtr, readsBufferPtr, bufferindex](){
                                //std::cerr << "buffer " << bufferindex << " running\n";
                                const int buffersize = bufferSizes[bufferindex];

                                //nvtx::push_range("process read batch", 0);
                                int nmodcounter = 0;

                                //nvtx::push_range("check read batch", 0);
                                for(int i = 0; i < buffersize; i++){
                                    read_number readId = (*indicesBufferPtr)[i];
                                    auto& read = (*readsBufferPtr)[i];
                                    checkRead(readId, read, nmodcounter);
                                }
                                //nvtx::pop_range();

                                //nvtx::push_range("insert read batch", 0);
                                setReads(
                                    &pforPool, 
                                    indicesBufferPtr->data(), 
                                    readsBufferPtr->data(), 
                                    buffersize
                                );
                                //nvtx::pop_range();

                                //TIMERSTARTCPU(clear);
                                bufferSizes[bufferindex] = 0;
                                //TIMERSTOPCPU(clear);
                                
                                std::lock_guard<std::mutex> l(mutex[bufferindex]);
                                canBeUsed[bufferindex] = true;
                                cv[bufferindex].notify_one();

                                //nvtx::pop_range();

                                //std::cerr << "buffer " << bufferindex << " finished\n";
                            });

                            bufferindex = (bufferindex + 1) % numBuffers; //swap buffers

                            //nvtx::pop_range();
                        }

                });
            }

            auto indicesBufferPtr = &indicesBuffers[bufferindex];
            auto readsBufferPtr = &readsBuffers[bufferindex];
            auto& buffersize = bufferSizes[bufferindex];

            if(buffersize > 0){
                if(!canBeUsed[bufferindex]){
                    std::unique_lock<std::mutex> ul(mutex[bufferindex]);
                    if(!canBeUsed[bufferindex]){
                        //std::cerr << "waiting for other buffer\n";
                        cv[bufferindex].wait(ul, [&](){ return canBeUsed[bufferindex]; });
                    }
                }

                int nmodcounter = 0;

                for(int i = 0; i < buffersize; i++){
                    read_number readId = (*indicesBufferPtr)[i];
                    auto& read = (*readsBufferPtr)[i];
                    checkRead(readId, read, nmodcounter);
                }

                setReads(&pforPool, indicesBufferPtr->data(), readsBufferPtr->data(), buffersize);

                buffersize = 0;
            }

            for(int i = 0; i < numBuffers; i++){
                std::unique_lock<std::mutex> ul(mutex[i]);
                if(!canBeUsed[i]){
                    //std::cerr << "Reading file completed. Waiting for buffer " << i << "\n";
                    cv[i].wait(ul, [&](){ return canBeUsed[i]; });
                }
            }

            progressThread.finished();

            // std::cerr << "occurences of n/N:\n";
            // for(const auto& p : nmap){
            //     std::cerr << p.first << " " << p.second << '\n';
            // }

            //constructionIsComplete();
        }

        std::size_t size() const{
            //assert(std::size_t(maximumNumberOfSequences) * maximum_allowed_sequence_bytes == sequence_data_bytes);

            std::size_t result = 0;
            result += sequence_data_bytes;
            result += lengthStorage.getRawSizeInBytes();

            if(useQualityScores){
                //assert(std::size_t(maximumNumberOfSequences) * sequenceLengthUpperBound * sizeof(char) == quality_data_bytes);
                result += quality_data_bytes;
            }

            return result;
        }

    	void resize(read_number nReads){
    		assert(getMaximumNumberOfSequences() >= nReads);

            maximumNumberOfSequences = nReads;
    	}

    	void destroy(){
            h_sequence_data.reset();
            h_quality_data.reset();
    	}

        Statistics getStatistics() const{
            return statistics;
        }

        read_number getNumberOfReads() const{
            return numberOfInsertedReads;
        }

        void setReadContainsN(read_number readId, bool contains){

            std::lock_guard<std::mutex> l(mutexUndeterminedBaseReads);

            auto pos = std::lower_bound(readIdsOfReadsWithUndeterminedBase.begin(),
                                                readIdsOfReadsWithUndeterminedBase.end(),
                                                readId);

            if(contains){
                if(pos != readIdsOfReadsWithUndeterminedBase.end()){
                    ; //already marked
                }else{                    
                    readIdsOfReadsWithUndeterminedBase.insert(pos, readId);
                }
            }else{
                if(pos != readIdsOfReadsWithUndeterminedBase.end()){
                    //remove mark
                    readIdsOfReadsWithUndeterminedBase.erase(pos);
                }else{
                    ; //already unmarked
                }
            }
        }

        bool readContainsN(read_number readId) const{

            auto pos = std::lower_bound(readIdsOfReadsWithUndeterminedBase.begin(),
                                                readIdsOfReadsWithUndeterminedBase.end(),
                                                readId);
            bool b2 = readIdsOfReadsWithUndeterminedBase.end() != pos && *pos == readId;

            return b2;
        }

        std::int64_t getNumberOfReadsWithN() const{
            return readIdsOfReadsWithUndeterminedBase.size();
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage info;
            info.host = sequence_data_bytes;
            info.host += lengthStorage.getRawSizeInBytes();
            info.host += sizeof(read_number) * readIdsOfReadsWithUndeterminedBase.capacity();

            if(useQualityScores){
                info.host += quality_data_bytes;
            }         

            return info;
        }

private:
        void insertSequence(read_number readNumber, const std::string& sequence){
            auto identity = [](auto i){return i;};

            const int sequencelength = sequence.length();

            unsigned int* dest = &h_sequence_data[std::size_t(readNumber) * sequenceDataPitchInInts];
            encodeSequence2Bit(
                dest,
                sequence.c_str(),
                sequence.length(),
                identity
            );

            statistics.minimumSequenceLength = std::min(statistics.minimumSequenceLength, sequencelength);
            statistics.maximumSequenceLength = std::max(statistics.maximumSequenceLength, sequencelength);

            lengthStorage.setLength(readNumber, Length_t(sequence.length()));

            read_number prev_value = numberOfInsertedReads;
            while(prev_value < readNumber+1 && !numberOfInsertedReads.compare_exchange_weak(prev_value, readNumber+1)){
                ;
            }
        }
public:

        void setReads(
            ThreadPool* threadPool, 
            const read_number* indices, 
            const Read* reads, 
            int numReads
        ){
            if(numReads == 0) return;
            
            //TIMERSTARTCPU(internalinit);
            auto lengthInRange = [&](Length_t length){
                return getSequenceLengthLowerBound() <= length && length <= getSequenceLengthUpperBound();
            };
            assert(numReads > 0);
            assert(std::all_of(indices, indices + numReads, [&](auto i){ return i < getMaximumNumberOfSequences();}));
            assert(std::all_of(reads, reads + numReads, [&](const auto& r){ return lengthInRange(Length_t(r.sequence.length()));}));
            
            if(canUseQualityScores()){
                assert(std::all_of(reads, reads + numReads, [&](const auto& r){ return r.sequence.length() == r.quality.length();}));
            }

            auto minmax = std::minmax_element(reads, reads + numReads, [](const auto& r1, const auto& r2){
                return r1.sequence.length() < r2.sequence.length();
            });

            statistics.minimumSequenceLength = std::min(statistics.minimumSequenceLength, int(minmax.first->sequence.length()));
            statistics.maximumSequenceLength = std::max(statistics.maximumSequenceLength, int(minmax.second->sequence.length()));

            read_number maxIndex = *std::max_element(indices, indices + numReads);

            read_number prev_value = numberOfInsertedReads;
            while(prev_value < maxIndex+1 && !numberOfInsertedReads.compare_exchange_weak(prev_value, maxIndex+1)){
                ;
            }


            ThreadPool::ParallelForHandle pforHandle;

            auto body = [&](auto begin, auto end, int /*threadid*/){
                for(auto i = begin; i < end; i++){
                    const auto& myRead = reads[i];
                    const read_number myReadNumber = indices[i];

                    insertRead(myReadNumber, myRead.sequence, myRead.quality);
                }
            };

            threadPool->parallelFor(pforHandle, 0, numReads, body);

        }


        void insertRead(read_number readNumber, const std::string& sequence){
            assert(readNumber < getMaximumNumberOfSequences());
            assert(int(sequence.length()) <= sequenceLengthUpperBound);

    		if(useQualityScores){
    			insertRead(readNumber, sequence, std::string(sequence.length(), 'A'));
    		}else{
    			insertSequence(readNumber, sequence);
    		}
    	}

        void insertRead(read_number readNumber, const std::string& sequence, const std::string& quality){
            assert(readNumber < getMaximumNumberOfSequences());
            assert(int(sequence.length()) <= sequenceLengthUpperBound);
            
            

    		insertSequence(readNumber, sequence);

    		if(useQualityScores){
                assert(int(quality.length()) <= sequenceLengthUpperBound);
                assert(sequence.length() == quality.length());
                
                std::memcpy(&h_quality_data[std::size_t(readNumber) * std::size_t(sequenceLengthUpperBound)],
                            quality.c_str(),
                            sizeof(char) * quality.length());
    		}
    	}

        const char* fetchQuality_ptr(read_number readNumber) const{
            if(useQualityScores){
                return &h_quality_data[std::size_t(readNumber) * std::size_t(sequenceLengthUpperBound)];
            }else{
                return nullptr;
            }
        }

        const unsigned int* fetchSequenceData_ptr(read_number readNumber) const{
        	return &h_sequence_data[std::size_t(readNumber) * std::size_t(sequenceDataPitchInInts)];
        }

        int fetchSequenceLength(read_number readNumber) const{
            return lengthStorage.getLength(readNumber);
        }

        template<class T, class GatherType>
        void gatherImpl(
                GatherHandle& handle,
                GatherType gatherType,
                const T* source,
                size_t sourcePitchElements,
                const read_number* readIds,
                int numReadIds,
                T* destination,
                size_t destinationPitchElements) const noexcept{
            
            if(numReadIds == 0){
                return;
            }

            if(gatherType == GatherType::Sorted){
                handle.permutation.resize(numReadIds);
                //handle.data.resize(sourcePitchElement * sizeof(T) * numReadIds);

                std::iota(
                    handle.permutation.begin(), 
                    handle.permutation.end(),
                    0
                );

                std::sort(
                    handle.permutation.begin(), 
                    handle.permutation.end(),
                    [&](const auto& l, const auto& r){
                        return readIds[l] < readIds[r];
                    }
                );
            }

            constexpr int prefetch_distance = 4;

            for(int i = 0; i < numReadIds && i < prefetch_distance; ++i) {
                const int index = gatherType == GatherType::Sorted ? handle.permutation[i] : i;
                const read_number nextReadId = readIds[index];
                const T* const nextData = source + sourcePitchElements * nextReadId;
                __builtin_prefetch(nextData, 0, 0);
            }

            for(int i = 0; i < numReadIds; i++){
                if(i + prefetch_distance < numReadIds) {
                    const int index = gatherType == GatherType::Sorted ? handle.permutation[i + prefetch_distance] : i + prefetch_distance;
                    const read_number nextReadId = readIds[index];
                    const T* const nextData = source + sourcePitchElements * nextReadId;
                    __builtin_prefetch(nextData, 0, 0);
                }

                const int index = gatherType == GatherType::Sorted ? handle.permutation[i] : i;
                const read_number readId = readIds[index];
                const T* const data = source + sourcePitchElements * readId;

                T* const destData = destination + destinationPitchElements * index;
                std::copy_n(data, sourcePitchElements, destData);
            }
        }


        void gatherSequenceData(
                GatherHandle& handle,
                const read_number* readIds,
                int numReadIds,
                unsigned int* destination,
                int destinationPitchElements) const noexcept{

            gatherImpl(
                handle,
                GatherType::Direct,
                &h_sequence_data[0],
                sequenceDataPitchInInts,
                readIds,
                numReadIds,
                destination,
                destinationPitchElements
            );
        }

        void gatherSequenceDataSpecial(
                GatherHandle& handle,
                const read_number* readIds,
                int numReadIds,
                unsigned int* destination,
                int destinationPitchElements) const noexcept{

            gatherImpl(
                handle,
                GatherType::Sorted,
                &h_sequence_data[0],
                sequenceDataPitchInInts,
                readIds,
                numReadIds,
                destination,
                destinationPitchElements
            );
        }

        void gatherSequenceQualities(
                GatherHandle& handle,
                const read_number* readIds,
                int numReadIds,
                char* destination,
                int destinationPitchElements) const noexcept{

            gatherImpl(
                handle,
                GatherType::Direct,
                &h_quality_data[0],
                sequenceQualitiesPitchInBytes,
                readIds,
                numReadIds,
                destination,
                destinationPitchElements
            );
        }

        void gatherSequenceQualitiesSpecial(
                GatherHandle& handle,
                const read_number* readIds,
                int numReadIds,
                char* destination,
                int destinationPitchElements) const noexcept{

            gatherImpl(
                handle,
                GatherType::Sorted,
                &h_quality_data[0],
                sequenceQualitiesPitchInBytes,
                readIds,
                numReadIds,
                destination,
                destinationPitchElements
            );
        }

        void gatherSequenceLengths(
                GatherHandle& handle,
                const read_number* readIds,
                int numReadIds,
                int* destination,
                int destinationPitchElements = 1) const noexcept{

            for(int i = 0; i < numReadIds; i++){
                int* const destLength = destination + i * destinationPitchElements;
                *destLength = fetchSequenceLength(readIds[i]);
            }            
        }

        void gatherSequenceLengthsSpecial(
                GatherHandle& handle,
                const read_number* readIds,
                int numReadIds,
                int* destination,
                int destinationPitchElements = 1) const noexcept{

            gatherSequenceLengths(
                handle,
                readIds,
                numReadIds,
                destination,
                destinationPitchElements
            );
        }





        bool canUseQualityScores() const{
            return useQualityScores;
        }

    	std::uint64_t getMaximumNumberOfSequences() const{
    		return maximumNumberOfSequences;
    	}

        int getMaximumAllowedSequenceBytes() const{
            return sequenceDataPitchInInts * sizeof(unsigned int);
        }

        int getSequenceLengthLowerBound() const{
            return sequenceLengthLowerBound;
        }

        int getSequenceLengthUpperBound() const{
            return sequenceLengthUpperBound;
        }

        void saveToFile(const std::string& filename) const{
            std::ofstream stream(filename, std::ios::binary);

            read_number inserted = getNumberOfReads();

            stream.write(reinterpret_cast<const char*>(&inserted), sizeof(read_number));
            stream.write(reinterpret_cast<const char*>(&sequenceLengthUpperBound), sizeof(int));
            stream.write(reinterpret_cast<const char*>(&sequenceLengthLowerBound), sizeof(int));
            stream.write(reinterpret_cast<const char*>(&useQualityScores), sizeof(bool));
            stream.write(reinterpret_cast<const char*>(&statistics), sizeof(Statistics));

            lengthStorage.writeToStream(stream);

            stream.write(reinterpret_cast<const char*>(&sequence_data_bytes), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(&h_sequence_data[0]), sequence_data_bytes); 
            stream.write(reinterpret_cast<const char*>(&quality_data_bytes), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(&h_quality_data[0]), quality_data_bytes);

            std::size_t numUndeterminedReads = readIdsOfReadsWithUndeterminedBase.size();
            stream.write(reinterpret_cast<const char*>(&numUndeterminedReads), sizeof(size_t));
            stream.write(reinterpret_cast<const char*>(readIdsOfReadsWithUndeterminedBase.data()), numUndeterminedReads * sizeof(read_number));

        }

        void loadFromFile(const std::string& filename){
            std::ifstream stream(filename, std::ios::binary);
            if(!stream)
                throw std::runtime_error("Cannot open file " + filename);

            destroy();

            read_number loaded_inserted = 0;
            int loaded_sequenceLengthLowerBound = 0;
            int loaded_sequenceLengthUpperBound = 0;
            bool loaded_useQualityScores = false;

            std::size_t loaded_sequence_data_bytes = 0;
            std::size_t loaded_quality_data_bytes = 0;            

            stream.read(reinterpret_cast<char*>(&loaded_inserted), sizeof(read_number));
            stream.read(reinterpret_cast<char*>(&loaded_sequenceLengthLowerBound), sizeof(int));
            stream.read(reinterpret_cast<char*>(&loaded_sequenceLengthUpperBound), sizeof(int));            
            stream.read(reinterpret_cast<char*>(&loaded_useQualityScores), sizeof(bool));

            init(loaded_inserted, loaded_useQualityScores, loaded_sequenceLengthLowerBound, loaded_sequenceLengthUpperBound);

            numberOfInsertedReads = loaded_inserted;

            stream.read(reinterpret_cast<char*>(&statistics), sizeof(Statistics));

            lengthStorage.readFromStream(stream);

            stream.read(reinterpret_cast<char*>(&loaded_sequence_data_bytes), sizeof(std::size_t));
            stream.read(reinterpret_cast<char*>(&h_sequence_data[0]), loaded_sequence_data_bytes);
            stream.read(reinterpret_cast<char*>(&loaded_quality_data_bytes), sizeof(std::size_t));            
            stream.read(reinterpret_cast<char*>(&h_quality_data[0]), loaded_quality_data_bytes);

            std::size_t numUndeterminedReads = 0;
            stream.read(reinterpret_cast<char*>(&numUndeterminedReads), sizeof(std::size_t));
            readIdsOfReadsWithUndeterminedBase.resize(numUndeterminedReads);
            stream.read(reinterpret_cast<char*>(readIdsOfReadsWithUndeterminedBase.data()), numUndeterminedReads * sizeof(read_number));
        }
    };

}



}

#endif
