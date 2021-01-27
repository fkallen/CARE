#ifndef DYNAMIC_2D_ARRAY_HPP
#define DYNAMIC_2D_ARRAY_HPP

#include <vector>
#include <cstddef>
#include <cassert>
#include <algorithm>
#include <functional>
#include <numeric>


template<class T>
class Dynamic2dArray{
public:
    using value_type = T;

    Dynamic2dArray(){
        offsetsPrefixSum.emplace_back(0);
    }

    void appendRow(const T* rowData, int length){
        assert(!isShaped);
        
        if(chunkArray.getNumChunks() == 0){
            chunkArray.addChunk(getMemoryLimitChunk());
            offsetsPrefixSum.emplace_back(0);
        }

        bool couldBeAppended = chunkArray.appendRow(chunkArray.getNumChunks() - 1, rowData, length);
        if(!couldBeAppended){
            chunkArray.addChunk(getMemoryLimitChunk());
            couldBeAppended = chunkArray.appendRow(chunkArray.getNumChunks() - 1, rowData, length);
            offsetsPrefixSum.emplace_back(offsetsPrefixSum.back() + 1);

            assert(couldBeAppended);
        }else{
            offsetsPrefixSum[chunkArray.getNumChunks()]++;
        }
    }

    void reshapeRows(std::size_t rowpitchelements){
        auto deallocVector = [](auto& vec){
            using W = typename std::remove_reference<decltype(vec)>::type;
            W tmp{};
            vec.swap(tmp);
        };

        if(!isShaped){

            for(std::size_t i = 0; i < chunkArray.getNumChunks(); i++){
                if(chunkArray.numRows[i] > 0){
                    const std::size_t maxLength = *std::max_element(
                        chunkArray.rowLengths[i].begin(),
                        chunkArray.rowLengths[i].end(),
                        std::less<int>{}
                    );

                    std::size_t newpitchelements = std::max(rowpitchelements, maxLength);

                    ShapedChunk newChunk{};
                    newChunk.numRows = chunkArray.numRows[i];
                    newChunk.rowPitchElements = newpitchelements;
                    newChunk.data.resize(chunkArray.numRows[i] * newpitchelements);

                    auto inputiter = chunkArray.data[i].begin();
                    auto outputiter = newChunk.data.begin();
                    for(std::size_t row = 0; row < chunkArray.numRows[i]; row++){
                        const int len = chunkArray.rowLengths[i][row];
                        std::copy_n(inputiter, len, outputiter);
                        outputiter += newpitchelements;
                        inputiter += len;
                    }

                    shapedChunkArray.emplace_back(std::move(newChunk));

                    deallocVector(chunkArray.data[i]);
                    deallocVector(chunkArray.rowLengths[i]);
                }
            }

            chunkArray = std::move(ChunkArray{});
        }else{
            for(std::size_t i = 0; i < shapedChunkArray.size(); i++){
                if(shapedChunkArray[i].numRows > 0){

                    std::size_t newpitchelements = std::max(rowpitchelements, shapedChunkArray[i].rowPitchElements);

                    std::vector<T> newchunkdata(shapedChunkArray[i].numRows * newpitchelements);

                    auto inputiter = shapedChunkArray[i].data.begin();
                    auto outputiter = newchunkdata.begin();
                    for(std::size_t row = 0; row < shapedChunkArray[i].numRows; row++){
                        const int len = shapedChunkArray[i].rowPitchElements;
                        outputiter = std::copy_n(inputiter, len, outputiter);
                        inputiter += len;
                    }

                    std::swap(shapedChunkArray[i].data, newchunkdata);
                    shapedChunkArray[i].rowPitchElements = newpitchelements;
                }
            }
        }

        // offsetsPrefixSum.resize(shapedChunkArray.size() + 1);

        // for(std::size_t i = 0; i < shapedChunkArray.size(); i++){
        //     offsetsPrefixSum[i+1] = offsetsPrefixSum[i] + shapedChunkArray[i].numRows;
        // }

        isShaped = true;
    }

    void reshapeRows(){
        reshapeRows(0);
    }

    template<class IndexGenerator>
    void gather(
        T* destination,
        std::size_t numColumns,
        IndexGenerator readIds,
        int numReadIds,
        size_t destinationPitchBytes
    ) const noexcept{
        
        if(numReadIds == 0){
            return;
        }

        if(!isShaped){
            assert(false && "not implemented");
        }else{
            constexpr int prefetch_distance = 4;

            for(int i = 0; i < numReadIds && i < prefetch_distance; ++i) {
                const int index = i;
                const std::size_t nextReadId = readIds(index);
                const T* const nextData = getPointerToRow(nextReadId);
                __builtin_prefetch(nextData, 0, 0);
            }

            for(int i = 0; i < numReadIds; i++){
                if(i + prefetch_distance < numReadIds) {
                    const int index = i + prefetch_distance;
                    const std::size_t nextReadId = readIds(index);
                    const T* const nextData = getPointerToRow(nextReadId);
                    __builtin_prefetch(nextData, 0, 0);
                }

                const std::size_t readId = readIds(i);
                const T* const data = getPointerToRow(readId);

                T* const destData = (T*)(((char*)destination) + destinationPitchBytes * i);
                std::copy_n(data, numColumns, destData);
            }
        }

        
    }

    void getMemoryInfo() const {
        if(isShaped){
            std::size_t used = 0;

            for(const auto& chunk : shapedChunkArray){
                used += sizeof(T) * chunk.data.capacity();
            }

            used += shapedChunkArray.capacity() * sizeof(ShapedChunk);
            used += offsetsPrefixSum.capacity() * sizeof(std::size_t);
            std::cerr << "shaped: " << used << "\n";
        }else{
            std::size_t used = 0;

            used += chunkArray.numRows.capacity() * sizeof(std::size_t);
            used += chunkArray.memoryLimitData.capacity() * sizeof(std::size_t);
            used += chunkArray.occupiedBytes.capacity() * sizeof(std::size_t);
            used += chunkArray.dataend.capacity() * sizeof(typename std::vector<T>::iterator);
            used += chunkArray.rowLengths.capacity() * sizeof(std::vector<int>);
            used += chunkArray.data.capacity() * sizeof(std::vector<T>);

            for(const auto& vec : chunkArray.rowLengths){
                used += sizeof(int) * vec.capacity();
            }

            for(const auto& vec : chunkArray.data){
                used += sizeof(T) * vec.capacity();
            }
            
            std::cerr << "unshaped: " << used << "\n";
        }
    }

    std::size_t getNumRows() const noexcept{
        if(isShaped){
            return offsetsPrefixSum.back();
        }else{
            return std::accumulate(
                chunkArray.numRows.begin(), 
                chunkArray.numRows.end(), 
                std::size_t{0}
            );
        }
    }

private:

    //SoA
    struct ChunkArray{
        void addChunk(std::size_t memoryLimitBytes){
            numRows.emplace_back(0);
            memoryLimitData.emplace_back(memoryLimitBytes);
            occupiedBytes.emplace_back(0);
            rowLengths.emplace_back();
            data.emplace_back(std::vector<T>(memoryLimitBytes / sizeof(T)));
            dataend.emplace_back(data.back().begin());
        }

        bool appendRow(std::size_t chunk, const T* rowData, int length){
            const std::size_t extraBytes = sizeof(T) * length;

            if(occupiedBytes[chunk] + extraBytes <= memoryLimitData[chunk]){
                dataend[chunk] = std::copy_n(rowData, length, dataend[chunk]);
                rowLengths[chunk].emplace_back(length);
                numRows[chunk]++;

                occupiedBytes[chunk] += extraBytes;
                return true;
            }else{
                return false;
            }
        }

        std::size_t getNumChunks() const noexcept{
            return data.size();
        }

        std::vector<std::size_t> numRows{};
        std::vector<std::size_t> memoryLimitData{};
        std::vector<std::size_t> occupiedBytes{};
        std::vector<typename std::vector<T>::iterator> dataend{};
        std::vector<std::vector<int>> rowLengths{};
        std::vector<std::vector<T>> data{};
    };

    struct ShapedChunk{
        const T* getPointerToRow(std::size_t rowIndex) const noexcept{
            assert(rowIndex < numRows);

            return data.data() + rowIndex * rowPitchElements;
        }

        std::size_t numRows{};
        std::size_t rowPitchElements{};
        std::vector<T> data{};
    };

    constexpr std::size_t getMemoryLimitChunk() const noexcept{
        constexpr std::size_t MB = 1ull << 20;
        return 16 * MB;
    }

    std::size_t getChunkIndexOfRow(std::size_t row) const noexcept{
        auto it = std::lower_bound(offsetsPrefixSum.begin(), offsetsPrefixSum.end(), row + 1);
        std::size_t chunkIndex = std::distance(offsetsPrefixSum.begin(), it) - 1;

        return chunkIndex;
    }

    std::size_t getRowIndexInChunk(std::size_t chunkIndex, std::size_t row) const noexcept{
        return row - offsetsPrefixSum[chunkIndex];
    }

    const T* getPointerToRow(std::size_t row) const noexcept{
        const std::size_t chunkIndex = getChunkIndexOfRow(row);
        const std::size_t rowInChunk = getRowIndexInChunk(chunkIndex, row);

        const T* const data = shapedChunkArray[chunkIndex].getPointerToRow(rowInChunk);
        return data;
    }

    bool isShaped = false;

    ChunkArray chunkArray{};
    std::vector<ShapedChunk> shapedChunkArray{};
    std::vector<std::size_t> offsetsPrefixSum{};
};





#endif