#ifndef DYNAMIC_2D_ARRAY_HPP
#define DYNAMIC_2D_ARRAY_HPP


#include <vector>
#include <cstddef>
#include <cassert>
#include <algorithm>
#include <functional>

template<class T>
class Dynamic2dArray{
public:
    Dynamic2dArray(){
        arraychunks.emplace_back(0);
    }

    void appendRow(const T* rowData, int length){
        assert(!isShaped);

        bool couldBeAppended = chunkArray.appendRow(getNumChunks() - 1, rowData, length);
        if(!couldBeAppended){
            chunkArray.addChunk(getMemoryLimitChunk());
            couldBeAppended = chunkArray.appendRow(getNumChunks() - 1, rowData, length);

            assert(couldBeAppended);
        }
    }

    void reshapeRows(std::size_t rowpitchelements){
        auto deallocVector = [](auto& vec){
            using T = typename std::remove_reference<decltype(vec)>::type;
            T tmp{};
            vec.swap(tmp);
        };

        if(!isShaped){

            for(std::size_t i = 0; i < chunkArray.getNumChunks(); i++){
                if(chunkArray.numRows[i] > 0){
                    const int maxLength = *std::max_element(
                        chunkArray.rowLengths[i].begin(),
                        chunkArray.rowLength[i].end(),
                        std::min<int>{}
                    );

                    std::size_t newpitchelements = std::max(rowpitchelements, maxLength);

                    std::vector<T> newchunkdata(chunkArray.numRows[i] * newpitchelements);

                    auto inputiter = chunkArray.data[i].begin();
                    auto outputiter = newchunkdata.begin();
                    for(std::size_t row = 0; row < chunkArray.numRows[i]; row++){
                        const int len = chunkArray.rowLength[i][row];
                        outputiter = std::copy_n(inputiter, len, outputiter);
                        inputiter += len;
                    }

                    shapedChunkArray.data.emplace_back(std::move(newchunkdata));
                    shapedChunkArray.numRows.emplace_back(chunkArray.numRows[i]);
                    shapedChunkArray.rowPitchElements.emplace_back(newpitchelements);

                    deallocVector(chunkArray.data[i]);
                    deallocVector(chunkArray.rowLength[i]);
                }
            }
        }else{
            for(std::size_t i = 0; i < shapedChunkArray.getNumChunks(); i++){
                if(shapedChunkArray.data[i].size() > 0){

                    std::size_t newpitchelements = std::max(rowpitchelements, shapedChunkArray.rowPitchElements[i]);

                    std::vector<T> newchunkdata(shapedChunkArray.numRows[i] * newpitchelements);

                    auto inputiter = shapedChunkArray.data[i].begin();
                    auto outputiter = newchunkdata.begin();
                    for(std::size_t row = 0; row < shapedChunkArray.numRows[i]; row++){
                        const int len = shapedChunkArray.rowPitchElements[i];
                        outputiter = std::copy_n(inputiter, len, outputiter);
                        inputiter += len;
                    }

                    std::swap(shapedChunkArray.data[i], newchunkdata)
                    shapedChunkArray.rowPitchElements[i] = newpitchelements;
                }
            }
        }

        isShaped = true;
    }

    void reshapeRows(){
        reshapeRows(0);
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

            if(occupiedBytes[chunk] + extraBytes[chunk] <= memoryLimitData[chunk]){
                dataend[chunk] = std::copy_n(rowData, length, dataend[chunk]);
                rowLengths[chunk].emplace_back(length);

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

    struct ShapedChunkArray{

        std::size_t getNumChunks() const noexcept{
            return data.size();
        }

        std::size_t numRows{};
        std::vector<std::size_t> rowPitchElements{};
        std::vector<std::vector<T>> data{};
    };


    constexpr std::size_t getMemoryLimitChunk() const noexcept{
        constexpr std::size_t MB = 1ull << 20;
        return 64 * MB;
    }

    bool isShaped = false;

    ChunkArray chunkArray{};
    ShapedChunkArray shapedChunkArray{};
};





#endif