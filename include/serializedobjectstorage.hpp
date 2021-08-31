#ifndef CARE_SERIALIZED_OBJECT_STORAGE_HPP
#define CARE_SERIALIZED_OBJECT_STORAGE_HPP

#include <mmapbuffer.hpp>
#include <filehelpers.hpp>
#include <memorymanagement.hpp>

#include <iostream>
#include <algorithm>
#include <memory>
namespace care{


class SerializedObjectStorage{
private:
    std::unique_ptr<FileBackedUVector<std::uint8_t>> databuffer;
    std::unique_ptr<FileBackedUVector<std::size_t>> offsetbuffer;
public:
    SerializedObjectStorage(std::size_t memoryLimitData_, std::size_t memoryLimitOffsets_, std::string filedirectory){
        std::string nametemplate = filedirectory + "serializedobjectstorage-XXXXXX";
        std::string databufferfilename = filehelpers::makeRandomFile(nametemplate);
        std::string offsetbufferfilename = filehelpers::makeRandomFile(nametemplate);

        databuffer = std::make_unique<FileBackedUVector<std::uint8_t>>(
            0, memoryLimitData_, databufferfilename);

        offsetbuffer = std::make_unique<FileBackedUVector<std::size_t>>(
            0, memoryLimitOffsets_, offsetbufferfilename);
    }

    void insert(const std::uint8_t* begin, const std::uint8_t* end){
        auto first = databuffer->insert(databuffer->end(), begin, end);
        offsetbuffer->push_back(std::distance(databuffer->begin(), first));
    }

    MemoryUsage getMemoryInfo() const{
        MemoryUsage result;
        result.host += databuffer->getCapacityInMemoryInBytes();
        result.host += offsetbuffer->getCapacityInMemoryInBytes();
        return result;
    }

    std::uint8_t* getPointer(std::size_t i) noexcept{
        const std::size_t offset = getOffset(i);
        return databuffer->data() + offset;
    }

    const std::uint8_t* getPointer(std::size_t i) const noexcept{
        const std::size_t offset = getOffset(i);
        return databuffer->data() + offset;
    }

    std::size_t getOffset(std::size_t i) const noexcept{
        return (*offsetbuffer)[i];
    }

    std::uint8_t* getDataBuffer() noexcept{
        return databuffer->data();
    }

    const std::uint8_t* getDataBuffer() const noexcept{
        return databuffer->data();
    }

    std::size_t* getOffsetBuffer() noexcept{
        return offsetbuffer->data();
    }

    const std::size_t* getOffsetBuffer() const noexcept{
        return offsetbuffer->data();
    }

    std::size_t size() const noexcept{
        return offsetbuffer->size();
    }

    std::size_t getNumElements() const noexcept{
        return offsetbuffer->size();
    }    

    std::size_t dataBytes() const noexcept{
        return sizeof(std::uint8_t) * databuffer->size();
    }

    std::size_t offsetBytes() const noexcept{
        return sizeof(std::size_t) * offsetbuffer->size();
    }
};


}





#endif