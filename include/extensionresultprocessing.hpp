#ifndef CARE_EXTENSION_RESULT_PROCESSING_HPP
#define CARE_EXTENSION_RESULT_PROCESSING_HPP


#include <config.hpp>
#include <sequence.hpp>
#include <hpc_helpers.cuh>
#include <memoryfile.hpp>
#include <readlibraryio.hpp>

#include <cstring>
#include <string>
#include <vector>

namespace care{

    // struct EncodedExtendedRead{
    //     std::uint32_t encodedflags{}; //contains size of data in bytes, and boolean flags
    //     read_number readId{};
    //     std::unique_ptr<std::uint8_t[]> data{};

    //     EncodedExtendedRead() = default;
    //     EncodedExtendedRead(EncodedExtendedRead&& rhs){
    //         *this = std::move(rhs);
    //     }

    //     EncodedExtendedRead& operator=(EncodedExtendedRead&& rhs){
    //         encodedflags = std::exchange(rhs.encodedflags, 0);
    //         readId = std::exchange(rhs.readId, 0);
    //         data = std::move(rhs.data);

    //         return *this;
    //     }

    //     EncodedExtendedRead(const EncodedExtendedRead& rhs){
    //         *this = rhs;
    //     }

    //     EncodedExtendedRead& operator=(const EncodedExtendedRead& rhs){
    //         encodedflags = rhs.encodedflags;
    //         readId = rhs.readId;

    //         const int numBytes = rhs.getNumBytes();
    //         data = std::make_unique<std::uint8_t[]>(numBytes);
    //         std::memcpy(data.get(), rhs.data.get(), numBytes);

    //         return *this;
    //     }

    //     EncodedExtendedRead& operator=(const TempCorrectedSequence& rhs);

    //     bool writeToBinaryStream(std::ostream& s) const;
    //     bool readFromBinaryStream(std::istream& s);

    //     std::uint8_t* copyToContiguousMemory(std::uint8_t*, std::uint8_t*) const;
    //     void copyFromContiguousMemory(const std::uint8_t*);

    //     bool operator==(const EncodedExtendedRead& rhs) const{
    //         const std::uint32_t numBytes = getNumBytes();
    //         return encodedflags == rhs.encodedflags && readId == rhs.readId 
    //                 && std::memcmp(data.get(), rhs.data.get(), numBytes);
    //     }

    //     bool operator!=(const EncodedExtendedRead& rhs) const{
    //         return !(operator==(rhs));
    //     }

    //     int getNumBytes() const{
    //         constexpr std::uint32_t mask = (std::uint32_t(1) << 29)-1;
    //         return (encodedflags & mask);
    //     }
    // };


    struct ExtendedReadDebug{
        bool reachedMate1;
        bool reachedMate2;
        read_number readId1;
        read_number readId2;
        std::string originalRead1;
        std::string originalRead2;
        std::string extendedRead1;
        std::string extendedRead2;

        std::uint8_t* copyToContiguousMemory(std::uint8_t* ptr, std::uint8_t* endPtr) const{
            const std::size_t requiredBytes = sizeof(bool)
                + sizeof(bool)
                + sizeof(read_number)
                + sizeof(read_number)
                + sizeof(int) + originalRead1.length()
                + sizeof(int) + originalRead2.length()
                + sizeof(int) + extendedRead1.length()
                + sizeof(int) + extendedRead2.length();

            const std::size_t availableBytes = std::distance(ptr, endPtr);

            if(requiredBytes <= availableBytes){                
                std::memcpy(ptr, &readId1, sizeof(read_number));
                ptr += sizeof(read_number);
                std::memcpy(ptr, &readId2, sizeof(read_number));
                ptr += sizeof(read_number);
                std::memcpy(ptr, &reachedMate1, sizeof(bool));
                ptr += sizeof(bool);
                std::memcpy(ptr, &reachedMate2, sizeof(bool));
                ptr += sizeof(bool);

                int l = 0;
                l = originalRead1.length();
                std::memcpy(ptr, &l, sizeof(int));
                ptr += sizeof(int);
                std::memcpy(ptr, originalRead1.c_str(), sizeof(char) * l);
                ptr += sizeof(char) * l;

                l = originalRead2.length();
                std::memcpy(ptr, &l, sizeof(int));
                ptr += sizeof(int);
                std::memcpy(ptr, originalRead2.c_str(), sizeof(char) * l);
                ptr += sizeof(char) * l;

                l = extendedRead1.length();
                std::memcpy(ptr, &l, sizeof(int));
                ptr += sizeof(int);
                std::memcpy(ptr, extendedRead1.c_str(), sizeof(char) * l);
                ptr += sizeof(char) * l;

                l = extendedRead2.length();
                std::memcpy(ptr, &l, sizeof(int));
                ptr += sizeof(int);
                std::memcpy(ptr, extendedRead2.c_str(), sizeof(char) * l);
                ptr += sizeof(char) * l;

                return ptr;
            }else{
                return nullptr;
            }        
        }

        void copyFromContiguousMemory(const std::uint8_t* ptr){
            std::memcpy(&readId1, ptr, sizeof(read_number));
            ptr += sizeof(read_number);
            std::memcpy(&readId2, ptr, sizeof(read_number));
            ptr += sizeof(read_number);
            std::memcpy(&reachedMate1, ptr, sizeof(bool));
            ptr += sizeof(bool);
            std::memcpy(&reachedMate2, ptr, sizeof(bool));
            ptr += sizeof(bool);            

            int l = 0;
            std::memcpy(&l, ptr, sizeof(int));
            ptr += sizeof(int);
            originalRead1.resize(l);
            std::memcpy(&originalRead1[0], ptr, sizeof(char) * l);
            ptr += l;

            std::memcpy(&l, ptr, sizeof(int));
            ptr += sizeof(int);
            originalRead2.resize(l);
            std::memcpy(&originalRead2[0], ptr, sizeof(char) * l);
            ptr += l;
            
            std::memcpy(&l, ptr, sizeof(int));
            ptr += sizeof(int);
            extendedRead1.resize(l);
            std::memcpy(&extendedRead1[0], ptr, sizeof(char) * l);
            ptr += l;

            std::memcpy(&l, ptr, sizeof(int));
            ptr += sizeof(int);
            extendedRead2.resize(l);
            std::memcpy(&extendedRead2[0], ptr, sizeof(char) * l);
            ptr += l;
        }

        bool writeToBinaryStream(std::ostream& os) const{
            os.write(reinterpret_cast<const char*>(&readId1), sizeof(read_number));
            os.write(reinterpret_cast<const char*>(&readId2), sizeof(read_number));
            os.write(reinterpret_cast<const char*>(&reachedMate1), sizeof(bool));
            os.write(reinterpret_cast<const char*>(&reachedMate2), sizeof(bool));

            int l = 0;
            l = originalRead1.length();
            os.write(reinterpret_cast<const char*>(&l), sizeof(int));
            os.write(reinterpret_cast<const char*>(originalRead1.c_str()), sizeof(char) * l);

            l = originalRead2.length();
            os.write(reinterpret_cast<const char*>(&l), sizeof(int));
            os.write(reinterpret_cast<const char*>(originalRead2.c_str()), sizeof(char) * l);

            l = extendedRead1.length();
            os.write(reinterpret_cast<const char*>(&l), sizeof(int));
            os.write(reinterpret_cast<const char*>(extendedRead1.c_str()), sizeof(char) * l);

            l = extendedRead2.length();
            os.write(reinterpret_cast<const char*>(&l), sizeof(int));
            os.write(reinterpret_cast<const char*>(extendedRead2.c_str()), sizeof(char) * l);

            return bool(os);
        }

        bool readFromBinaryStream(std::istream& is){
            is.read(reinterpret_cast<char*>(&readId1), sizeof(read_number));
            is.read(reinterpret_cast<char*>(&readId2), sizeof(read_number));
            is.read(reinterpret_cast<char*>(&reachedMate1), sizeof(bool));
            is.read(reinterpret_cast<char*>(&reachedMate2), sizeof(bool));

            int l = 0;
            is.read(reinterpret_cast<char*>(&l), sizeof(int));
            originalRead1.resize(l);
            is.read(reinterpret_cast<char*>(&originalRead1[0]), sizeof(char) * l);

            is.read(reinterpret_cast<char*>(&l), sizeof(int));
            originalRead2.resize(l);
            is.read(reinterpret_cast<char*>(&originalRead2[0]), sizeof(char) * l);

            is.read(reinterpret_cast<char*>(&l), sizeof(int));
            extendedRead1.resize(l);
            is.read(reinterpret_cast<char*>(&extendedRead1[0]), sizeof(char) * l);

            is.read(reinterpret_cast<char*>(&l), sizeof(int));
            extendedRead2.resize(l);
            is.read(reinterpret_cast<char*>(&extendedRead2[0]), sizeof(char) * l);

            return bool(is);
        }
    };


    struct ExtendedRead{
        bool reachedMate;
        read_number readId;
        std::string extendedSequence;

        ExtendedRead() = default;

        ExtendedRead(const ExtendedReadDebug& rhs){
            *this = rhs;            
        }

        ExtendedRead& operator=(const ExtendedReadDebug& rhs){
            auto select_longest = [&](){
                const int len1 = rhs.extendedRead1.length();
                const int len2 = rhs.extendedRead2.length();

                if(len1 > len2){
                    reachedMate = rhs.reachedMate1;
                    readId = rhs.readId1;
                    extendedSequence = rhs.extendedRead1;
                }else if(len1 < len2){
                    reachedMate = rhs.reachedMate2;
                    readId = rhs.readId2;
                    extendedSequence = rhs.extendedRead2;
                }else{
                    reachedMate = rhs.reachedMate1;
                    readId = rhs.readId1;
                    extendedSequence = rhs.extendedRead1;
                }
            };

            if(rhs.reachedMate1 && !rhs.reachedMate2){
                reachedMate = true;
                readId = rhs.readId1;
                extendedSequence = rhs.extendedRead1;
            }else if(!rhs.reachedMate1 && rhs.reachedMate2){
                reachedMate = true;
                readId = rhs.readId2;
                extendedSequence = rhs.extendedRead2;
            }else if(rhs.reachedMate1 && rhs.reachedMate2){
                select_longest();
            }else{
                //!reachedMate1 && !reachedMate2
                select_longest();
            }

            return *this;
        }

        std::uint8_t* copyToContiguousMemory(std::uint8_t* ptr, std::uint8_t* endPtr) const{
            const std::size_t requiredBytes = sizeof(bool)
                + sizeof(bool)
                + sizeof(read_number)
                + sizeof(int) + extendedSequence.length();

            const std::size_t availableBytes = std::distance(ptr, endPtr);

            if(requiredBytes <= availableBytes){                
                std::memcpy(ptr, &readId, sizeof(read_number));
                ptr += sizeof(read_number);
                std::memcpy(ptr, &reachedMate, sizeof(bool));
                ptr += sizeof(bool);

                int l = 0;
                l = extendedSequence.length();
                std::memcpy(ptr, &l, sizeof(int));
                ptr += sizeof(int);
                std::memcpy(ptr, extendedSequence.c_str(), sizeof(char) * l);
                ptr += sizeof(char) * l;

                return ptr;
            }else{
                return nullptr;
            }        
        }

        void copyFromContiguousMemory(const std::uint8_t* ptr){
            std::memcpy(&readId, ptr, sizeof(read_number));
            ptr += sizeof(read_number);
            std::memcpy(&reachedMate, ptr, sizeof(bool));
            ptr += sizeof(bool);    

            int l = 0;
            std::memcpy(&l, ptr, sizeof(int));
            ptr += sizeof(int);
            extendedSequence.resize(l);
            std::memcpy(&extendedSequence[0], ptr, sizeof(char) * l);
            ptr += l;
        }

        bool writeToBinaryStream(std::ostream& os) const{
            os.write(reinterpret_cast<const char*>(&readId), sizeof(read_number));
            os.write(reinterpret_cast<const char*>(&reachedMate), sizeof(bool));

            int l = 0;
            l = extendedSequence.length();
            os.write(reinterpret_cast<const char*>(&l), sizeof(int));
            os.write(reinterpret_cast<const char*>(extendedSequence.c_str()), sizeof(char) * l);

            return bool(os);
        }

        bool readFromBinaryStream(std::istream& is){
            is.read(reinterpret_cast<char*>(&readId), sizeof(read_number));
            is.read(reinterpret_cast<char*>(&reachedMate), sizeof(bool));

            int l = 0;
            is.read(reinterpret_cast<char*>(&l), sizeof(int));
            extendedSequence.resize(l);
            is.read(reinterpret_cast<char*>(&extendedSequence[0]), sizeof(char) * l);

            return bool(is);
        }
    };





    void constructOutputFileFromExtensionResults(
        const std::string& tempdir,
        const std::vector<std::string>& originalReadFiles,
        MemoryFileFixedSize<ExtendedRead>& partialResults, 
        std::size_t memoryForSorting,
        FileFormat outputFormat,
        const std::vector<std::string>& outputfiles,
        bool isSorted
    );


}





#endif