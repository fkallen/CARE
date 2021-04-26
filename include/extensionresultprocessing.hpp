#ifndef CARE_EXTENSION_RESULT_PROCESSING_HPP
#define CARE_EXTENSION_RESULT_PROCESSING_HPP


#include <config.hpp>
#include <sequencehelpers.hpp>
#include <hpc_helpers.cuh>
#include <memoryfile.hpp>
#include <readlibraryio.hpp>
#include <options.hpp>

#include <cstring>
#include <string>
#include <vector>

namespace care{

    enum class ExtendedReadStatus : unsigned char{
        FoundMate,
        MSANoExtension,
        LengthAbort,
        CandidateAbort
    };

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
        ExtendedReadStatus status1;
        ExtendedReadStatus status2;
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
                std::memcpy(ptr, &status1, sizeof(ExtendedReadStatus));
                ptr += sizeof(ExtendedReadStatus);
                std::memcpy(ptr, &status2, sizeof(ExtendedReadStatus));
                ptr += sizeof(ExtendedReadStatus);

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
            std::memcpy(&status1, ptr, sizeof(ExtendedReadStatus));
            ptr += sizeof(ExtendedReadStatus);
            std::memcpy(&status2, ptr, sizeof(ExtendedReadStatus));
            ptr += sizeof(ExtendedReadStatus);            

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
            os.write(reinterpret_cast<const char*>(&status1), sizeof(ExtendedReadStatus));
            os.write(reinterpret_cast<const char*>(&status2), sizeof(ExtendedReadStatus));

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
            is.read(reinterpret_cast<char*>(&status1), sizeof(ExtendedReadStatus));
            is.read(reinterpret_cast<char*>(&status2), sizeof(ExtendedReadStatus));

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

        ExtendedReadStatus status;
        read_number readId;
        int read1begin = 0;
        int read1end = 0;
        int read2begin = 0;
        int read2end = 0;
        std::string extendedSequence;
        std::string qualityScores;

        ExtendedRead() = default;

        #if 0
        ExtendedRead(const ExtendedReadDebug& rhs){
            *this = rhs;            
        }

        ExtendedRead& operator=(const ExtendedReadDebug& rhs){
            auto select_longest = [&](){
                const int len1 = rhs.extendedRead1.length();
                const int len2 = rhs.extendedRead2.length();

                if(len1 > len2){
                    status = rhs.status1;
                    readId = rhs.readId1;
                    extendedSequence = rhs.extendedRead1;
                }else if(len1 < len2){
                    status = rhs.status2;
                    readId = rhs.readId2;
                    extendedSequence = rhs.extendedRead2;
                }else{
                    status = rhs.status1;
                    readId = rhs.readId1;
                    extendedSequence = rhs.extendedRead1;
                }
            };

            if(rhs.status1 == ExtendedReadStatus::FoundMate && rhs.status2 != ExtendedReadStatus::FoundMate){
                status = ExtendedReadStatus::FoundMate;
                readId = rhs.readId1;
                extendedSequence = rhs.extendedRead1;
            }else if(rhs.status1 != ExtendedReadStatus::FoundMate && rhs.status2 == ExtendedReadStatus::FoundMate){
                status = ExtendedReadStatus::FoundMate;
                readId = rhs.readId2;
                extendedSequence = rhs.extendedRead2;
            }else if(rhs.status1 == ExtendedReadStatus::FoundMate && rhs.status2 == ExtendedReadStatus::FoundMate){
                select_longest();
            }else{
                //!FoundMate 1 && !FoundMate 2
                select_longest();
            }

            return *this;
        }
        #endif

        std::uint8_t* copyToContiguousMemory(std::uint8_t* ptr, std::uint8_t* endPtr) const{
            const std::size_t requiredBytes = sizeof(read_number) //readid
                + sizeof(ExtendedReadStatus) //status
                + sizeof(int) * 4  //original ranges
                + sizeof(int) + extendedSequence.length(); //sequence

            const std::size_t availableBytes = std::distance(ptr, endPtr);

            if(requiredBytes <= availableBytes){                
                std::memcpy(ptr, &readId, sizeof(read_number));
                ptr += sizeof(read_number);
                std::memcpy(ptr, &status, sizeof(ExtendedReadStatus));
                ptr += sizeof(ExtendedReadStatus);

                std::memcpy(ptr, &read1begin, sizeof(int));
                ptr += sizeof(int);
                std::memcpy(ptr, &read1end, sizeof(int));
                ptr += sizeof(int);
                std::memcpy(ptr, &read2begin, sizeof(int));
                ptr += sizeof(int);
                std::memcpy(ptr, &read2end, sizeof(int));
                ptr += sizeof(int);

                int l = 0;
                l = extendedSequence.length();
                std::memcpy(ptr, &l, sizeof(int));
                ptr += sizeof(int);
                std::memcpy(ptr, extendedSequence.c_str(), sizeof(char) * l);
                ptr += sizeof(char) * l;

                int m = 0;
                m = qualityScores.length();
                std::memcpy(ptr, &m, sizeof(int));
                ptr += sizeof(int);
                std::memcpy(ptr, qualityScores.c_str(), sizeof(char) * m);
                ptr += sizeof(char) * m;

                return ptr;
            }else{
                return nullptr;
            }        
        }

        void copyFromContiguousMemory(const std::uint8_t* ptr){
            std::memcpy(&readId, ptr, sizeof(read_number));
            ptr += sizeof(read_number);
            std::memcpy(&status, ptr, sizeof(ExtendedReadStatus));
            ptr += sizeof(ExtendedReadStatus);

            std::memcpy(&read1begin, ptr, sizeof(int));
            ptr += sizeof(int);
            std::memcpy(&read1end, ptr, sizeof(int));
            ptr += sizeof(int);
            std::memcpy(&read2begin, ptr, sizeof(int));
            ptr += sizeof(int);
            std::memcpy(&read2end, ptr, sizeof(int));
            ptr += sizeof(int);

            int l = 0;
            std::memcpy(&l, ptr, sizeof(int));
            ptr += sizeof(int);
            extendedSequence.resize(l);
            std::memcpy(&extendedSequence[0], ptr, sizeof(char) * l);
            ptr += l;

            int m = 0;
            std::memcpy(&m, ptr, sizeof(int));
            ptr += sizeof(int);
            qualityScores.resize(m);
            std::memcpy(&qualityScores[0], ptr, sizeof(char) * m);
            ptr += m;
        }

        bool writeToBinaryStream(std::ostream& os) const{
            os.write(reinterpret_cast<const char*>(&readId), sizeof(read_number));
            os.write(reinterpret_cast<const char*>(&status), sizeof(ExtendedReadStatus));

            os.write(reinterpret_cast<const char*>(&read1begin), sizeof(int));
            os.write(reinterpret_cast<const char*>(&read1end), sizeof(int));
            os.write(reinterpret_cast<const char*>(&read2begin), sizeof(int));
            os.write(reinterpret_cast<const char*>(&read2end), sizeof(int));

            int l = 0;
            l = extendedSequence.length();
            os.write(reinterpret_cast<const char*>(&l), sizeof(int));
            os.write(reinterpret_cast<const char*>(extendedSequence.c_str()), sizeof(char) * l);

            int m = 0;
            m = qualityScores.length();
            os.write(reinterpret_cast<const char*>(&m), sizeof(int));
            os.write(reinterpret_cast<const char*>(qualityScores.c_str()), sizeof(char) * m);

            return bool(os);
        }

        bool readFromBinaryStream(std::istream& is){
            is.read(reinterpret_cast<char*>(&readId), sizeof(read_number));
            is.read(reinterpret_cast<char*>(&status), sizeof(ExtendedReadStatus));

            is.read(reinterpret_cast<char*>(&read1begin), sizeof(int));
            is.read(reinterpret_cast<char*>(&read1end), sizeof(int));
            is.read(reinterpret_cast<char*>(&read2begin), sizeof(int));
            is.read(reinterpret_cast<char*>(&read2end), sizeof(int));

            int l = 0;
            is.read(reinterpret_cast<char*>(&l), sizeof(int));
            extendedSequence.resize(l);
            is.read(reinterpret_cast<char*>(&extendedSequence[0]), sizeof(char) * l);

            int m = 0;
            is.read(reinterpret_cast<char*>(&m), sizeof(int));
            qualityScores.resize(m);
            is.read(reinterpret_cast<char*>(&qualityScores[0]), sizeof(char) * m);

            return bool(is);
        }

        //from serialized object beginning at ptr, return the read id of this object
        static read_number parseReadId(const std::uint8_t* ptr){
            read_number id;
            std::memcpy(&id, ptr, sizeof(read_number));
            return id;
        }

        read_number getReadId() const noexcept{
            return readId;
        }
    };





    void constructOutputFileFromExtensionResults(
        const std::string& tempdir,
        const std::vector<std::string>& originalReadFiles,
        MemoryFileFixedSize<ExtendedRead>& partialResults, 
        std::size_t memoryForSorting,
        FileFormat outputFormat,
        const std::string& extendedOutputfile,
        const std::vector<std::string>& outputfiles, //one output file per original file
        SequencePairType pairmode,
        bool isSorted,
        bool outputToSingleFile
    );


}





#endif