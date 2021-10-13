#ifndef CARE_EXTENDEDREAD_HPP
#define CARE_EXTENDEDREAD_HPP


#include <config.hpp>
#include <sequencehelpers.hpp>
#include <hpc_helpers.cuh>
#include <serializedobjectstorage.hpp>
#include <readlibraryio.hpp>
#include <options.hpp>

#include <cstring>
#include <string>
#include <vector>

namespace care{

    enum class ExtendedReadStatus : unsigned char{
        FoundMate = 1,
        MSANoExtension = 2,
        LengthAbort = 4,
        CandidateAbort = 8,
        Repeated = 16
    };

    // struct ExtendedReadStatus{
    //     static constexpr unsigned char FoundMate = 1;
    //     static constexpr unsigned char MSANoExtension = 2;
    //     static constexpr unsigned char LengthAbort = 4;
    //     static constexpr unsigned char CandidateAbort = 8;
    //     static constexpr unsigned char Repeated = 16;

    //     unsigned char status;
    // };

    struct EncodedExtendedRead{
        std::uint32_t encodedflags{}; //contains size of data in bytes, and mergedFromReadsWithoutMate
        read_number readId{};
        std::unique_ptr<std::uint8_t[]> data{};

        EncodedExtendedRead() = default;
        EncodedExtendedRead(const EncodedExtendedRead& rhs){
            auto bytes = rhs.getNumBytes();
            encodedflags = rhs.encodedflags;
            readId = rhs.readId;
            data = std::make_unique<std::uint8_t[]>(bytes);
            std::copy(rhs.data.get(), rhs.data.get() + bytes, data.get());
        }
        EncodedExtendedRead(EncodedExtendedRead&& rhs){
            *this = std::move(rhs);
        }

        EncodedExtendedRead& operator=(const EncodedExtendedRead& rhs){
            auto bytes = rhs.getNumBytes();
            encodedflags = rhs.encodedflags;
            readId = rhs.readId;
            data = std::make_unique<std::uint8_t[]>(bytes);
            std::copy(rhs.data.get(), rhs.data.get() + bytes, data.get());

            return *this;
        }

        EncodedExtendedRead& operator=(EncodedExtendedRead&& rhs){
            encodedflags = std::exchange(rhs.encodedflags, 0);
            readId = std::exchange(rhs.readId, 0);
            data = std::move(rhs.data);

            return *this;
        }

        // EncodedExtendedRead(const EncodedExtendedRead& rhs){
        //     *this = rhs;
        // }

        // EncodedExtendedRead& operator=(const EncodedExtendedRead& rhs){
        //     encodedflags = rhs.encodedflags;
        //     readId = rhs.readId;

        //     const int numBytes = rhs.getNumBytes();
        //     data = std::make_unique<std::uint8_t[]>(numBytes);
        //     std::memcpy(data.get(), rhs.data.get(), numBytes);

        //     return *this;
        // }

        int getSerializedNumBytes() const noexcept{
            const int dataBytes = getNumBytes();
            return sizeof(read_number) + sizeof(std::uint32_t) + dataBytes;
        }

        std::uint8_t* copyToContiguousMemory(std::uint8_t* ptr, std::uint8_t* endPtr) const{
            const int dataBytes = getNumBytes();

            const std::size_t availableBytes = std::distance(ptr, endPtr);
            const std::size_t requiredBytes = getSerializedNumBytes();
            if(requiredBytes <= availableBytes){
                std::memcpy(ptr, &readId, sizeof(read_number));
                ptr += sizeof(read_number);
                std::memcpy(ptr, &encodedflags, sizeof(std::uint32_t));
                ptr += sizeof(std::uint32_t);
                std::memcpy(ptr, data.get(), dataBytes);
                ptr += dataBytes;
                return ptr;
            }else{
                return nullptr;
            } 
        }

        void copyFromContiguousMemory(const std::uint8_t* ptr){
            std::memcpy(&readId, ptr, sizeof(read_number));
            ptr += sizeof(read_number);
            std::memcpy(&encodedflags, ptr, sizeof(std::uint32_t));
            ptr += sizeof(read_number);

            const int numBytes = getNumBytes();
            data = std::make_unique<std::uint8_t[]>(numBytes);

            std::memcpy(data.get(), ptr, numBytes);
            //ptr += numBytes;
        }


        int getNumBytes() const{
            constexpr std::uint32_t mask = (std::uint32_t(1) << 31)-1;
            return (encodedflags & mask);
        }

        static read_number parseReadId(const std::uint8_t* ptr){
            read_number id;
            std::memcpy(&id, ptr, sizeof(read_number));
            return id;
        }

        read_number getReadId() const noexcept{
            return readId;
        }
    };



    struct ExtendedRead{
        bool mergedFromReadsWithoutMate = false;
        ExtendedReadStatus status{};
        read_number readId{};
        int read1begin = 0;
        int read1end = 0;
        int read2begin = 0;
        int read2end = 0;
        std::string extendedSequence{};
        std::string qualityScores{};

        ExtendedRead() = default;

        bool operator==(const ExtendedRead& rhs) const noexcept{
            if(mergedFromReadsWithoutMate != rhs.mergedFromReadsWithoutMate) return false;
            if(status != rhs.status) return false;
            if(readId != rhs.readId) return false;
            if(read1begin != rhs.read1begin) return false;
            if(read1end != rhs.read1end) return false;
            if(read2begin != rhs.read2begin) return false;
            if(read2end != rhs.read2end) return false;
            if(extendedSequence != rhs.extendedSequence) return false;
            if(qualityScores != rhs.qualityScores) return false;
            return true;
        }

        bool operator!=(const ExtendedRead& rhs) const noexcept{
            return !(operator==(rhs));
        }

        int getSerializedNumBytes() const noexcept{
            return sizeof(bool) // mergedFromReadsWithoutMate
                + sizeof(ExtendedReadStatus) //status
                + sizeof(read_number) //readid
                + sizeof(int) * 4  //original ranges
                + sizeof(int) + extendedSequence.length() //sequence
                + sizeof(int) + qualityScores.length(); // quality scores
        }

        std::uint8_t* copyToContiguousMemory(std::uint8_t* ptr, std::uint8_t* endPtr) const{
            const std::size_t requiredBytes = getSerializedNumBytes();                

            const std::size_t availableBytes = std::distance(ptr, endPtr);

            if(requiredBytes <= availableBytes){                
                std::memcpy(ptr, &readId, sizeof(read_number));
                ptr += sizeof(read_number);
                std::memcpy(ptr, &mergedFromReadsWithoutMate, sizeof(bool));
                ptr += sizeof(bool);
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
            std::memcpy(&mergedFromReadsWithoutMate, ptr, sizeof(bool));
            ptr += sizeof(bool);
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

        //from serialized object beginning at ptr, return the read id of this object
        static read_number parseReadId(const std::uint8_t* ptr){
            read_number id;
            std::memcpy(&id, ptr, sizeof(read_number));
            return id;
        }

        read_number getReadId() const noexcept{
            return readId;
        }

        void encodeInto(EncodedExtendedRead& target) const{

            const int numEncodedSequenceInts = SequenceHelpers::getEncodedNumInts2Bit(extendedSequence.size());
            std::size_t requiredBytes = 0;
            requiredBytes += sizeof(ExtendedReadStatus); // status
            requiredBytes += sizeof(int); // read1begin
            requiredBytes += sizeof(int); // read1end
            requiredBytes += sizeof(int); // read2begin
            requiredBytes += sizeof(int); // read2end
            requiredBytes += sizeof(int); // seq length
            requiredBytes += sizeof(int); // qual length
            requiredBytes += sizeof(unsigned int) * numEncodedSequenceInts; // enc seq
            requiredBytes += sizeof(char) * qualityScores.size(); //qual

            assert(requiredBytes < (1u << 31)); // 1 bit reserved for flag

            if(int(requiredBytes) > target.getNumBytes()){
                target.data = std::make_unique<std::uint8_t[]>(requiredBytes);
            }else{
                ; //reuse buffer
            }

            target.readId = readId;
            target.encodedflags = (std::uint32_t(mergedFromReadsWithoutMate) << 31);
            target.encodedflags |= std::uint32_t(requiredBytes);

            //fill buffer

            std::uint8_t* ptr = target.data.get();
            auto saveint = [&](int value){
                std::memcpy(ptr, &value, sizeof(int)); ptr += sizeof(int);
            };

            std::memcpy(ptr, &status, sizeof(ExtendedReadStatus)); ptr += sizeof(ExtendedReadStatus); 

            saveint(read1begin);
            saveint(read1end);
            saveint(read2begin);
            saveint(read2end);
            saveint(extendedSequence.size());
            saveint(qualityScores.size());

            SequenceHelpers::encodeSequence2Bit(
                reinterpret_cast<unsigned int*>(ptr), 
                extendedSequence.data(), 
                extendedSequence.size()
            );
            ptr += sizeof(unsigned int) * numEncodedSequenceInts;

            ptr = std::copy(qualityScores.begin(), qualityScores.end(), ptr);

            assert(target.data.get() + requiredBytes == ptr);
        }
    
        EncodedExtendedRead encode() const{
            EncodedExtendedRead result;
            encodeInto(result);
            return result;
        }
    
        void decode(const EncodedExtendedRead& rhs){
            mergedFromReadsWithoutMate = bool(rhs.encodedflags >> 31);
            readId = rhs.readId;

            const std::uint8_t* ptr = rhs.data.get();
            auto loadint = [&](int& value){
                std::memcpy(&value, ptr, sizeof(int)); ptr += sizeof(int);
            };

            std::memcpy(&status, ptr, sizeof(ExtendedReadStatus)); ptr += sizeof(ExtendedReadStatus); 

            loadint(read1begin);
            loadint(read1end);
            loadint(read2begin);
            loadint(read2end);
            int seqlen;
            int quallen;
            loadint(seqlen);
            loadint(quallen);

            extendedSequence.resize(seqlen);
            qualityScores.resize(quallen);

            const int numEncodedSequenceInts = SequenceHelpers::getEncodedNumInts2Bit(seqlen);

            SequenceHelpers::decode2BitSequence(extendedSequence.data(), reinterpret_cast<const unsigned int*>(ptr), seqlen);
            ptr += sizeof(unsigned int) * numEncodedSequenceInts;

            std::copy(ptr, ptr + quallen, qualityScores.begin());

            assert(rhs.data.get() + rhs.getNumBytes() == ptr + quallen);
        }
    
    };

}





#endif