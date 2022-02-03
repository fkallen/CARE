#ifndef CARE_EXTENDEDREAD_HPP
#define CARE_EXTENDEDREAD_HPP


#include <config.hpp>
#include <sequencehelpers.hpp>
#include <hpc_helpers.cuh>
#include <readlibraryio.hpp>
#include <options.hpp>
#include <bitcompressedstring.hpp>

#include <cstring>
#include <string>
#include <vector>

#define CARE_USE_BIT_COMPRESED_QUALITY_FOR_ENCODED_EXTENDED

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

        const std::uint8_t* copyFromContiguousMemory(const std::uint8_t* ptr){
            std::memcpy(&readId, ptr, sizeof(read_number));
            ptr += sizeof(read_number);
            std::memcpy(&encodedflags, ptr, sizeof(std::uint32_t));
            ptr += sizeof(read_number);

            const int numBytes = getNumBytes();
            data = std::make_unique<std::uint8_t[]>(numBytes);

            std::memcpy(data.get(), ptr, numBytes);
            ptr += numBytes;

            return ptr;
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
    public:
        bool mergedFromReadsWithoutMate = false;
        ExtendedReadStatus status{};
        read_number readId{};
        int read1begin = 0;
        int read1end = 0;
        int read2begin = 0;
        int read2end = 0;
    private:
        std::string extendedSequence_raw{};
        std::string qualityScores_raw{};
        //std::string_view extendedSequence_sv;
        //std::string_view qualityScores_sv;
    public:

        ExtendedRead() = default;

        bool operator==(const ExtendedRead& rhs) const noexcept{
            if(mergedFromReadsWithoutMate != rhs.mergedFromReadsWithoutMate) return false;
            if(status != rhs.status) return false;
            if(readId != rhs.readId) return false;
            if(read1begin != rhs.read1begin) return false;
            if(read1end != rhs.read1end) return false;
            if(read2begin != rhs.read2begin) return false;
            if(read2end != rhs.read2end) return false;
            if(getSequence() != rhs.getSequence()) return false;
            if(getQuality() != rhs.getQuality()) return false;
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
                + sizeof(int) + getSequence().length() //sequence
                + sizeof(int) + getQuality().length(); // quality scores
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
                l = getSequence().length();
                std::memcpy(ptr, &l, sizeof(int));
                ptr += sizeof(int);
                std::memcpy(ptr, getSequence().data(), sizeof(char) * l);
                ptr += sizeof(char) * l;

                int m = 0;
                m = getQuality().length();
                std::memcpy(ptr, &m, sizeof(int));
                ptr += sizeof(int);
                std::memcpy(ptr, getQuality().data(), sizeof(char) * m);
                ptr += sizeof(char) * m;

                return ptr;
            }else{
                return nullptr;
            }        
        }

        const std::uint8_t* copyFromContiguousMemory(const std::uint8_t* ptr){
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
            extendedSequence_raw.resize(l);
            std::memcpy(&extendedSequence_raw[0], ptr, sizeof(char) * l);
            ptr += sizeof(char) * l;

            //extendedSequence_sv = extendedSequence_raw;

            int m = 0;
            std::memcpy(&m, ptr, sizeof(int));
            ptr += sizeof(int);
            qualityScores_raw.resize(m);
            std::memcpy(&qualityScores_raw[0], ptr, sizeof(char) * m);
            ptr += sizeof(char) * m;

            //qualityScores_sv = qualityScores_raw;

            return ptr;
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

            const int numEncodedSequenceInts = SequenceHelpers::getEncodedNumInts2Bit(getSequence().size());
            std::size_t requiredBytes = 0;
            requiredBytes += sizeof(ExtendedReadStatus); // status
            requiredBytes += sizeof(int); // read1begin
            requiredBytes += sizeof(int); // read1end
            requiredBytes += sizeof(int); // read2begin
            requiredBytes += sizeof(int); // read2end
            requiredBytes += sizeof(int); // seq length
            requiredBytes += sizeof(int); // qual length
            requiredBytes += sizeof(unsigned int) * numEncodedSequenceInts; // enc seq

            #ifdef CARE_USE_BIT_COMPRESED_QUALITY_FOR_ENCODED_EXTENDED
            BitCompressedString bitcompressedquality(getQuality());
            requiredBytes += bitcompressedquality.getSerializedNumBytes();
            #else
            requiredBytes += sizeof(char) * getQuality().size(); //qual
            #endif

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
            saveint(getSequence().size());
            saveint(getQuality().size());

            SequenceHelpers::encodeSequence2Bit(
                reinterpret_cast<unsigned int*>(ptr), 
                getSequence().data(), 
                getSequence().size()
            );
            ptr += sizeof(unsigned int) * numEncodedSequenceInts;

            #ifdef CARE_USE_BIT_COMPRESED_QUALITY_FOR_ENCODED_EXTENDED
            ptr = bitcompressedquality.copyToContiguousMemory(ptr, ptr + bitcompressedquality.getSerializedNumBytes());
            assert(ptr != nullptr);
            #else
            ptr = std::copy(getQuality().begin(), getQuality().end(), ptr);
            #endif

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

            extendedSequence_raw.resize(seqlen);

            const int numEncodedSequenceInts = SequenceHelpers::getEncodedNumInts2Bit(seqlen);

            SequenceHelpers::decode2BitSequence(extendedSequence_raw.data(), reinterpret_cast<const unsigned int*>(ptr), seqlen);
            ptr += sizeof(unsigned int) * numEncodedSequenceInts;
            //extendedSequence_sv = extendedSequence_raw;

            #ifdef CARE_USE_BIT_COMPRESED_QUALITY_FOR_ENCODED_EXTENDED
            BitCompressedString bitcompressedqual;
            ptr = bitcompressedqual.copyFromContiguousMemory(ptr);
            assert(rhs.data.get() + rhs.getNumBytes() == ptr);
            qualityScores_raw = bitcompressedqual.getString();
            #else
            qualityScores_raw.resize(quallen);
            std::copy(ptr, ptr + quallen, qualityScores_raw.begin());
            assert(rhs.data.get() + rhs.getNumBytes() == ptr + quallen);
            #endif
            //qualityScores_sv = qualityScores_raw;

        }

        void removeOutwardExtension(){
            const int newlength = (read2end == -1) ? extendedSequence_raw.size() : (read2end - read1begin);

            if(qualityScores_raw.size() == extendedSequence_raw.size()){
                qualityScores_raw.erase(qualityScores_raw.begin(), qualityScores_raw.begin() + read1begin);
                qualityScores_raw.erase(qualityScores_raw.begin() + newlength, qualityScores_raw.end());
            }else{
                assert(qualityScores_raw.size() == 0);
            }

            extendedSequence_raw.erase(extendedSequence_raw.begin(), extendedSequence_raw.begin() + read1begin);
            extendedSequence_raw.erase(extendedSequence_raw.begin() + newlength, extendedSequence_raw.end());

            const int curRead1begin = read1begin;
            read1begin -= curRead1begin;
            read1end -= curRead1begin;
            if(read2begin != -1){
                read2begin -= curRead1begin;
                read2end -= curRead1begin;

                assert(read2end - read1begin == newlength);
            }
        }

        void setSequence(std::string newseq){
            extendedSequence_raw = std::move(newseq);
            //extendedSequence_sv = extendedSequence_raw;
        }

        void setQuality(std::string newqual){
            qualityScores_raw = std::move(newqual);
            //qualityScores_sv = qualityScores_raw;
        }

        std::string_view getSequence() const noexcept{
            return extendedSequence_raw;
        }

        std::string_view getQuality() const noexcept{
            return qualityScores_raw;
        }
    
    };

}



#ifdef CARE_USE_BIT_COMPRESED_QUALITY_FOR_ENCODED_EXTENDED
#undef CARE_USE_BIT_COMPRESED_QUALITY_FOR_ENCODED_EXTENDED
#endif




#endif