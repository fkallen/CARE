#ifndef CARE_CORRECTION_RESULT_PROCESSING_HPP
#define CARE_CORRECTION_RESULT_PROCESSING_HPP

#include <config.hpp>
#include <hpc_helpers.cuh>
#include <readlibraryio.hpp>
#include <sequencehelpers.hpp>
#include <serializedobjectstorage.hpp>

#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace care{

    enum class TempCorrectedSequenceType : int {Anchor, Candidate};

    struct TempCorrectedSequence; //forward declaration

    struct EncodedTempCorrectedSequence{
        std::uint32_t encodedflags{}; //contains size of data in bytes, and boolean flags
        read_number readId{};
        std::unique_ptr<std::uint8_t[]> data{};

        EncodedTempCorrectedSequence() = default;
        EncodedTempCorrectedSequence(EncodedTempCorrectedSequence&& rhs){
            *this = std::move(rhs);
        }

        EncodedTempCorrectedSequence& operator=(EncodedTempCorrectedSequence&& rhs){
            encodedflags = std::exchange(rhs.encodedflags, 0);
            readId = std::exchange(rhs.readId, 0);
            data = std::move(rhs.data);

            return *this;
        }

        EncodedTempCorrectedSequence(const EncodedTempCorrectedSequence& rhs){
            *this = rhs;
        }

        EncodedTempCorrectedSequence& operator=(const EncodedTempCorrectedSequence& rhs){
            encodedflags = rhs.encodedflags;
            readId = rhs.readId;

            const int numBytes = rhs.getNumBytes();
            data = std::make_unique<std::uint8_t[]>(numBytes);
            std::memcpy(data.get(), rhs.data.get(), numBytes);

            return *this;
        }

        EncodedTempCorrectedSequence& operator=(const TempCorrectedSequence& rhs);

        bool writeToBinaryStream(std::ostream& s) const;
        bool readFromBinaryStream(std::istream& s);

        std::uint8_t* copyToContiguousMemory(std::uint8_t*, std::uint8_t*) const;
        void copyFromContiguousMemory(const std::uint8_t*);

        bool operator==(const EncodedTempCorrectedSequence& rhs) const{
            const std::uint32_t numBytes = getNumBytes();
            return encodedflags == rhs.encodedflags && readId == rhs.readId 
                    && std::memcmp(data.get(), rhs.data.get(), numBytes);
        }

        bool operator!=(const EncodedTempCorrectedSequence& rhs) const{
            return !(operator==(rhs));
        }

        int getNumBytes() const{
            constexpr std::uint32_t mask = (std::uint32_t(1) << 29)-1;
            return (encodedflags & mask);
        }

        int getSerializedNumBytes() const noexcept{
            return sizeof(read_number) + sizeof(std::uint32_t) + getNumBytes();
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

        bool isHQ() const noexcept{
            return (encodedflags >> 31) & std::uint32_t(1);
        }

        bool useEdits() const noexcept{
            return (encodedflags >> 30) & std::uint32_t(1);
        }

        int getNumEdits() const noexcept{
            if(useEdits()){
                //num edits is stored in the first int of encoded data
                int num;
                std::memcpy(&num, data.get(), sizeof(int));

                return num;
            }else{
                return 0;
            }
        }

        TempCorrectedSequenceType getType() const noexcept{
            return TempCorrectedSequenceType((encodedflags >> 29) & std::uint32_t(1));
        }
    };

    // represents a sequence produced by the correction of a read.
    // Will be saved to file during correction.
    // Will be loaded from file during mergeResultFiles
    struct TempCorrectedSequence{
        
        struct EncodedEdit{
            // char b;
            // int p;
            std::uint16_t data;

            EncodedEdit() = default;
            
            HOSTDEVICEQUALIFIER
            explicit EncodedEdit(int p, char b){
                data = p;
                data = data << 2;
                std::uint16_t enc = SequenceHelpers::encodeBase(b);
                data |= enc;

                // this->p = p;
                // this->b = b;
            }

            HOSTDEVICEQUALIFIER
            explicit EncodedEdit(int p, std::uint8_t b){
                data = p;
                data = data << 2;
                data |= b;

                // this->p = p;
                // this->b = convertIntToDNACharNoIf(b);
            }

            HOSTDEVICEQUALIFIER
            bool operator==(const EncodedEdit& rhs) const{
                return data == rhs.data;
            }

            HOSTDEVICEQUALIFIER
            bool operator!=(const EncodedEdit& rhs) const{
                return !(operator==(rhs));
            }

            HOSTDEVICEQUALIFIER
            int pos() const{
                return data >> 2;
                //return p;
            }

            HOSTDEVICEQUALIFIER
            char base() const{
                std::uint8_t enc = data & 0x03;
                return SequenceHelpers::decodeBase(enc);
                //return b;
            }

            HOSTDEVICEQUALIFIER
            void pos(int i){
                std::uint16_t a = i;
                data = (a << 2) | (data & 0x03);
                //p = i;
            }

            HOSTDEVICEQUALIFIER
            void base(char b){
                std::uint16_t enc = SequenceHelpers::convertDNACharToIntNoIf(b);
                data = (data & ~0x03) | enc;
                //this->b = b;
            }

            HOSTDEVICEQUALIFIER
            void base(std::uint8_t b){
                std::uint16_t enc = b;
                data = (data & ~0x03) | enc;

                //this->b = convertIntToDNACharNoIf(b);
            }
        };
        struct Edit{
            char base_;
            int pos_;

            Edit() = default;
            HOSTDEVICEQUALIFIER
            Edit(int p, char b) : base_(b), pos_(p){}

            HOSTDEVICEQUALIFIER
            Edit(const EncodedEdit& rhs) : base_(rhs.base()), pos_(rhs.pos()){}

            HOSTDEVICEQUALIFIER
            bool operator==(const Edit& rhs) const{
                return base() == rhs.base() && pos() == rhs.pos();
            }

            HOSTDEVICEQUALIFIER
            bool operator!=(const Edit& rhs) const{
                return !(operator==(rhs));
            }

            Edit& operator=(const Edit& rhs) = default;

            HOSTDEVICEQUALIFIER
            Edit& operator=(const EncodedEdit& rhs){
                base(rhs.base());
                pos(rhs.pos());
                return *this;
            }

            HOSTDEVICEQUALIFIER
            char base() const noexcept{
                return base_;
            }

            HOSTDEVICEQUALIFIER
            int pos() const noexcept{
                return pos_;
            }

            HOSTDEVICEQUALIFIER
            void base(char b) noexcept{
                base_ = b;
            }

            HOSTDEVICEQUALIFIER
            void pos(int i) noexcept{
                pos_ = i;
            }
        };

        static constexpr char AnchorChar = 'a';
        static constexpr char CandidateChar = 'c';

        TempCorrectedSequence() = default;
        TempCorrectedSequence(const TempCorrectedSequence&) = default;
        TempCorrectedSequence(TempCorrectedSequence&&) = default;
        TempCorrectedSequence& operator=(const TempCorrectedSequence&) = default;
        TempCorrectedSequence& operator=(TempCorrectedSequence&&) = default;

        TempCorrectedSequence(const EncodedTempCorrectedSequence&);
        TempCorrectedSequence& operator=(const EncodedTempCorrectedSequence&);

        EncodedTempCorrectedSequence encode() const;
        void encodeInto(EncodedTempCorrectedSequence&) const;
        void decode(const EncodedTempCorrectedSequence&);

        bool writeToBinaryStream(std::ostream& s) const;
        bool readFromBinaryStream(std::istream& s);

        bool hq = false; //if anchor
        bool useEdits = false;
        TempCorrectedSequenceType type = TempCorrectedSequenceType::Anchor;
        int shift = 0; //if candidate
        read_number readId = 0;

        std::string sequence = "";
        std::vector<Edit> edits;

        bool operator==(const TempCorrectedSequence& rhs) const{
            return hq == rhs.hq && useEdits == rhs.useEdits && type == rhs.type && shift == rhs.shift && readId == rhs.readId
                && sequence == rhs.sequence && edits == rhs.edits;
        }

        bool operator!=(const TempCorrectedSequence& rhs) const{
            return !(operator==(rhs));
        }

        
    };

    void encodeDataIntoEncodedCorrectedSequence(
        EncodedTempCorrectedSequence& target,
        read_number readId,
        bool hq,
        bool useEdits,
        TempCorrectedSequenceType type,
        int shift,
        int numEdits,
        const TempCorrectedSequence::Edit* edits,
        int sequenceLength,
        const char* sequence
    );



    /*
        Merges temporary results with unordered reads into single file outputfile with ordered reads.
        Quality scores and missing sequences are taken from original file.
        Temporary result files are expected to be in format:

        readnumber
        sequence
        readnumber
        sequence
        ...
    */

    void constructOutputFileFromCorrectionResults(
        const std::vector<std::string>& originalReadFiles,
        SerializedObjectStorage& partialResults, 
        FileFormat outputFormat,
        const std::vector<std::string>& outputfiles,
        bool showProgress
    );



    std::ostream& operator<<(std::ostream& os, const TempCorrectedSequence& tmp);
    std::istream& operator>>(std::istream& is, TempCorrectedSequence& tmp);




}


#endif