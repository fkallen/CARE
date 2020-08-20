#ifndef CARE_CORRECTION_RESULT_PROCESSING_HPP
#define CARE_CORRECTION_RESULT_PROCESSING_HPP

#include <config.hpp>
#include <hpc_helpers.cuh>
#include <memoryfile.hpp>
#include <readlibraryio.hpp>
#include <sequence.hpp>

#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace care{

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
    };

    // represents a sequence produced by the correction of a read.
    // Will be saved to file during correction.
    // Will be loaded from file during mergeResultFiles
    struct TempCorrectedSequence{
        enum class Type : int {Anchor, Candidate};
        struct EncodedEdit{
            // char b;
            // int p;
            std::uint16_t data;

            EncodedEdit() = default;
            
            HOSTDEVICEQUALIFIER
            explicit EncodedEdit(int p, char b){
                data = p;
                data = data << 2;
                std::uint16_t enc = convertDNACharToIntNoIf(b);
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
                return convertIntToDNACharNoIf(enc);
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
                std::uint16_t enc = convertDNACharToIntNoIf(b);
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

        // struct EncodedEdit{
        //     char b;
        //     int p;

        //     EncodedEdit() = default;
        //     HOSTDEVICEQUALIFIER
        //     EncodedEdit(int p, char b) : b(b), p(p){}

        //     HOSTDEVICEQUALIFIER
        //     bool operator==(const EncodedEdit& rhs) const{
        //         return b == rhs.b && p == rhs.p;
        //     }

        //     HOSTDEVICEQUALIFIER
        //     bool operator!=(const EncodedEdit& rhs) const{
        //         return !(operator==(rhs));
        //     }

        //     int pos() const{
        //         return p;
        //     }

        //     char base() const{
        //         return b;
        //     }

        //     void pos(int i){
        //         p = i;
        //     }

        //     void base(char c){
        //         b = c;
        //     }
        // };

        struct Edit{
            char base;
            int pos;

            Edit() = default;
            HOSTDEVICEQUALIFIER
            Edit(int p, char b) : base(b), pos(p){}

            HOSTDEVICEQUALIFIER
            bool operator==(const Edit& rhs) const{
                return base == rhs.base && pos == rhs.pos;
            }

            HOSTDEVICEQUALIFIER
            bool operator!=(const Edit& rhs) const{
                return !(operator==(rhs));
            }

            Edit& operator=(const Edit& rhs) = default;
            Edit& operator=(const EncodedEdit& rhs){
                base = rhs.base();
                pos = rhs.pos();
                return *this;
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
        Type type = Type::Anchor;
        int shift = 0; //if candidate
        read_number readId = 0;

        std::string sequence = "";
        std::vector<Edit> edits;
        std::vector<int> uncorrectedPositionsNoConsensus{}; //if anchor

        bool operator==(const TempCorrectedSequence& rhs) const{
            return hq == rhs.hq && useEdits == rhs.useEdits && type == rhs.type && shift == rhs.shift && readId == rhs.readId
                && sequence == rhs.sequence && edits == rhs.edits && uncorrectedPositionsNoConsensus == rhs.uncorrectedPositionsNoConsensus;
        }

        bool operator!=(const TempCorrectedSequence& rhs) const{
            return !(operator==(rhs));
        }

        
    };



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
        const std::string& tempdir,
        const std::vector<std::string>& originalReadFiles,
        MemoryFileFixedSize<EncodedTempCorrectedSequence>& partialResults, 
        std::size_t memoryForSorting,
        FileFormat outputFormat,
        const std::vector<std::string>& outputfiles,
        bool isSorted
    );





    std::ostream& operator<<(std::ostream& os, const TempCorrectedSequence& tmp);
    std::istream& operator>>(std::istream& is, TempCorrectedSequence& tmp);




}


#endif