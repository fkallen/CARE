#ifndef CARE_CORRECTION_RESULT_PROCESSING_HPP
#define CARE_CORRECTION_RESULT_PROCESSING_HPP

#include <config.hpp>
#include <hpc_helpers.cuh>
#include <memoryfile.hpp>
#include <readlibraryio.hpp>

#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace care{


    struct EncodedTempCorrectedSequence{
        std::uint32_t encodedflags; //contains size of data in bytes, and boolean flags
        read_number readId;
        std::unique_ptr<std::uint8_t[]> data;

        EncodedTempCorrectedSequence() = default;
        EncodedTempCorrectedSequence(EncodedTempCorrectedSequence&&) = default;
        EncodedTempCorrectedSequence& operator=(EncodedTempCorrectedSequence&&) = default;

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

        bool writeToBinaryStream(std::ostream& s) const;
        bool readFromBinaryStream(std::istream& s);

        std::uint8_t* copyToContiguousMemory(std::uint8_t*, std::uint8_t*) const;
        void copyFromContiguousMemory(const std::uint8_t*);

        bool operator==(const EncodedTempCorrectedSequence& rhs) const{
            std::uint32_t numBytes = 123;
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


    void constructOutputFileFromResults(
                        const std::string& tempdir,
                        std::uint32_t expectedNumReads, 
                        const std::string& originalReadFile,
                        FileFormat originalFormat,
                        MemoryFileFixedSize<EncodedTempCorrectedSequence>& partialResults, 
                        std::size_t memoryForSorting,
                        const std::string& outputfile,
                        bool isSorted);

    void constructOutputFileFromResults2(
                    const std::string& tempdir,
                    const std::vector<std::string>& originalReadFiles,
                    MemoryFileFixedSize<EncodedTempCorrectedSequence>& partialResults, 
                    std::size_t memoryForSorting,
                    FileFormat outputFormat,
                    const std::vector<std::string>& outputfiles,
                    bool isSorted);





    std::ostream& operator<<(std::ostream& os, const TempCorrectedSequence& tmp);
    std::istream& operator>>(std::istream& is, TempCorrectedSequence& tmp);




}


#endif