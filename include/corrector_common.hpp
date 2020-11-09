#ifndef CARE_CORRECTOR_COMMON_HPP
#define CARE_CORRECTOR_COMMON_HPP

#include <config.hpp>
#include <correctionresultprocessing.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

namespace care{

    class CorrectionOutput{
    public:
        void encode(){
            encodedAnchorCorrections.resize(anchorCorrections.size());
            encodedCandidateCorrections.resize(candidateCorrections.size());

            for(std::size_t i = 0; i < anchorCorrections.size(); i++){
                anchorCorrections[i].encodeInto(encodedAnchorCorrections[i]);
            }

            for(std::size_t i = 0; i < candidateCorrections.size(); i++){
                candidateCorrections[i].encodeInto(encodedCandidateCorrections[i]);
            }
        }

        std::vector<TempCorrectedSequence> anchorCorrections;
        std::vector<TempCorrectedSequence> candidateCorrections;
        std::vector<EncodedTempCorrectedSequence> encodedAnchorCorrections;
        std::vector<EncodedTempCorrectedSequence> encodedCandidateCorrections;
    };

    class ReadCorrectionFlags{
    public:
        ReadCorrectionFlags() = default;

        ReadCorrectionFlags(std::size_t numReads)
            : size(numReads), flags(std::make_unique<std::uint8_t[]>(numReads)){
            std::fill(flags.get(), flags.get() + size, 0);
        }

        std::size_t sizeInBytes() const noexcept{
            return size * sizeof(std::uint8_t);
        }

        bool isCorrectedAsHQAnchor(std::int64_t position) const noexcept{
            return (flags[position] & readCorrectedAsHQAnchor()) > 0;
        }

        bool isNotCorrectedAsAnchor(std::int64_t position) const noexcept{
            return (flags[position] & readCouldNotBeCorrectedAsAnchor()) > 0;
        }

        void setCorrectedAsHqAnchor(std::int64_t position) const noexcept{
            flags[position] = readCorrectedAsHQAnchor();
        }

        void setCouldNotBeCorrectedAsAnchor(std::int64_t position) const noexcept{
            flags[position] = readCouldNotBeCorrectedAsAnchor();
        }

    private:
        static constexpr std::uint8_t readCorrectedAsHQAnchor() noexcept{ return 1; };
        static constexpr std::uint8_t readCouldNotBeCorrectedAsAnchor() noexcept{ return 2; };

        std::size_t size;
        std::unique_ptr<std::uint8_t[]> flags{};
    };


    class ReadProvider{
    public:
        bool readContainsN(read_number readId) const{
            return readContainsN_impl(readId);
        }

        void gatherSequenceLengths(const read_number* readIds, int numIds, int* lengths) const{
            gatherSequenceLengths_impl(readIds, numIds, lengths);
        }

        void gatherSequenceData(const read_number* readIds, int numIds, unsigned int* sequenceData, std::size_t encodedSequencePitchInInts) const{
            gatherSequenceData_impl(readIds, numIds, sequenceData, encodedSequencePitchInInts);
        }

        void gatherSequenceQualities(const read_number* readIds, int numIds, char* qualities, std::size_t qualityPitchInBytes) const{
            gatherSequenceQualities_impl(readIds, numIds, qualities, qualityPitchInBytes);
        }

    private:
        virtual bool readContainsN_impl(read_number readId) const = 0;

        virtual void gatherSequenceLengths_impl(const read_number* readIds, int numIds, int* lengths) const = 0;

        virtual void gatherSequenceData_impl(const read_number* readIds, int numIds, unsigned int* sequenceData, std::size_t encodedSequencePitchInInts) const = 0;

        virtual void gatherSequenceQualities_impl(const read_number* readIds, int numIds, char* qualities, std::size_t qualityPitchInBytes) const = 0;
    };

    class CandidateIdsProvider{
    public: 
        void getCandidates(std::vector<read_number>& ids, const char* anchor, const int size) const{
            getCandidates_impl(ids, anchor, size);
        }
    private:
        virtual void getCandidates_impl(std::vector<read_number>& ids, const char* anchor, const int size) const = 0;
    };

}


#endif