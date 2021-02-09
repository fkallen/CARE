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


}


#endif