#ifndef CARE_CORRECTOR_COMMON_HPP
#define CARE_CORRECTOR_COMMON_HPP

#include <config.hpp>
#include <correctedsequence.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

namespace care{

    class CorrectionOutput{
    public:
        std::vector<TempCorrectedSequence> anchorCorrections{};
        std::vector<TempCorrectedSequence> candidateCorrections{};
    };

    class EncodedCorrectionOutput{
    public:
        EncodedCorrectionOutput() = default;
        EncodedCorrectionOutput(const EncodedCorrectionOutput&) = default;
        EncodedCorrectionOutput(EncodedCorrectionOutput&&) = default;

        EncodedCorrectionOutput(const CorrectionOutput& rhs){
            encodedAnchorCorrections.resize(rhs.anchorCorrections.size());
            encodedCandidateCorrections.resize(rhs.candidateCorrections.size());

            for(std::size_t i = 0; i < rhs.anchorCorrections.size(); i++){
                rhs.anchorCorrections[i].encodeInto(encodedAnchorCorrections[i]);
            }

            for(std::size_t i = 0; i < rhs.candidateCorrections.size(); i++){
                rhs.candidateCorrections[i].encodeInto(encodedCandidateCorrections[i]);
            }
        }

        EncodedCorrectionOutput& operator=(EncodedCorrectionOutput rhs){
            std::swap(*this, rhs);
            return *this;
        }

        EncodedCorrectionOutput& operator=(const CorrectionOutput& rhs){
            EncodedCorrectionOutput tmp(rhs);
            std::swap(*this, tmp);
            return *this;
        }

        std::vector<EncodedTempCorrectedSequence> encodedAnchorCorrections{};
        std::vector<EncodedTempCorrectedSequence> encodedCandidateCorrections{};
    };

    class SerializedEncodedCorrectionOutput{
    public:
        SerializedEncodedCorrectionOutput() = default;
        SerializedEncodedCorrectionOutput(const SerializedEncodedCorrectionOutput&) = default;
        SerializedEncodedCorrectionOutput(SerializedEncodedCorrectionOutput&&) = default;

        SerializedEncodedCorrectionOutput& operator=(SerializedEncodedCorrectionOutput rhs){
            std::swap(*this, rhs);
            return *this;
        }

        int numAnchors = 0;
        int numCandidates = 0;
        std::vector<std::uint8_t> serializedEncodedAnchorCorrections{};
        std::vector<std::size_t> beginOffsetsAnchors{};
        std::vector<std::uint8_t> serializedEncodedCandidateCorrections{};
        std::vector<std::size_t> beginOffsetsCandidates{};
    };

    class ReadCorrectionFlags{
    public:
        ReadCorrectionFlags() = default;

        ReadCorrectionFlags(std::size_t numReads)
            : size(numReads), flags(std::make_unique<std::uint8_t[]>(numReads)){
            std::fill(flags.get(), flags.get() + size, 0);

            #ifdef __CUDACC__
            cudaError_t status = cudaHostRegister(flags.get(), sizeInBytes(), cudaHostRegisterDefault);
            assert(status == cudaSuccess);
            #endif
        }

        ~ReadCorrectionFlags(){
            cudaHostUnregister(flags.get());
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

        #ifdef __CUDACC__
        void isCorrectedAsHQAnchor(bool* output, const read_number* readIds, int numIds, cudaStream_t stream) const noexcept{
            helpers::lambda_kernel<<<SDIV(numIds, 128), 128, 0, stream>>>(
                [
                    flags = flags.get(),
                    output,
                    readIds,
                    numIds
                ] __device__(){
                    constexpr std::uint8_t HQVALUE = 1;
                    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    const int stride = blockDim.x * gridDim.x;

                    for(int i = tid; i < numIds; i += stride){
                        if(flags[readIds[i]] == HQVALUE){
                            output[i] = true;
                        }else{
                            output[i] = false;
                        }
                    }
                }
            );
            CUDACHECKASYNC;
        }
        #endif

    private:
        static constexpr std::uint8_t readCorrectedAsHQAnchor() noexcept{ return 1; };
        static constexpr std::uint8_t readCouldNotBeCorrectedAsAnchor() noexcept{ return 2; };

        std::size_t size;
        std::unique_ptr<std::uint8_t[]> flags{};
    };


}


#endif