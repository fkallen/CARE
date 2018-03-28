#include "../inc/binarysequencehelpers.hpp"

#include <cstring>
#include <cstdint>
#include <cstdio>

#include <memory>
#include <string>

namespace care{


const uint8_t BITS_PER_BASE = 2;
const uint8_t BASE_A = 0x00;
const uint8_t BASE_C = 0x01;
const uint8_t BASE_G = 0x02;
const uint8_t BASE_T = 0x03;
const uint8_t BASES_PER_BYTE = (sizeof(uint8_t)*8 / BITS_PER_BASE);

static_assert(BASES_PER_BYTE == 4, "unexpected size of uint8_t"); // only tested for sizeof(std::uint8_t) == 1

// REVERSE_COMPLEMENT_2BIT[ X ] is the reverse complement of X , where X is the encoded value of 4 bases
const uint8_t REVERSE_COMPLEMENT_2BIT[256] = {
        0xff, 0xbf, 0x7f, 0x3f, 0xef, 0xaf, 0x6f, 0x2f, 0xdf, 0x9f, 0x5f, 0x1f, 0xcf, 0x8f, 0x4f, 0xf,
        0xfb, 0xbb, 0x7b, 0x3b, 0xeb, 0xab, 0x6b, 0x2b, 0xdb, 0x9b, 0x5b, 0x1b, 0xcb, 0x8b, 0x4b, 0xb,
        0xf7, 0xb7, 0x77, 0x37, 0xe7, 0xa7, 0x67, 0x27, 0xd7, 0x97, 0x57, 0x17, 0xc7, 0x87, 0x47, 0x7,
        0xf3, 0xb3, 0x73, 0x33, 0xe3, 0xa3, 0x63, 0x23, 0xd3, 0x93, 0x53, 0x13, 0xc3, 0x83, 0x43, 0x3,
        0xfe, 0xbe, 0x7e, 0x3e, 0xee, 0xae, 0x6e, 0x2e, 0xde, 0x9e, 0x5e, 0x1e, 0xce, 0x8e, 0x4e, 0xe,
        0xfa, 0xba, 0x7a, 0x3a, 0xea, 0xaa, 0x6a, 0x2a, 0xda, 0x9a, 0x5a, 0x1a, 0xca, 0x8a, 0x4a, 0xa,
        0xf6, 0xb6, 0x76, 0x36, 0xe6, 0xa6, 0x66, 0x26, 0xd6, 0x96, 0x56, 0x16, 0xc6, 0x86, 0x46, 0x6,
        0xf2, 0xb2, 0x72, 0x32, 0xe2, 0xa2, 0x62, 0x22, 0xd2, 0x92, 0x52, 0x12, 0xc2, 0x82, 0x42, 0x2,
        0xfd, 0xbd, 0x7d, 0x3d, 0xed, 0xad, 0x6d, 0x2d, 0xdd, 0x9d, 0x5d, 0x1d, 0xcd, 0x8d, 0x4d, 0xd,
        0xf9, 0xb9, 0x79, 0x39, 0xe9, 0xa9, 0x69, 0x29, 0xd9, 0x99, 0x59, 0x19, 0xc9, 0x89, 0x49, 0x9,
        0xf5, 0xb5, 0x75, 0x35, 0xe5, 0xa5, 0x65, 0x25, 0xd5, 0x95, 0x55, 0x15, 0xc5, 0x85, 0x45, 0x5,
        0xf1, 0xb1, 0x71, 0x31, 0xe1, 0xa1, 0x61, 0x21, 0xd1, 0x91, 0x51, 0x11, 0xc1, 0x81, 0x41, 0x1,
        0xfc, 0xbc, 0x7c, 0x3c, 0xec, 0xac, 0x6c, 0x2c, 0xdc, 0x9c, 0x5c, 0x1c, 0xcc, 0x8c, 0x4c, 0xc,
        0xf8, 0xb8, 0x78, 0x38, 0xe8, 0xa8, 0x68, 0x28, 0xd8, 0x98, 0x58, 0x18, 0xc8, 0x88, 0x48, 0x8,
        0xf4, 0xb4, 0x74, 0x34, 0xe4, 0xa4, 0x64, 0x24, 0xd4, 0x94, 0x54, 0x14, 0xc4, 0x84, 0x44, 0x4,
        0xf0, 0xb0, 0x70, 0x30, 0xe0, 0xa0, 0x60, 0x20, 0xd0, 0x90, 0x50, 0x10, 0xc0, 0x80, 0x40, 0x0
};

bool encode(const char * sequence, int sequencelength, int k_, uint8_t* encoded, int encodedlength, bool failOnUnknownBase){

        int k = sequencelength < k_ ? sequencelength : k_;

        bool success = encodedlength * 8 >= BITS_PER_BASE * k && encoded != nullptr && sequence != nullptr;

        if(success) {
		        memset(encoded, 0, encodedlength);

                for(int j = 0; j< k && success; ++j) {
                        int byteIndex = j / 4;
                        int pos = j % 4;
                        switch(sequence[j]) {
                        case 'A':
                                encoded[byteIndex] |= BASE_A << (2*(3-pos));
                                break;
                        case 'C':
                                encoded[byteIndex] |= BASE_C << (2*(3-pos));
                                break;
                        case 'G':
                                encoded[byteIndex] |= BASE_G << (2*(3-pos));
                                break;
                        case 'T':
                                encoded[byteIndex] |= BASE_T << (2*(3-pos));
                                break;
                        default:
                                success = !failOnUnknownBase;
                                encoded[byteIndex] |= BASE_A << (2*(3-pos));
                                break;
                        }
                }
        }

        return success;
}


bool decode(const uint8_t* encoded, int encodedlength, int k_, char* sequence, int sequencelength){

        const int k = encodedlength * BASES_PER_BYTE < k_ ? encodedlength * BASES_PER_BYTE : k_;
        // sequence must store at least k chars + '\0'
        const bool success = sequencelength > k && encoded != nullptr && sequence != nullptr;

        if(success) {
            for(int i = 0; i < k; i++){
                const int byteIndex = i / 4;
                const int pos = i % 4;
                const std::uint8_t base = (encoded[byteIndex] >> (2*(3-pos))) & 0x03;
                switch(base){
                    case BASE_A: sequence[i] = 'A';
                            break;
                    case BASE_C: sequence[i] = 'C';
                            break;
                    case BASE_G: sequence[i] = 'G';
                            break;
                    case BASE_T: sequence[i] = 'T';
                            break;
                    default: // this cannot happen
                            break;
                }
            }

            sequence[k] = '\0';
        }

        return success;
}

bool encoded_to_reverse_complement_encoded(const uint8_t* encoded, int encodedlength, uint8_t* rcencoded, int rcencodedlength, int k_){

        const int k = encodedlength * BASES_PER_BYTE < k_ ? encodedlength * BASES_PER_BYTE : k_;
        const bool success = rcencodedlength * BASES_PER_BYTE >= k && encoded != nullptr && rcencoded != nullptr;

        if(success) {
                // be safe. don't assume rcencoded is already zero'd
                memset(rcencoded, 0, rcencodedlength);
                const int bytes = (k + 3) / 4;
                const int unusedPositions = bytes * 4 - k;

                for(int i = 0; i < bytes; i++){
                    rcencoded[i] = REVERSE_COMPLEMENT_2BIT[encoded[bytes - 1 - i]];
                }

                if(unusedPositions > 0){
                    rcencoded[0] <<= (2 * unusedPositions);
                    for(int i = 1; i < bytes; i++){
                        rcencoded[i-1] |= rcencoded[i] >> (2 * (4-unusedPositions));
                        rcencoded[i] <<= (2 * unusedPositions);
                    }
                }
        }

        return success;
}



std::pair<std::unique_ptr<std::uint8_t[]>, std::size_t> encode_2bit(const std::string& sequence){
	const std::size_t l = sequence.length();
	const std::size_t bytes = (l + 3) / 4;

	std::unique_ptr<std::uint8_t[]> encoded = std::make_unique<std::uint8_t[]>(bytes);

	std::memset(encoded.get(), 0, bytes);

	for(std::size_t i = 0; i < l; i++){
		const std::size_t byteIndex = i / 4;
		const std::size_t pos = i % 4;
        switch(sequence[i]) {
        case 'A':
                encoded[byteIndex] |= BASE_A << (2*(3-pos));
                break;
        case 'C':
                encoded[byteIndex] |= BASE_C << (2*(3-pos));
                break;
        case 'G':
                encoded[byteIndex] |= BASE_G << (2*(3-pos));
                break;
        case 'T':
                encoded[byteIndex] |= BASE_T << (2*(3-pos));
                break;
        default:
                encoded[byteIndex] |= BASE_A << (2*(3-pos));
                break;
        }
	}

	return {std::move(encoded), bytes};
}

std::string decode_2bit(const std::unique_ptr<std::uint8_t[]>& encoded, std::size_t bases){
	std::string sequence;
	sequence.reserve(bases);

	for(std::size_t i = 0; i < bases; i++){
		const std::size_t byteIndex = i / 4;
		const std::size_t pos = i % 4;
        const std::uint8_t base = (encoded[byteIndex] >> (2*(3-pos))) & 0x03;
        switch(base){
            case BASE_A: sequence.push_back('A'); break;
            case BASE_C: sequence.push_back('C'); break;
            case BASE_G: sequence.push_back('G'); break;
            case BASE_T: sequence.push_back('T'); break;
        default: sequence.push_back('_'); break; // cannot happen
        }
	}

	return sequence;
}

}
