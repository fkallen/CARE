#ifndef BINARY_SEQUENCE_HELPERS_HPP
#define BINARY_SEQUENCE_HELPERS_HPP

#include <cstring>
#include <cstdint>

#include <memory>
#include <string>

namespace care{

// encode the first min(sequencelenght, k_) bases in sequence and store in encoded which is encodelength bytes
// sequencelength includes '\0'
// if failOnUnknownBase is true, encode fails if a base not A C G or T
// the bases are aligned to the right,
// i.e. if encoded can store 12 bases and 5 bases are stored in it, the bases will be stored at positions 7 to 11
// return true on success, false otherwise
bool encode(const char * sequence, int sequencelength, int k_, uint8_t* encoded, int encodedlength, bool failOnUnknownBase);

// decode the last min(encodedlength * BASES_PER_BYTE, k_) bases of encoded and store them in sequence
// '\0' will be added after last decoded base
// return true on success, false otherwise
bool decode(const uint8_t* encoded, int encodedlength, int k_, char* sequence, int sequencelength);


// store the reverse complement of encoded in rcencoded
// return true on success, false otherwise
bool encoded_to_reverse_complement_encoded(const uint8_t* encoded, int encodedlength, uint8_t* rcencoded, int rcencodedlength, int k_);

std::pair<std::unique_ptr<std::uint8_t[]>, std::size_t> encode_2bit(const std::string& sequence);
std::string decode_2bit(const std::unique_ptr<std::uint8_t[]>& encoded, std::size_t bases);
std::string decode_2bit(const std::uint8_t* encoded, std::size_t bases);
std::pair<std::unique_ptr<std::uint8_t[]>, int> reverse_complement_2bit(const uint8_t* encoded, int bases);


std::pair<std::unique_ptr<std::uint8_t[]>, int> encode_2bit_hilo(const char* sequence, int nbases);
std::pair<std::unique_ptr<std::uint8_t[]>, int> encode_2bit_hilo(const std::string& sequence);
std::string decode_2bit_hilo(const std::uint8_t* encoded, int bases);
std::string decode_2bit_hilo(const std::unique_ptr<std::uint8_t[]>& encoded, int bases);
void reverse_complement_2bit_hilo(const uint8_t* encoded, int bases, uint8_t* rcencoded);
std::pair<std::unique_ptr<std::uint8_t[]>, int> reverse_complement_2bit_hilo(const uint8_t* encoded, int bases);



}

#endif
