#include "../inc/util.hpp"

#include <algorithm>


#define SDIV(x,y)(((x)+(y)-1)/(y))

void shiftBitsLeftBy(unsigned char* array, int bytes, int shiftamount){
	constexpr int maxshiftPerIter = 7;
	const int iters = SDIV(shiftamount, maxshiftPerIter);
	for(int iter = 0; iter < iters-1; ++iter){
		for(int i = 0; i < bytes - 1; ++i){
			array[i] = (array[i] << maxshiftPerIter) | (array[i+1] >> (8 - maxshiftPerIter));
		}
		array[bytes - 1] <<= maxshiftPerIter;

		shiftamount -= maxshiftPerIter;
	}

	for(int i = 0; i < bytes - 1; ++i){
		array[i] = (array[i] << shiftamount) | (array[i+1] >> (8 - shiftamount));
	}
	array[bytes - 1] <<= shiftamount;
}

void shiftBitsRightBy(unsigned char* array, int bytes, int shiftamount){
	constexpr int maxshiftPerIter = 7;
	const int iters = SDIV(shiftamount, maxshiftPerIter);
	for(int iter = 0; iter < iters-1; ++iter){
		for(int i = bytes - 1; i > 0; --i){
			array[i] = (array[i] >> maxshiftPerIter) | (array[i-1] << (8 - maxshiftPerIter));
		}
		array[0] >>= maxshiftPerIter;

		shiftamount -= maxshiftPerIter;
	}

	for(int i = bytes - 1; i > 0; --i){
		array[i] = (array[i] >> shiftamount) | (array[i-1] << (8 - shiftamount));
	}
	array[0] >>= shiftamount;
}

void shiftBitsBy(unsigned char* array, int bytes, int shiftamount){
	if(shiftamount < 0){
		shiftamount = -shiftamount;
		shiftBitsLeftBy(array, bytes, shiftamount);
	}else{
		shiftBitsRightBy(array, bytes, shiftamount);
	}
}

int hammingdistanceHiLo(const std::uint8_t* l, const std::uint8_t* r, int length_l, int length_r, int bytes){
	using T = int;

	const int overlap = length_l < length_r ? length_l : length_r;

	const int halfbytes = bytes/2;
	const int partitions = (overlap / 8) / sizeof(T);

	const T* const lhi = (const T*)l;
	const T* const llo = (const T*)(l + halfbytes);

	const T* const rhi = (const T*)r;
	const T* const rlo = (const T*)(r + halfbytes);

	int result = 0;

	for(int i = 0; i < partitions; i++){
		const T hixor = lhi[i] ^ rhi[i];
		const T loxor = llo[i] ^ rlo[i];
		const T bits = hixor | loxor;
		result += __builtin_popcount(bits);
	}

	int remainingbits = overlap - partitions * sizeof(T) * 8;
	if(remainingbits != 0){
		const int charpartitions = (remainingbits / 8) / sizeof(std::uint8_t);

		const std::uint8_t* const lhichar = (const std::uint8_t*)(lhi + partitions);
		const std::uint8_t* const llochar = (const std::uint8_t*)(llo + partitions);
		const std::uint8_t* const rhichar = (const std::uint8_t*)(rhi + partitions);
		const std::uint8_t* const rlochar = (const std::uint8_t*)(rlo + partitions);

		for(int i = 0; i < charpartitions; i++){
			const std::uint8_t hixorchar = lhichar[i] ^ rhichar[i];
			const std::uint8_t loxorchar = llochar[i] ^ rlochar[i];
			const std::uint8_t bitschar = hixorchar | loxorchar;
			result += __builtin_popcount(bitschar);
		}

		remainingbits = remainingbits - charpartitions * sizeof(std::uint8_t) * 8;

		if(remainingbits != 0){
			std::uint8_t mask = 0xFF << (sizeof(std::uint8_t)*8 - remainingbits);
			const T hixorchar2 = lhichar[charpartitions] ^ rhichar[charpartitions];
			const T loxorchar2 = llochar[charpartitions] ^ rlochar[charpartitions];
			const T bitschar2 = hixorchar2 | loxorchar2;
			result += __builtin_popcount(bitschar2 & mask);
		}

	}

	return result;
}







#undef SDIV
