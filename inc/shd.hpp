#ifndef SHD_HPP
#define SHD_HPP

#include "hammingtools.hpp"
#include "alignment.hpp"

namespace hammingtools{

	namespace alignment{

		AlignResultCompact cpu_shifted_hamming_distance(const char* subject, const char* query, int ns, int nq);

#ifdef __NVCC__

		void call_shd_kernel(const SHDdata& buffer);
		void call_shd_kernel_async(const SHDdata& buffer);

#endif

	}

}

#endif
