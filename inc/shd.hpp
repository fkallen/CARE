#ifndef SHD_HPP
#define SHD_HPP

#include "hammingtools.hpp"
#include "alignment.hpp"
#include "options.hpp"

namespace care{

namespace hammingtools{

	namespace alignment{

		AlignResultCompact cpu_shifted_hamming_distance(const GoodAlignmentProperties& prop, const char* subject, const char* query, int ns, int nq);

#ifdef __NVCC__

		struct shdparams{
            GoodAlignmentProperties props;
			int max_sequence_bytes;
			int sequencepitch;
			int n_queries;
			int subjectlength;
			const int* __restrict__ querylengths;
			const char* __restrict__ subjectdata;
			const char* __restrict__ queriesdata;
			AlignResultCompact* __restrict__ results;
		};

		void call_shd_kernel(const shdparams& buffer, int maxQueryLength, cudaStream_t stream);
		void call_shd_kernel_async(const shdparams& buffer, int maxQueryLength, cudaStream_t stream);

#endif

	}

}

}

#endif
