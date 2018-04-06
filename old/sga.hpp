#ifndef SGA_HPP
#define SGA_HPP

#include "graphtools.hpp"
#include "alignment.hpp"
#include "hpc_helpers.cuh"

namespace care{
namespace graphtools{

	//forward decl
	struct AlignerDataArrays;

	namespace alignment{

		AlignResult cpu_semi_global_alignment(const AlignerDataArrays* buffers, const char* subject, const char* query, int ns, int nq);


#ifdef __NVCC__

		struct sgaparams{
			int max_sequence_length;
			int max_ops_per_alignment;
			int sequencepitch;
			int n_queries;
			int subjectlength;
			int ALIGNMENTSCORE_MATCH = 1;
			int ALIGNMENTSCORE_SUB = -1;
			int ALIGNMENTSCORE_INS = -1;
			int ALIGNMENTSCORE_DEL = -1;
			const int* __restrict__ querylengths;
			const char* __restrict__ subjectdata;
			const char* __restrict__ queriesdata;
			AlignResultCompact* __restrict__ results;
			AlignOp* __restrict__ ops;
		};

		void call_cuda_semi_global_alignment_kernel_async(const sgaparams& buffers, cudaStream_t stream);
		void call_cuda_semi_global_alignment_kernel(const sgaparams& buffers, cudaStream_t stream);

#endif

	}

}
}
#endif
