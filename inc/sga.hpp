#ifndef SGA_HPP
#define SGA_HPP

#include "graphtools.hpp"
#include "alignment.hpp"

namespace graphtools{

	namespace alignment{

		AlignResult cpu_semi_global_alignment(const char* subject, const char* query, int ns, int nq,
				int SCORE_EQUAL, int SCORE_SUBSTITUTE, int SCORE_INSERT, int SCORE_DELETE);


#ifdef __NVCC__
		void call_cuda_semi_global_alignment_kernel_async(const AlignerDataArrays& buffers);

		void call_cuda_semi_global_align_kernel(const AlignerDataArrays& buffers);

#endif

	}

}

#endif
