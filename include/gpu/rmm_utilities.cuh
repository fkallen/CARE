#ifndef CARE_RMM_UTILITIES_CUH
#define CARE_RMM_UTILITIES_CUH


#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <gpu/cudaerrorcheck.cuh>

#include <cassert>
#include <cstdint>

template<class T>
void erase(rmm::device_uvector<T>& vec, T* first, T* last, rmm::cuda_stream_view stream){
    auto currentend = vec.data() + vec.size();
    assert(first >= vec.data());
    assert(last <= currentend);
    assert(first <= last);


    if(last < currentend){
        const std::size_t elementsAfterRangeToErase = currentend - last;

        CUDACHECK(cudaMemcpyAsync(
            first,
            last,
            sizeof(T) * elementsAfterRangeToErase,
            D2D,
            stream
        ));

    }

    const std::size_t numErased = std::distance(first, last);
    vec.resize(vec.size() - numErased, stream);
}

template<class T>
void append(rmm::device_uvector<T>& vec, const T* rangeBegin, const T* rangeEnd, rmm::cuda_stream_view stream){
    const std::size_t rangesize = std::distance(rangeBegin, rangeEnd);
    if(rangesize > 0){
        const std::size_t oldsize = vec.size();
        vec.resize(oldsize + rangesize, stream);

        CUDACHECK(cudaMemcpyAsync(
            vec.data() + oldsize,
            rangeBegin,
            sizeof(T) * rangesize,
            cudaMemcpyDefault,
            stream
        ));
    }
}




#endif