#ifndef CARE_RMM_UTILITIES_CUH
#define CARE_RMM_UTILITIES_CUH


#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <gpu/cudaerrorcheck.cuh>

#include <cassert>
#include <cstdint>

#include <cub/cub.cuh>

class CubRMMResource : public rmm::mr::device_memory_resource {
public:

    CubRMMResource(      
        cub::CachingDeviceAllocator& pool
    ) : cubAllocator(&pool){

    }
    ~CubRMMResource() override{
    }

    CubRMMResource(CubRMMResource const&) = delete;
    CubRMMResource(CubRMMResource&&)      = delete;
    CubRMMResource& operator=(CubRMMResource const&) = delete;
    CubRMMResource& operator=(CubRMMResource&&) = delete;

    [[nodiscard]] bool supports_streams() const noexcept override { return true; }

    [[nodiscard]] bool supports_get_mem_info() const noexcept override { return false; }

private:
    cub::CachingDeviceAllocator* cubAllocator;

    void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override{
        void* ptr = nullptr;

        if (bytes > 0) {
            CUDACHECK(cubAllocator->DeviceAllocate(&ptr, bytes, stream.value()));
        }

        return ptr;
    }


    void do_deallocate(void* ptr, std::size_t, rmm::cuda_stream_view) override{
        if (ptr != nullptr) {
            CUDACHECK(cubAllocator->DeviceFree(ptr));
        }
    }

    [[nodiscard]] bool do_is_equal(rmm::mr::device_memory_resource const& other) const noexcept override{
        return dynamic_cast<CubRMMResource const*>(&other) != nullptr;
    }

    [[nodiscard]] std::pair<std::size_t, std::size_t> do_get_mem_info(rmm::cuda_stream_view) const override{
        return std::make_pair(0, 0);
    }
};

class MyRMMCudaAsyncResource : public rmm::mr::device_memory_resource {
public:

    MyRMMCudaAsyncResource(      
        cudaMemPool_t pool
    ) : cuda_pool_handle_(pool){

    }

    [[nodiscard]] cudaMemPool_t pool_handle() const noexcept { return cuda_pool_handle_; }

    ~MyRMMCudaAsyncResource() override{

    }

    MyRMMCudaAsyncResource(MyRMMCudaAsyncResource const&) = delete;
    MyRMMCudaAsyncResource(MyRMMCudaAsyncResource&&)      = delete;
    MyRMMCudaAsyncResource& operator=(MyRMMCudaAsyncResource const&) = delete;
    MyRMMCudaAsyncResource& operator=(MyRMMCudaAsyncResource&&) = delete;

    [[nodiscard]] bool supports_streams() const noexcept override { return true; }

    [[nodiscard]] bool supports_get_mem_info() const noexcept override { return false; }

private:
    cudaMemPool_t cuda_pool_handle_;

    void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override{
        void* ptr = nullptr;

        if (bytes > 0) {
            CUDACHECK(cudaMallocFromPoolAsync(&ptr, bytes, pool_handle(), stream.value()));
        }

        return ptr;
    }


    void do_deallocate(void* ptr, std::size_t, rmm::cuda_stream_view stream) override{
        if (ptr != nullptr) {
            CUDACHECK(cudaFreeAsync(ptr, stream.value()));
        }
    }

    [[nodiscard]] bool do_is_equal(rmm::mr::device_memory_resource const& other) const noexcept override{
        return dynamic_cast<MyRMMCudaAsyncResource const*>(&other) != nullptr;
    }

    [[nodiscard]] std::pair<std::size_t, std::size_t> do_get_mem_info(rmm::cuda_stream_view) const override{
        return std::make_pair(0, 0);
    }
};





template<class T>
void resizeUninitialized(rmm::device_uvector<T>& vec, size_t newsize, rmm::cuda_stream_view stream){
    vec = rmm::device_uvector<T>(newsize, stream, vec.memory_resource());
}

template<class T>
void reserve(rmm::device_uvector<T>& vec, size_t newsize, rmm::cuda_stream_view stream){
    vec.resize(newsize, stream);
}

template<class T>
void clear(rmm::device_uvector<T>& vec, rmm::cuda_stream_view stream){
    vec.resize(0, stream);
}

template<class T>
void destroy(rmm::device_uvector<T>& vec, rmm::cuda_stream_view stream){
    vec.resize(0, stream);
    vec.shrink_to_fit(stream);
}

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
            stream.value()
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
            stream.value()
        ));
    }
}




#endif