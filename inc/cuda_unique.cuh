#ifndef CUDA_UNIQUE_CUH
#define CUDA_UNIQUE_CUH

#include "ganja/hpc_helpers.cuh"

//#include <cuda.h>
//#include <cuda_runtime.h>

#include <memory>


struct CudaDeleter
{
	int deviceId;

	CudaDeleter(){}

	CudaDeleter(const CudaDeleter& other){
		*this = other;
	}

	CudaDeleter(const CudaDeleter&& other){
		*this = std::move(other);
	}

	CudaDeleter(int id):deviceId(id){}

	CudaDeleter& operator=(const CudaDeleter& other){
		deviceId = other.deviceId;
		return *this;
	}

	CudaDeleter& operator=(const CudaDeleter&& other){
		deviceId = other.deviceId;
		return *this;
	}

	void operator()(void *p){
		cudaSetDevice(deviceId); CUERR;
		cudaFree(p);// CUERR;
		
        cudaError_t err;                                                       
        if ((err = cudaGetLastError()) != cudaSuccess) {                       
            std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "    
                      << __FILE__ << ", line " << __LINE__ << std::endl;       
        } 		
	}
};

struct PinnedCudaDeleter
{
	void operator()(void *p){
		cudaFreeHost(p); CUERR;
	}
};




template <class T>
using unique_dev_ptr = std::unique_ptr<T, CudaDeleter>;

template <class T>
using shared_dev_ptr = std::shared_ptr<T>;

template <class T>
using unique_pinned_ptr = std::unique_ptr<T, PinnedCudaDeleter>;

template <class T>
using shared_pinned_ptr = std::shared_ptr<T>;




template <class T>
unique_dev_ptr<T> make_unique_dev(int deviceId, size_t elements){
	cudaSetDevice(deviceId); CUERR;

	T* mem = nullptr;

	cudaMalloc(&mem, sizeof(T) * elements); CUERR;

	return unique_dev_ptr<T>(mem, CudaDeleter{deviceId});
}

template <class T>
shared_dev_ptr<T> make_shared_dev(int deviceId, size_t elements){
	cudaSetDevice(deviceId); CUERR;

	T* mem = nullptr;

	cudaMalloc(&mem, sizeof(T) * elements); CUERR;

	return shared_dev_ptr<T>(mem, CudaDeleter{deviceId});
}

template <class T>
unique_pinned_ptr<T> make_unique_pinned(size_t elements){
	T* mem = nullptr;

	cudaMallocHost(&mem, sizeof(T) * elements); CUERR;

	return unique_pinned_ptr<T>(mem, PinnedCudaDeleter{});
}

template <class T>
shared_pinned_ptr<T> make_shared_pinned(size_t elements){
	T* mem = nullptr;

	cudaMallocHost(&mem, sizeof(T) * elements); CUERR;

	return shared_pinned_ptr<T>(mem, PinnedCudaDeleter{});
}



// wrap existing device pointer into unique pointer
template <class T>
unique_dev_ptr<T> make_unique_dev(int deviceId, T* ptr){
	return std::unique_ptr<T, CudaDeleter>(ptr, CudaDeleter{deviceId});
}

// wrap existing device pointer into shared pointer
template <class T>
shared_dev_ptr<T> make_shared_dev(int deviceId, T* ptr){
	return std::shared_ptr<T>(ptr, CudaDeleter{deviceId});
}






#endif
