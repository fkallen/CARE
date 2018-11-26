#ifndef MY_THRUST_CUSTOM_ALLOCATORS_HPP
#define MY_THRUST_CUSTOM_ALLOCATORS_HPP

#include <stdexcept>
#include <exception>
#include <iostream>

#ifdef __NVCC__

#include <thrust/device_malloc_allocator.h>

template<class T, bool allowFallback = true>
struct ThrustFallbackDeviceAllocator : thrust::device_malloc_allocator<T>
{
	using value_type = T;

	using super_t = thrust::device_malloc_allocator<T>;

	using pointer = typename super_t::pointer;
	using size_type = typename super_t::size_type;
	using reference = typename super_t::reference;
	using const_reference = typename super_t::const_reference;

	pointer allocate(size_type n){
		//std::cerr << "alloc" << std::endl;

		T* ptr = nullptr;
		cudaError_t status = cudaMalloc(&ptr, n * sizeof(T));
		if(status == cudaSuccess){
			//std::cerr << "cudaMalloc\n";
		}else{
			cudaGetLastError(); //reset the error of failed allocation

	    	if(!allowFallback){
    			throw std::bad_alloc();
    		}

	    	status = cudaMallocManaged(&ptr, n * sizeof(T));
    		if(status != cudaSuccess){
    			throw std::bad_alloc();
    		}
    		int deviceId = 0;
    		status = cudaGetDevice(&deviceId);
    		if(status != cudaSuccess){
    			throw std::bad_alloc();
    		}
    		status = cudaMemAdvise(ptr, n * sizeof(T), cudaMemAdviseSetAccessedBy, deviceId);
    		if(status != cudaSuccess){
    			throw std::bad_alloc();
    		}
			//std::cerr << "cudaMallocManaged\n";
		}
		return thrust::device_pointer_cast(ptr);
	}

    void deallocate(pointer ptr, size_type n){
    	//std::cerr << "dealloc" << std::endl;

    	cudaError_t status = cudaFree(ptr.get());
    	if(status != cudaSuccess){
    		throw std::bad_alloc();
    	}
    }
};

#endif

#endif
