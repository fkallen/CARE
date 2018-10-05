#ifndef MY_THRUST_CUSTOM_ALLOCATORS_HPP
#define MY_THRUST_CUSTOM_ALLOCATORS_HPP

#include <stdexcept>
#include <exception>
#include <iostream>

#ifdef __NVCC__

struct DefaultDeviceAllocator
{
	using value_type = char;
	
	DefaultDeviceAllocator() {}

	~DefaultDeviceAllocator(){}

	char* allocate(std::ptrdiff_t num_bytes){
		char* ptr = nullptr;
		cudaError_t status = cudaMalloc(&ptr, num_bytes);
		if(status != cudaSuccess){
			throw std::bad_alloc(); //("cudaMalloc: " + cudaGetErrorString(status));
		}
		return ptr;
	}

    void deallocate(char* ptr, std::size_t n){
		cudaError_t status = cudaFree(ptr);
		if(status != cudaSuccess){
			throw std::bad_alloc(); //("cudaFree: " + cudaGetErrorString(status));
		}
    }
};

struct ManagedDeviceAllocator
{
	using value_type = char;
	
	ManagedDeviceAllocator() {}

	~ManagedDeviceAllocator(){}

	char* allocate(std::ptrdiff_t num_bytes){
		char* ptr = nullptr;
		cudaError_t status = cudaMallocManaged(&ptr, num_bytes);
		if(status != cudaSuccess){
			throw std::bad_alloc(); //("cudaMallocManaged: " + cudaGetErrorString(status));
		}
		return ptr;
	}

    void deallocate(char* ptr, size_t n)
    {
		cudaError_t status = cudaFree(ptr);
		if(status != cudaSuccess){
			throw std::bad_alloc(); //("cudaFree: " + cudaGetErrorString(status));
		}
    }
};

#endif

#endif
