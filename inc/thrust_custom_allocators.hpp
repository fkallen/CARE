#ifndef MY_THRUST_CUSTOM_ALLOCATORS_HPP
#define MY_THRUST_CUSTOM_ALLOCATORS_HPP

#include <stdexcept>
#include <exception>
#include <iostream>
#include <vector>

#ifdef __NVCC__

struct DefaultDeviceAllocator
{
	using value_type = char;
	
	std::vector<char*> allocations;
	int nAllocations = 0;
	
	DefaultDeviceAllocator() {}

	~DefaultDeviceAllocator(){
		if(nAllocations > 0){
			std::cout << "Would have leaked " << nAllocations << " allocations from DefaultDeviceAllocator." << std::endl;
			for(auto ptr : allocations)
				cudaFree(ptr);
			
			nAllocations = 0;
		}		
	}

	char* allocate(std::ptrdiff_t num_bytes){
		//std::cout << "alloc" << std::endl;
		char* ptr = nullptr;
		cudaError_t status = cudaMalloc(&ptr, num_bytes);
		if(status != cudaSuccess){
			throw std::bad_alloc(); //("cudaMalloc: " + cudaGetErrorString(status));
		}
		allocations.emplace_back(ptr);
		++nAllocations;
		return ptr;
	}

    void deallocate(char* ptr, std::size_t n){
		//std::cout << "dealloc" << std::endl;
		cudaError_t status = cudaFree(ptr);
		if(status != cudaSuccess){
			throw std::bad_alloc(); //("cudaFree: " + cudaGetErrorString(status));
		}
		auto it = std::find(allocations.begin(), allocations.end(), ptr);
		if(it != allocations.end()){
			--nAllocations;
			allocations.erase(it);
		}
    }
};

struct ManagedDeviceAllocator
{
	using value_type = char;
	
	std::vector<char*> allocations;
	int nAllocations = 0;
	
	ManagedDeviceAllocator() {}

	~ManagedDeviceAllocator(){
		if(nAllocations > 0){
			std::cout << "Would have leaked " << nAllocations << " allocations from ManagedDeviceAllocator." << std::endl;
			for(auto ptr : allocations)
				cudaFree(ptr);
			
			nAllocations = 0;
		}
	}

	char* allocate(std::ptrdiff_t num_bytes){
		//std::cout << "alloc" << std::endl;
		char* ptr = nullptr;
		cudaError_t status = cudaMallocManaged(&ptr, num_bytes);
		if(status != cudaSuccess){
			throw std::bad_alloc(); //("cudaMallocManaged: " + cudaGetErrorString(status));
		}
		allocations.emplace_back(ptr);
		++nAllocations;
		return ptr;
	}

    void deallocate(char* ptr, size_t n)
    {
		//std::cout << "dealloc" << std::endl;
		cudaError_t status = cudaFree(ptr);
		if(status != cudaSuccess){
			throw std::bad_alloc(); //("cudaFree: " + cudaGetErrorString(status));
		}
		auto it = std::find(allocations.begin(), allocations.end(), ptr);
		if(it != allocations.end()){
			--nAllocations;
			allocations.erase(it);
		}
    }
};

#endif

#endif
