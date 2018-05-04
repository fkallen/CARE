CXX=g++
CUDACC=nvcc
HOSTLINKER=g++

CXXFLAGS = -std=c++14
CFLAGS = -Wall -g -fopenmp 
NVCCFLAGS = -x cu -lineinfo -rdc=true --expt-extended-lambda --expt-relaxed-constexpr

#TODO CUDA_PATH =

CUDA_ARCH = -gencode=arch=compute_61,code=sm_61

LDFLAGSGPU = -lpthread -lgomp -lstdc++fs
LDFLAGSCPU = -lpthread -lgomp -lstdc++fs 

SOURCES = $(wildcard src/*.cpp)
OBJECTS_CPU = $(patsubst src/%.cpp, buildcpu/%.o, $(SOURCES))
OBJECTS_GPU = $(patsubst src/%.cpp, buildgpu/%.o, $(SOURCES))

PATH_CORRECTOR=$(shell pwd)

INC_CORRECTOR=$(PATH_CORRECTOR)/inc

GPU_VERSION = errorcorrector_gpu
CPU_VERSION = errorcorrector_cpu

all: cpu

cpu:	$(CPU_VERSION)
gpu:	$(GPU_VERSION)

$(GPU_VERSION) : $(OBJECTS_GPU)
	@echo Linking $(GPU_VERSION)
	@$(CUDACC) $(CUDA_ARCH) $(OBJECTS_GPU) $(LDFLAGSGPU) -o $(GPU_VERSION)

$(CPU_VERSION) : $(OBJECTS_CPU)
	@echo Linking $(CPU_VERSION)
	@$(HOSTLINKER) $(OBJECTS_CPU) $(LDFLAGSCPU) -o $(CPU_VERSION)

buildcpu/%.o : src/%.cpp
	@echo Compiling $< to $@
	@$(CXX) $(CXXFLAGS) $(CFLAGS) -c $< -o $@

buildgpu/%.o : src/%.cpp
	@echo Compiling $< to $@
	@$(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -c $< -o $@


clean:
	rm $(GPU_VERSION) $(CPU_VERSION) $(OBJECTS_GPU) $(OBJECTS_CPU)
